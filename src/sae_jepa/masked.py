"""Stage 1: shared full/masked encoder consistency plus two-view SIGReg.

Masks are independent Bernoulli draws per token/coordinate, applied AFTER
train-only scalar normalization. Hidden coordinates become zero (train mean),
without inverse-keep-probability scaling. Both branches receive gradients.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import config_from_dict, load_config, parse_override, update_config
from .data import DataSource, eval_batches, mix_seed
from .evaluate import (_Moments, _ProjectionDiagnostics, _TopRows, write_evaluation)
from .models import build_model
from .normalization import check_normalization_matches
from .sigreg import epps_pulley_sigreg, fixed_projections, gaussian_expected_value, integration_grid
from .train import Trainer, autocast_context, resolve_device, rms

ARCHITECTURE_ID = "masked_sigreg_encoder_v1"


def mask_coordinates(x, probability, generator):
    """Dedicated CPU RNG keeps masks independent of model/data/projection RNGs."""
    keep = torch.rand(x.shape, generator=generator) >= probability
    return x * keep.to(device=x.device, dtype=x.dtype)


class MaskedTrainer(Trainer):
    def __init__(self, cfg):
        if cfg.model.type != "masked_sigreg_encoder":
            raise ValueError("sj-masked requires model.type=masked_sigreg_encoder")
        super().__init__(cfg)
        self.mask_generator = torch.Generator().manual_seed(
            mix_seed(cfg.train.seed, cfg.masking.seed_offset)
        )
        self.convention.update({
            "objective": "MSE(y_mask,y_full) + lambda * (SIGReg(y_full)+SIGReg(y_mask))/2",
            "views": "shared encoder; gradients through both branches; shared projections per step",
            "masking": "independent Bernoulli per token/coordinate; normalized zeros; no rescaling",
            "mask_probability": cfg.masking.probability,
        })
        if not cfg.masking.enabled:
            self.convention.update({"objective": "lambda * SIGReg(y_full)",
                                    "views": "full input only; no consistency loss",
                                    "masking": "disabled", "mask_probability": 0.0})

    def loss(self, h, diagnostics):
        if not self.cfg.masking.enabled:
            return self.sigreg_only_loss(h, diagnostics)
        with autocast_context(self.device, self.amp_dtype):
            full = self.model(h)
            masked = self.model.encode_normalized(mask_coordinates(
                full["x"], self.cfg.masking.probability, self.mask_generator))
        y = full["y"]
        consistency = F.mse_loss(masked.float(), y.float())
        # Match the official multi-view reduction: per-view ECF over N samples,
        # same random directions, then mean the view losses (never pool 2N).
        projections = self.sigreg.projections()
        terms = [epps_pulley_sigreg(v, projections, self.sigreg.t, self.sigreg.weights,
                                   self.sigreg.scale_by_batch_size) for v in (y, masked)]
        sigreg = (terms[0] + terms[1]) / 2
        weighted = self.cfg.sigreg.weight * sigreg
        loss = consistency + weighted
        if not torch.isfinite(loss):
            raise ValueError("nonfinite masked consistency loss")
        metrics = {}
        if diagnostics:
            for name, term in [("consistency", consistency), ("sigreg_weighted", weighted)]:
                gradients = torch.autograd.grad(term, (y, masked), retain_graph=True)
                for view, gradient in zip(("full", "masked"), gradients):
                    metrics[f"grad_rms_y/{view}/{name}"] = rms(gradient)
            metrics.update({"loss": float(loss.detach()),
                            "consistency_mse": float(consistency.detach()),
                            "sigreg": float(sigreg.detach()),
                            "sigreg_full": float(terms[0].detach()),
                            "sigreg_masked": float(terms[1].detach()),
                            "sigreg_weighted": float(weighted.detach())})
            for name, v in [("full", y), ("masked", masked)]:
                vf = v.detach().float()
                metrics[f"{name}/mean_sq_per_dim"] = float(vf.mean(0).square().mean())
                metrics[f"{name}/variance_mean"] = float(vf.var(0, unbiased=False).mean())
        return loss, metrics

    def sigreg_only_loss(self, h, diagnostics):
        with autocast_context(self.device, self.amp_dtype):
            y = self.model.encode_dense(h)
        sigreg = self.sigreg(y)
        loss = self.cfg.sigreg.weight * sigreg
        if not torch.isfinite(loss):
            raise ValueError("nonfinite SIGReg-only loss")
        metrics = {}
        if diagnostics:
            gradient = torch.autograd.grad(loss, y, retain_graph=True)[0]
            yf = y.detach().float()
            metrics = {"loss": float(loss.detach()), "sigreg": float(sigreg.detach()),
                       "sigreg_full": float(sigreg.detach()), "sigreg_weighted": float(loss.detach()),
                       "grad_rms_y/full/sigreg_weighted": rms(gradient),
                       "full/mean_sq_per_dim": float(yf.mean(0).square().mean()),
                       "full/variance_mean": float(yf.var(0, unbiased=False).mean())}
        return loss, metrics

    def checkpoint_state(self):
        state = super().checkpoint_state()
        state["architecture_id"] = ARCHITECTURE_ID
        state["rng"]["mask"] = self.mask_generator.get_state()
        return state

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("architecture_id") != ARCHITECTURE_ID or "mask" not in state.get("rng", {}):
            raise ValueError("not a resumable masked encoder checkpoint")
        super().load_checkpoint(path)
        self.mask_generator.set_state(state["rng"]["mask"])

    def validate(self, detailed=False, split="validation"):
        batches = self.cfg.eval.batches if detailed else self.cfg.train.validation_batches
        return evaluate_masked(self.model, self.source, self.cfg, self.device,
                               split=split, detailed=detailed, batches=batches)


@torch.no_grad()
def evaluate_masked(model, source, cfg, device, *, split="validation", detailed=True, batches=None,
                    norm_diagnostics=None):
    """Fixed evaluation masks/directions, full and masked covariance, no fake FVU."""
    was_training = model.training
    model.eval()
    try:
        return _evaluate(model, source, cfg, device, split, detailed, batches, norm_diagnostics)
    finally:
        model.train(was_training)


def _evaluate(model, source, cfg, device, split, detailed, batches, norm_diagnostics):
    ec, sc, d = cfg.eval, cfg.sigreg, cfg.model.d_latent
    masks = torch.Generator().manual_seed(cfg.masking.validation_seed)
    reference = torch.Generator().manual_seed(ec.gaussian_reference_seed)
    t, weights = integration_grid(sc.t_min, sc.t_max, sc.num_points)
    projections = {"heldout_": fixed_projections(d, sc.validation_projections,
                                                sc.validation_seed, "validation").to(device)}
    if detailed:
        projections["diagnostic_"] = fixed_projections(d, ec.diagnostic_projections,
                                                      ec.diagnostic_seed, "diagnostic").to(device)
    names = ("gaussian", "masked", "reference") if cfg.masking.enabled else ("gaussian", "reference")
    moments = {name: _Moments(d, device, covariance=detailed) for name in names}
    tests = {name: {key: _ProjectionDiagnostics(
        a, t, weights, sc.scale_by_batch_size,
        tuple(ec.quantiles) if key == "diagnostic_" else None,
        ec.maximum_quantile_samples if key == "diagnostic_" else 0,
        w2=key == "diagnostic_") for key, a in projections.items()} for name in names}
    tops = {name: _TopRows(ec.outlier_buffer, d, device) for name in names} if detailed else {}
    norms = {name: [] for name in names}
    n, count, consistency_sse = 0, 0, 0.0
    for item in eval_batches(source, split, ec.batch_size,
                            ec.batches if batches is None else batches, seed=ec.sample_seed,
                            with_metadata=norm_diagnostics is not None):
        h, metadata = item if norm_diagnostics is not None else (item, None)
        with autocast_context(device, cfg.optim.amp_dtype):
            full = model(h.to(device))
            if cfg.masking.enabled:
                masked_x = mask_coordinates(full["x"], cfg.masking.probability, masks)
                masked = model.encode_normalized(masked_x)
        if norm_diagnostics is not None:
            norm_diagnostics.add(h, full["x"], full["y"], metadata, source.paths(split),
                                 masked_x=masked_x if cfg.masking.enabled else None,
                                 masked_y=masked if cfg.masking.enabled else None)
        values = {"gaussian": full["y"].float(),
                  "reference": torch.randn(len(h), d, generator=reference).to(device)}
        if cfg.masking.enabled:
            values["masked"] = masked.float()
            consistency_sse += float((values["gaussian"].double() - values["masked"].double()).square().sum())
        for name, v in values.items():
            if not torch.isfinite(v).all():
                raise ValueError("nonfinite evaluation representation")
            moments[name].add(v)
            for test in tests[name].values():
                test.add(v)
            if detailed:
                sq = v.square().mean(1)
                norms[name].append(sq.cpu())
                tops[name].add(v, sq, torch.arange(n, n + len(h), device=device))
        n += len(h)
        count += 1
    if n == 0:
        raise ValueError(f"split {split!r} did not yield a full evaluation batch")
    result = {"objective": "masked_consistency", "split": split, "positions": n,
              "batches": count, "batch_size": ec.batch_size, "detailed": detailed,
              "mask_probability": cfg.masking.probability,
              "mask_validation_seed": cfg.masking.validation_seed,
              "consistency/mse": consistency_sse / (n * d),
              "sigreg_gaussian_expected_value": gaussian_expected_value(t, weights)}
    if not cfg.masking.enabled:
        result.update(objective="sigreg_only", mask_probability=0.0)
        del result["consistency/mse"]
        del result["mask_validation_seed"]
    spectra = {}
    for name in names:
        result.update(moments[name].summary(f"{name}/"))
        for key, test in tests[name].items():
            result.update(test.summary(f"{name}/{key}"))
        if detailed:
            spectra[name] = moments[name].eigenvalues
            sq = torch.cat(norms[name]).double()
            for q in ec.norm_quantiles:
                result[f"{name}/y_sqnorm_q{q:g}"] = float(torch.quantile(sq, q))
            k = min(math.ceil(n * ec.outlier_fraction), ec.outlier_buffer, n - 1)
            if k > 0:
                trimmed = moments[name].without(tops[name].top(k)[0])
                result.update(trimmed.summary(f"{name}/trimmed/"))
                spectra[f"{name}_trimmed"] = trimmed.eigenvalues
            result[f"{name}/trimmed/count"] = k
    if cfg.masking.enabled:
        result["sigreg/two_view_mean"] = (
            result["gaussian/heldout_sigreg"] + result["masked/heldout_sigreg"]) / 2
    if detailed:
        result["_spectra"] = spectra
    return result


def report(run_root):
    """Compare consistency and isotropy, without ranking by SIGReg alone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(run_root)
    rows = []
    for path in sorted(root.rglob("eval-*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("objective") in {"masked_consistency", "sigreg_only"}:
            rows.append((path, row))
    if not rows:
        raise ValueError("no masked evaluations found")
    output = root / "report"
    output.mkdir(exist_ok=True)
    for split in sorted({r["split"] for _, r in rows}):
        subset = [(p, r) for p, r in rows if r["split"] == split]
        lines = [f"# Masked stage 1: {split}", "",
                 "Full input is the downstream representation. Compare both views against the finite-sample Gaussian reference. No reconstruction decoder/FVU is available.", "",
                 "| run | objective | mask | lambda | consistency | SIGReg full | masked | ref | effective rank full | masked | ref |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
        for path, row in subset:
            label = f"{row.get('run_name', path.parent.name)} step={row.get('step', '?')}"
            keys = ["mask_probability", "sigreg_weight", "consistency/mse", "gaussian/heldout_sigreg",
                    "masked/heldout_sigreg", "reference/heldout_sigreg", "gaussian/cov_effective_rank",
                    "masked/cov_effective_rank", "reference/cov_effective_rank"]
            lines.append("| " + label + " | " + row["objective"] + " | " + " | ".join(f"{row[k]:.6g}" if k in row else "" for k in keys) + " |")
            if "spectra_path" in row:
                spectra = torch.load(path.parent / row["spectra_path"], map_location="cpu", weights_only=True)
                for ax, name in zip(axes[0], ("gaussian", "masked")):
                    if name not in spectra:
                        continue
                    values = spectra[name]
                    ax.loglog(range(1, len(values) + 1), values.clamp_min(1e-10), label=label)
                    if len(ax.lines) == 1:
                        ref = spectra["reference"]
                        ax.loglog(range(1, len(ref) + 1), ref.clamp_min(1e-10), color="gray", label="Gaussian reference")
        for ax, title in zip(axes[0], ("Full input", "Masked input")):
            ax.set(title=title, xlabel="Eigenvalue index", ylabel="Covariance eigenvalue")
            if ax.lines:
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"{split}_spectra.png", dpi=140)
        plt.close(fig)
        (output / f"{split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (output / f"{split}.json").write_text(json.dumps([r for _, r in subset], indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("report")
    summary.add_argument("--run-root", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--set", action="append", default=[])
    train.add_argument("--resume", default="auto")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--activation-manifest")
    evaluate.add_argument("--output")
    evaluate.add_argument("--norm-diagnostics", action="store_true",
                          help="write paired raw/normalized input and output norms per sample")
    evaluate.add_argument("--norm-outlier-fraction", type=float, default=.01,
                          help="top output-norm fraction for paired diagnostics (default .01)")
    evaluate.add_argument("--set", action="append", default=[], help="eval.* overrides only")
    args = parser.parse_args(argv)
    if args.command == "report":
        report(args.run_root)
        return
    if args.command == "train":
        cfg = load_config(args.config, args.set)
        trainer = MaskedTrainer(cfg)
        latest = Path(cfg.train.output_dir) / "checkpoints/latest.pt"
        if args.resume == "auto" and latest.exists():
            trainer.load_checkpoint(latest)
        elif args.resume not in {"auto", "none"}:
            trainer.load_checkpoint(args.resume)
        elif args.resume == "none" and latest.exists():
            raise ValueError("checkpoint already exists; use a new output directory")
        trainer.run()
        return
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("not a masked encoder checkpoint")
    cfg = config_from_dict(state["config"])
    for override in args.set:
        values = parse_override(override)
        if set(values) != {"eval"}:
            raise ValueError("only eval.* overrides are allowed")
        update_config(cfg, values)
    cfg.validate()
    device = resolve_device(args.device)
    with torch.random.fork_rng(devices=[]):
        model = build_model(cfg.model, state["normalization"]["mean"], state["normalization"]["scale"])
    model.load_state_dict(state["model"])
    source = DataSource(args.activation_manifest or cfg.data.activation_manifest,
                        skip_burn_in=cfg.data.skip_burn_in,
                        skip_leading_positions=cfg.data.skip_leading_positions,
                        test_split=cfg.data.test_split,
                        holdout_test_fraction=cfg.data.holdout_test_fraction)
    if source.fingerprint != state["data_manifest"]["fingerprint"] or source.splits != state["data_manifest"]["splits"]:
        raise ValueError("evaluation data differs from training data/splits")
    check_normalization_matches(state["normalization"], source)
    from .norm_diagnostics import NormDiagnostics
    diagnostic = NormDiagnostics(args.norm_outlier_fraction) if args.norm_diagnostics else None
    results = evaluate_masked(model.to(device), source, cfg, device, split=args.split,
                              norm_diagnostics=diagnostic)
    results.update({"checkpoint": args.checkpoint, "step": state["step"], "seed": cfg.train.seed,
                    "sigreg_weight": cfg.sigreg.weight, "run_name": cfg.name})
    output = Path(args.output) if args.output else Path(args.checkpoint).parent.parent / f"eval-{args.split}-step-{state['step']:07d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if diagnostic is not None:
        results["norm_diagnostics"] = diagnostic.write(output, {
            "checkpoint": str(args.checkpoint), "step": state["step"], "split": args.split,
            "activation_manifest": str(args.activation_manifest or cfg.data.activation_manifest),
            "data_fingerprint": source.fingerprint, "sample_seed": cfg.eval.sample_seed,
            "batch_size": cfg.eval.batch_size, "batches": results["batches"],
            "masking_enabled": cfg.masking.enabled, "mask_probability": results["mask_probability"],
            "mask_validation_seed": cfg.masking.validation_seed if cfg.masking.enabled else None,
            "amp_dtype": cfg.optim.amp_dtype})
    write_evaluation(results, output)
    print(f"{args.split}: objective={results['objective']}, "
          f"SIGReg full={results['gaussian/heldout_sigreg']:.3f}")


if __name__ == "__main__":
    main()
