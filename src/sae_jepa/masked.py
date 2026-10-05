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
from .covariance import (gaussian_covariance_expected_value, sample_orthonormal_sketch,
                         sketched_covariance_loss)
from .data import DataSource, eval_batches, mix_seed, write_json
from .evaluate import (_Moments, _ProjectionDiagnostics, _TopRows, write_evaluation)
from .models import model_from_checkpoint
from .normalization import check_normalization_matches
from .sigreg import epps_pulley_sigreg, fixed_projections, gaussian_expected_value, integration_grid
from .train import Trainer, autocast_context, resolve_device, rms

ARCHITECTURE_ID = "masked_sigreg_encoder_v1"
OUTPUT_INIT_SEED_OFFSET = 3_000_017
OUTPUT_INIT_BATCHES = 8
READOUT_SEED_OFFSET = 5_000_011


def mask_coordinates(x, probability, generator):
    """Dedicated CPU RNG keeps masks independent of model/data/projection RNGs."""
    keep = torch.rand(x.shape, generator=generator) >= probability
    return x * keep.to(device=x.device, dtype=x.dtype)


@torch.no_grad()
def calibrate_output_layer(model, source, cfg, device):
    """Rescale and center the encoder's last Linear so y has the target variance.

    Uses a fixed sample of train rows drawn with its own seed, so the training
    data order, the model RNG and all other generators are untouched.  The new
    output is ``gain * (y - mean(y))``: the spectrum shape of the default
    initialization is kept, only its overall scale and mean change.
    """
    target = cfg.model.init_output_variance
    rows = torch.cat(list(eval_batches(
        source, "train", cfg.eval.batch_size, OUTPUT_INIT_BATCHES,
        seed=mix_seed(cfg.train.seed, OUTPUT_INIT_SEED_OFFSET), allow_train=True)))
    y = model.encode_dense(rows.to(device)).double()
    mean = y.mean(0)
    variance = float(y.var(0, unbiased=False).mean())
    if not variance > 0:
        raise ValueError("cannot calibrate the output layer: initial output has zero variance")
    gain = math.sqrt(target / variance)
    last = model.encoder[-1]
    last.weight.mul_(gain)
    last.bias.copy_((gain * (last.bias.double() - mean)).to(last.bias.dtype))
    return {"target_variance": target, "initial_variance_mean": variance,
            "initial_mean_sq_per_dim": float(mean.square().mean()), "gain": gain,
            "samples": len(rows), "centered": True,
            "sampling": "fixed train sample, dedicated seed; training order unchanged"}


def _readout_rows(source, split, cfg, batches, seed):
    return eval_batches(source, split, cfg.eval.batch_size, batches, seed=seed,
                        allow_train=split == "train")


@torch.no_grad()
def fit_linear_readout(model, source, cfg, device, *, batches, ridge, seed):
    """Closed-form ridge readout y -> scalar-normalized x on a fixed train sample.

    The encoder is frozen and run exactly as stage 2 runs it (same autocast).
    ``W = (Cov_yy + ridge * tr(Cov_yy)/d I)^-1 Cov_yx``, ``b = mean_x - mean_y W``,
    accumulated in float64.  FVU is invariant to the scalar normalization, so
    the reported values equal original-space FVU.
    """
    d, d_in = cfg.model.d_latent, cfg.model.d_in
    options = {"dtype": torch.float64, "device": device}
    n, sum_xx = 0, 0.0
    sum_y, sum_x = torch.zeros(d, **options), torch.zeros(d_in, **options)
    yy, yx = torch.zeros(d, d, **options), torch.zeros(d, d_in, **options)
    for h in _readout_rows(source, "train", cfg, batches, seed):
        h = h.to(device)
        x = ((h.float() - model.input_mean) / model.input_scale).double()
        with autocast_context(device, cfg.optim.amp_dtype):
            y = model.encode_dense(h)
        y = y.double()
        n += len(h)
        sum_y += y.sum(0)
        sum_x += x.sum(0)
        sum_xx += float(x.square().sum())
        yy += y.T @ y
        yx += y.T @ x
    if n <= d:
        raise ValueError("readout fit needs more train samples than latent dimensions")
    mean_y, mean_x = sum_y / n, sum_x / n
    cov_yy = yy / n - torch.outer(mean_y, mean_y)
    cov_yx = yx / n - torch.outer(mean_y, mean_x)
    penalty = ridge * torch.trace(cov_yy) / d
    weight = torch.linalg.solve(cov_yy + penalty * torch.eye(d, **options), cov_yx)
    bias = mean_x - mean_y @ weight
    total = sum_xx / n - float(mean_x.square().sum())
    explained = float(2 * (weight * cov_yx).sum() - (weight * (cov_yy @ weight)).sum())
    readout = model.attach_readout()
    readout.weight.copy_(weight.T.float())
    readout.bias.copy_(bias.float())
    return {"form": "linear; y -> (h - mean)/scale", "solver": "closed-form ridge, float64",
            "ridge": ridge, "ridge_penalty": float(penalty), "samples": n, "split": "train",
            "seed": seed, "batch_size": cfg.eval.batch_size, "batches": batches,
            "amp_dtype": cfg.optim.amp_dtype, "train_fvu": (total - explained) / total}


@torch.no_grad()
def readout_fvu(model, source, cfg, device, split, batches, seed):
    """Original-space FVU of encoder + readout on fixed held-out batches."""
    n, sse, sum_xx, sum_x = 0, 0.0, 0.0, None
    for h in _readout_rows(source, split, cfg, batches, seed):
        h = h.to(device)
        x = ((h.float() - model.input_mean) / model.input_scale).double()
        with autocast_context(device, cfg.optim.amp_dtype):
            x_hat = model.decode_normalized(model.encode_dense(h))
        sse += float((x_hat.double() - x).square().sum())
        sum_x = x.sum(0) if sum_x is None else sum_x + x.sum(0)
        sum_xx += float(x.square().sum())
        n += len(h)
    if not n:
        raise ValueError(f"no full batch for {split}")
    return sse / (sum_xx - float(sum_x.square().sum()) / n)


class MaskedTrainer(Trainer):
    def __init__(self, cfg):
        if cfg.model.type != "masked_sigreg_encoder":
            raise ValueError("sj-masked requires model.type=masked_sigreg_encoder")
        super().__init__(cfg)
        self.input_whitening = None
        if cfg.data.input_whitening_path:
            from .input_whitening import install, validate
            self.input_whitening = torch.load(cfg.data.input_whitening_path, map_location='cpu', weights_only=False)
            validate(self.input_whitening, self.source, self.normalization)
            install(self.model, self.input_whitening)
        self.output_init = None
        if cfg.model.init_output_variance > 0:
            # After whitening: the calibration must see the actual encoder input.
            self.output_init = calibrate_output_layer(self.model, self.source, cfg, self.device)
        self.mask_generator = torch.Generator().manual_seed(
            mix_seed(cfg.train.seed, cfg.masking.seed_offset)
        )
        self.covariance_generator = None
        if cfg.covariance.weight > 0:
            self.covariance_generator = torch.Generator().manual_seed(
                mix_seed(cfg.train.seed, cfg.covariance.seed_offset))
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
        self.convention['input_transform'] = 'zca' if self.input_whitening is not None else 'scalar'
        if self.input_whitening is not None:
            self.convention['input_whitening'] = {k: self.input_whitening[k] for k in
                ('epsilon', 'count', 'sampling', 'convention')}
        if self.output_init is not None:
            self.convention['output_init'] = self.output_init
        if self.covariance_generator is not None:
            self.convention["objective"] += " + beta * mean_view(CovLoss(y_view))"
            self.convention["covariance"] = {
                "formula": ("mean((Cov(y @ R) - I_k)^2); centered; denominator B-1"
                            if cfg.covariance.estimator == "plugin" else
                            "mean((Cov(y[:B//2] @ R) - I_k) * (Cov(y[B//2:] @ R) - I_k)); "
                            "each half centered; denominator n-1"),
                "estimator": cfg.covariance.estimator,
                "weight": cfg.covariance.weight, "sketch_dim": cfg.covariance.sketch_dim,
                "projection": "orthonormal columns; fresh each step; shared between views",
                "precision": "float32; autocast disabled",
                "gaussian_expected_value": gaussian_covariance_expected_value(
                    cfg.optim.batch_size, cfg.covariance.sketch_dim, cfg.covariance.estimator),
            }

    def covariance_term(self, views, diagnostics):
        """One independent sketch per step, separate centering for each view."""
        projection = sample_orthonormal_sketch(
            self.cfg.model.d_latent, self.cfg.covariance.sketch_dim,
            self.covariance_generator, self.device)
        terms = [sketched_covariance_loss(v, projection, self.cfg.covariance.estimator)
                 for v in views]
        covariance = sum(terms) / len(terms)
        weighted = self.cfg.covariance.weight * covariance
        metrics = {}
        if diagnostics:
            metrics = {"covariance": float(covariance.detach()),
                       "covariance_weighted": float(weighted.detach())}
            gradients = torch.autograd.grad(weighted, views, retain_graph=True)
            for name, term, gradient in zip(("full", "masked"), terms, gradients):
                metrics[f"covariance_{name}"] = float(term.detach())
                metrics[f"grad_rms_y/{name}/covariance_weighted"] = rms(gradient)
        return weighted, metrics

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
        metrics = {}
        if self.covariance_generator is not None:
            cov_weighted, metrics = self.covariance_term((y, masked), diagnostics)
            loss = loss + cov_weighted
        if not torch.isfinite(loss):
            raise ValueError("nonfinite masked consistency loss")
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
        weighted = self.cfg.sigreg.weight * sigreg
        loss = weighted
        metrics = {}
        if self.covariance_generator is not None:
            cov_weighted, metrics = self.covariance_term((y,), diagnostics)
            loss = loss + cov_weighted
        if not torch.isfinite(loss):
            raise ValueError("nonfinite SIGReg-only loss")
        if diagnostics:
            gradient = torch.autograd.grad(weighted, y, retain_graph=True)[0]
            yf = y.detach().float()
            metrics.update({"loss": float(loss.detach()), "sigreg": float(sigreg.detach()),
                       "sigreg_full": float(sigreg.detach()), "sigreg_weighted": float(weighted.detach()),
                       "grad_rms_y/full/sigreg_weighted": rms(gradient),
                       "full/mean_sq_per_dim": float(yf.mean(0).square().mean()),
                       "full/variance_mean": float(yf.var(0, unbiased=False).mean())})
        return loss, metrics

    def checkpoint_state(self):
        state = super().checkpoint_state()
        state["architecture_id"] = ARCHITECTURE_ID
        state["rng"]["mask"] = self.mask_generator.get_state()
        if self.covariance_generator is not None:
            state["rng"]["covariance"] = self.covariance_generator.get_state()
        if self.input_whitening is not None:
            state['input_whitening'] = self.input_whitening
        return state

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("architecture_id") != ARCHITECTURE_ID or "mask" not in state.get("rng", {}):
            raise ValueError("not a resumable masked encoder checkpoint")
        saved = state.get('input_whitening')
        if (saved is None) != (self.input_whitening is None):
            raise ValueError('cannot resume: input whitening changed')
        if saved is not None:
            for key in ('matrix', 'center', 'mean'):
                if not torch.equal(saved[key], self.input_whitening[key]):
                    raise ValueError('cannot resume: input whitening statistics changed')
            if saved['epsilon'] != self.input_whitening['epsilon']:
                raise ValueError('cannot resume: input whitening epsilon changed')
        if (self.covariance_generator is not None and
                state.get("config", {}).get("covariance", {}).get("weight", 0) > 0 and
                "covariance" not in state.get("rng", {})):
            raise ValueError("covariance RNG state missing from checkpoint")
        super().load_checkpoint(path)
        self.mask_generator.set_state(state["rng"]["mask"])
        if self.covariance_generator is not None:
            self.covariance_generator.set_state(state["rng"]["covariance"])

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
    cov_projection = None
    cov_totals = {name: 0.0 for name in names}
    if cfg.covariance.weight > 0:
        cov_projection = sample_orthonormal_sketch(
            d, cfg.covariance.sketch_dim,
            torch.Generator().manual_seed(cfg.covariance.validation_seed), device)
    # Training-time validation tracks output rank too; the reference spectrum is fixed.
    moments = {name: _Moments(d, device, covariance=detailed or (
        ec.training_covariance and name != "reference")) for name in names}
    tests = {name: {key: _ProjectionDiagnostics(
        a, t, weights, sc.scale_by_batch_size,
        tuple(ec.quantiles) if key == "diagnostic_" else None,
        ec.maximum_quantile_samples if key == "diagnostic_" else 0,
        w2=key == "diagnostic_") for key, a in projections.items()} for name in names}
    tops = {name: _TopRows(ec.outlier_buffer, d, device) for name in names} if detailed else {}
    norms = {name: [] for name in names}
    input_moments = _Moments(cfg.model.d_in, device, covariance=True) if detailed and ec.input_diagnostics else None
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
        if input_moments is not None:
            input_moments.add(full['x'].float())
        values = {"gaussian": full["y"].float(),
                  "reference": torch.randn(len(h), d, generator=reference).to(device)}
        if cfg.masking.enabled:
            values["masked"] = masked.float()
            consistency_sse += float((values["gaussian"].double() - values["masked"].double()).square().sum())
        for name, v in values.items():
            if not torch.isfinite(v).all():
                raise ValueError("nonfinite evaluation representation")
            moments[name].add(v)
            if cov_projection is not None:
                cov_totals[name] += float(sketched_covariance_loss(
                    v, cov_projection, cfg.covariance.estimator))
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
    if cov_projection is not None:
        result.update({"covariance_weight": cfg.covariance.weight,
                       "covariance_sketch_dim": cfg.covariance.sketch_dim,
                       "covariance_estimator": cfg.covariance.estimator,
                       "covariance_validation_seed": cfg.covariance.validation_seed,
                       "covariance_gaussian_expected_value": gaussian_covariance_expected_value(
                           ec.batch_size, cfg.covariance.sketch_dim, cfg.covariance.estimator)})
        for name in names:
            result[f"{name}/sketched_covariance"] = cov_totals[name] / count
    result['input_transform'] = 'zca' if hasattr(model, 'whitening_matrix') else 'scalar'
    if hasattr(model, 'whitening_matrix'):
        result['input_whitening_epsilon'] = model.whitening_epsilon
        result['input_whitening_fit_count'] = model.whitening_count
    if input_moments is not None:
        result.update(input_moments.summary('encoder_input/'))
        spectra['encoder_input'] = input_moments.eigenvalues
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
                 "| run | objective | mask | lambda | beta | sketch k | consistency | SIGReg full | masked | ref | effective rank full | masked | ref |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
        for path, row in subset:
            label = f"{row.get('run_name', path.parent.name)} step={row.get('step', '?')}"
            keys = ["mask_probability", "sigreg_weight", "covariance_weight", "covariance_sketch_dim",
                    "consistency/mse", "gaussian/heldout_sigreg",
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
    readout = sub.add_parser("fit-readout", help="fit a frozen-encoder linear readout for stage 2")
    readout.add_argument("--checkpoint", required=True)
    readout.add_argument("--output", required=True, help="new checkpoint with the readout added")
    readout.add_argument("--batches", type=int, default=512, help="train batches of eval.batch_size")
    readout.add_argument("--ridge", type=float, default=1e-4, help="relative to mean latent variance")
    readout.add_argument("--device", default="cuda")
    readout.add_argument("--activation-manifest")
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
    for override in getattr(args, "set", []):
        values = parse_override(override)
        if set(values) != {"eval"}:
            raise ValueError("only eval.* overrides are allowed")
        update_config(cfg, values)
    cfg.validate()
    device = resolve_device(args.device)
    whitening = state.get('input_whitening')
    model = model_from_checkpoint(state)
    source = DataSource(args.activation_manifest or cfg.data.activation_manifest,
                        skip_burn_in=cfg.data.skip_burn_in,
                        skip_leading_positions=cfg.data.skip_leading_positions,
                        test_split=cfg.data.test_split,
                        holdout_test_fraction=cfg.data.holdout_test_fraction)
    if source.fingerprint != state["data_manifest"]["fingerprint"] or source.splits != state["data_manifest"]["splits"]:
        raise ValueError("evaluation data differs from training data/splits")
    check_normalization_matches(state["normalization"], source)
    if whitening is not None:
        from .input_whitening import validate
        validate(whitening, source, state['normalization'])
    if args.command == "fit-readout":
        if "readout" in state:
            raise ValueError("checkpoint already has a readout; start from the training checkpoint")
        output = Path(args.output)
        if output.exists():
            raise ValueError(f"{output} exists; choose a new path")
        model = model.to(device).eval()
        seed = mix_seed(cfg.train.seed, READOUT_SEED_OFFSET)
        info = fit_linear_readout(model, source, cfg, device, batches=args.batches,
                                  ridge=args.ridge, seed=seed)
        info["validation_fvu"] = readout_fvu(model, source, cfg, device, "validation",
                                             cfg.eval.batches, cfg.eval.sample_seed)
        info["source_checkpoint"] = str(args.checkpoint)
        # Frozen artifact for stage 2: no optimizer state, so it cannot be resumed by mistake.
        state = {k: v for k, v in state.items() if k not in {"optimizer", "scheduler"}}
        state["model"] = {k: v.cpu() for k, v in model.state_dict().items()}
        state["readout"] = info
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, output)
        write_json(output.with_suffix(".json"), info)
        print(f"readout: train FVU={info['train_fvu']:.5f} validation FVU={info['validation_fvu']:.5f}")
        return
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
            "amp_dtype": cfg.optim.amp_dtype, "input_transform": results['input_transform']})
    write_evaluation(results, output)
    print(f"{args.split}: objective={results['objective']}, "
          f"SIGReg full={results['gaussian/heldout_sigreg']:.3f}")


if __name__ == "__main__":
    main()
