"""Checkpoint-only diagnostics on one shared, randomly sampled validation set.

Activations and latents are cached on CPU, one model at a time. Subsets retain
the original batch boundaries; Gaussian references use identical batch sizes.
No training, normalization fitting, or dataset-wide activation scan is needed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from .config import config_from_dict
from .data import (DataSource, LEJEPA_FORMAT, _safetensors_array, eval_batches,
                   parse_entry, write_json)
from .evaluate import _Moments, _ProjectionDiagnostics, _autocast, load_checkpoint_model
from .normalization import check_normalization_matches
from .sigreg import fixed_projections, integration_grid


def keep_without_largest(scores, fraction):
    """Exact, stable top-k removal (ties broken by original sample index)."""
    k = math.ceil(len(scores) * fraction)
    if not 0 <= fraction < 1 or len(scores) - k < 2:
        raise ValueError("trim fraction must retain at least two samples")
    keep = torch.ones(len(scores), dtype=torch.bool)
    keep[torch.argsort(scores, descending=True, stable=True)[:k]] = False
    return keep


def diagnose_subset(y, reference, keep, reference_keep, *, batch_size, device,
                    sigreg_cfg, seeds, projections):
    """Reference masks must match each model batch's retained sample count."""
    t, weights = integration_grid(sigreg_cfg.t_min, sigreg_cfg.t_max, sigreg_cfg.num_points)
    moments = [_Moments(y.shape[1], device, True) for _ in range(2)]
    for start in range(0, len(y), batch_size):
        stop = start + batch_size
        for m, values, mask in zip(moments, [y, reference], [keep, reference_keep]):
            m.add(values[start:stop][mask[start:stop]].to(device))
    metrics = {"count": int(keep.sum()), "covariance": moments[0].summary(""),
               "reference_covariance": moments[1].summary(""), "projection_seeds": []}
    for seed in seeds:
        directions = fixed_projections(y.shape[1], projections, seed, "diagnostic").to(device)
        sets = [_ProjectionDiagnostics(directions, t, weights,
                sigreg_cfg.scale_by_batch_size, None, 0, w2=True) for _ in range(2)]
        sizes = []
        for start in range(0, len(y), batch_size):
            stop = start + batch_size
            sizes.append(int(keep[start:stop].sum()))
            if sizes[-1] == 0:
                continue
            assert sizes[-1] == int(reference_keep[start:stop].sum())
            for diagnostic, values, mask in zip(sets, [y, reference], [keep, reference_keep]):
                diagnostic.add(values[start:stop][mask[start:stop]].to(device))
        metrics["projection_seeds"].append({"seed": seed, "model": sets[0].summary(""),
                                             "reference": sets[1].summary("")})
    metrics["batch_sizes"] = sizes
    metrics["seed_summary"] = {}
    for kind in ["model", "reference"]:
        metrics["seed_summary"][kind] = {}
        for key in ["sigreg", "w2_sq"]:
            values = [r[kind][key] for r in metrics["projection_seeds"]]
            metrics["seed_summary"][kind][key] = {
                "mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.,
                "min": min(values), "max": max(values)}
    return metrics, {"model": moments[0].eigenvalues, "reference": moments[1].eigenvalues}


def reference_mask(reference, keep, batch_size, trim_by_norm):
    """For latent trimming, trim Gaussian norms to the same count in each batch.

    This is a size-matched norm-selection reference, not a Gaussian null for a
    model-dependent global order statistic. Input selection uses the same IDs.
    """
    if not trim_by_norm:
        return keep.clone()
    result = torch.zeros_like(keep)
    for start in range(0, len(reference), batch_size):
        stop = start + batch_size
        count = int(keep[start:stop].sum())
        order = torch.argsort(reference[start:stop].square().mean(1), stable=True)
        result[start + order[:count]] = True
    return result


class TokenContexts:
    def __init__(self, source, radius, tokenizer=None):
        self.source, self.radius, self.tokenizer = source, radius, tokenizer
        raw = json.loads(source.manifest_path.read_text(encoding="utf-8"))
        self.shards = {s["file"]: s for s in raw.get("shards", [])} if source.format == LEJEPA_FORMAT else {}
        self.arrays = {}

    def get(self, entry, sequence, position):
        path = parse_entry(entry)[0]
        if path not in self.shards:
            return {"context_status": "token context unavailable for this manifest format"}
        if path not in self.arrays:
            try:
                self.arrays[path] = _safetensors_array(self.source.root / path, "token_ids")[0]
            except ValueError as error:
                return {"context_status": str(error)}
        record = self.shards[path]["sequences"][sequence]
        offset, length = int(record["offset"]), int(record["length"])
        lo, hi = max(0, position - self.radius), min(length, position + self.radius + 1)
        ids = self.arrays[path][offset + lo:offset + hi].tolist()
        result = {"context_status": "available", "document_id": record.get("document_id"),
                  "segment_index": record.get("segment_index"), "context_start_position": lo,
                  "target_index": position - lo, "token_id": int(ids[position - lo]),
                  "context_token_ids": ids}
        if self.tokenizer is not None:
            result["context_text"] = self.tokenizer.decode(ids, skip_special_tokens=False)
            result["token_text"] = self.tokenizer.decode([result["token_id"]], skip_special_tokens=False)
        return result


def outlier_table(scores, x_scores, errors, meta, entries, contexts, limit):
    result = []
    for i in torch.argsort(scores, descending=True, stable=True)[:limit].tolist():
        entry = entries[int(meta["entry"][i])]
        sequence, position = int(meta["sequence"][i]), int(meta["position"][i])
        result.append({"sample_index": i, "entry": entry, "shard": parse_entry(entry)[0],
                       "sequence": sequence, "position": position, "score": float(scores[i]),
                       "x_sqnorm": float(x_scores[i]), "normalized_mse": float(errors[i]),
                       **contexts.get(entry, sequence, position)})
    return result


def check_source(state, source):
    recorded = state["data_manifest"]
    if any(recorded[k] != source.record()[k] for k in ["fingerprint", "splits", "burn_in_excluded"]):
        raise ValueError("checkpoint manifest, split assignment or exclusion policy differs")
    check_normalization_matches(state["normalization"], source)


def norm_summary(scores):
    levels = torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0], dtype=torch.float64)
    return {"mean": float(scores.double().mean()), "quantiles": {
        f"{q:g}": float(v) for q, v in zip(levels.tolist(), torch.quantile(scores.double(), levels))}}


@torch.no_grad()
def run(args):
    if args.batches <= 0 or args.batch_size < 2 or args.projections < 1:
        raise ValueError("use positive bounded batches/projections and batch_size >= 2")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("projection seeds must be nonempty and unique")
    if not args.trim_fractions or any(not 0 < f < 1 for f in args.trim_fractions):
        raise ValueError("trim fractions must be strictly between zero and one")
    if len(set(args.trim_fractions)) != len(args.trim_fractions):
        raise ValueError("trim fractions must be unique")
    if args.context_radius < 0 or args.top < 1:
        raise ValueError("context radius must be nonnegative and top must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("output directory must be empty; use a new directory for each diagnosis")
    device = torch.device(args.device)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    cached = meta = base_mean = base_scale = base_convention = base_dim = None
    records = []
    for index, checkpoint in enumerate(args.checkpoints):
        print(f"[{index + 1}/{len(args.checkpoints)}] {checkpoint}", flush=True)
        # Model construction consumes RNG; isolate it even in library calls.
        with torch.random.fork_rng(devices=[]):
            model, state = load_checkpoint_model(checkpoint, device)
        cfg = config_from_dict(state["config"])
        checkpoint_step = int(state["step"])
        source = DataSource(args.activation_manifest or cfg.data.activation_manifest,
                            skip_burn_in=cfg.data.skip_burn_in,
                            skip_leading_positions=cfg.data.skip_leading_positions,
                            test_split=cfg.data.test_split,
                            holdout_test_fraction=cfg.data.holdout_test_fraction)
        check_source(state, source)
        convention = [cfg.sigreg.t_min, cfg.sigreg.t_max, cfg.sigreg.num_points, cfg.sigreg.scale_by_batch_size]
        if cfg.eval.diagnostic_seed in args.seeds:
            raise ValueError("choose fresh diagnostic seeds distinct from checkpoint eval.diagnostic_seed")
        if cached is None:
            batches = list(eval_batches(source, "validation", args.batch_size, args.batches,
                                        seed=args.sample_seed, with_metadata=True))
            if not batches:
                raise ValueError("validation does not contain a full evaluation batch")
            cached = torch.cat([h.float() for h, _ in batches])
            meta = {k: torch.cat([m[k] for _, m in batches]) for k in batches[0][1]}
            del batches
            for fraction in args.trim_fractions:
                keep_without_largest(torch.zeros(len(cached)), fraction)
            base_source = source.record()
            base_mean, base_scale = model.input_mean.cpu().clone(), model.input_scale.cpu().clone()
            base_convention, base_dim = convention, cfg.model.d_latent
            ids_hash = hashlib.sha256()
            ids_hash.update(json.dumps(source.paths("validation")).encode())
            for k in sorted(meta):
                ids_hash.update(k.encode()); ids_hash.update(meta[k].numpy().tobytes())
            sample_hash = ids_hash.hexdigest()
            torch.save({**meta, "entries": source.paths("validation"), "sample_sha256": sample_hash}, output / "samples.pt")
        elif (source.record() != base_source or convention != base_convention or cfg.model.d_latent != base_dim
              or not torch.equal(base_mean, model.input_mean.cpu())
              or not torch.equal(base_scale, model.input_scale.cpu())):
            raise ValueError("comparison requires identical source, exclusion, normalization, latent dimension and SIGReg convention")
        latents, errors, x_scores = [], [], []
        for h in cached.split(args.batch_size):
            with _autocast(device, cfg.optim.amp_dtype):
                result = model(h.to(device))
            latents.append(result["y"].float().cpu())
            errors.append((result["x_hat"].float() - result["x"].float()).square().mean(1).cpu())
            x_scores.append(result["x"].float().square().mean(1).cpu())
        y, errors, x_scores = torch.cat(latents), torch.cat(errors), torch.cat(x_scores)
        del latents, result, model, state
        if not all(torch.isfinite(v).all() for v in [cached, y, errors, x_scores]):
            raise ValueError("nonfinite input, latent or reconstruction error")
        reference = torch.randn(y.shape, generator=torch.Generator().manual_seed(args.reference_seed))
        y_scores = y.square().mean(1)
        contexts = TokenContexts(source, args.context_radius, tokenizer)
        record = {"checkpoint": str(checkpoint), "step": checkpoint_step, "lambda": cfg.sigreg.weight,
                  "sample_sha256": sample_hash, "config": asdict(cfg), "subsets": {},
                  "squared_norm_per_dim": {"input": norm_summary(x_scores),
                      "latent": norm_summary(y_scores), "reference": norm_summary(reference.square().mean(1))},
                  "top_input": outlier_table(x_scores, x_scores, errors, meta, source.paths("validation"), contexts, args.top),
                  "top_latent": outlier_table(y_scores, x_scores, errors, meta, source.paths("validation"), contexts, args.top)}
        subsets = {"full": torch.ones(len(y), dtype=torch.bool)}
        for fraction in args.trim_fractions:
            for kind, scores in [("input", x_scores), ("latent", y_scores)]:
                subsets[f"trim_{kind}_{fraction:g}"] = keep_without_largest(scores, fraction)
        spectra = {}
        for name, keep in subsets.items():
            print(f"  {name}: {int(keep.sum())} samples", flush=True)
            ref_keep = reference_mask(reference, keep, args.batch_size, name.startswith("trim_latent"))
            metrics, spectrum = diagnose_subset(y, reference, keep, ref_keep, batch_size=args.batch_size,
                    device=device, sigreg_cfg=cfg.sigreg, seeds=args.seeds, projections=args.projections)
            h = cached[keep].double()
            variance = (h - h.mean(0)).square().sum()
            metrics["fvu"] = float(errors[keep].double().sum() * base_scale.double().square() * h.shape[1] / variance.clamp_min(1e-30))
            del h
            metrics["removed_indices"] = (~keep).nonzero().flatten().tolist()
            metrics["removed_input_energy_share"] = float(x_scores[~keep].double().sum() / x_scores.double().sum().clamp_min(1e-30))
            metrics["removed_latent_energy_share"] = float(y_scores[~keep].double().sum() / y_scores.double().sum().clamp_min(1e-30))
            metrics["reference_selection"] = "within-batch smallest Gaussian norms, matched counts" if name.startswith("trim_latent") else "same sample indices"
            record["subsets"][name], spectra[name] = metrics, spectrum
        stem = f"model-{index:02d}"
        torch.save(spectra, output / f"{stem}-spectra.pt")
        record["spectra_path"] = f"{stem}-spectra.pt"
        write_json(output / f"{stem}.json", record)
        records.append(record)
    manifest = {"arguments": vars(args), "source": base_source, "sample_sha256": sample_hash,
                "positions": len(cached), "notes": [
                    "Seed std is projection Monte Carlo variation, not a training-seed confidence interval.",
                    "SIGReg and W2 are means of original batches with retained rows; subset batch sizes may differ.",
                    "Input trimming shares token IDs across models; latent trimming is model-dependent.",
                    "Trimmed FVU uses each subset's centered input variance.",
                    "Context is bounded by the stored sequence, not the complete document."],
                "models": [f"model-{i:02d}.json" for i in range(len(records))]}
    write_json(output / "diagnosis.json", manifest)
    write_summary(output, records)


def write_summary(output, records):
    lines = ["# Fixed-validation checkpoint diagnostics", "",
             "FVU uses original activation space. SIGReg/W2 show mean ± sample std across projection seeds.",
             "This std is not uncertainty across training seeds. See diagnosis.json for sampling and reference semantics.", "",
             "| Model | λ | Subset | N | FVU | SIGReg | W2² | Effective rank | Gaussian rank |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for i, record in enumerate(records):
        for name, r in record["subsets"].items():
            s = r["seed_summary"]["model"]
            lines.append(f"| {i:02d} | {record['lambda']:g} | {name} | {r['count']} | {r['fvu']:.6g} | "
                         f"{s['sigreg']['mean']:.4g} ± {s['sigreg']['std']:.3g} | "
                         f"{s['w2_sq']['mean']:.4g} ± {s['w2_sq']['std']:.3g} | "
                         f"{r['covariance']['cov_effective_rank']:.1f} | {r['reference_covariance']['cov_effective_rank']:.1f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(records), figsize=(6 * len(records), 4), squeeze=False)
    for ax, record in zip(axes[0], records):
        spectra = torch.load(output / record["spectra_path"], weights_only=True)
        for name, values in spectra.items():
            ax.plot(values["model"].clamp_min(1e-12).numpy(), label=name)
        ax.plot(spectra["full"]["reference"].numpy(), "k--", label="Gaussian (full)")
        ax.set(title=f"lambda={record['lambda']:g}", xlabel="Eigenvalue index (descending)",
               ylabel="Covariance eigenvalue", yscale="log")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output / "spectra.png", dpi=160); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--activation-manifest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=31337)
    parser.add_argument("--reference-seed", type=int, default=4242)
    parser.add_argument("--seeds", nargs="+", type=int, default=[901, 902, 903])
    parser.add_argument("--projections", type=int, default=256)
    parser.add_argument("--trim-fractions", nargs="+", type=float, default=[0.001, 0.01])
    parser.add_argument("--top", type=int, default=32)
    parser.add_argument("--context-radius", type=int, default=16)
    parser.add_argument("--tokenizer", help="optional local tokenizer directory; never downloads")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
