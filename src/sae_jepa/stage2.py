"""Frozen dense front-end -> Top-K SAE pilot training, evaluation and comparison."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import yaml

from .config import _update, config_from_dict, parse_override
from .data import DataSource, TrainBatches, eval_batches, write_json
from .evaluate import _autocast, load_checkpoint_model
from .models import build_model
from .normalization import check_normalization_matches
from .topk import TopKSAE
from .train import lr_multiplier

FORMAT = "sae-jepa-topk-checkpoint-v1"


@dataclass
class Stage2Config:
    dictionary_size: int = 16384
    k: int = 64
    seed: int = 42
    steps: int = 10000
    batch_size: int = 512
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 500
    decay_fraction: float = 0.2
    gradient_clip: float = 1.0
    amp_dtype: str = "bfloat16"
    shards_per_window: int = 4
    calibration_batches: int = 64
    calibration_seed: int = 91001
    eval_batch_size: int = 512
    eval_batches: int = 64
    eval_seed: int = 31337
    validation_every: int = 1000
    validation_batches: int = 16
    log_every: int = 100
    checkpoint_every: int = 1000
    required_splits: tuple[str, ...] = ("validation",)

    def validate(self):
        for key in ["dictionary_size", "k", "steps", "batch_size", "shards_per_window",
                    "calibration_batches", "eval_batch_size", "eval_batches", "validation_batches",
                    "log_every", "checkpoint_every"]:
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.k > self.dictionary_size:
            raise ValueError("k must not exceed dictionary_size")
        if self.warmup_steps < 0 or self.validation_every < 0:
            raise ValueError("warmup_steps and validation_every must be nonnegative")
        if not 0 <= self.decay_fraction <= 1 or self.amp_dtype not in {"none", "bfloat16"}:
            raise ValueError("invalid decay_fraction or amp_dtype")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in [self.weight_decay, self.gradient_clip]):
            raise ValueError("weight_decay and gradient_clip must be finite and nonnegative")
        if not self.required_splits or any(s not in {"validation", "test"} for s in self.required_splits):
            raise ValueError("required_splits must contain validation and/or test")


def config(values=None, overrides=()):
    cfg = Stage2Config()
    _update(cfg, values or {})
    for override in overrides:
        _update(cfg, parse_override(override))
    cfg.validate()
    return cfg


def tensor_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_for(front, manifest=None):
    data = front["config"]["data"]
    source = DataSource(manifest or data["activation_manifest"],
        skip_burn_in=data.get("skip_burn_in", True),
        skip_leading_positions=data.get("skip_leading_positions", 0),
        test_split=data.get("test_split", "auto"), holdout_test_fraction=data.get("holdout_test_fraction", .5))
    record = source.record()
    if any(record[k] != front["data_manifest"][k] for k in ["fingerprint", "splits", "burn_in_excluded"]):
        raise ValueError("front-end data identity, split assignment or exclusion policy differs")
    check_normalization_matches(front["normalization"], source)
    return source


def load_front(path):
    with torch.random.fork_rng(devices=[]):
        model, state = load_checkpoint_model(path, torch.device("cpu"))
    front = {k: state[k] for k in ["config", "data_manifest", "normalization", "model", "step"]}
    front["path"] = str(path)
    front["sha256"] = tensor_hash(front["model"])
    del state
    return model, front


def build_front(front, device):
    cfg = config_from_dict(front["config"])
    with torch.random.fork_rng(devices=[]):
        model = build_model(cfg.model, front["normalization"]["mean"], front["normalization"]["scale"])
    model.load_state_dict(front["model"])
    return model.to(device).eval().requires_grad_(False)


def check_splits(source, cfg):
    for split in set(cfg.required_splits) | {"validation"}:
        if source.split_positions(split) < cfg.eval_batch_size:
            raise ValueError(f"{split} needs at least {cfg.eval_batch_size} usable tokens")
    if source.split_positions("train") < cfg.batch_size:
        raise ValueError("train split cannot provide one batch")


@torch.no_grad()
def calibrate(frontend, source, cfg, device):
    """Train-only sampled mean and scalar RMS, preserving latent anisotropy."""
    batches = TrainBatches(source, cfg.batch_size, cfg.calibration_seed, cfg.shards_per_window)
    total = torch.zeros(frontend.cfg.d_latent, dtype=torch.float64, device=device)
    total_sq = torch.zeros_like(total)
    count = 0
    for _ in range(cfg.calibration_batches):
        with _autocast(device, cfg.amp_dtype):
            y = frontend.encode_dense(next(batches).to(device)).float()
        if not torch.isfinite(y).all():
            raise ValueError("nonfinite calibration activations")
        total += y.double().sum(0); total_sq += y.double().square().sum(0)
        count += len(y)
    mean = total / count
    variance = (total_sq / count - mean.square()).mean()
    if not torch.isfinite(variance) or variance <= 1e-12:
        raise ValueError("latent calibration variance is zero or nonfinite")
    return {"mean": mean.float().cpu(), "scale": float(variance.sqrt()), "count": count,
            "split": "train", "seed": cfg.calibration_seed, "batches": cfg.calibration_batches,
            "convention": "u=(y-train_sample_mean)/train_sample_scalar_RMS; no whitening"}


class ReconstructionMoments:
    def __init__(self, d, device):
        self.n = 0
        self.sum = torch.zeros(d, dtype=torch.float64, device=device)
        self.sum_sq = 0.
        self.sse = 0.

    def add(self, target, prediction):
        target, prediction = target.double(), prediction.double()
        self.n += len(target)
        self.sum += target.sum(0)
        self.sum_sq += float(target.square().sum())
        self.sse += float((prediction - target).square().sum())

    def summary(self):
        variance = self.sum_sq - float(self.sum.square().sum()) / self.n
        return {"mse": self.sse / (self.n * len(self.sum)), "fvu": self.sse / max(variance, 1e-30)}


@torch.no_grad()
def evaluate(frontend, sae, calibration, source, cfg, device, split="validation", batches=None):
    was_training = sae.training
    sae.eval()
    mean = calibration["mean"].to(device)
    scale = calibration["scale"]
    moments = {"frontend": ReconstructionMoments(source.d_in, device),
               "end_to_end": ReconstructionMoments(source.d_in, device),
               "latent": ReconstructionMoments(frontend.cfg.d_latent, device)}
    counts = torch.zeros(cfg.dictionary_size, dtype=torch.long, device=device)
    n, l0_total, l0_min, l0_max = 0, 0, cfg.k, 0
    sample_identity = {"fingerprint": source.fingerprint, "split": split,
                       "entries": source.paths(split), "burn_in_excluded": source.burn_in}
    digest = hashlib.sha256(json.dumps(sample_identity, sort_keys=True).encode())
    try:
        for h, meta in eval_batches(source, split, cfg.eval_batch_size,
                    cfg.eval_batches if batches is None else batches, seed=cfg.eval_seed, with_metadata=True):
            for key in sorted(meta):
                digest.update(meta[key].numpy().tobytes())
            h = h.to(device).float()
            with _autocast(device, cfg.amp_dtype):
                y = frontend.encode_dense(h).float()
                u = (y - mean) / scale
                u_hat, z = sae(u)
                y_hat = u_hat.float() * scale + mean
                h_front = frontend.denormalize(frontend.decode_normalized(y))
                h_hat = frontend.denormalize(frontend.decode_normalized(y_hat))
            if not all(torch.isfinite(v).all() for v in [y, u_hat, h_hat]):
                raise ValueError("nonfinite evaluation outputs")
            moments["frontend"].add(h, h_front)
            moments["end_to_end"].add(h, h_hat)
            moments["latent"].add(y, y_hat)
            active = z > 0
            counts += active.sum(0)
            l0 = active.sum(1)
            n += len(h); l0_total += int(l0.sum())
            l0_min, l0_max = min(l0_min, int(l0.min())), max(l0_max, int(l0.max()))
        if not n:
            raise ValueError(f"no full evaluation batch for {split}")
        metrics = {key: value.summary() for key, value in moments.items()}
        metrics.update({"split": split, "positions": n, "sample_sha256": digest.hexdigest(),
            "extra_fvu": metrics["end_to_end"]["fvu"] - metrics["frontend"]["fvu"],
            "l0_mean": l0_total / n, "l0_min": l0_min, "l0_max": l0_max,
            "inactive_features": int((counts == 0).sum()),
            "inactive_fraction": float((counts == 0).float().mean()),
            "feature_firing_counts": counts.cpu().tolist()})
        return metrics
    finally:
        sae.train(was_training)


class Stage2Trainer:
    def __init__(self, frontend_checkpoint, output, cfg, device="cuda", manifest=None, resume=False):
        cfg.validate()
        self.cfg, self.device, self.output = cfg, torch.device(device), Path(output)
        saved = None
        path = self.output / "checkpoints/latest.pt"
        if resume:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("format") != FORMAT:
                raise ValueError("not a stage-2 checkpoint")
            if config(saved["config"]) != cfg:
                raise ValueError("stage-2 config changed; resume requires the original settings including total steps")
        elif self.output.exists() and any(self.output.iterdir()):
            raise ValueError("output is not empty; use --resume or a new output directory")
        front_model, self.front = load_front(frontend_checkpoint)
        if saved and saved["frontend"]["sha256"] != self.front["sha256"]:
            raise ValueError("front-end weights/normalization changed since checkpoint")
        self.source = source_for(self.front, manifest)
        if saved:
            old_source = saved["data_manifest"]
            if any(old_source[k] != self.source.record()[k] for k in ["fingerprint", "splits", "burn_in_excluded"]):
                raise ValueError("data source changed since checkpoint")
        check_splits(self.source, cfg)
        self.frontend = front_model.to(self.device).eval().requires_grad_(False)
        self.calibration = saved["calibration"] if saved else calibrate(self.frontend, self.source, cfg, self.device)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg.seed)
            self.sae = TopKSAE(self.frontend.cfg.d_latent, cfg.dictionary_size, cfg.k).to(self.device)
        self.initial_sha256 = tensor_hash(self.sae.state_dict())
        self.optimizer = torch.optim.AdamW(self.sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            lambda done: lr_multiplier(done, cfg.steps, cfg.warmup_steps, cfg.decay_fraction))
        self.data = TrainBatches(self.source, cfg.batch_size, cfg.seed, cfg.shards_per_window)
        self.step = 0
        self.firing_counts = torch.zeros(cfg.dictionary_size, dtype=torch.long, device=self.device)
        self.mean = self.calibration["mean"].to(self.device)
        if saved:
            self.sae.load_state_dict(saved["sae"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.scheduler.load_state_dict(saved["scheduler"])
            self.data.load_state_dict(saved["data_state"])
            self.firing_counts.copy_(saved["firing_counts"])
            self.step = saved["step"]
            self.initial_sha256 = saved["initial_sha256"]
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "resolved_config.json", asdict(cfg))
        write_json(self.output / "provenance.json", {"frontend_path": str(frontend_checkpoint),
            "frontend_sha256": self.front["sha256"], "frontend_step": self.front["step"],
            "frontend_config": self.front["config"], "data_manifest": self.source.record(),
            "initial_sae_sha256": self.initial_sha256, "calibration": {
                k: v for k, v in self.calibration.items() if k != "mean"}})
        # A crash can leave log entries newer than the last saved weights.
        if saved and (self.output / "metrics.jsonl").exists():
            log = self.output / "metrics.jsonl"
            rows = []
            for line in log.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["step"] <= self.step:
                    rows.append(json.dumps(row))
            log.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    def train_step(self):
        cfg = self.cfg
        self.sae.train()
        h = next(self.data).to(self.device)
        with torch.no_grad(), _autocast(self.device, cfg.amp_dtype):
            y = self.frontend.encode_dense(h).float()
            u = (y - self.mean) / self.calibration["scale"]
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast(self.device, cfg.amp_dtype):
            prediction, z = self.sae(u)
            loss = (prediction.float() - u).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite stage-2 training loss")
        loss.backward()
        self.sae.project_decoder_gradient()
        norm = torch.nn.utils.clip_grad_norm_(self.sae.parameters(), cfg.gradient_clip or float("inf"), error_if_nonfinite=True)
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step(); self.sae.normalize_decoder(); self.scheduler.step()
        self.step += 1
        with torch.no_grad():
            active = z > 0
            self.firing_counts += active.sum(0)
        return {"normalized_latent_mse": float(loss.detach()), "lr": lr, "gradient_norm": float(norm),
                "l0_mean": float(active.sum(1).float().mean()),
                "never_fired_fraction": float((self.firing_counts == 0).float().mean())}

    def save(self):
        path = self.output / "checkpoints/latest.pt"
        path.parent.mkdir(exist_ok=True)
        partial = path.with_suffix(".pt.partial")
        torch.save({"format": FORMAT, "step": self.step, "config": asdict(self.cfg),
                    "frontend": self.front, "calibration": self.calibration,
                    "data_manifest": self.source.record(), "sae": self.sae.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                    "data_state": self.data.state_dict(), "firing_counts": self.firing_counts.cpu(),
                    "initial_sha256": self.initial_sha256}, partial)
        partial.replace(path)
        return path

    def run(self, max_steps=None):
        stop = min(self.cfg.steps, max_steps) if max_steps is not None else self.cfg.steps
        started = time.time()
        if self.step == 0:
            self.save()
        with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            while self.step < stop:
                metrics = self.train_step()
                row = {"step": self.step}
                if self.step == 1 or self.step % self.cfg.log_every == 0 or self.step == stop:
                    row["train"] = metrics
                    row["elapsed_seconds"] = time.time() - started
                    print(f"step {self.step}: loss={metrics['normalized_latent_mse']:.5f} L0={metrics['l0_mean']:.1f}", flush=True)
                if self.cfg.validation_every and (self.step % self.cfg.validation_every == 0):
                    row["validation"] = evaluate(self.frontend, self.sae, self.calibration,
                        self.source, self.cfg, self.device, batches=self.cfg.validation_batches)
                    row["validation"].pop("feature_firing_counts")
                if len(row) > 1:
                    log.write(json.dumps(row) + "\n"); log.flush()
                if self.step % self.cfg.checkpoint_every == 0:
                    self.save()
        self.save()
        if self.step == self.cfg.steps:
            for split in self.cfg.required_splits:
                results = evaluate(self.frontend, self.sae, self.calibration, self.source, self.cfg, self.device, split)
                results.update({"step": self.step, "frontend_lambda": self.front["config"]["sigreg"]["weight"],
                    "frontend_sha256": self.front["sha256"], "stage2_config": asdict(self.cfg),
                    "initial_sae_sha256": self.initial_sha256,
                    "train_positions": self.step * self.cfg.batch_size,
                    "train_never_fired_fraction": float((self.firing_counts == 0).float().mean())})
                write_json(self.output / f"eval-{split}.json", results)


def preflight(checkpoints, cfg, manifest=None):
    """Check all candidates before the first run, including shared data policy."""
    baseline = None
    rows = []
    for path in checkpoints:
        model, front = load_front(path)
        source = source_for(front, manifest)
        check_splits(source, cfg)
        identity = {k: source.record()[k] for k in ["fingerprint", "splits", "burn_in_excluded"]}
        identity["model"] = asdict(model.cfg)
        identity["normalization"] = tensor_hash({"mean": model.input_mean, "scale": model.input_scale.reshape(1)})
        if baseline is not None and identity != baseline:
            raise ValueError("comparison candidates differ in data, exclusions, dimensions or input normalization")
        baseline = identity
        rows.append({"path": str(path), "lambda": front["config"]["sigreg"]["weight"],
                     "step": front["step"], "sha256": front["sha256"]})
    if len({r["sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate front-end checkpoint in comparison")
    return rows


def write_report(root, split="validation"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    paths = sorted(root.glob(f"model-*/eval-{split}.json"))
    if not paths:
        raise ValueError(f"no {split} evaluations found")
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    for r in rows[1:]:
        if any(r[k] != rows[0][k] for k in ["sample_sha256", "stage2_config", "initial_sae_sha256", "step"]):
            raise ValueError("refusing report: evaluations do not share sampling, initialization and training budget")
    lines = [f"# Stage 2 ({split})", "", "FVU is in original activation space; latent FVU is reported separately.",
             "Inactive means no positive activation on this evaluation sample, not permanently dead.", "",
             "| Run | λ | Frontend FVU (%) | End-to-end FVU (%) | Extra FVU (pp) | Latent FVU (%) | Mean L0 | Inactive (%) | Never fired in training (%) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    compact = []
    for p, r in zip(paths, rows):
        compact.append({k: v for k, v in r.items() if k != "feature_firing_counts"})
        lines.append(f"| {p.parent.name} | {r['frontend_lambda']:g} | {r['frontend']['fvu']*100:.4f} | "
                     f"{r['end_to_end']['fvu']*100:.4f} | {r['extra_fvu']*100:.4f} | "
                     f"{r['latent']['fvu']*100:.4f} | {r['l0_mean']:.2f} | {r['inactive_fraction']*100:.2f} | "
                     f"{r['train_never_fired_fraction']*100:.2f} |")
    report = root / "report"; report.mkdir(exist_ok=True)
    (report / f"{split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report / f"{split}.json", compact)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(rows)))
    ax.bar([v - .18 for v in x], [r['frontend']['fvu']*100 for r in rows], width=.36, label="Frontend only")
    ax.bar([v + .18 for v in x], [r['end_to_end']['fvu']*100 for r in rows], width=.36, label="Frontend + Top-K SAE")
    ax.set_xticks(x, [f"lambda={r['frontend_lambda']:g}" for r in rows])
    ax.set_ylabel("Original-space FVU (%)"); ax.legend(); fig.tight_layout()
    fig.savefig(report / f"{split}.png", dpi=160); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "sweep"]:
        p = sub.add_parser(name)
        p.add_argument("--checkpoints", nargs="+", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--config")
        p.add_argument("--set", action="append", default=[])
        p.add_argument("--activation-manifest")
        p.add_argument("--device", default="cuda")
        p.add_argument("--resume", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--activation-manifest")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p = sub.add_parser("report")
    p.add_argument("--run-root", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    args = parser.parse_args(argv)
    if args.command == "report":
        return write_report(args.run_root, args.split)
    if args.command == "evaluate":
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if state.get("format") != FORMAT:
            raise ValueError("not a stage-2 checkpoint")
        cfg, device = config(state["config"]), torch.device(args.device)
        front = build_front(state["frontend"], device)
        source = source_for(state["frontend"], args.activation_manifest)
        with torch.random.fork_rng(devices=[]):
            sae = TopKSAE(front.cfg.d_latent, cfg.dictionary_size, cfg.k).to(device)
        sae.load_state_dict(state["sae"])
        results = evaluate(front, sae, state["calibration"], source, cfg, device, args.split)
        results.update({"step": state["step"], "frontend_lambda": state["frontend"]["config"]["sigreg"]["weight"],
                        "frontend_sha256": state["frontend"]["sha256"], "stage2_config": asdict(cfg),
                        "initial_sae_sha256": state["initial_sha256"], "train_positions": state["step"] * cfg.batch_size,
                        "train_never_fired_fraction": float((state["firing_counts"] == 0).float().mean())})
        return write_json(Path(args.output), results)
    values = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    cfg = config(values, args.set)
    candidates = preflight(args.checkpoints, cfg, args.activation_manifest)
    if args.command == "train":
        if len(candidates) != 1:
            raise ValueError("train takes exactly one checkpoint; use sweep for comparisons")
        return Stage2Trainer(args.checkpoints[0], args.output, cfg, args.device, args.activation_manifest, args.resume).run()
    root = Path(args.output)
    definition = {"candidates": candidates, "config": asdict(cfg)}
    # JSON round-trip normalizes tuple/list representation.
    definition = json.loads(json.dumps(definition))
    if root.exists() and any(root.iterdir()):
        if not args.resume or not (root / "comparison.json").exists():
            raise ValueError("comparison output exists; use --resume or a new directory")
        if json.loads((root / "comparison.json").read_text()) != definition:
            raise ValueError("comparison candidates or settings changed")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "comparison.json", definition)
    for i, checkpoint in enumerate(args.checkpoints):
        output = root / f"model-{i:02d}"
        resume = args.resume and (output / "checkpoints/latest.pt").exists()
        Stage2Trainer(checkpoint, output, cfg, args.device, args.activation_manifest, resume).run()
    for split in cfg.required_splits:
        write_report(root, split)


if __name__ == "__main__":
    main()
