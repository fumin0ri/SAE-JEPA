"""Stage-1 training: reconstruction (+ lambda * SIGReg) on a dense representation.

    L = L_rec + lambda_sigreg * L_SIGReg(y),   L_rec = mean((x_hat - x)^2)

``L_rec`` is measured in the normalized space ``x = (h - mu)/s`` and therefore
equals the fraction of (train-scale) variance left unexplained.  With
``sigreg.weight = 0`` (Dense-AE) SIGReg is neither computed nor does it consume
its random generator, so Dense-AE and Dense-SIGReg-AE runs with the same seed
see identical data order and identical initial weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .config import ExperimentConfig, config_to_dict, load_config
from .data import DataSource, TrainBatches, mix_seed, write_json
from .evaluate import evaluate_model, write_evaluation
from .models import ARCHITECTURE_ID, DenseSIGRegAE, build_model
from .normalization import (
    compute_train_statistics,
    load_normalization,
    save_normalization,
)
from .sigreg import SIGRegLoss, loss_convention


CHECKPOINT_FORMAT = "sae-jepa-dense-checkpoint-v1"


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def configure_accelerator(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def autocast_context(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype == "bfloat16"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def lr_multiplier(completed_steps: int, total_steps: int, warmup_steps: int, decay_fraction: float) -> float:
    """Linear warmup, constant, then linear decay to zero over the final fraction.

    ``completed_steps`` is the number of optimizer updates already taken, so the
    upcoming update is number ``u = completed_steps + 1`` (1-indexed).
    """
    u = completed_steps + 1
    warm = min(1.0, u / warmup_steps) if warmup_steps > 0 else 1.0
    decay_steps = int(round(decay_fraction * total_steps))
    decay = 1.0
    if decay_steps > 0:
        decay = min(1.0, max(0.0, (total_steps - u + 1) / decay_steps))
    return min(warm, decay)


# Settings that may change between a checkpoint and its resumption: they only
# affect logging cadence, file locations, hardware, or offline evaluation.
# The manifest and normalization paths are compared by content instead.
RESUMABLE_KEYS = {
    "name",
    "data.activation_manifest",
    "data.normalization_path",
    "train.output_dir",
    "train.device",
    "train.log_every",
    "train.validation_every",
    "train.validation_batches",
    "train.checkpoint_every",
    "train.keep_checkpoints",
}
RESUMABLE_SECTIONS = {"eval"}


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = list(value) if isinstance(value, tuple) else value
    return flat


def training_config_differences(saved: dict[str, Any], current: ExperimentConfig) -> list[str]:
    """Every training-relevant setting whose value differs from the checkpoint."""
    old = _flatten(saved)
    new = _flatten(config_to_dict(current))
    differences = []
    for key in sorted(set(old) | set(new)):
        if key in RESUMABLE_KEYS or key.split(".", 1)[0] in RESUMABLE_SECTIONS:
            continue
        if old.get(key, "<missing>") != new.get(key, "<missing>"):
            differences.append(f"{key}: {old.get(key, '<missing>')!r} -> {new.get(key, '<missing>')!r}")
    return differences


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


class Trainer:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        if not cfg.train.output_dir:
            raise ValueError("train.output_dir must be set")
        if not cfg.data.activation_manifest:
            raise ValueError("data.activation_manifest must be set")
        self.output_dir = Path(cfg.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(cfg.train.device)
        configure_accelerator(self.device)
        self.amp_dtype = cfg.optim.amp_dtype if self.device.type == "cuda" else "none"

        self.source = DataSource(
            cfg.data.activation_manifest,
            skip_burn_in=cfg.data.skip_burn_in,
            test_split=cfg.data.test_split,
            holdout_test_fraction=cfg.data.holdout_test_fraction,
        )
        self._check_required_splits()
        if cfg.model.d_in == 0:
            cfg.model.d_in = self.source.d_in
        elif cfg.model.d_in != self.source.d_in:
            raise ValueError("model.d_in does not match the activation manifest")
        self.normalization = self._load_or_compute_normalization()

        # Model initialization depends only on the run seed.
        torch.manual_seed(cfg.train.seed)
        self.model: DenseSIGRegAE = build_model(
            cfg.model, self.normalization["mean"], self.normalization["scale"]
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda completed: lr_multiplier(
                completed, cfg.optim.steps, cfg.optim.warmup_steps, cfg.optim.decay_fraction
            ),
        )
        self.data = TrainBatches(
            self.source,
            cfg.optim.batch_size,
            seed=cfg.train.seed,
            shards_per_window=cfg.data.shards_per_window,
        )
        self.sigreg: SIGRegLoss | None = None
        if cfg.sigreg.weight > 0:
            self.sigreg = SIGRegLoss(
                cfg.model.d_latent,
                num_projections=cfg.sigreg.num_projections,
                t_min=cfg.sigreg.t_min,
                t_max=cfg.sigreg.t_max,
                num_points=cfg.sigreg.num_points,
                scale_by_batch_size=cfg.sigreg.scale_by_batch_size,
                resample_every_step=cfg.sigreg.resample_every_step,
                seed=mix_seed(cfg.train.seed, cfg.sigreg.seed_offset),
            )
        self.step = 0
        self.convention = loss_convention(
            num_projections=cfg.sigreg.num_projections,
            t_min=cfg.sigreg.t_min,
            t_max=cfg.sigreg.t_max,
            num_points=cfg.sigreg.num_points,
            scale_by_batch_size=cfg.sigreg.scale_by_batch_size,
            resample_every_step=cfg.sigreg.resample_every_step,
            batch_size=cfg.optim.batch_size,
        )

    # ------------------------------------------------------------------ setup
    def _check_required_splits(self) -> None:
        """Fail before training if a split that will be evaluated is unusable."""
        for split in self.cfg.eval.required_splits:
            if split not in ("validation", "test"):
                raise ValueError(f"eval.required_splits: unknown held-out split {split!r}")
            if not self.source.splits[split]:
                raise ValueError(
                    f"required split {split!r} is empty (data.test_split="
                    f"{self.cfg.data.test_split!r}); provide it or drop it from "
                    "eval.required_splits"
                )
            positions = self.source.split_positions(split)
            if positions < self.cfg.eval.batch_size:
                raise ValueError(
                    f"required split {split!r} has {positions} positions, fewer than "
                    f"one evaluation batch of {self.cfg.eval.batch_size}"
                )

    def _load_or_compute_normalization(self) -> dict[str, Any]:
        path = self.cfg.data.normalization_path
        if path:
            return load_normalization(path, self.source)
        local = self.output_dir / "normalization.pt"
        if local.exists():
            return load_normalization(local, self.source)
        stats = compute_train_statistics(self.source, progress=True)
        save_normalization(stats, local)
        self.cfg.data.normalization_path = str(local)
        return stats

    # ------------------------------------------------------------ objective
    def loss(
        self, h: torch.Tensor, diagnostics: bool
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weight = self.cfg.sigreg.weight
        with autocast_context(self.device, self.amp_dtype):
            out = self.model(h)
        y = out["y"]
        reconstruction = F.mse_loss(out["x_hat"].float(), out["x"])
        sigreg = self.sigreg(y) if self.sigreg is not None else None
        loss = reconstruction if sigreg is None else reconstruction + weight * sigreg
        metrics: dict[str, float] = {}
        if diagnostics:
            grad_rec = torch.autograd.grad(reconstruction, y, retain_graph=True)[0]
            metrics["grad_rms_y/reconstruction"] = rms(grad_rec)
            if sigreg is not None:
                grad_sig = torch.autograd.grad(weight * sigreg, y, retain_graph=True)[0]
                metrics["grad_rms_y/sigreg_weighted"] = rms(grad_sig)
                metrics["grad_rms_y/sigreg_to_reconstruction"] = metrics[
                    "grad_rms_y/sigreg_weighted"
                ] / max(metrics["grad_rms_y/reconstruction"], 1e-30)
            with torch.no_grad():
                yf = y.detach().float()
                metrics.update(
                    {
                        "loss": float(loss.detach()),
                        "reconstruction_normalized_mse": float(reconstruction.detach()),
                        "y/mean_sq_per_dim": float(yf.mean(0).square().mean()),
                        "y/variance_mean": float(yf.var(0, unbiased=False).mean()),
                    }
                )
                if sigreg is not None:
                    metrics["sigreg"] = float(sigreg.detach())
                    metrics["sigreg_weighted"] = float(weight * sigreg.detach())
        return loss, metrics

    # ---------------------------------------------------------- checkpoints
    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT,
            "architecture_id": ARCHITECTURE_ID,
            "step": self.step,
            "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "sigreg": self.sigreg.state_dict() if self.sigreg is not None else None,
                "data": self.data.state_dict(),
            },
            "normalization": {
                "mean": self.normalization["mean"],
                "scale": self.normalization["scale"],
                "count": self.normalization["count"],
                "shards": self.normalization["shards"],
                "burn_in_excluded": self.normalization["burn_in_excluded"],
                "manifest_fingerprint": self.normalization["manifest_fingerprint"],
                "path": self.cfg.data.normalization_path,
            },
            "data_manifest": self.source.record(),
            "config": config_to_dict(self.cfg),
            "sigreg_convention": self.convention,
        }

    def save_checkpoint(self) -> Path:
        directory = self.output_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        state = self.checkpoint_state()
        path = directory / f"step-{self.step:07d}.pt"
        partial = path.with_suffix(".pt.partial")
        torch.save(state, partial)
        partial.replace(path)
        latest = directory / "latest.pt"
        partial = latest.with_suffix(".pt.partial")
        torch.save(state, partial)
        partial.replace(latest)
        if not self.cfg.train.keep_checkpoints:
            for old in directory.glob("step-*.pt"):
                if old != path:
                    old.unlink()
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint {path}")
        if state["data_manifest"]["fingerprint"] != self.source.fingerprint:
            raise ValueError("checkpoint was trained on a different activation manifest")
        differences = training_config_differences(state["config"], self.cfg)
        if differences:
            raise ValueError(
                "cannot resume: settings that change training differ from the "
                "checkpoint (start a new run instead): " + "; ".join(differences)
            )
        if state["data_manifest"]["splits"] != self.source.splits:
            raise ValueError("cannot resume: split assignment differs from the checkpoint")
        saved_norm = state["normalization"]
        if not (
            torch.equal(saved_norm["mean"], self.normalization["mean"])
            and saved_norm["scale"] == self.normalization["scale"]
        ):
            raise ValueError("cannot resume: normalization statistics differ from the checkpoint")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        if self.sigreg is not None:
            self.sigreg.load_state_dict(state["rng"]["sigreg"])
        self.data.load_state_dict(state["rng"]["data"])
        self.step = int(state["step"])

    # ---------------------------------------------------------------- loop
    def validate(self, detailed: bool = False, split: str = "validation") -> dict[str, Any]:
        return evaluate_model(
            self.model,
            self.source,
            split=split,
            batch_size=self.cfg.eval.batch_size,
            maximum_batches=self.cfg.train.validation_batches if not detailed else self.cfg.eval.batches,
            device=self.device,
            amp_dtype=self.amp_dtype,
            sigreg_cfg=self.cfg.sigreg,
            eval_cfg=self.cfg.eval,
            detailed=detailed,
        )

    def train_step(self, diagnostics: bool) -> dict[str, float]:
        h = next(self.data).to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.loss(h, diagnostics)
        loss.backward()
        if self.cfg.optim.gradient_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.gradient_clip)
            if diagnostics:
                metrics["grad_norm_params"] = float(norm)
        if diagnostics:
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return metrics

    def run(self, max_steps: int | None = None) -> dict[str, Any]:
        cfg = self.cfg
        write_json(self.output_dir / "resolved_config.json", config_to_dict(cfg))
        write_json(self.output_dir / "sigreg_convention.json", self.convention)
        write_json(self.output_dir / "data_manifest.json", self.source.record())
        log_path = self.output_dir / "metrics.jsonl"
        stop = cfg.optim.steps if max_steps is None else min(cfg.optim.steps, max_steps)
        started = time.time()
        with log_path.open("a", encoding="utf-8") as log:
            while self.step < stop:
                upcoming = self.step + 1
                diagnostics = upcoming == 1 or upcoming % cfg.train.log_every == 0 or upcoming == cfg.optim.steps
                metrics = self.train_step(diagnostics)
                record: dict[str, Any] = {}
                if diagnostics:
                    record["train"] = metrics
                    record["elapsed_seconds"] = time.time() - started
                if cfg.train.validation_every > 0 and (
                    self.step % cfg.train.validation_every == 0 or self.step == cfg.optim.steps
                ):
                    record["validation"] = self.validate(detailed=False)
                if record:
                    record["step"] = self.step
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    if "train" in record:
                        t = record["train"]
                        print(
                            f"step {self.step}: rec={t['reconstruction_normalized_mse']:.4f}"
                            + (f" sigreg={t['sigreg']:.3f}" if "sigreg" in t else "")
                            + f" lr={t['lr']:.2e}",
                            flush=True,
                        )
                if cfg.train.checkpoint_every > 0 and self.step % cfg.train.checkpoint_every == 0:
                    self.save_checkpoint()
        if self.step == cfg.optim.steps:
            self.save_checkpoint()
            final = self.validate(detailed=True)
            final.update(
                {
                    "checkpoint": str(self.output_dir / "checkpoints" / "latest.pt"),
                    "step": self.step,
                    "sigreg_weight": cfg.sigreg.weight,
                    "run_name": cfg.name,
                    "seed": cfg.train.seed,
                }
            )
            write_evaluation(final, self.output_dir / f"eval-validation-step-{self.step:07d}.json")
            return final
        return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Dense-AE / Dense-SIGReg-AE (stage 1)")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument(
        "--resume",
        default="auto",
        help="'auto' resumes from OUTPUT/checkpoints/latest.pt when present, "
        "'none' starts fresh, or a checkpoint path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config, args.set)
    trainer = Trainer(cfg)
    latest = Path(cfg.train.output_dir) / "checkpoints" / "latest.pt"
    if args.resume == "auto" and latest.exists():
        trainer.load_checkpoint(latest)
        print(f"resumed from {latest} at step {trainer.step}")
    elif args.resume not in {"auto", "none"}:
        trainer.load_checkpoint(args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
