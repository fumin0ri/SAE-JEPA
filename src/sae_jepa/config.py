"""Experiment configuration: YAML file + ``--set section.key=value`` overrides."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    activation_manifest: str = ""
    # Train-only statistics from ``sj-compute-normalization``.  When empty,
    # they are computed at startup and saved next to the run.
    normalization_path: str = ""
    input_whitening_path: str = ""  # frozen train-fit ZCA; SIGReg-only ablation
    skip_burn_in: bool = True
    # Drop token positions < k of every stored sequence from normalization,
    # training and evaluation (1 = drop the first token of each segment, whose
    # residual norm is an extreme outlier in Pythia).  0 keeps every position.
    skip_leading_positions: int = 0
    shards_per_window: int = 4
    test_split: str = "auto"
    holdout_test_fraction: float = 0.5


@dataclass
class ModelConfig:
    type: str = "dense_sigreg_ae"
    d_in: int = 0  # 0: resolved from the activation manifest
    d_hidden: int = 4096
    d_latent: int = 4096
    activation: str = "gelu"
    # masked_sigreg_encoder only.  > 0: after building (and whitening) the
    # encoder, rescale and center its last Linear layer on held-aside train rows
    # so the initial output has this mean per-dimension variance and zero mean.
    # 0 keeps PyTorch's default initialization (the original behavior).
    init_output_variance: float = 0.0


@dataclass
class SIGRegConfig:
    weight: float = 0.0
    num_projections: int = 256
    t_min: float = -5.0
    t_max: float = 5.0
    num_points: int = 17
    scale_by_batch_size: bool = True
    resample_every_step: bool = True
    # The train projection generator is seeded by (run seed, seed_offset);
    # validation uses fixed projections from validation_seed.
    seed_offset: int = 1_000_003
    validation_seed: int = 20_240_917
    validation_projections: int = 256


@dataclass
class CovarianceConfig:
    weight: float = 0.0  # beta; zero preserves the original objective and RNGs
    sketch_dim: int = 64
    # "plugin" (original; biased toward shrunken variance), "split_half"
    # (unbiased estimate of ||Cov - I||^2 from two independent batch halves) or
    # "split_half_corr" (off-diagonal correlations only; scale left to SIGReg)
    estimator: str = "plugin"
    # Ramp beta linearly from 0 at step 0 to full weight at this step (0 = off),
    # so SIGReg can grow the output scale before decorrelation acts.
    warmup_steps: int = 0
    seed_offset: int = 4_000_037
    validation_seed: int = 91_003


@dataclass
class OptimConfig:
    batch_size: int = 512
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    warmup_steps: int = 1000
    decay_fraction: float = 0.2  # linear decay to zero over the final 20%
    steps: int = 10_000
    gradient_clip: float = 0.0  # 0 disables clipping
    amp_dtype: str = "bfloat16"  # bfloat16 | none


@dataclass
class TrainConfig:
    seed: int = 42
    output_dir: str = ""
    device: str = "cuda"
    log_every: int = 100
    validation_every: int = 1000
    validation_batches: int = 16
    checkpoint_every: int = 2000
    # False: only checkpoints/latest.pt (~0.6 GB for the 4096-wide model incl.
    # AdamW state) is written.  True: also keep a step-XXXXXXX.pt copy per save.
    keep_checkpoints: bool = False


@dataclass
class EvalConfig:
    input_diagnostics: bool = False  # covariance of the actual encoder input
    # Masked/SIGReg-only encoder: also accumulate the output covariance during
    # periodic training validation (and at step 0), so effective rank and
    # participation ratio are logged over training, not only at the end.
    training_covariance: bool = True
    split: str = "validation"
    batch_size: int = 512  # same N as training so SIGReg values are comparable
    # Evaluation batches are a fixed random sample of positions drawn from the
    # whole split (all sequences and shards) and shuffled into batches with
    # sample_seed, so every run and every lambda sees identical batches.
    batches: int = 64  # 0 = every full batch of the split
    sample_seed: int = 31_337
    # Splits that must be non-empty before training starts (the sweep evaluates them).
    required_splits: tuple[str, ...] = ("validation", "test")
    # Diagnostic projections are never used by training or validation SIGReg.
    diagnostic_projections: int = 256
    diagnostic_seed: int = 777
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    maximum_quantile_samples: int = 65_536
    gaussian_reference_seed: int = 4242
    # Outlier diagnostics (detailed evaluation only).  Token positions below
    # leading_positions (position 0 = first token of each stored sequence) are
    # reported separately; covariance is also reported without them and
    # without the top outlier_fraction of samples by ||y||^2.
    leading_positions: int = 1
    outlier_fraction: float = 0.001
    outlier_buffer: int = 1024  # max samples that can be trimmed
    outlier_table_size: int = 32
    norm_quantiles: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999, 1.0)


@dataclass
class MaskingConfig:
    enabled: bool = True  # False: full-input SIGReg only, no consistency loss
    probability: float = 0.25
    seed_offset: int = 2_000_003
    validation_seed: int = 81_001


@dataclass
class ExperimentConfig:
    name: str = "dense_sigreg_ae"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    covariance: CovarianceConfig = field(default_factory=CovarianceConfig)

    def validate(self) -> None:
        cc = self.covariance
        if not math.isfinite(cc.weight) or cc.weight < 0:
            raise ValueError("covariance.weight must be finite and non-negative")
        if type(cc.sketch_dim) is not int or cc.sketch_dim < 1:
            raise ValueError("covariance.sketch_dim must be a positive integer")
        if type(cc.warmup_steps) is not int or cc.warmup_steps < 0:
            raise ValueError("covariance.warmup_steps must be a non-negative integer")
        if cc.estimator not in ("plugin", "split_half", "split_half_corr"):
            raise ValueError("covariance.estimator must be 'plugin', 'split_half' or 'split_half_corr'")
        if cc.weight > 0:
            if self.model.type not in {"dense_sigreg_ae", "masked_sigreg_encoder"}:
                raise ValueError("covariance requires dense_sigreg_ae or masked_sigreg_encoder")
            if self.model.type == "dense_sigreg_ae" and not self.sigreg.weight > 0:
                raise ValueError("dense covariance (reconstruction + SIGReg + cov) requires sigreg.weight > 0")
            if cc.sketch_dim > self.model.d_latent or cc.sketch_dim >= self.optim.batch_size:
                raise ValueError("covariance.sketch_dim must be <= d_latent and < training batch size")
            if cc.sketch_dim >= self.eval.batch_size:
                raise ValueError("covariance.sketch_dim must be < evaluation batch size")
            if cc.estimator != "plugin" and min(self.optim.batch_size, self.eval.batch_size) < 4:
                raise ValueError(f"covariance.estimator={cc.estimator} requires batch sizes >= 4")
        if self.data.input_whitening_path and (self.model.type != "masked_sigreg_encoder" or self.masking.enabled):
            raise ValueError("input whitening currently requires SIGReg-only (masked encoder, masking.enabled=false)")
        if not isinstance(self.masking.enabled, bool):
            raise ValueError("masking.enabled must be a boolean")
        if not self.masking.enabled and self.model.type != "masked_sigreg_encoder":
            raise ValueError("masking.enabled=false requires model.type=masked_sigreg_encoder")
        if self.model.type not in {"dense_sigreg_ae", "masked_sigreg_encoder"}:
            raise ValueError(f"unknown model.type {self.model.type!r}")
        if not 0 < self.masking.probability < 1:
            raise ValueError("masking.probability must lie strictly between 0 and 1")
        if self.model.type == "masked_sigreg_encoder" and not self.sigreg.weight > 0:
            raise ValueError("masked consistency requires sigreg.weight > 0 to oppose collapse")
        if self.model.init_output_variance < 0:
            raise ValueError("model.init_output_variance must be non-negative")
        if self.model.init_output_variance > 0 and self.model.type != "masked_sigreg_encoder":
            raise ValueError("model.init_output_variance requires model.type=masked_sigreg_encoder")
        if self.sigreg.weight < 0:
            raise ValueError("sigreg.weight must be non-negative")
        if self.data.skip_leading_positions < 0:
            raise ValueError("data.skip_leading_positions cannot be negative")
        if self.optim.batch_size < 2 or self.optim.steps < 1:
            raise ValueError("batch_size must be >= 2 and steps >= 1")
        if not 0.0 <= self.optim.decay_fraction <= 1.0:
            raise ValueError("optim.decay_fraction must lie in [0, 1]")
        if self.optim.amp_dtype not in {"bfloat16", "none"}:
            raise ValueError("optim.amp_dtype must be bfloat16 or none")
        if self.sigreg.num_projections < 1 or self.sigreg.num_points < 2:
            raise ValueError("SIGReg needs >= 1 projection and >= 2 quadrature points")


def _coerce(value: Any, current: Any) -> Any:
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(current, float) and not isinstance(value, bool) and isinstance(value, (int, str)):
        return float(value)  # PyYAML reads "1e-4" (no dot) as a string
    return value


def _update(target: Any, values: dict[str, Any], prefix: str = "") -> None:
    names = {f.name for f in fields(target)}
    for key, value in values.items():
        if key not in names:
            raise KeyError(f"unknown config key {prefix}{key}")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{prefix}{key} must be a mapping")
            _update(current, value, f"{prefix}{key}.")
        else:
            setattr(target, key, _coerce(value, current))


def update_config(cfg: ExperimentConfig, values: dict[str, Any]) -> None:
    """Apply a nested mapping of settings to ``cfg`` (unknown keys raise)."""
    _update(cfg, values)


def parse_override(text: str) -> dict[str, Any]:
    if "=" not in text:
        raise ValueError(f"override must look like section.key=value, got {text!r}")
    dotted, raw = text.split("=", 1)
    value: Any = yaml.safe_load(raw)
    for part in reversed(dotted.strip().split(".")):
        value = {part: value}
    return value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        _update(cfg, values)
    for override in overrides or []:
        _update(cfg, parse_override(override))
    cfg.validate()
    return cfg


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)


def config_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig()
    _update(cfg, values)
    return cfg
