"""Training Engine — Chapter 6 / Stage 3: Training configuration, optimizers, checkpoints, gradients."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    LION = "lion"


class ScheduleType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    WARMUP_COSINE = "warmup_cosine"
    EXPONENTIAL = "exponential"


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    MIXED = "mixed"


@dataclass
class TrainingConfig:
    learning_rate: float = 5e-5
    batch_size: int = 8
    epochs: int = 3
    optimizer: OptimizerType = OptimizerType.ADAMW
    schedule: ScheduleType = ScheduleType.COSINE
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    precision: Precision = Precision.MIXED
    gradient_accumulation_steps: int = 1
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 10
    seed: int = 42


@dataclass
class OptimizerState:
    name: str
    lr: float
    params: Dict[str, float]
    step: int = 0
    momentum: Dict[str, float] = field(default_factory=dict)
    variance: Dict[str, float] = field(default_factory=dict)


@dataclass
class Checkpoint:
    step: int
    metrics: Dict[str, float]
    timestamp: float
    model_state: Dict[str, Any] = field(default_factory=dict)


class TrainingConfigurator:
    """Configure training runs with validation and estimation."""

    def create_config(self, params: Dict[str, Any]) -> TrainingConfig:
        config = TrainingConfig()
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def validate_config(self, config: TrainingConfig) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        if config.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        elif config.learning_rate > 1.0:
            warnings.append("learning_rate > 1.0 is unusually high")
        if config.batch_size <= 0:
            errors.append("batch_size must be positive")
        if config.epochs <= 0:
            errors.append("epochs must be positive")
        if config.max_grad_norm <= 0:
            errors.append("max_grad_norm must be positive")
        if config.warmup_steps < 0:
            errors.append("warmup_steps must be non-negative")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def estimate_training_time(self, config: TrainingConfig, data_size: int) -> Dict[str, float]:
        steps_per_epoch = math.ceil(data_size / config.batch_size)
        total_steps = steps_per_epoch * config.epochs
        total_seconds = total_steps * 0.5
        return {
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "estimated_seconds": total_seconds,
            "estimated_minutes": round(total_seconds / 60, 1),
            "estimated_hours": round(total_seconds / 3600, 2),
        }

    def suggest_hyperparameters(self, task: str, model_size: str) -> Dict[str, Any]:
        lr_map = {"small": 1e-4, "medium": 5e-5, "large": 2e-5, "xlarge": 1e-5}
        bs_map = {"small": 32, "medium": 16, "large": 8, "xlarge": 4}
        return {
            "learning_rate": lr_map.get(model_size, 5e-5),
            "batch_size": bs_map.get(model_size, 8),
            "epochs": 3, "warmup_ratio": 0.06,
            "weight_decay": 0.01, "max_grad_norm": 1.0, "schedule": "cosine",
        }


class OptimizerFactory:
    """Create and manage optimizers with schedulers."""

    def create_optimizer(self, name: str, params: Dict[str, float], lr: float = 5e-5) -> OptimizerState:
        return OptimizerState(
            name=name, lr=lr, params=copy.deepcopy(params),
            momentum={k: 0.0 for k in params}, variance={k: 0.0 for k in params},
        )

    def get_scheduler(self, optimizer: OptimizerState, schedule: ScheduleType, steps: int) -> List[float]:
        lrs: List[float] = []
        base_lr = optimizer.lr
        for step in range(steps):
            if schedule == ScheduleType.COSINE:
                lr = base_lr * 0.5 * (1 + math.cos(math.pi * step / steps))
            elif schedule == ScheduleType.LINEAR:
                lr = base_lr * (1 - step / steps)
            elif schedule == ScheduleType.CONSTANT:
                lr = base_lr
            elif schedule == ScheduleType.WARMUP_COSINE:
                warmup = min(100, steps // 10)
                if step < warmup:
                    lr = base_lr * step / warmup
                else:
                    progress = (step - warmup) / max(steps - warmup, 1)
                    lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            elif schedule == ScheduleType.EXPONENTIAL:
                lr = base_lr * (0.99 ** step)
            else:
                lr = base_lr
            lrs.append(max(0, lr))
        return lrs

    def compute_lr(self, step: int, warmup: int, total: int) -> float:
        if step < warmup:
            return min(1.0, step / max(warmup, 1))
        return max(0.0, 1.0 - (step - warmup) / max(total - warmup, 1))

    def get_gradient_stats(self, optimizer: OptimizerState) -> Dict[str, float]:
        if not optimizer.params:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
        values = list(optimizer.params.values())
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        return {"mean": round(mean, 6), "std": round(std, 6), "max": round(max(values), 6), "min": round(min(values), 6)}


class CheckpointManager:
    """Manage training checkpoints."""

    def __init__(self) -> None:
        self._checkpoints: Dict[str, List[Checkpoint]] = {}

    def save_checkpoint(self, model: Dict[str, Any], step: int, metrics: Dict[str, float], run_id: str = "default") -> Checkpoint:
        ckpt = Checkpoint(step=step, metrics=dict(metrics), timestamp=time.time(), model_state=copy.deepcopy(model))
        self._checkpoints.setdefault(run_id, []).append(ckpt)
        return ckpt

    def load_checkpoint(self, path: str) -> Optional[Checkpoint]:
        if ":" in path:
            run_id, step_str = path.rsplit(":", 1)
            step = int(step_str)
            for ckpt in self._checkpoints.get(run_id, []):
                if ckpt.step == step:
                    return ckpt
        else:
            ckpts = self._checkpoints.get(path, [])
            if ckpts:
                return ckpts[-1]
        return None

    def list_checkpoints(self, run_id: str = "default") -> List[Checkpoint]:
        return list(self._checkpoints.get(run_id, []))

    def prune_checkpoints(self, keep_n: int = 3, run_id: str = "default") -> int:
        ckpts = self._checkpoints.get(run_id, [])
        if len(ckpts) <= keep_n:
            return 0
        removed = len(ckpts) - keep_n
        self._checkpoints[run_id] = ckpts[-keep_n:]
        return removed


class GradientManager:
    """Manage gradient computation and manipulation."""

    def compute_gradients(self, model: Dict[str, float], batch: Dict[str, float]) -> Dict[str, float]:
        grads: Dict[str, float] = {}
        for key in model:
            loss_grad = batch.get(key, 0.0) - model[key]
            grads[key] = loss_grad * 2.0  # simplified gradient
        return grads

    def clip_gradients(self, gradients: Dict[str, float], max_norm: float = 1.0) -> Dict[str, float]:
        norm = math.sqrt(sum(v ** 2 for v in gradients.values()))
        if norm <= max_norm or norm == 0:
            return dict(gradients)
        scale = max_norm / norm
        return {k: v * scale for k, v in gradients.items()}

    def accumulate_gradients(self, batches: List[Dict[str, float]]) -> Dict[str, float]:
        if not batches:
            return {}
        accumulated: Dict[str, float] = {}
        n = len(batches)
        for batch in batches:
            for key, val in batch.items():
                accumulated[key] = accumulated.get(key, 0.0) + val
        return {k: v / n for k, v in accumulated.items()}

    def check_gradient_health(self, gradients: Dict[str, float]) -> Dict[str, Any]:
        if not gradients:
            return {"healthy": False, "issue": "empty gradients"}
        values = list(gradients.values())
        norm = math.sqrt(sum(v ** 2 for v in values))
        has_nan = any(v != v for v in values)  # NaN check
        has_inf = any(abs(v) == float('inf') for v in values)
        issues: List[str] = []
        healthy = True
        if has_nan:
            issues.append("NaN gradients detected")
            healthy = False
        if has_inf:
            issues.append("Inf gradients detected")
            healthy = False
        if norm > 1000:
            issues.append("Gradient explosion detected")
            healthy = False
        if norm < 1e-10 and not has_nan:
            issues.append("Vanishing gradients detected")
            healthy = False
        return {"healthy": healthy, "norm": round(norm, 6), "issues": issues}
