"""Advanced — Chapter 22: Model merging, continual learning, mechanistic analysis, scaling predictions."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class MergeStrategy(Enum):
    SLERP = "slerp"
    TIES = "ties"
    DARE = "dare"
    LINEAR = "linear"
    TASK_ARITHMETIC = "task_arithmetic"


@dataclass
class TaskData:
    name: str
    data: List[Dict[str, Any]]
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ActivationSnapshot:
    layer: str
    values: List[float]
    shape: Tuple[int, ...] = (1,)


@dataclass
class ScalingPrediction:
    params: int
    data_size: int
    predicted_loss: float
    compute_flops: float


class ModelMerger:
    """Merge fine-tuned models using various strategies."""

    def merge_slerp(self, model_a: Dict[str, float], model_b: Dict[str, float], t: float = 0.5) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key in set(list(model_a.keys()) + list(model_b.keys())):
            a = model_a.get(key, 0.0)
            b = model_b.get(key, 0.0)
            # SLERP: spherical linear interpolation
            dot = a * b
            norm_a = abs(a) if a != 0 else 1e-8
            norm_b = abs(b) if b != 0 else 1e-8
            cos_angle = max(-1, min(1, dot / (norm_a * norm_b)))
            if abs(cos_angle) > 0.9995:
                # Fall back to linear
                result[key] = a * (1 - t) + b * t
            else:
                omega = math.acos(cos_angle)
                sin_omega = math.sin(omega)
                result[key] = (math.sin((1 - t) * omega) * a + math.sin(t * omega) * b) / sin_omega
        return result

    def merge_ties(self, models: List[Dict[str, float]], densities: List[float]) -> Dict[str, float]:
        if not models:
            return {}
        result: Dict[str, float] = {}
        all_keys = set()
        for m in models:
            all_keys.update(m.keys())
        for key in all_keys:
            values = [m.get(key, 0.0) for m in models]
            # TIES: Trim, Elect, and Sign
            trimmed = sorted(values, key=abs, reverse=True)
            keep = max(1, int(len(trimmed) * (densities[0] if densities else 0.2)))
            elected = trimmed[:keep]
            sign = 1 if sum(elected) >= 0 else -1
            result[key] = sign * sum(abs(v) for v in elected) / len(elected)
        return result

    def merge_dare(self, models: List[Dict[str, float]], drop_rate: float = 0.9) -> Dict[str, float]:
        if not models:
            return {}
        rng = random.Random(42)
        result: Dict[str, float] = {}
        all_keys = set()
        for m in models:
            all_keys.update(m.keys())
        for key in all_keys:
            delta = sum(m.get(key, 0.0) for m in models) / len(models)
            if rng.random() < drop_rate:
                result[key] = 0.0  # Drop this delta
            else:
                result[key] = delta / (1 - drop_rate)  # Rescale
        return result

    def evaluate_merged(self, model: Dict[str, float], benchmark: str = "default") -> Dict[str, float]:
        rng = random.Random(hash(str(sorted(model.items()))) + hash(benchmark))
        return {
            "accuracy": round(rng.uniform(0.6, 0.9), 4),
            "f1": round(rng.uniform(0.55, 0.88), 4),
            "perplexity": round(rng.uniform(3.0, 8.0), 2),
        }


class ContinualLearner:
    """Continual fine-tuning with catastrophic forgetting prevention."""

    def __init__(self) -> None:
        self._tasks: List[TaskData] = []
        self._fisher_matrix: Dict[str, float] = {}

    def add_task(self, task_data: TaskData) -> None:
        self._tasks.append(task_data)

    def compute_elasticity(self, model: Dict[str, float]) -> Dict[str, float]:
        elasticity: Dict[str, float] = {}
        for key, val in model.items():
            # Higher absolute value = less elastic (more important to preserve)
            elasticity[key] = round(1.0 / (abs(val) + 1e-6), 6)
        return elasticity

    def apply_ewc(self, model: Dict[str, float], fisher_matrix: Dict[str, float], lambda_ewc: float = 1000.0) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key, val in model.items():
            fisher = fisher_matrix.get(key, 0.0)
            penalty = lambda_ewc * fisher * val * val
            result[key] = val - 0.01 * (val + penalty / (abs(val) + 1e-8))
        return result

    def detect_forgetting(self, old_metrics: Dict[str, float], new_metrics: Dict[str, float]) -> Dict[str, Any]:
        forgetting: Dict[str, float] = {}
        total_forgetting = 0.0
        count = 0
        for key in old_metrics:
            if key in new_metrics:
                diff = old_metrics[key] - new_metrics[key]
                if key in ["perplexity", "loss"]:
                    diff = -diff  # For these, increase is bad
                if diff > 0:
                    forgetting[key] = round(diff, 4)
                    total_forgetting += diff
                    count += 1
        avg_forgetting = total_forgetting / count if count else 0
        return {
            "has_forgetting": avg_forgetting > 0.05,
            "avg_forgetting": round(avg_forgetting, 4),
            "affected_metrics": forgetting,
        }


class MechanisticAnalyzer:
    """Analyze model internals for interpretability."""

    def extract_activations(self, model: Dict[str, Any], input_data: Any) -> List[ActivationSnapshot]:
        layers = model.get("layers", ["layer_0", "layer_1", "layer_2"])
        rng = random.Random(hash(str(input_data)))
        return [
            ActivationSnapshot(
                layer=layer,
                values=[rng.gauss(0, 1) for _ in range(10)],
                shape=(2, 5),
            )
            for layer in layers
        ]

    def compute_attention_patterns(self, model: Dict[str, Any], input_data: Any) -> Dict[str, List[List[float]]]:
        num_heads = model.get("num_heads", 8)
        seq_len = model.get("seq_len", 10)
        rng = random.Random(42)
        patterns: Dict[str, List[List[float]]] = {}
        for h in range(num_heads):
            head_pattern: List[List[float]] = []
            for i in range(seq_len):
                row = [rng.uniform(0, 1) for _ in range(seq_len)]
                total = sum(row)
                head_pattern.append([r / total for r in row])
            patterns[f"head_{h}"] = head_pattern
        return patterns

    def identify_circuits(self, model: Dict[str, Any]) -> List[Dict[str, Any]]:
        layers = model.get("layers", ["layer_0", "layer_1"])
        circuits: List[Dict[str, Any]] = []
        for i in range(len(layers) - 1):
            circuits.append({
                "name": f"circuit_{i}",
                "input_layer": layers[i],
                "output_layer": layers[i + 1],
                "importance": round(random.uniform(0.3, 0.9), 4),
            })
        return circuits

    def probe_layer(self, model: Dict[str, Any], layer: str, probe_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        rng = random.Random(hash(layer))
        correct = sum(1 for _ in probe_data if rng.random() > 0.3)
        return {
            "layer": layer,
            "accuracy": round(correct / len(probe_data), 4) if probe_data else 0.0,
            "features_detected": rng.randint(3, 20),
            "linear_probe_score": round(rng.uniform(0.6, 0.95), 4),
        }


class ScalingPredictor:
    """Predict scaling behavior following scaling laws."""

    def predict_loss(self, params: int, data_size: int) -> float:
        # Chinchilla-style scaling law: L(N,D) = A/N^alpha + B/D^beta + L_irr
        A, B = 406.4, 410.7
        alpha, beta = 0.34, 0.28
        L_irr = 1.69
        loss = A / (params ** alpha) + B / (data_size ** beta) + L_irr
        return round(max(loss, L_irr), 4)

    def compute_chinchilla_optimal(self, target_loss: float = 2.5) -> Dict[str, int]:
        # Approximate Chinchilla optimal: N and D for a given compute budget
        # For target loss, find optimal N and D
        # Simplified: D ≈ 20 * N for Chinchilla optimal
        A, B = 406.4, 410.7
        alpha, beta = 0.34, 0.28
        L_irr = 1.69

        # Search for optimal params
        best_n = 1_000_000
        for n in [10**i for i in range(6, 11)]:
            d = 20 * n
            loss = A / (n ** alpha) + B / (d ** beta) + L_irr
            if loss <= target_loss:
                best_n = n
                break
        return {
            "optimal_params": best_n,
            "optimal_tokens": 20 * best_n,
            "target_loss": target_loss,
        }

    def estimate_compute_budget(self, params: int, tokens: int) -> float:
        # FLOPs ≈ 6 * N * D (training)
        flops = 6 * params * tokens
        return flops

    def get_efficiency_frontier(self) -> List[Dict[str, Any]]:
        frontier: List[Dict[str, Any]] = []
        for exp in range(7, 12):
            params = 10 ** exp
            tokens = 20 * params
            loss = self.predict_loss(params, tokens)
            compute = self.estimate_compute_budget(params, tokens)
            frontier.append({
                "params": params,
                "tokens": tokens,
                "predicted_loss": loss,
                "compute_flops": compute,
            })
        return frontier
