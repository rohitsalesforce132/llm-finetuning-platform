"""Deployment — Chapter 19: Model serving, inference optimization, adapter serving, auto-scaling."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ServingBackend(Enum):
    VLLM = "vllm"
    TGI = "tgi"
    TRITON = "triton"
    CUSTOM = "custom"


class ScalingMetric(Enum):
    CPU = "cpu"
    GPU = "gpu"
    REQUEST_RATE = "request_rate"
    LATENCY = "latency"


@dataclass
class ServingStats:
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0


@dataclass
class ScalingPolicy:
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu: float = 0.7
    target_latency_ms: float = 100.0
    scale_up_cooldown: int = 60
    scale_down_cooldown: int = 300


@dataclass
class CapacityPrediction:
    current_rps: float
    max_rps: float
    utilization: float
    recommended_replicas: int


class ModelServer:
    """Serve fine-tuned models for inference."""

    def __init__(self) -> None:
        self._model: Optional[Dict[str, Any]] = None
        self._stats = ServingStats()
        self._request_times: List[float] = []

    def load_model(self, path: str, device: str = "cpu") -> Dict[str, Any]:
        self._model = {"path": path, "device": device, "loaded": True}
        self._stats = ServingStats()
        return self._model

    def predict(self, input_text: str) -> Dict[str, Any]:
        if self._model is None:
            return {"error": "No model loaded"}
        start = time.time()
        # Simulate prediction
        output = f"Generated response for: {input_text[:50]}"
        elapsed = (time.time() - start) * 1000
        self._stats.total_requests += 1
        self._request_times.append(elapsed)
        self._stats.avg_latency_ms = sum(self._request_times) / len(self._request_times)
        return {"output": output, "latency_ms": round(elapsed, 2)}

    def batch_predict(self, inputs: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(inp) for inp in inputs]

    def get_serving_stats(self) -> ServingStats:
        if self._request_times:
            sorted_times = sorted(self._request_times)
            idx = min(int(len(sorted_times) * 0.99), len(sorted_times) - 1)
            self._stats.p99_latency_ms = round(sorted_times[idx], 2)
        self._stats.throughput_rps = round(1000 / self._stats.avg_latency_ms, 2) if self._stats.avg_latency_ms > 0 else 0
        return self._stats


class InferenceOptimizer:
    """Optimize inference performance."""

    def enable_speculative_decoding(self, model: Dict[str, Any], draft_model: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "speculative_decoding": True,
            "draft_model": draft_model.get("name", "draft"),
            "target_model": model.get("name", "target"),
            "expected_speedup": "1.5-2x",
        }

    def setup_continuous_batching(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "continuous_batching": True,
            "max_batch_size": config.get("max_batch_size", 32),
            "max_tokens": config.get("max_tokens", 2048),
            "scheduling_policy": config.get("scheduling", "fcfs"),
        }

    def apply_kv_cache(self, model: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(model)
        result["kv_cache_enabled"] = True
        result["kv_cache_config"] = {
            "max_seq_length": model.get("context_length", 4096),
            "cache_dtype": "fp16",
            "block_size": 16,
        }
        return result

    def estimate_throughput(self, model: Dict[str, Any], batch_sizes: List[int]) -> Dict[str, float]:
        base_latency = model.get("latency_ms", 50.0)
        results: Dict[str, float] = {}
        for bs in batch_sizes:
            # Throughput scales sublinearly with batch size
            latency = base_latency * (1 + 0.3 * math.log2(max(bs, 1)))
            throughput = bs / (latency / 1000)
            results[f"batch_{bs}"] = round(throughput, 2)
        return results


class AdapterServer:
    """Serve multiple LoRA adapters concurrently."""

    def __init__(self) -> None:
        self._adapters: Dict[str, Dict[str, Any]] = {}
        self._active: Optional[str] = None
        self._stats: Dict[str, int] = {}

    def load_adapter(self, name: str, path: str) -> Dict[str, Any]:
        adapter = {"name": name, "path": path, "loaded": True}
        self._adapters[name] = adapter
        self._stats[name] = 0
        if self._active is None:
            self._active = name
        return adapter

    def switch_adapter(self, name: str) -> bool:
        if name in self._adapters:
            self._active = name
            return True
        return False

    def batch_serve(self, requests: List[Dict[str, str]], adapter_map: Dict[str, str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for req in requests:
            adapter_name = adapter_map.get(req.get("id", ""), self._active or "default")
            if adapter_name in self._adapters:
                self._stats[adapter_name] = self._stats.get(adapter_name, 0) + 1
            results.append({
                "input": req.get("text", ""),
                "adapter": adapter_name,
                "output": f"Response via {adapter_name}",
            })
        return results

    def get_adapter_stats(self) -> Dict[str, Any]:
        return {
            "adapters": list(self._adapters.keys()),
            "active": self._active,
            "request_counts": dict(self._stats),
            "total_adapters": len(self._adapters),
        }


class AutoScaler:
    """Auto-scale inference replicas."""

    def __init__(self) -> None:
        self._policy = ScalingPolicy()
        self._current_replicas = 1
        self._metrics: List[Dict[str, float]] = []

    def set_scaling_policy(self, min_replicas: int = 1, max_replicas: int = 10) -> ScalingPolicy:
        self._policy = ScalingPolicy(min_replicas=min_replicas, max_replicas=max_replicas)
        return self._policy

    def get_current_load(self) -> Dict[str, float]:
        if not self._metrics:
            return {"cpu": 0.5, "gpu": 0.3, "request_rate": 10.0, "latency": 50.0}
        return self._metrics[-1]

    def predict_capacity(self, horizon_minutes: int = 30) -> CapacityPrediction:
        current = self.get_current_load()
        current_rps = current.get("request_rate", 10.0)
        max_per_replica = 100.0  # rps per replica
        max_rps = max_per_replica * self._current_replicas
        utilization = current_rps / max_rps if max_rps > 0 else 1.0
        recommended = max(self._policy.min_replicas,
                         min(self._policy.max_replicas,
                             math.ceil(current_rps / (max_per_replica * self._policy.target_cpu))))
        return CapacityPrediction(
            current_rps=current_rps,
            max_rps=max_rps,
            utilization=round(utilization, 4),
            recommended_replicas=recommended,
        )

    def compute_scaling_decision(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        self._metrics.append(metrics)
        cpu = metrics.get("cpu", 0.5)
        latency = metrics.get("latency", 50.0)
        action = "maintain"
        if cpu > self._policy.target_cpu or latency > self._policy.target_latency_ms:
            if self._current_replicas < self._policy.max_replicas:
                action = "scale_up"
                self._current_replicas += 1
        elif cpu < self._policy.target_cpu * 0.5 and latency < self._policy.target_latency_ms * 0.5:
            if self._current_replicas > self._policy.min_replicas:
                action = "scale_down"
                self._current_replicas -= 1
        return {
            "action": action,
            "current_replicas": self._current_replicas,
            "cpu": cpu,
            "latency": latency,
        }
