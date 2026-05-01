"""PEFT Methods — Chapter 11: LoRA, QLoRA, DoRA, and adapter management."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class AdapterType(Enum):
    LORA = "lora"
    QLORA = "qlora"
    DORA = "dora"
    PREFIX = "prefix"
    ADAPTER = "adapter"


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    dropout: float = 0.05


@dataclass
class AdapterInfo:
    name: str
    adapter_type: AdapterType
    config: Dict[str, Any]
    trainable_params: int = 0
    total_params: int = 0


class LoRAAdapter:
    """Low-Rank Adaptation for parameter-efficient fine-tuning."""

    def apply(self, model: Dict[str, Any], rank: int = 8, alpha: float = 16.0) -> Dict[str, Any]:
        result = copy.deepcopy(model)
        lora_params: Dict[str, Dict[str, Any]] = {}
        for key, val in result.get("weights", {}).items():
            if isinstance(val, (int, float)):
                size = 1
            elif isinstance(val, list):
                size = len(val)
            else:
                size = 64
            lora_params[key] = {
                "rank": rank,
                "alpha": alpha,
                "A": [0.01] * min(rank, size),
                "B": [0.0] * min(rank, size),
                "scaling": alpha / rank,
            }
        result["lora_params"] = lora_params
        result["lora_applied"] = True
        return result

    def merge_weights(self, model: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(model)
        if "lora_params" not in result:
            return result
        weights = result.get("weights", {})
        for key, lora in result["lora_params"].items():
            if key in weights:
                delta = sum(a * b for a, b in zip(lora["A"], lora["B"])) * lora["scaling"]
                if isinstance(weights[key], (int, float)):
                    weights[key] += delta
        result.pop("lora_params", None)
        result["lora_applied"] = False
        result["lora_merged"] = True
        return result

    def get_lora_params(self, model: Dict[str, Any]) -> Dict[str, Any]:
        return model.get("lora_params", {})

    def compute_rank_importance(self, model: Dict[str, Any]) -> Dict[str, float]:
        lora_params = model.get("lora_params", {})
        importance: Dict[str, float] = {}
        for key, lora in lora_params.items():
            a_norm = math.sqrt(sum(x ** 2 for x in lora.get("A", [0])))
            b_norm = math.sqrt(sum(x ** 2 for x in lora.get("B", [0])))
            importance[key] = round(a_norm * b_norm * lora.get("scaling", 1.0), 6)
        return importance


class QLoRAAdapter:
    """Quantized LoRA for memory-efficient fine-tuning."""

    def quantize_model(self, model: Dict[str, Any], bits: int = 4) -> Dict[str, Any]:
        result = copy.deepcopy(model)
        quantized_weights: Dict[str, Any] = {}
        for key, val in result.get("weights", {}).items():
            if isinstance(val, (int, float)):
                levels = 2 ** bits
                quantized_weights[key] = round(val * levels) / levels
            else:
                quantized_weights[key] = val
        result["weights"] = quantized_weights
        result["quantization_bits"] = bits
        result["quantized"] = True
        return result

    def apply_lora(self, model: Dict[str, Any], rank: int = 8) -> Dict[str, Any]:
        lora = LoRAAdapter()
        return lora.apply(model, rank=rank, alpha=rank * 2)

    def prepare_for_training(self, model: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(model)
        result["trainable"] = True
        result["frozen_base"] = True
        result["gradient_checkpointing"] = True
        result["paged_optimizers"] = True
        return result

    def compute_memory_savings(self, model: Dict[str, Any]) -> Dict[str, float]:
        bits = model.get("quantization_bits", 32)
        total_params = model.get("total_params", 1_000_000)
        original_mb = total_params * 32 / 8 / (1024 * 1024)
        quantized_mb = total_params * bits / 8 / (1024 * 1024)
        lora_params = model.get("lora_rank", 8) * 2 * len(model.get("weights", {}))
        lora_mb = lora_params * 32 / 8 / (1024 * 1024)
        return {
            "original_mb": round(original_mb, 2),
            "quantized_mb": round(quantized_mb, 2),
            "lora_overhead_mb": round(lora_mb, 4),
            "savings_pct": round((1 - (quantized_mb + lora_mb) / original_mb) * 100, 1),
        }


class DoRAAdapter:
    """Weight-Decomposed Low-Rank Adaptation."""

    def decompose_weights(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        weights = layer.get("weights", {})
        magnitude: Dict[str, float] = {}
        direction: Dict[str, float] = {}
        for key, val in weights.items():
            if isinstance(val, (int, float)):
                norm = abs(val) if val != 0 else 1.0
                magnitude[key] = norm
                direction[key] = val / norm
            else:
                magnitude[key] = 1.0
                direction[key] = val
        return {"magnitude": magnitude, "direction": direction}

    def apply_dora(self, model: Dict[str, Any], rank: int = 8) -> Dict[str, Any]:
        result = copy.deepcopy(model)
        decomposed = self.decompose_weights(result)
        result["dora"] = {
            "magnitude": decomposed["magnitude"],
            "direction": decomposed["direction"],
            "rank": rank,
            "lora_A": [0.01] * rank,
            "lora_B": [0.0] * rank,
        }
        result["dora_applied"] = True
        return result

    def compute_norm_gradient(self, model: Dict[str, Any]) -> Dict[str, float]:
        dora = model.get("dora", {})
        magnitude = dora.get("magnitude", {})
        grad: Dict[str, float] = {}
        for key, mag in magnitude.items():
            grad[key] = round(1.0 / (mag + 1e-8), 6)
        return grad

    def compare_with_lora(self, dora_metrics: Dict[str, float], lora_metrics: Dict[str, float]) -> Dict[str, Any]:
        dora_loss = dora_metrics.get("loss", 1.0)
        lora_loss = lora_metrics.get("loss", 1.0)
        improvement = ((lora_loss - dora_loss) / lora_loss * 100) if lora_loss != 0 else 0
        return {
            "dora_loss": dora_loss,
            "lora_loss": lora_loss,
            "improvement_pct": round(improvement, 2),
            "better": dora_loss <= lora_loss,
            "extra_params_pct": round(dora_metrics.get("extra_params", 0), 2),
        }


class AdapterRegistry:
    """Registry for managing PEFT adapters."""

    def __init__(self) -> None:
        self._adapters: Dict[str, AdapterInfo] = {}

    def register(self, name: str, adapter_type: AdapterType, config: Dict[str, Any]) -> AdapterInfo:
        info = AdapterInfo(name=name, adapter_type=adapter_type, config=config)
        self._adapters[name] = info
        return info

    def load(self, name: str) -> Optional[AdapterInfo]:
        return self._adapters.get(name)

    def list_adapters(self) -> List[AdapterInfo]:
        return list(self._adapters.values())

    def switch_adapter(self, model: Dict[str, Any], name: str) -> Dict[str, Any]:
        result = copy.deepcopy(model)
        info = self._adapters.get(name)
        if info:
            result["active_adapter"] = name
            result["adapter_config"] = info.config
        return result
