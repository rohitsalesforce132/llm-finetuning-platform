"""Model Selection — Chapter 5 / Stage 2: Base model selection, initialization, tokenization, quantization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ModelFamily(Enum):
    """Supported model families."""
    GPT = "gpt"
    LLAMA = "llama"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    CUSTOM = "custom"


class QuantizationType(Enum):
    """Quantization approaches."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"
    GPTQ = "gptq"
    AWQ = "awq"


class TaskType(Enum):
    """Fine-tuning task types."""
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QA = "qa"
    INSTRUCTION = "instruction"


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    family: ModelFamily
    parameters: int  # in millions
    context_length: int
    vocab_size: int
    hidden_size: int
    num_layers: int
    memory_gb: float


@dataclass
class ModelCriteria:
    """Criteria for model evaluation."""
    max_parameters: Optional[int] = None
    max_memory_gb: Optional[float] = None
    min_context_length: Optional[int] = None
    task: Optional[TaskType] = None


# Pre-built model registry
_MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "gpt2": ModelInfo("gpt2", ModelFamily.GPT, 124, 1024, 50257, 768, 12, 0.5),
    "gpt2-medium": ModelInfo("gpt2-medium", ModelFamily.GPT, 355, 1024, 50257, 1024, 24, 1.4),
    "gpt2-large": ModelInfo("gpt2-large", ModelFamily.GPT, 774, 1024, 50257, 1280, 36, 3.0),
    "llama-7b": ModelInfo("llama-7b", ModelFamily.LLAMA, 7000, 4096, 32000, 4096, 32, 14.0),
    "llama-13b": ModelInfo("llama-13b", ModelFamily.LLAMA, 13000, 4096, 32000, 5120, 40, 26.0),
    "mistral-7b": ModelInfo("mistral-7b", ModelFamily.MISTRAL, 7000, 8192, 32000, 4096, 32, 14.0),
    "gemma-2b": ModelInfo("gemma-2b", ModelFamily.GEMMA, 2000, 8192, 256000, 2048, 18, 4.0),
}


class ModelSelector:
    """Select base models for fine-tuning based on criteria."""

    def evaluate_model(self, model: str, criteria: ModelCriteria) -> float:
        """Evaluate a model against criteria, returning a score 0-1."""
        info = self.get_model_info(model)
        if info is None:
            return 0.0

        score = 1.0
        if criteria.max_parameters and info.parameters > criteria.max_parameters:
            score *= 0.5
        if criteria.max_memory_gb and info.memory_gb > criteria.max_memory_gb:
            score *= 0.5
        if criteria.min_context_length and info.context_length < criteria.min_context_length:
            score *= 0.7

        return round(score, 3)

    def rank_models(self, models: List[str], weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """Rank models by composite score."""
        if weights is None:
            weights = {"memory": 0.3, "parameters": 0.3, "context": 0.2, "vocab": 0.2}

        scores: List[Tuple[str, float]] = []
        for name in models:
            info = self.get_model_info(name)
            if info is None:
                scores.append((name, 0.0))
                continue

            # Normalize and weight
            mem_score = max(0, 1 - info.memory_gb / 30)
            param_score = min(1, info.parameters / 13000)
            ctx_score = min(1, info.context_length / 8192)
            vocab_score = min(1, info.vocab_size / 256000)

            total = (
                weights.get("memory", 0) * mem_score
                + weights.get("parameters", 0) * param_score
                + weights.get("context", 0) * ctx_score
                + weights.get("vocab", 0) * vocab_score
            )
            scores.append((name, round(total, 3)))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def recommend(self, task: TaskType, constraints: ModelCriteria) -> Optional[str]:
        """Recommend the best model for a task given constraints."""
        candidates = list(_MODEL_REGISTRY.keys())
        ranked = self.rank_models(candidates)
        for name, score in ranked:
            if self.evaluate_model(name, constraints) > 0.5:
                return name
        return ranked[0][0] if ranked else None

    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return _MODEL_REGISTRY.get(model)


class InitializationChecker:
    """Validate model initialization and configuration."""

    def check_config(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Check model configuration for issues."""
        issues: List[str] = []
        warnings: List[str] = []

        required_keys = ["hidden_size", "num_layers", "vocab_size"]
        for key in required_keys:
            if key not in model:
                issues.append(f"Missing required config key: {key}")

        if "hidden_size" in model and model["hidden_size"] % 64 != 0:
            warnings.append("hidden_size not divisible by 64 may reduce efficiency")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }

    def validate_layers(self, model: Dict[str, Any]) -> bool:
        """Validate model layer configuration."""
        num_layers = model.get("num_layers", 0)
        if num_layers <= 0:
            return False
        if "layer_dims" in model:
            if len(model["layer_dims"]) != num_layers:
                return False
        return True

    def count_parameters(self, model: Dict[str, Any]) -> int:
        """Count total parameters in the model."""
        total = 0
        hidden = model.get("hidden_size", 0)
        vocab = model.get("vocab_size", 0)
        layers = model.get("num_layers", 0)

        # Embedding params
        total += hidden * vocab
        # Per-layer params (Q, K, V, O projections + FFN)
        total += layers * (4 * hidden * hidden + 2 * hidden * 4 * hidden)
        # Output head
        total += hidden * vocab

        return total

    def get_memory_estimate(self, model: Dict[str, Any], batch_size: int = 1) -> Dict[str, float]:
        """Estimate memory usage in GB."""
        params = self.count_parameters(model)
        bytes_per_param = 4  # float32
        param_memory_gb = (params * bytes_per_param) / (1024 ** 3)

        # Activation memory estimate (rough)
        hidden = model.get("hidden_size", 768)
        layers = model.get("num_layers", 12)
        seq_len = model.get("context_length", 512)
        activation_memory_gb = (batch_size * seq_len * hidden * layers * bytes_per_param) / (1024 ** 3)

        return {
            "parameters_gb": round(param_memory_gb, 2),
            "activations_gb": round(activation_memory_gb, 2),
            "total_gb": round(param_memory_gb + activation_memory_gb, 2),
        }


class TokenizerManager:
    """Manage tokenizers for training and inference."""

    def __init__(self) -> None:
        self._tokenizers: Dict[str, Dict[str, Any]] = {}

    def load_tokenizer(self, name: str) -> Dict[str, Any]:
        """Load a tokenizer by name (simulated)."""
        if name not in self._tokenizers:
            # Create a simple simulated tokenizer
            self._tokenizers[name] = {
                "name": name,
                "vocab_size": 50257 if "gpt" in name else 32000,
                "vocab": {f"token_{i}": i for i in range(50257 if "gpt" in name else 32000)},
                "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
            }
        return self._tokenizers[name]

    def check_vocab_coverage(self, tokenizer: Dict[str, Any], data: List[str]) -> Dict[str, Any]:
        """Check how well tokenizer vocabulary covers the data."""
        vocab = set(tokenizer.get("vocab", {}).keys())
        total_words = 0
        covered_words = 0

        for text in data:
            words = text.lower().split()
            total_words += len(words)
            covered_words += sum(1 for w in words if w in vocab)

        coverage = covered_words / total_words if total_words else 0.0
        return {
            "total_words": total_words,
            "covered_words": covered_words,
            "coverage_ratio": round(coverage, 4),
            "oov_words": total_words - covered_words,
        }

    def extend_tokenizer(self, tokenizer: Dict[str, Any], new_tokens: List[str]) -> Dict[str, Any]:
        """Extend tokenizer with new tokens."""
        updated = dict(tokenizer)
        vocab = dict(updated.get("vocab", {}))
        start_id = max(vocab.values()) + 1 if vocab else 0

        for token in new_tokens:
            if token not in vocab:
                vocab[token] = start_id
                start_id += 1

        updated["vocab"] = vocab
        updated["vocab_size"] = len(vocab)
        return updated

    def compute_token_stats(self, tokenizer: Dict[str, Any], data: List[str]) -> Dict[str, Any]:
        """Compute tokenization statistics for data."""
        vocab = tokenizer.get("vocab", {})
        lengths: List[int] = []

        for text in data:
            # Simple word-level tokenization simulation
            tokens = text.lower().split()
            lengths.append(len(tokens))

        avg_len = sum(lengths) / len(lengths) if lengths else 0
        return {
            "num_samples": len(data),
            "avg_tokens": round(avg_len, 2),
            "min_tokens": min(lengths) if lengths else 0,
            "max_tokens": max(lengths) if lengths else 0,
            "total_tokens": sum(lengths),
        }


class QuantizationPrep:
    """Prepare models for quantization."""

    def analyze_quantization_impact(self, model: Dict[str, Any], bits: int = 4) -> Dict[str, Any]:
        """Analyze the impact of quantization on model quality."""
        total_params = model.get("total_params", 1_000_000)

        original_bits = 32
        compression_ratio = original_bits / bits
        memory_savings = (1 - bits / original_bits) * 100

        # Estimated quality loss (heuristic)
        if bits >= 16:
            quality_loss = 0.0
        elif bits >= 8:
            quality_loss = 1.0
        elif bits >= 4:
            quality_loss = 3.0
        else:
            quality_loss = 8.0

        return {
            "bits": bits,
            "compression_ratio": round(compression_ratio, 1),
            "memory_savings_pct": round(memory_savings, 1),
            "estimated_quality_loss_pct": quality_loss,
            "original_size_mb": round(total_params * original_bits / 8 / (1024 * 1024), 1),
            "quantized_size_mb": round(total_params * bits / 8 / (1024 * 1024), 1),
        }

    def prepare_for_quantization(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare model for quantization (calibration config)."""
        return {
            "calibration_dataset_size": 128,
            "calibration_seq_length": model.get("context_length", 2048),
            "quantize_embeddings": False,
            "quantize_head": True,
            "per_channel": True,
            "symmetric": True,
        }

    def estimate_memory_savings(self, model: Dict[str, Any], bits: int = 4) -> float:
        """Estimate memory savings in GB from quantization."""
        total_params = model.get("total_params", 1_000_000)
        original_gb = total_params * 32 / 8 / (1024 ** 3)
        quantized_gb = total_params * bits / 8 / (1024 ** 3)
        return round(original_gb - quantized_gb, 2)

    def validate_quantized_model(self, model: Dict[str, Any]) -> bool:
        """Validate that a quantized model is functional."""
        required = ["quantization_config", "weights"]
        return all(key in model for key in required)
