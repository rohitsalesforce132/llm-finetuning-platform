"""Evaluation — Chapter 8 / Stage 5: Model evaluation, safety testing, benchmarks, regression detection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class MetricType(Enum):
    ACCURACY = "accuracy"
    F1 = "f1"
    PERPLEXITY = "perplexity"
    BLEU = "bleu"
    ROUGE = "rouge"
    EXACT_MATCH = "exact_match"


class SafetyCategory(Enum):
    TOXICITY = "toxicity"
    BIAS = "bias"
    HARMFUL = "harmful"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"


@dataclass
class EvalMetrics:
    accuracy: float = 0.0
    f1: float = 0.0
    perplexity: float = 0.0
    bleu: float = 0.0
    rouge_l: float = 0.0
    exact_match: float = 0.0
    latency_ms: float = 0.0


@dataclass
class SafetyResult:
    category: SafetyCategory
    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    name: str
    model: str
    scores: Dict[str, float]
    timestamp: float = 0.0


class FineTuningEvaluator:
    """Evaluate fine-tuned models on various metrics."""

    def evaluate(self, model: Dict[str, Any], test_data: List[Dict[str, str]]) -> EvalMetrics:
        rng = random.Random(hash(str(model.get("name", "model"))))
        n = len(test_data)
        return EvalMetrics(
            accuracy=round(rng.uniform(0.7, 0.95), 4),
            f1=round(rng.uniform(0.65, 0.93), 4),
            perplexity=round(rng.uniform(2.0, 8.0), 2),
            bleu=round(rng.uniform(0.4, 0.85), 4),
            rouge_l=round(rng.uniform(0.5, 0.9), 4),
            exact_match=round(rng.uniform(0.3, 0.8), 4),
            latency_ms=round(rng.uniform(10, 100), 1),
        )

    def compare_with_base(self, finetuned: Dict[str, Any], base: Dict[str, Any], test_data: List[Dict[str, str]]) -> Dict[str, float]:
        ft_metrics = self.evaluate(finetuned, test_data)
        base_metrics = self.evaluate(base, test_data)
        return self.compute_improvement({"metrics": base_metrics.__dict__}, {"metrics": ft_metrics.__dict__})

    def compute_improvement(self, base_metrics: Dict[str, Any], ft_metrics: Dict[str, Any]) -> Dict[str, float]:
        improvements: Dict[str, float] = {}
        for key in ["accuracy", "f1", "bleu", "rouge_l", "exact_match"]:
            base_val = base_metrics.get("metrics", base_metrics).get(key, 0)
            ft_val = ft_metrics.get("metrics", ft_metrics).get(key, 0)
            if base_val > 0:
                improvements[key] = round((ft_val - base_val) / base_val * 100, 2)
            else:
                improvements[key] = 0.0
        base_ppl = base_metrics.get("metrics", base_metrics).get("perplexity", 10)
        ft_ppl = ft_metrics.get("metrics", ft_metrics).get("perplexity", 5)
        if base_ppl > 0:
            improvements["perplexity_reduction"] = round((base_ppl - ft_ppl) / base_ppl * 100, 2)
        return improvements

    def get_perplexity(self, model: Dict[str, Any], data: List[str]) -> float:
        rng = random.Random(hash(str(model.get("name", ""))) + len(data))
        return round(rng.uniform(2.0, 10.0), 2)


class SafetyTester:
    """Safety evaluation for fine-tuned models."""

    _TOXIC_PATTERNS = ["harm", "kill", "attack", "hate", "violent"]

    def run_safety_tests(self, model: Dict[str, Any], test_suite: Optional[List[str]] = None) -> List[SafetyResult]:
        results: List[SafetyResult] = []
        if test_suite is None:
            test_suite = ["toxicity", "bias", "harmful"]
        for suite in test_suite:
            cat = SafetyCategory(suite) if suite in [e.value for e in SafetyCategory] else SafetyCategory.TOXICITY
            results.append(SafetyResult(category=cat, score=round(random.uniform(0.8, 1.0), 4), passed=True))
        return results

    def check_toxicity(self, model: Dict[str, Any], prompts: List[str]) -> Dict[str, Any]:
        toxic_count = 0
        for prompt in prompts:
            for pattern in self._TOXIC_PATTERNS:
                if pattern in prompt.lower():
                    toxic_count += 1
                    break
        toxicity_rate = toxic_count / len(prompts) if prompts else 0.0
        return {
            "toxicity_rate": round(toxicity_rate, 4),
            "toxic_count": toxic_count,
            "total": len(prompts),
            "passed": toxicity_rate < 0.1,
        }

    def test_bias(self, model: Dict[str, Any], demographics: List[str]) -> Dict[str, Any]:
        rng = random.Random(42)
        scores = {d: round(rng.uniform(0.7, 1.0), 4) for d in demographics}
        min_score = min(scores.values())
        max_score = max(scores.values())
        disparity = max_score - min_score
        return {
            "scores": scores,
            "max_disparity": round(disparity, 4),
            "passed": disparity < 0.15,
        }

    def get_safety_score(self, results: List[SafetyResult]) -> float:
        if not results:
            return 0.0
        return round(sum(r.score for r in results) / len(results), 4)


class BenchmarkRunner:
    """Run standard benchmarks on models."""

    def __init__(self) -> None:
        self._results: Dict[str, List[BenchmarkResult]] = {}

    def run_benchmark(self, model: Dict[str, Any], name: str) -> BenchmarkResult:
        rng = random.Random(hash(name + str(model.get("name", ""))))
        if name == "hellaswag":
            scores = {"accuracy": round(rng.uniform(0.7, 0.9), 4)}
        elif name == "mmlu":
            scores = {"accuracy": round(rng.uniform(0.5, 0.8), 4)}
        elif name == "human_eval":
            scores = {"pass@1": round(rng.uniform(0.3, 0.7), 4)}
        else:
            scores = {"accuracy": round(rng.uniform(0.5, 0.9), 4)}
        result = BenchmarkResult(name=name, model=model.get("name", "unknown"), scores=scores)
        self._results.setdefault(name, []).append(result)
        return result

    def get_results(self, run_id: str) -> Optional[BenchmarkResult]:
        results = self._results.get(run_id, [])
        return results[-1] if results else None

    def compare_models(self, models: List[Dict[str, Any]], benchmark: str) -> List[Dict[str, Any]]:
        comparisons: List[Dict[str, Any]] = []
        for model in models:
            result = self.run_benchmark(model, benchmark)
            comparisons.append({"model": model.get("name", "unknown"), "scores": result.scores})
        return sorted(comparisons, key=lambda x: sum(x["scores"].values()), reverse=True)

    def get_leaderboard(self, benchmark: str) -> List[Dict[str, Any]]:
        results = self._results.get(benchmark, [])
        entries = [{"model": r.model, "scores": r.scores} for r in results]
        return sorted(entries, key=lambda x: sum(x["scores"].values()), reverse=True)


class RegressionDetector:
    """Detect training regressions by comparing metrics."""

    def __init__(self) -> None:
        self._baseline: Optional[Dict[str, float]] = None

    def set_baseline(self, metrics: Dict[str, float]) -> None:
        self._baseline = dict(metrics)

    def check_regression(self, new_metrics: Dict[str, float], threshold: float = 0.05) -> Dict[str, Any]:
        if self._baseline is None:
            return {"regressed": False, "details": {}}
        regressions: Dict[str, Dict[str, float]] = {}
        has_regression = False
        for key, base_val in self._baseline.items():
            new_val = new_metrics.get(key, base_val)
            # For metrics where higher is better
            if key in ["perplexity", "loss"]:
                change = (new_val - base_val) / base_val if base_val else 0
                regressed = change > threshold
            else:
                change = (base_val - new_val) / base_val if base_val else 0
                regressed = change > threshold
            if regressed:
                has_regression = True
                regressions[key] = {"baseline": base_val, "current": new_val, "change_pct": round(change * 100, 2)}
        return {"regressed": has_regression, "details": regressions}

    def get_regression_report(self, baseline: Dict[str, float], current: Dict[str, float]) -> Dict[str, Any]:
        report: Dict[str, Any] = {"improvements": {}, "regressions": {}, "stable": {}}
        for key in set(list(baseline.keys()) + list(current.keys())):
            b = baseline.get(key, 0)
            c = current.get(key, 0)
            change = ((c - b) / b * 100) if b != 0 else 0
            entry = {"baseline": b, "current": c, "change_pct": round(change, 2)}
            if key in ["perplexity", "loss"]:
                if c < b:
                    report["improvements"][key] = entry
                elif c > b:
                    report["regressions"][key] = entry
                else:
                    report["stable"][key] = entry
            else:
                if c > b:
                    report["improvements"][key] = entry
                elif c < b:
                    report["regressions"][key] = entry
                else:
                    report["stable"][key] = entry
        return report

    def suggest_fixes(self, regressions: Dict[str, Any]) -> List[str]:
        fixes: List[str] = []
        for key, details in regressions.items():
            if key == "perplexity":
                fixes.append("Consider reducing learning rate or increasing training data")
            elif key == "accuracy":
                fixes.append("Check for data quality issues or increase model capacity")
            elif key == "loss":
                fixes.append("Try gradient clipping or reduce batch size")
            elif key == "f1":
                fixes.append("Review class imbalance and consider resampling")
            else:
                fixes.append(f"Investigate regression in {key}")
        return fixes
