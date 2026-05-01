"""Monitoring — Chapter 20: Drift detection, hallucination monitoring, cost tracking, feedback loops."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DriftType(Enum):
    COVARIATE = "covariate"
    CONCEPT = "concept"
    PREDICTION = "prediction"
    LABEL = "label"


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftReport:
    drift_score: float
    drift_type: DriftType
    detected: bool
    features_affected: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class FeedbackEntry:
    response_id: str
    rating: float
    timestamp: float
    comment: str = ""


@dataclass
class CostEntry:
    tokens: int
    model: str
    cost_usd: float
    timestamp: float


class DriftDetector:
    """Detect model drift in production."""

    def __init__(self) -> None:
        self._baseline: Optional[List[float]] = None
        self._history: List[float] = []

    def baseline(self, embeddings: List[float]) -> None:
        self._baseline = list(embeddings)

    def check_drift(self, current_embeddings: List[float]) -> DriftReport:
        if self._baseline is None:
            return DriftReport(drift_score=0.0, drift_type=DriftType.COVARIATE, detected=False)
        # Compute cosine distance between baseline and current
        dot = sum(a * b for a, b in zip(self._baseline, current_embeddings))
        norm_a = math.sqrt(sum(a ** 2 for a in self._baseline))
        norm_b = math.sqrt(sum(b ** 2 for b in current_embeddings))
        if norm_a == 0 or norm_b == 0:
            drift_score = 1.0
        else:
            similarity = dot / (norm_a * norm_b)
            drift_score = 1.0 - similarity
        self._history.append(drift_score)
        detected = drift_score > 0.3
        return DriftReport(
            drift_score=round(drift_score, 4),
            drift_type=DriftType.COVARIATE,
            detected=detected,
            features_affected=["embedding_space"] if detected else [],
            recommendation="Consider retraining" if detected else "No action needed",
        )

    def get_drift_report(self) -> Dict[str, Any]:
        if not self._history:
            return {"measurements": 0, "current_drift": 0.0, "trend": "unknown"}
        trend = "increasing" if len(self._history) > 1 and self._history[-1] > self._history[0] else "stable"
        return {
            "measurements": len(self._history),
            "current_drift": round(self._history[-1], 4),
            "max_drift": round(max(self._history), 4),
            "avg_drift": round(sum(self._history) / len(self._history), 4),
            "trend": trend,
        }

    def suggest_retraining(self, drift_score: float) -> Dict[str, Any]:
        if drift_score < 0.2:
            return {"retrain": False, "urgency": "none", "reason": "Drift within acceptable range"}
        elif drift_score < 0.4:
            return {"retrain": True, "urgency": "low", "reason": "Moderate drift detected"}
        elif drift_score < 0.6:
            return {"retrain": True, "urgency": "medium", "reason": "Significant drift detected"}
        else:
            return {"retrain": True, "urgency": "high", "reason": "Severe drift detected, immediate action needed"}


class HallucinationMonitor:
    """Monitor hallucinations in production outputs."""

    def __init__(self) -> None:
        self._checks: List[Dict[str, Any]] = []
        self._rate: float = 0.0

    def check_response(self, response: str, context: str) -> Dict[str, Any]:
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        if not response_words:
            return {"hallucination_score": 0.0, "flagged": False}
        # Words in response not in context
        novel_words = response_words - context_words
        score = len(novel_words) / len(response_words) if response_words else 0
        flagged = score > 0.5
        result = {"hallucination_score": round(score, 4), "flagged": flagged, "novel_word_count": len(novel_words)}
        self._checks.append(result)
        self._rate = sum(c["hallucination_score"] for c in self._checks) / len(self._checks)
        return result

    def get_hallucination_rate(self) -> float:
        return round(self._rate, 4)

    def alert_if_above(self, threshold: float = 0.3) -> Optional[AlertLevel]:
        if self._rate > threshold * 2:
            return AlertLevel.CRITICAL
        elif self._rate > threshold:
            return AlertLevel.WARNING
        return None

    def suggest_corrections(self, response: str) -> List[str]:
        suggestions: List[str] = []
        if len(response.split()) > 200:
            suggestions.append("Response is very long, consider grounding with retrieval")
        if "?" in response and "don't know" not in response.lower():
            suggestions.append("Consider hedging uncertain claims")
        if any(str(i) in response for i in range(2020, 2030)):
            suggestions.append("Verify temporal claims against current data")
        if not suggestions:
            suggestions.append("No corrections suggested")
        return suggestions


class CostMonitor:
    """Track and optimize inference costs."""

    def __init__(self) -> None:
        self._entries: List[CostEntry] = []
        self._pricing: Dict[str, float] = {"gpt-4": 0.03, "gpt-3.5": 0.002, "llama": 0.0001}

    def track_request(self, tokens: int, model: str) -> CostEntry:
        price_per_1k = self._pricing.get(model, 0.01)
        cost = tokens / 1000 * price_per_1k
        entry = CostEntry(tokens=tokens, model=model, cost_usd=round(cost, 6), timestamp=time.time())
        self._entries.append(entry)
        return entry

    def get_cost_report(self, period_hours: int = 24) -> Dict[str, Any]:
        cutoff = time.time() - period_hours * 3600
        recent = [e for e in self._entries if e.timestamp >= cutoff]
        if not recent:
            return {"total_cost": 0.0, "total_tokens": 0, "requests": 0}
        return {
            "total_cost": round(sum(e.cost_usd for e in recent), 4),
            "total_tokens": sum(e.tokens for e in recent),
            "requests": len(recent),
            "avg_cost_per_request": round(sum(e.cost_usd for e in recent) / len(recent), 6),
            "by_model": {m: round(sum(e.cost_usd for e in recent if e.model == m), 4)
                        for m in set(e.model for e in recent)},
        }

    def estimate_monthly_cost(self) -> float:
        if not self._entries:
            return 0.0
        daily_cost = sum(e.cost_usd for e in self._entries) / max(1, len(self._entries) / 100)
        return round(daily_cost * 30, 2)

    def suggest_cost_optimizations(self) -> List[str]:
        suggestions: List[str] = []
        if not self._entries:
            return ["No data available for optimization"]
        model_counts: Dict[str, int] = {}
        for e in self._entries:
            model_counts[e.model] = model_counts.get(e.model, 0) + 1
        expensive = [m for m in model_counts if self._pricing.get(m, 0) > 0.01]
        if expensive:
            suggestions.append(f"Consider using cheaper models for non-critical requests: {expensive}")
        avg_tokens = sum(e.tokens for e in self._entries) / len(self._entries)
        if avg_tokens > 1000:
            suggestions.append("High average token count — consider prompt compression")
        if len(model_counts) == 1:
            suggestions.append("Consider model routing based on task complexity")
        return suggestions if suggestions else ["Costs are optimized"]


class FeedbackLoop:
    """Production feedback loop for continuous improvement."""

    def __init__(self) -> None:
        self._feedback: List[FeedbackEntry] = []

    def collect_feedback(self, response_id: str, rating: float, comment: str = "") -> FeedbackEntry:
        entry = FeedbackEntry(response_id=response_id, rating=rating, timestamp=time.time(), comment=comment)
        self._feedback.append(entry)
        return entry

    def analyze_feedback(self, window_hours: int = 24) -> Dict[str, Any]:
        cutoff = time.time() - window_hours * 3600
        recent = [f for f in self._feedback if f.timestamp >= cutoff]
        if not recent:
            return {"count": 0, "avg_rating": 0.0, "negative_ratio": 0.0}
        ratings = [f.rating for f in recent]
        negative = sum(1 for r in ratings if r < 3)
        return {
            "count": len(recent),
            "avg_rating": round(sum(ratings) / len(ratings), 2),
            "negative_ratio": round(negative / len(recent), 4),
            "min_rating": min(ratings),
            "max_rating": max(ratings),
        }

    def should_trigger_retraining(self, feedback: Optional[Dict[str, Any]] = None) -> bool:
        if feedback is None:
            feedback = self.analyze_feedback()
        return feedback.get("negative_ratio", 0) > 0.3 or feedback.get("avg_rating", 5) < 3.0

    def get_feedback_stats(self) -> Dict[str, Any]:
        if not self._feedback:
            return {"total": 0, "avg_rating": 0.0, "distribution": {}}
        ratings = [f.rating for f in self._feedback]
        dist: Dict[str, int] = {}
        for r in ratings:
            bucket = f"{int(r)}-star"
            dist[bucket] = dist.get(bucket, 0) + 1
        return {
            "total": len(self._feedback),
            "avg_rating": round(sum(ratings) / len(ratings), 2),
            "distribution": dist,
        }
