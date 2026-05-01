"""Preference Optimization — Chapters 16-18: PPO, DPO, ORPO, Reward Models."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class PreferenceMethod(Enum):
    PPO = "ppo"
    DPO = "dpo"
    ORPO = "orpo"
    RLHF = "rlhf"
    KTO = "kto"


@dataclass
class PreferencePair:
    prompt: str
    preferred: str
    rejected: str
    preferred_score: float = 1.0
    rejected_score: float = 0.0


@dataclass
class PPOConfig:
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95


class PPOTrainer:
    """Proximal Policy Optimization for RLHF."""

    def compute_advantages(self, rewards: List[float], values: List[float], gamma: float = 0.99, lam: float = 0.95) -> List[float]:
        advantages: List[float] = []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[i + 1]
            delta = rewards[i] + gamma * next_value - values[i]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        return advantages

    def clip_ratio(self, new_prob: float, old_prob: float, epsilon: float = 0.2) -> float:
        if old_prob == 0:
            return 1.0
        ratio = new_prob / old_prob
        return max(1 - epsilon, min(1 + epsilon, ratio))

    def compute_ppo_loss(self, advantages: List[float], ratios: List[float], clip_epsilon: float = 0.2) -> Dict[str, float]:
        if not advantages or not ratios:
            return {"total_loss": 0.0, "clipped_loss": 0.0, "unclipped_loss": 0.0}
        clipped_ratios = [max(1 - clip_epsilon, min(1 + clip_epsilon, r)) for r in ratios]
        unclipped = [-a * r for a, r in zip(advantages, ratios)]
        clipped = [-a * cr for a, cr in zip(advantages, clipped_ratios)]
        losses = [max(u, c) for u, c in zip(unclipped, clipped)]
        return {
            "total_loss": round(sum(losses) / len(losses), 6),
            "clipped_loss": round(sum(clipped) / len(clipped), 6),
            "unclipped_loss": round(sum(unclipped) / len(unclipped), 6),
        }

    def train_step(self, model: Dict[str, Any], batch: List[Dict[str, float]]) -> Dict[str, Any]:
        rewards = [b.get("reward", 0.0) for b in batch]
        values = [b.get("value", 0.5) for b in batch]
        old_probs = [b.get("old_prob", 0.5) for b in batch]
        advantages = self.compute_advantages(rewards, values)
        ratios = [1.0 + random.Random(i).uniform(-0.1, 0.1) for i in range(len(batch))]
        loss_dict = self.compute_ppo_loss(advantages, ratios)
        return {"loss": loss_dict["total_loss"], "advantages": advantages, "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0}


class DPOTrainer:
    """Direct Preference Optimization."""

    def compute_dpo_loss(self, preferred_logps: List[float], rejected_logps: List[float], beta: float = 0.1) -> Dict[str, float]:
        if not preferred_logps:
            return {"loss": 0.0, "accuracies": 0.0}
        losses: List[float] = []
        correct = 0
        for pref_lp, rej_lp in zip(preferred_logps, rejected_logps):
            log_ratio = beta * (pref_lp - rej_lp)
            loss = -math.log(math.exp(log_ratio) / (1 + math.exp(log_ratio)) + 1e-8)
            losses.append(loss)
            if pref_lp > rej_lp:
                correct += 1
        return {
            "loss": round(sum(losses) / len(losses), 6),
            "accuracies": round(correct / len(losses), 4),
            "margin": round(sum(p - r for p, r in zip(preferred_logps, rejected_logps)) / len(preferred_logps), 6),
        }

    def compute_logprobs(self, model: Dict[str, Any], pairs: List[PreferencePair]) -> Tuple[List[float], List[float]]:
        rng = random.Random(hash(str(model.get("name", ""))))
        pref_logps: List[float] = []
        rej_logps: List[float] = []
        for pair in pairs:
            pref_logps.append(rng.gauss(-1.0, 0.5))
            rej_logps.append(rng.gauss(-2.0, 0.5))
        return pref_logps, rej_logps

    def train_step(self, model: Dict[str, Any], preferences: List[PreferencePair]) -> Dict[str, Any]:
        pref_logps, rej_logps = self.compute_logprobs(model, preferences)
        ref_pref, ref_rej = self.get_reference_logps(model, preferences)
        # DPO uses the difference from reference
        adjusted_pref = [p - r for p, r in zip(pref_logps, ref_pref)]
        adjusted_rej = [p - r for p, r in zip(rej_logps, ref_rej)]
        loss_dict = self.compute_dpo_loss(adjusted_pref, adjusted_rej)
        return loss_dict

    def get_reference_logps(self, model: Dict[str, Any], pairs: List[PreferencePair]) -> Tuple[List[float], List[float]]:
        rng = random.Random(42)
        pref = [rng.gauss(-1.5, 0.3) for _ in pairs]
        rej = [rng.gauss(-2.5, 0.3) for _ in pairs]
        return pref, rej


class ORPOTrainer:
    """Odds Ratio Preference Optimization."""

    def compute_odds_ratio(self, preferred_prob: float, rejected_prob: float) -> float:
        if rejected_prob <= 0 or rejected_prob >= 1:
            return 1.0
        pref_odds = preferred_prob / (1 - preferred_prob + 1e-8)
        rej_odds = rejected_prob / (1 - rejected_prob + 1e-8)
        return pref_odds / (rej_odds + 1e-8)

    def compute_orpo_loss(self, sft_loss: float, odds_ratio: float, lamda: float = 0.1) -> Dict[str, float]:
        log_odds = math.log(odds_ratio + 1e-8)
        orpo_component = -log_odds
        total = sft_loss + lamda * orpo_component
        return {
            "total_loss": round(total, 6),
            "sft_loss": round(sft_loss, 6),
            "preference_loss": round(lamda * orpo_component, 6),
            "odds_ratio": round(odds_ratio, 6),
        }

    def train_step(self, model: Dict[str, Any], preferences: List[PreferencePair]) -> Dict[str, Any]:
        rng = random.Random(42)
        total_orpo_loss = 0.0
        for pair in preferences:
            pref_prob = rng.uniform(0.5, 0.95)
            rej_prob = rng.uniform(0.05, 0.5)
            odds = self.compute_odds_ratio(pref_prob, rej_prob)
            sft = rng.uniform(0.5, 2.0)
            loss_dict = self.compute_orpo_loss(sft, odds)
            total_orpo_loss += loss_dict["total_loss"]
        avg_loss = total_orpo_loss / len(preferences) if preferences else 0
        return {"loss": round(avg_loss, 6)}

    def combine_sft_orpo(self, sft_loss: float, preference_loss: float, lamda: float = 0.1) -> float:
        return round(sft_loss + lamda * preference_loss, 6)


class RewardModel:
    """Preference reward model for RLHF."""

    def __init__(self) -> None:
        self._scores: List[Dict[str, Any]] = []

    def score(self, prompt: str, response: str) -> float:
        score = 0.5 + 0.3 * (len(response) / max(len(prompt), 1))
        score = min(max(score, 0.0), 1.0)
        self._scores.append({"prompt": prompt, "response": response, "score": score})
        return round(score, 4)

    def batch_score(self, prompts: List[str], responses: List[str]) -> List[float]:
        return [self.score(p, r) for p, r in zip(prompts, responses)]

    def train_on_preferences(self, preferences: List[PreferencePair]) -> Dict[str, float]:
        total_loss = 0.0
        correct = 0
        for pref in preferences:
            s_diff = pref.preferred_score - pref.rejected_score
            if s_diff > 0:
                correct += 1
            loss = -math.log(math.exp(s_diff) / (1 + math.exp(s_diff)) + 1e-8)
            total_loss += loss
        return {
            "loss": round(total_loss / len(preferences), 6) if preferences else 0.0,
            "accuracy": round(correct / len(preferences), 4) if preferences else 0.0,
        }

    def detect_reward_hacking(self, scores: List[float], threshold: float = 0.95) -> Dict[str, Any]:
        if not scores:
            return {"hacking": False, "avg_score": 0.0}
        avg = sum(scores) / len(scores)
        high_ratio = sum(1 for s in scores if s > threshold) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        hacking = high_ratio > 0.8 or variance < 0.001
        return {
            "hacking": hacking,
            "avg_score": round(avg, 4),
            "high_score_ratio": round(high_ratio, 4),
            "variance": round(variance, 6),
        }
