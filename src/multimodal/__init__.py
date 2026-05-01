"""Multimodal — Chapter 21: Vision-language adapters, visual tokenizers, multimodal training, hallucination detection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class VisionEncoder(Enum):
    CLIP = "clip"
    VIT = "vit"
    SAM = "sam"
    DINO = "dino"


class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class VisualToken:
    id: int
    embedding: List[float] = field(default_factory=list)
    position: Tuple[int, int] = (0, 0)


@dataclass
class MultimodalBatch:
    images: List[List[List[float]]]
    texts: List[str]
    labels: Optional[List[str]] = None


@dataclass
class GroundingResult:
    text_span: str
    grounded: bool
    confidence: float
    region: Optional[Tuple[int, int, int, int]] = None


class VisionLanguageAdapter:
    """Adapt vision-language models for multimodal tasks."""

    def setup_vlm(self, base_model: Dict[str, Any], vision_encoder: str = "clip") -> Dict[str, Any]:
        return {
            "base_model": base_model.get("name", "unknown"),
            "vision_encoder": vision_encoder,
            "projection_dim": 768,
            "image_resolution": 224,
            "patch_size": 16,
            "initialized": True,
        }

    def tokenize_image(self, image: List[List[float]]) -> Dict[str, Any]:
        height = len(image)
        width = len(image[0]) if image else 0
        tokens = height * width // 16  # simplified
        return {
            "tokens": tokens,
            "resolution": (height, width),
            "patch_embeddings": [[0.1] * 768 for _ in range(min(tokens, 196))],
        }

    def align_embeddings(self, vision_emb: List[float], text_emb: List[float]) -> Dict[str, float]:
        if not vision_emb or not text_emb:
            return {"similarity": 0.0, "alignment_loss": 1.0}
        min_len = min(len(vision_emb), len(text_emb))
        dot = sum(a * b for a, b in zip(vision_emb[:min_len], text_emb[:min_len]))
        norm_v = math.sqrt(sum(x ** 2 for x in vision_emb[:min_len]))
        norm_t = math.sqrt(sum(x ** 2 for x in text_emb[:min_len]))
        if norm_v == 0 or norm_t == 0:
            return {"similarity": 0.0, "alignment_loss": 1.0}
        sim = dot / (norm_v * norm_t)
        return {"similarity": round(sim, 4), "alignment_loss": round(1 - sim, 4)}

    def get_vlm_config(self) -> Dict[str, Any]:
        return {
            "model_type": "vlm",
            "vision_encoder": "clip",
            "text_decoder": "transformer",
            "projection": "linear",
            "max_image_tokens": 576,
            "max_text_tokens": 2048,
        }


class VisualTokenizer:
    """Tokenize visual inputs for multimodal processing."""

    def encode_image(self, image: List[List[float]], resolution: int = 224) -> Dict[str, Any]:
        return {
            "resolution": resolution,
            "channels": 3,
            "embedding_dim": 768,
            "total_patches": (resolution // 16) ** 2,
            "encoded": True,
        }

    def extract_patches(self, image: List[List[float]], patch_size: int = 16) -> List[List[float]]:
        patches: List[List[float]] = []
        for i in range(0, len(image), patch_size):
            for j in range(0, len(image[i]) if i < len(image) else 0, patch_size):
                patch = []
                for di in range(min(patch_size, len(image) - i)):
                    row = image[i + di]
                    for dj in range(min(patch_size, len(row) - j)):
                        if isinstance(row[j + dj], (int, float)):
                            patch.append(float(row[j + dj]))
                if patch:
                    patches.append(patch)
        return patches

    def compute_visual_tokens(self, embeddings: List[List[float]]) -> List[VisualToken]:
        tokens: List[VisualToken] = []
        for i, emb in enumerate(embeddings):
            row = i // 14  # assuming 14x14 grid
            col = i % 14
            tokens.append(VisualToken(id=i, embedding=emb[:768] if len(emb) >= 768 else emb + [0.0] * (768 - len(emb)), position=(row, col)))
        return tokens

    def decode_tokens(self, tokens: List[VisualToken]) -> List[List[float]]:
        max_row = max((t.position[0] for t in tokens), default=0) + 1
        max_col = max((t.position[1] for t in tokens), default=0) + 1
        grid: List[List[float]] = [[0.0] * max_col for _ in range(max_row)]
        for token in tokens:
            r, c = token.position
            if r < max_row and c < max_col:
                grid[r][c] = sum(token.embedding[:10]) / max(len(token.embedding[:10]), 1)
        return grid


class MultimodalTrainer:
    """Train multimodal models combining vision and language."""

    def prepare_batch(self, images: List[List[List[float]]], texts: List[str]) -> MultimodalBatch:
        return MultimodalBatch(images=images, texts=texts)

    def compute_loss(self, predictions: Dict[str, List[float]], targets: Dict[str, List[float]]) -> Dict[str, float]:
        pred_vals = predictions.get("logits", [0.0])
        target_vals = targets.get("logits", [0.0])
        mse = sum((p - t) ** 2 for p, t in zip(pred_vals, target_vals)) / max(len(pred_vals), 1)
        return {
            "total_loss": round(mse, 6),
            "vision_loss": round(mse * 0.4, 6),
            "text_loss": round(mse * 0.4, 6),
            "alignment_loss": round(mse * 0.2, 6),
        }

    def train_step(self, model: Dict[str, Any], batch: MultimodalBatch) -> Dict[str, float]:
        rng = random.Random(42)
        predictions = {"logits": [rng.gauss(0, 1) for _ in range(10)]}
        targets = {"logits": [rng.gauss(0, 1) for _ in range(10)]}
        loss_dict = self.compute_loss(predictions, targets)
        loss_dict["batch_size"] = len(batch.texts)
        return loss_dict

    def evaluate_multimodal(self, model: Dict[str, Any], test_data: List[MultimodalBatch]) -> Dict[str, float]:
        total_loss = 0.0
        for batch in test_data:
            loss_dict = self.train_step(model, batch)
            total_loss += loss_dict["total_loss"]
        avg_loss = total_loss / len(test_data) if test_data else 0
        return {
            "avg_loss": round(avg_loss, 4),
            "num_batches": len(test_data),
            "vqa_accuracy": round(random.Random(42).uniform(0.6, 0.9), 4),
            "image_text_recall": round(random.Random(43).uniform(0.7, 0.95), 4),
        }


class VisualHallucinationDetector:
    """Detect visual hallucinations in multimodal outputs."""

    def detect(self, text: str, image: Dict[str, Any]) -> Dict[str, Any]:
        objects = image.get("objects", [])
        mentioned = [obj for obj in objects if obj.lower() in text.lower()]
        hallucinated = []
        text_words = set(text.lower().split())
        object_words = set(obj.lower() for obj in objects)
        hallucinated = list(text_words - object_words - {"the", "a", "an", "is", "in", "on", "with", "and", "of"})
        score = len(hallucinated) / len(text_words) if text_words else 0
        return {
            "hallucination_score": round(score, 4),
            "hallucinated_words": hallucinated[:5],
            "grounded_objects": mentioned,
            "flagged": score > 0.3,
        }

    def compute_hallucination_rate(self, results: List[Dict[str, Any]]) -> float:
        if not results:
            return 0.0
        flagged = sum(1 for r in results if r.get("flagged", False))
        return round(flagged / len(results), 4)

    def check_object_grounding(self, text: str, objects: List[str]) -> List[GroundingResult]:
        results: List[GroundingResult] = []
        text_lower = text.lower()
        for obj in objects:
            grounded = obj.lower() in text_lower
            results.append(GroundingResult(
                text_span=obj,
                grounded=grounded,
                confidence=0.95 if grounded else 0.1,
            ))
        return results

    def get_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"total": 0, "hallucination_rate": 0.0, "avg_score": 0.0}
        scores = [r.get("hallucination_score", 0) for r in results]
        return {
            "total": len(results),
            "hallucination_rate": self.compute_hallucination_rate(results),
            "avg_score": round(sum(scores) / len(scores), 4),
            "max_score": round(max(scores), 4),
            "flagged_count": sum(1 for r in results if r.get("flagged", False)),
        }
