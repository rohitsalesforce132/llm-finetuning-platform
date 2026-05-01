"""Data Pipeline — Chapter 4 / Stage 1: Data collection, quality, splitting, and augmentation."""

from __future__ import annotations

import copy
import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class DataFormat(Enum):
    """Supported data formats."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    JSONL = "jsonl"
    TEXT = "text"


class QualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class SplitStrategy(Enum):
    """Split strategies."""
    RANDOM = "random"
    STRATIFIED = "stratified"
    SEQUENTIAL = "sequential"


@dataclass
class DataRecord:
    """A single data record."""
    id: str
    text: str
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Report on data quality."""
    total_records: int
    duplicate_count: int
    empty_count: int
    quality_level: QualityLevel
    issues: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class DataSplit:
    """Container for split datasets."""
    train: List[DataRecord]
    val: List[DataRecord]
    test: List[DataRecord]


@dataclass
class SplitStats:
    """Statistics about a data split."""
    train_size: int
    val_size: int
    test_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    label_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)


class DataCollector:
    """Collect and format training data from various sources."""

    def __init__(self) -> None:
        self._sources: Dict[str, List[DataRecord]] = {}

    def collect(self, source: str, fmt: DataFormat = DataFormat.JSON) -> List[DataRecord]:
        """Collect data from a named source in the given format."""
        if source not in self._sources:
            return []
        records = self._sources[source]
        return copy.deepcopy(records)

    def add_source(self, name: str, records: List[DataRecord]) -> None:
        """Register a data source (simulated)."""
        self._sources[name] = copy.deepcopy(records)

    def validate_format(self, data: List[DataRecord]) -> bool:
        """Validate that data conforms to expected format."""
        if not data:
            return False
        for record in data:
            if not isinstance(record.id, str) or not record.id:
                return False
            if not isinstance(record.text, str):
                return False
        return True

    def get_schema(self) -> Dict[str, str]:
        """Return the expected data schema."""
        return {
            "id": "string (required)",
            "text": "string (required)",
            "label": "string (optional)",
            "metadata": "dict (optional)",
        }

    def transform(self, data: List[DataRecord], target_format: DataFormat) -> List[Dict[str, Any]]:
        """Transform data to target format representation."""
        if target_format == DataFormat.JSON:
            return [{"id": r.id, "text": r.text, "label": r.label} for r in data]
        elif target_format == DataFormat.CSV:
            rows = []
            for r in data:
                row = f"{r.id}|{r.text}|{r.label or ''}"
                rows.append({"line": row})
            return rows
        elif target_format == DataFormat.JSONL:
            import json
            return [{"line": json.dumps({"id": r.id, "text": r.text})} for r in data]
        elif target_format == DataFormat.TEXT:
            return [{"text": r.text} for r in data]
        else:
            return [{"id": r.id, "text": r.text} for r in data]


class DataQualityChecker:
    """Validate and report on data quality."""

    def check_quality(self, data: List[DataRecord]) -> QualityLevel:
        """Check overall data quality and return a level."""
        if not data:
            return QualityLevel.POOR
        report = self.get_quality_report(data)
        return report.quality_level

    def detect_duplicates(self, data: List[DataRecord]) -> List[DataRecord]:
        """Detect and return duplicate records."""
        seen_texts: Dict[str, DataRecord] = {}
        duplicates: List[DataRecord] = []
        for record in data:
            text_hash = hashlib.md5(record.text.encode()).hexdigest()
            if text_hash in seen_texts:
                duplicates.append(record)
            else:
                seen_texts[text_hash] = record
        return duplicates

    def check_distribution(self, data: List[DataRecord]) -> Dict[str, int]:
        """Check label distribution in the dataset."""
        distribution: Dict[str, int] = {}
        for record in data:
            label = record.label or "unlabeled"
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def get_quality_report(self, data: List[DataRecord]) -> QualityReport:
        """Generate a comprehensive quality report."""
        if not data:
            return QualityReport(
                total_records=0,
                duplicate_count=0,
                empty_count=0,
                quality_level=QualityLevel.POOR,
                issues=["Dataset is empty"],
            )

        duplicates = self.detect_duplicates(data)
        empty_count = sum(1 for r in data if not r.text.strip())
        issues: List[str] = []

        if duplicates:
            issues.append(f"Found {len(duplicates)} duplicate records")
        if empty_count:
            issues.append(f"Found {empty_count} empty text records")

        dup_ratio = len(duplicates) / len(data)
        empty_ratio = empty_count / len(data)

        quality_score = 1.0 - dup_ratio - empty_ratio
        if quality_score >= 0.9:
            level = QualityLevel.EXCELLENT
        elif quality_score >= 0.7:
            level = QualityLevel.GOOD
        elif quality_score >= 0.5:
            level = QualityLevel.FAIR
        else:
            level = QualityLevel.POOR

        return QualityReport(
            total_records=len(data),
            duplicate_count=len(duplicates),
            empty_count=empty_count,
            quality_level=level,
            issues=issues,
            scores={"quality_score": round(quality_score, 3)},
        )


class ImbalanceHandler:
    """Handle imbalanced datasets with various strategies."""

    def detect_imbalance(self, data: List[DataRecord], threshold: float = 0.3) -> bool:
        """Detect if dataset is imbalanced beyond threshold."""
        distribution: Dict[str, int] = {}
        for record in data:
            label = record.label or "unlabeled"
            distribution[label] = distribution.get(label, 0) + 1

        if len(distribution) < 2:
            return False

        counts = list(distribution.values())
        ratio = min(counts) / max(counts)
        return ratio < threshold

    def oversample(self, data: List[DataRecord], strategy: str = "minority") -> List[DataRecord]:
        """Oversample minority classes to balance dataset."""
        distribution: Dict[str, List[DataRecord]] = {}
        for record in data:
            label = record.label or "unlabeled"
            distribution.setdefault(label, []).append(record)

        if not distribution:
            return data

        max_count = max(len(v) for v in distribution.values())
        result: List[DataRecord] = []

        for label, records in distribution.items():
            result.extend(records)
            if strategy == "minority" and len(records) < max_count:
                needed = max_count - len(records)
                for i in range(needed):
                    result.append(records[i % len(records)])

        return result

    def undersample(self, data: List[DataRecord], strategy: str = "majority") -> List[DataRecord]:
        """Undersample majority classes to balance dataset."""
        distribution: Dict[str, List[DataRecord]] = {}
        for record in data:
            label = record.label or "unlabeled"
            distribution.setdefault(label, []).append(record)

        if not distribution:
            return data

        min_count = min(len(v) for v in distribution.values())
        result: List[DataRecord] = []

        for label, records in distribution.items():
            result.extend(records[:min_count])

        return result

    def compute_class_weights(self, data: List[DataRecord]) -> Dict[str, float]:
        """Compute class weights inversely proportional to frequency."""
        distribution: Dict[str, int] = {}
        for record in data:
            label = record.label or "unlabeled"
            distribution[label] = distribution.get(label, 0) + 1

        if not distribution:
            return {}

        total = sum(distribution.values())
        n_classes = len(distribution)
        weights: Dict[str, float] = {}
        for label, count in distribution.items():
            weights[label] = round(total / (n_classes * count), 4)

        return weights


class DataSplitter:
    """Create train/val/test splits from datasets."""

    def split(
        self, data: List[DataRecord], ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ) -> DataSplit:
        """Split data into train/val/test sets."""
        shuffled = list(data)
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        return DataSplit(
            train=shuffled[:train_end],
            val=shuffled[train_end:val_end],
            test=shuffled[val_end:],
        )

    def stratified_split(
        self, data: List[DataRecord], ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ) -> DataSplit:
        """Stratified split maintaining label proportions."""
        by_label: Dict[str, List[DataRecord]] = {}
        for record in data:
            label = record.label or "unlabeled"
            by_label.setdefault(label, []).append(record)

        train: List[DataRecord] = []
        val: List[DataRecord] = []
        test: List[DataRecord] = []

        for label, records in by_label.items():
            n = len(records)
            train_end = int(n * ratios[0])
            val_end = train_end + int(n * ratios[1])
            train.extend(records[:train_end])
            val.extend(records[train_end:val_end])
            test.extend(records[val_end:])

        return DataSplit(train=train, val=val, test=test)

    def cross_validate(self, data: List[DataRecord], k: int = 5) -> List[Tuple[List[DataRecord], List[DataRecord]]]:
        """Create k-fold cross-validation splits."""
        shuffled = list(data)
        random.shuffle(shuffled)

        fold_size = len(shuffled) // k
        folds: List[Tuple[List[DataRecord], List[DataRecord]]] = []

        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(shuffled)
            val_fold = shuffled[start:end]
            train_fold = shuffled[:start] + shuffled[end:]
            folds.append((train_fold, val_fold))

        return folds

    def get_split_stats(self, splits: DataSplit) -> SplitStats:
        """Get statistics about the split."""
        total = len(splits.train) + len(splits.val) + len(splits.test)

        label_dist: Dict[str, Dict[str, int]] = {}
        for name, subset in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
            dist: Dict[str, int] = {}
            for r in subset:
                label = r.label or "unlabeled"
                dist[label] = dist.get(label, 0) + 1
            label_dist[name] = dist

        return SplitStats(
            train_size=len(splits.train),
            val_size=len(splits.val),
            test_size=len(splits.test),
            train_ratio=len(splits.train) / total if total else 0,
            val_ratio=len(splits.val) / total if total else 0,
            test_ratio=len(splits.test) / total if total else 0,
            label_distribution=label_dist,
        )


class SyntheticDataGenerator:
    """Generate synthetic training data through templates and augmentation."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def generate(self, template: str, n: int) -> List[DataRecord]:
        """Generate n synthetic records from a template."""
        records: List[DataRecord] = []
        for i in range(n):
            text = template.replace("{i}", str(i)).replace("{n}", str(n))
            records.append(DataRecord(id=f"syn_{i}", text=text, label="synthetic"))
        return records

    def augment(self, data: List[DataRecord], strategies: Optional[List[str]] = None) -> List[DataRecord]:
        """Augment data using specified strategies."""
        if strategies is None:
            strategies = ["shuffle_words"]

        augmented: List[DataRecord] = []
        for record in data:
            aug_text = record.text
            for strategy in strategies:
                if strategy == "shuffle_words":
                    words = aug_text.split()
                    if len(words) > 1:
                        self._rng.shuffle(words)
                    aug_text = " ".join(words)
                elif strategy == "duplicate_words":
                    words = aug_text.split()
                    if words:
                        idx = self._rng.randint(0, len(words) - 1)
                        words.insert(idx, words[idx])
                        aug_text = " ".join(words)
                elif strategy == "truncate":
                    words = aug_text.split()
                    if len(words) > 3:
                        keep = self._rng.randint(3, len(words))
                        aug_text = " ".join(words[:keep])

            augmented.append(DataRecord(
                id=f"aug_{record.id}",
                text=aug_text,
                label=record.label,
            ))

        return augmented

    def validate_synthetic(self, real: List[DataRecord], synthetic: List[DataRecord]) -> Dict[str, Any]:
        """Validate quality of synthetic data against real data."""
        real_lens = [len(r.text.split()) for r in real]
        syn_lens = [len(r.text.split()) for r in synthetic]

        real_avg = sum(real_lens) / len(real_lens) if real_lens else 0
        syn_avg = sum(syn_lens) / len(syn_lens) if syn_lens else 0

        return {
            "real_avg_length": round(real_avg, 2),
            "synthetic_avg_length": round(syn_avg, 2),
            "length_ratio": round(syn_avg / real_avg, 3) if real_avg else 0,
            "real_count": len(real),
            "synthetic_count": len(synthetic),
            "valid": abs(syn_avg - real_avg) / max(real_avg, 1) < 0.5 if real_avg else False,
        }

    def blend(
        self, real: List[DataRecord], synthetic: List[DataRecord], ratio: float = 0.5
    ) -> List[DataRecord]:
        """Blend real and synthetic data at given ratio."""
        n_synthetic = int(len(real) * ratio / (1 - ratio)) if ratio < 1 else len(synthetic)
        n_synthetic = min(n_synthetic, len(synthetic))
        blended = list(real) + synthetic[:n_synthetic]
        return blended
