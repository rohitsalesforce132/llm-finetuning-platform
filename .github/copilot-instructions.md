# GitHub Copilot Instructions

## Project Overview
LLM Fine-Tuning Platform — complete fine-tuning lifecycle from data preparation through production deployment. Based on the LLM Fine-Tuning Interview Handbook 2026.

## Architecture
```
src/
├── data_preparation/      # Data collection, quality, imbalance, splits, synthetic data
├── model_selection/       # Base model selection, tokenizer management, quantization prep
├── training_engine/       # Training config, optimizers, checkpoints, gradient management
├── peft_methods/          # LoRA, QLoRA, DoRA adapters, adapter registry
├── preference_optimization/ # PPO (RLHF), DPO, ORPO, reward models
├── evaluation/            # Fine-tuning eval, safety testing, benchmarks, regression detection
├── deployment/            # Model serving, inference optimization, adapter serving, autoscaling
├── monitoring/            # Drift detection, hallucination monitoring, cost tracking, feedback loops
├── multimodal/            # Vision-language models, visual tokenization, multimodal training
└── advanced/              # Model merging, continual learning, mechanistic analysis, scaling prediction
```

## Conventions
- Pure Python stdlib only — zero external dependencies
- Type hints on all public methods
- Dataclasses for structured data, Enums for vocabularies
- Tests in tests/test_all.py using pytest

## Running
```bash
pytest tests/test_all.py -v  # Run all 194 tests
```
