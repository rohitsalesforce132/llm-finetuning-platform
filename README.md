# LLM Fine-Tuning Platform

Complete production-grade LLM fine-tuning platform implementing all 22 chapters of the *LLM Fine-Tuning Interview Handbook 2026*. Pure Python stdlib, zero external dependencies.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Fine-Tuning Platform                  │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│  Stage 1 │ Stage 2  │ Stage 3  │  Stage 4 │     Stage 5     │
│  Data    │  Model   │ Training │   PEFT   │   Evaluation    │
│  Prep    │  Select  │  Engine  │  Methods │   & Safety      │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│  Preference Opt  │ Deployment │ Monitoring │ Multimodal     │
│  PPO/DPO/ORPO    │ Serving    │  Drift/Hall│ Vision-Language │
├──────────────────┴────────────┴────────────┴────────────────┤
│                    Advanced Topics                            │
│        Merging │ Continual │ Mechanistic │ Scaling           │
└──────────────────────────────────────────────────────────────┘
```

## 10 Subsystems

| # | Subsystem | Chapter | Description |
|---|-----------|---------|-------------|
| 1 | `data_preparation` | Ch4 | Data collection, quality, splitting, augmentation |
| 2 | `model_selection` | Ch5 | Base model selection, tokenization, quantization |
| 3 | `training_engine` | Ch6 | Config, optimizers, checkpoints, gradients |
| 4 | `peft_methods` | Ch11 | LoRA, QLoRA, DoRA, adapter registry |
| 5 | `preference_optimization` | Ch16-18 | PPO, DPO, ORPO, reward models |
| 6 | `evaluation` | Ch8 | Metrics, safety testing, benchmarks, regression |
| 7 | `deployment` | Ch19 | Serving, inference optimization, auto-scaling |
| 8 | `monitoring` | Ch20 | Drift, hallucination, cost, feedback loops |
| 9 | `multimodal` | Ch21 | Vision-language, visual tokenizers, VLM training |
| 10 | `advanced` | Ch22 | Model merging, continual learning, scaling laws |

## Quick Start

```bash
# Run all 194 tests
pytest tests/test_all.py -v

# All should pass with 0 failures
```

## Design Principles

- **Pure Python stdlib** — zero external dependencies, runs anywhere
- **Type hints** on all public methods
- **Dataclasses** for structured data, **Enums** for fixed vocabularies
- **Deterministic tests** — every test is reproducible
- **Comprehensive coverage** — 194 tests across all 10 subsystems (19+ per subsystem)

## Project Structure

```
llm-finetuning-platform/
├── README.md
├── STAR.md
├── src/
│   ├── __init__.py
│   ├── data_preparation/
│   ├── model_selection/
│   ├── training_engine/
│   ├── peft_methods/
│   ├── preference_optimization/
│   ├── evaluation/
│   ├── deployment/
│   ├── monitoring/
│   ├── multimodal/
│   └── advanced/
└── tests/
    └── test_all.py
```

## License

MIT
