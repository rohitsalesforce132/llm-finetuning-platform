"""Comprehensive tests for the LLM Fine-Tuning Platform — 10 subsystems, 150+ tests."""

import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# === Subsystem 1: Data Preparation ===
from src.data_preparation import (
    DataCollector, DataQualityChecker, ImbalanceHandler, DataSplitter,
    SyntheticDataGenerator, DataFormat, QualityLevel, DataRecord, DataSplit,
)


class TestDataCollector:
    def setup_method(self):
        self.collector = DataCollector()
        self.sample_data = [
            DataRecord(id="1", text="Hello world", label="greeting"),
            DataRecord(id="2", text="Goodbye world", label="farewell"),
            DataRecord(id="3", text="How are you", label="question"),
        ]

    def test_collect_empty_source(self):
        result = self.collector.collect("nonexistent")
        assert result == []

    def test_collect_returns_data(self):
        self.collector.add_source("test", self.sample_data)
        result = self.collector.collect("test")
        assert len(result) == 3

    def test_collect_returns_copy(self):
        self.collector.add_source("test", self.sample_data)
        result = self.collector.collect("test")
        result[0].text = "modified"
        assert self.collector.collect("test")[0].text == "Hello world"

    def test_validate_format_valid(self):
        assert self.collector.validate_format(self.sample_data) is True

    def test_validate_format_empty(self):
        assert self.collector.validate_format([]) is False

    def test_validate_format_missing_text(self):
        bad_data = [DataRecord(id="1", text="", label="x")]
        # empty text is still valid format (text is str)
        assert self.collector.validate_format(bad_data) is True

    def test_get_schema(self):
        schema = self.collector.get_schema()
        assert "id" in schema
        assert "text" in schema

    def test_transform_json(self):
        result = self.collector.transform(self.sample_data, DataFormat.JSON)
        assert len(result) == 3
        assert "id" in result[0]

    def test_transform_csv(self):
        result = self.collector.transform(self.sample_data, DataFormat.CSV)
        assert "line" in result[0]

    def test_transform_text(self):
        result = self.collector.transform(self.sample_data, DataFormat.TEXT)
        assert "text" in result[0]


class TestDataQualityChecker:
    def setup_method(self):
        self.checker = DataQualityChecker()
        self.good_data = [
            DataRecord(id=str(i), text=f"Sample text {i}", label=f"class_{i % 3}")
            for i in range(20)
        ]

    def test_check_quality_good(self):
        level = self.checker.check_quality(self.good_data)
        assert level in [QualityLevel.EXCELLENT, QualityLevel.GOOD]

    def test_check_quality_empty(self):
        level = self.checker.check_quality([])
        assert level == QualityLevel.POOR

    def test_detect_duplicates_none(self):
        dups = self.checker.detect_duplicates(self.good_data)
        assert len(dups) == 0

    def test_detect_duplicates_found(self):
        dup_data = self.good_data + [DataRecord(id="dup", text="Sample text 0", label="x")]
        dups = self.checker.detect_duplicates(dup_data)
        assert len(dups) == 1

    def test_check_distribution(self):
        dist = self.checker.check_distribution(self.good_data)
        assert "class_0" in dist
        assert "class_1" in dist
        assert "class_2" in dist

    def test_get_quality_report(self):
        report = self.checker.get_quality_report(self.good_data)
        assert report.total_records == 20
        assert report.quality_level in list(QualityLevel)


class TestImbalanceHandler:
    def setup_method(self):
        self.handler = ImbalanceHandler()
        self.balanced = [
            DataRecord(id=str(i), text=f"Text {i}", label=f"class_{i % 2}")
            for i in range(20)
        ]
        self.imbalanced = [
            DataRecord(id=str(i), text=f"Text {i}", label="majority")
            for i in range(18)
        ] + [
            DataRecord(id=str(i), text=f"Text {i}", label="minority")
            for i in range(18, 20)
        ]

    def test_detect_imbalance_balanced(self):
        assert self.handler.detect_imbalance(self.balanced) is False

    def test_detect_imbalance_imbalanced(self):
        assert self.handler.detect_imbalance(self.imbalanced) is True

    def test_oversample(self):
        result = self.handler.oversample(self.imbalanced)
        minority = [r for r in result if r.label == "minority"]
        majority = [r for r in result if r.label == "majority"]
        assert len(minority) >= len(majority)

    def test_undersample(self):
        result = self.handler.undersample(self.imbalanced)
        minority = [r for r in result if r.label == "minority"]
        majority = [r for r in result if r.label == "majority"]
        assert len(minority) == len(majority)

    def test_compute_class_weights(self):
        weights = self.handler.compute_class_weights(self.imbalanced)
        assert "majority" in weights
        assert "minority" in weights
        assert weights["minority"] > weights["majority"]


class TestDataSplitter:
    def setup_method(self):
        self.splitter = DataSplitter()
        self.data = [
            DataRecord(id=str(i), text=f"Text {i}", label=f"class_{i % 3}")
            for i in range(100)
        ]

    def test_split_ratios(self):
        split = self.splitter.split(self.data, (0.8, 0.1, 0.1))
        assert len(split.train) == 80
        assert len(split.val) == 10
        assert len(split.test) == 10

    def test_split_no_overlap(self):
        split = self.splitter.split(self.data, (0.8, 0.1, 0.1))
        all_ids = [r.id for r in split.train + split.val + split.test]
        assert len(set(all_ids)) == 100

    def test_stratified_split(self):
        split = self.splitter.stratified_split(self.data, (0.8, 0.1, 0.1))
        assert len(split.train) > 0
        assert len(split.val) > 0

    def test_cross_validate(self):
        folds = self.splitter.cross_validate(self.data, k=5)
        assert len(folds) == 5
        for train, val in folds:
            assert len(train) > len(val)

    def test_get_split_stats(self):
        split = self.splitter.split(self.data, (0.8, 0.1, 0.1))
        stats = self.splitter.get_split_stats(split)
        assert stats.train_size == 80
        assert abs(stats.train_ratio - 0.8) < 0.01


class TestSyntheticDataGenerator:
    def setup_method(self):
        self.gen = SyntheticDataGenerator(seed=42)

    def test_generate(self):
        records = self.gen.generate("Sample {i} of {n}", 5)
        assert len(records) == 5
        assert records[0].label == "synthetic"

    def test_augment_shuffle(self):
        data = [DataRecord(id="1", text="hello world foo bar", label="x")]
        augmented = self.gen.augment(data, ["shuffle_words"])
        assert len(augmented) == 1
        assert augmented[0].id.startswith("aug_")

    def test_validate_synthetic(self):
        real = [DataRecord(id=str(i), text=f"Word " * 10, label="x") for i in range(5)]
        syn = [DataRecord(id=str(i), text=f"Word " * 8, label="x") for i in range(5)]
        result = self.gen.validate_synthetic(real, syn)
        assert "real_avg_length" in result
        assert "synthetic_avg_length" in result

    def test_blend(self):
        real = [DataRecord(id=str(i), text=f"Real {i}", label="x") for i in range(10)]
        syn = [DataRecord(id=str(i), text=f"Syn {i}", label="syn") for i in range(10)]
        blended = self.gen.blend(real, syn, ratio=0.5)
        assert len(blended) == 20  # 10 real + 10 synthetic (ratio=0.5 means synthetic:real = 0.5:0.5 = 1:1)

    def test_blend_full_ratio(self):
        real = [DataRecord(id="1", text="Real", label="x")]
        syn = [DataRecord(id="2", text="Syn", label="syn")]
        blended = self.gen.blend(real, syn, ratio=1.0)
        assert len(blended) == 2


# === Subsystem 2: Model Selection ===
from src.model_selection import (
    ModelSelector, InitializationChecker, TokenizerManager, QuantizationPrep,
    ModelFamily, ModelCriteria, TaskType,
)


class TestModelSelector:
    def setup_method(self):
        self.selector = ModelSelector()

    def test_get_model_info_known(self):
        info = self.selector.get_model_info("gpt2")
        assert info is not None
        assert info.family == ModelFamily.GPT

    def test_get_model_info_unknown(self):
        assert self.selector.get_model_info("unknown_model") is None

    def test_evaluate_model(self):
        criteria = ModelCriteria(max_parameters=1000)
        score = self.selector.evaluate_model("gpt2", criteria)
        assert 0 <= score <= 1

    def test_rank_models(self):
        ranked = self.selector.rank_models(["gpt2", "llama-7b", "mistral-7b"])
        assert len(ranked) == 3
        assert all(isinstance(s, float) for _, s in ranked)

    def test_recommend(self):
        criteria = ModelCriteria(max_memory_gb=2.0)
        result = self.selector.recommend(TaskType.CLASSIFICATION, criteria)
        assert result is not None


class TestInitializationChecker:
    def setup_method(self):
        self.checker = InitializationChecker()
        self.model = {
            "hidden_size": 768,
            "num_layers": 12,
            "vocab_size": 50257,
            "context_length": 1024,
        }

    def test_check_config_valid(self):
        result = self.checker.check_config(self.model)
        assert result["valid"] is True

    def test_check_config_missing_key(self):
        result = self.checker.check_config({"hidden_size": 768})
        assert result["valid"] is False

    def test_validate_layers(self):
        assert self.checker.validate_layers(self.model) is True

    def test_validate_layers_zero(self):
        assert self.checker.validate_layers({"num_layers": 0}) is False

    def test_count_parameters(self):
        params = self.checker.count_parameters(self.model)
        assert params > 0

    def test_get_memory_estimate(self):
        mem = self.checker.get_memory_estimate(self.model, batch_size=4)
        assert mem["total_gb"] > 0
        assert "parameters_gb" in mem


class TestTokenizerManager:
    def setup_method(self):
        self.manager = TokenizerManager()

    def test_load_tokenizer(self):
        tok = self.manager.load_tokenizer("gpt2")
        assert tok["name"] == "gpt2"
        assert tok["vocab_size"] > 0

    def test_check_vocab_coverage(self):
        tok = self.manager.load_tokenizer("gpt2")
        result = self.manager.check_vocab_coverage(tok, ["token_1 token_2", "token_3"])
        assert "coverage_ratio" in result

    def test_extend_tokenizer(self):
        tok = self.manager.load_tokenizer("gpt2")
        original_size = tok["vocab_size"]
        updated = self.manager.extend_tokenizer(tok, ["new_token_1", "new_token_2"])
        assert updated["vocab_size"] == original_size + 2

    def test_compute_token_stats(self):
        tok = self.manager.load_tokenizer("gpt2")
        stats = self.manager.compute_token_stats(tok, ["hello world", "foo bar baz"])
        assert stats["num_samples"] == 2
        assert stats["avg_tokens"] > 0


class TestQuantizationPrep:
    def setup_method(self):
        self.prep = QuantizationPrep()
        self.model = {"total_params": 7_000_000_000}

    def test_analyze_quantization_impact(self):
        result = self.prep.analyze_quantization_impact(self.model, bits=4)
        assert result["compression_ratio"] == 8.0
        assert result["memory_savings_pct"] == 87.5

    def test_prepare_for_quantization(self):
        config = self.prep.prepare_for_quantization(self.model)
        assert config["calibration_dataset_size"] == 128

    def test_estimate_memory_savings(self):
        savings = self.prep.estimate_memory_savings(self.model, bits=4)
        assert savings > 0

    def test_validate_quantized_model_valid(self):
        model = {"quantization_config": {}, "weights": {}}
        assert self.prep.validate_quantized_model(model) is True

    def test_validate_quantized_model_invalid(self):
        assert self.prep.validate_quantized_model({}) is False


# === Subsystem 3: Training Engine ===
from src.training_engine import (
    TrainingConfigurator, OptimizerFactory, CheckpointManager, GradientManager,
    TrainingConfig, OptimizerType, ScheduleType,
)


class TestTrainingConfigurator:
    def setup_method(self):
        self.configurator = TrainingConfigurator()

    def test_create_config(self):
        config = self.configurator.create_config({"learning_rate": 1e-4, "batch_size": 16})
        assert config.learning_rate == 1e-4
        assert config.batch_size == 16

    def test_validate_config_valid(self):
        config = TrainingConfig()
        result = self.configurator.validate_config(config)
        assert result["valid"] is True

    def test_validate_config_invalid_lr(self):
        config = TrainingConfig(learning_rate=-1.0)
        result = self.configurator.validate_config(config)
        assert result["valid"] is False

    def test_validate_config_high_lr_warning(self):
        config = TrainingConfig(learning_rate=2.0)
        result = self.configurator.validate_config(config)
        assert len(result["warnings"]) > 0

    def test_estimate_training_time(self):
        config = TrainingConfig(batch_size=8, epochs=3)
        est = self.configurator.estimate_training_time(config, data_size=1000)
        assert est["total_steps"] == 375
        assert est["estimated_minutes"] > 0

    def test_suggest_hyperparameters(self):
        result = self.configurator.suggest_hyperparameters("classification", "large")
        assert result["learning_rate"] == 2e-5
        assert result["batch_size"] == 8


class TestOptimizerFactory:
    def setup_method(self):
        self.factory = OptimizerFactory()
        self.params = {"w1": 0.5, "w2": -0.3}

    def test_create_optimizer(self):
        opt = self.factory.create_optimizer("adamw", self.params, lr=1e-4)
        assert opt.name == "adamw"
        assert opt.lr == 1e-4

    def test_get_scheduler_cosine(self):
        opt = self.factory.create_optimizer("adam", self.params)
        lrs = self.factory.get_scheduler(opt, ScheduleType.COSINE, steps=100)
        assert len(lrs) == 100
        assert lrs[0] > lrs[-1]

    def test_get_scheduler_constant(self):
        opt = self.factory.create_optimizer("adam", self.params, lr=5e-5)
        lrs = self.factory.get_scheduler(opt, ScheduleType.CONSTANT, steps=10)
        assert all(lr == 5e-5 for lr in lrs)

    def test_compute_lr_warmup(self):
        assert self.factory.compute_lr(5, 10, 100) == 0.5

    def test_get_gradient_stats(self):
        opt = self.factory.create_optimizer("adam", self.params)
        stats = self.factory.get_gradient_stats(opt)
        assert "mean" in stats
        assert "std" in stats


class TestCheckpointManager:
    def setup_method(self):
        self.mgr = CheckpointManager()
        self.model = {"weights": {"w1": 0.5}}

    def test_save_and_load(self):
        self.mgr.save_checkpoint(self.model, step=100, metrics={"loss": 0.5})
        ckpt = self.mgr.load_checkpoint("default:100")
        assert ckpt is not None
        assert ckpt.step == 100

    def test_load_latest(self):
        self.mgr.save_checkpoint(self.model, step=100, metrics={"loss": 0.5})
        self.mgr.save_checkpoint(self.model, step=200, metrics={"loss": 0.3})
        ckpt = self.mgr.load_checkpoint("default")
        assert ckpt.step == 200

    def test_list_checkpoints(self):
        self.mgr.save_checkpoint(self.model, step=100, metrics={"loss": 0.5})
        self.mgr.save_checkpoint(self.model, step=200, metrics={"loss": 0.3})
        assert len(self.mgr.list_checkpoints()) == 2

    def test_prune_checkpoints(self):
        for i in range(5):
            self.mgr.save_checkpoint(self.model, step=i*100, metrics={"loss": 0.5})
        removed = self.mgr.prune_checkpoints(keep_n=2)
        assert removed == 3
        assert len(self.mgr.list_checkpoints()) == 2


class TestGradientManager:
    def setup_method(self):
        self.gm = GradientManager()
        self.model = {"w1": 0.5, "w2": -0.3}
        self.batch = {"w1": 0.8, "w2": -0.1}

    def test_compute_gradients(self):
        grads = self.gm.compute_gradients(self.model, self.batch)
        assert "w1" in grads
        assert "w2" in grads

    def test_clip_gradients_within_norm(self):
        grads = {"w1": 0.1, "w2": 0.1}
        clipped = self.gm.clip_gradients(grads, max_norm=1.0)
        assert clipped["w1"] == 0.1

    def test_clip_gradients_exceeds_norm(self):
        grads = {"w1": 100.0, "w2": 100.0}
        clipped = self.gm.clip_gradients(grads, max_norm=1.0)
        norm = math.sqrt(sum(v**2 for v in clipped.values()))
        assert norm <= 1.01

    def test_accumulate_gradients(self):
        batches = [{"w1": 0.2, "w2": 0.4}, {"w1": 0.6, "w2": 0.8}]
        accumulated = self.gm.accumulate_gradients(batches)
        assert abs(accumulated["w1"] - 0.4) < 0.001

    def test_check_gradient_health_healthy(self):
        grads = {"w1": 0.01, "w2": 0.02}
        health = self.gm.check_gradient_health(grads)
        assert health["healthy"] is True

    def test_check_gradient_health_explosion(self):
        grads = {"w1": 1000.0, "w2": 1000.0}
        health = self.gm.check_gradient_health(grads)
        assert "explosion" in str(health["issues"]).lower()


# === Subsystem 4: PEFT Methods ===
from src.peft_methods import (
    LoRAAdapter, QLoRAAdapter, DoRAAdapter, AdapterRegistry, AdapterType,
)


class TestLoRAAdapter:
    def setup_method(self):
        self.lora = LoRAAdapter()
        self.model = {"weights": {"w1": 0.5, "w2": -0.3}, "name": "test"}

    def test_apply(self):
        result = self.lora.apply(self.model, rank=4, alpha=8.0)
        assert result["lora_applied"] is True
        assert "lora_params" in result

    def test_merge_weights(self):
        applied = self.lora.apply(self.model)
        merged = self.lora.merge_weights(applied)
        assert merged.get("lora_merged") is True
        assert "lora_params" not in merged

    def test_get_lora_params(self):
        applied = self.lora.apply(self.model)
        params = self.lora.get_lora_params(applied)
        assert "w1" in params

    def test_compute_rank_importance(self):
        applied = self.lora.apply(self.model)
        importance = self.lora.compute_rank_importance(applied)
        assert "w1" in importance


class TestQLoRAAdapter:
    def setup_method(self):
        self.qlora = QLoRAAdapter()
        self.model = {"weights": {"w1": 0.567, "w2": -0.321}, "total_params": 1000000}

    def test_quantize_model(self):
        result = self.qlora.quantize_model(self.model, bits=4)
        assert result["quantized"] is True
        assert result["quantization_bits"] == 4

    def test_apply_lora(self):
        result = self.qlora.apply_lora(self.model, rank=4)
        assert result.get("lora_applied") is True

    def test_prepare_for_training(self):
        result = self.qlora.prepare_for_training(self.model)
        assert result["trainable"] is True
        assert result["frozen_base"] is True

    def test_compute_memory_savings(self):
        result = self.qlora.compute_memory_savings(self.model)
        assert "savings_pct" in result
        # With default 32-bit model, savings are minimal without quantization


class TestDoRAAdapter:
    def setup_method(self):
        self.dora = DoRAAdapter()
        self.model = {"weights": {"w1": 3.0, "w2": -4.0}, "name": "test"}

    def test_decompose_weights(self):
        result = self.dora.decompose_weights(self.model)
        assert "magnitude" in result
        assert "direction" in result

    def test_apply_dora(self):
        result = self.dora.apply_dora(self.model, rank=4)
        assert result["dora_applied"] is True

    def test_compute_norm_gradient(self):
        applied = self.dora.apply_dora(self.model)
        grad = self.dora.compute_norm_gradient(applied)
        assert "w1" in grad

    def test_compare_with_lora(self):
        dora_metrics = {"loss": 0.8, "extra_params": 0.5}
        lora_metrics = {"loss": 1.0}
        comparison = self.dora.compare_with_lora(dora_metrics, lora_metrics)
        assert comparison["better"] is True


class TestAdapterRegistry:
    def setup_method(self):
        self.registry = AdapterRegistry()

    def test_register(self):
        info = self.registry.register("adapter1", AdapterType.LORA, {"rank": 8})
        assert info.name == "adapter1"

    def test_load(self):
        self.registry.register("adapter1", AdapterType.LORA, {"rank": 8})
        loaded = self.registry.load("adapter1")
        assert loaded is not None

    def test_load_nonexistent(self):
        assert self.registry.load("nonexistent") is None

    def test_list_adapters(self):
        self.registry.register("a1", AdapterType.LORA, {})
        self.registry.register("a2", AdapterType.DORA, {})
        assert len(self.registry.list_adapters()) == 2

    def test_switch_adapter(self):
        self.registry.register("a1", AdapterType.LORA, {"rank": 8})
        model = {"name": "test"}
        result = self.registry.switch_adapter(model, "a1")
        assert result["active_adapter"] == "a1"


# === Subsystem 5: Preference Optimization ===
from src.preference_optimization import (
    PPOTrainer, DPOTrainer, ORPOTrainer, RewardModel, PreferencePair,
)


class TestPPOTrainer:
    def setup_method(self):
        self.ppo = PPOTrainer()

    def test_compute_advantages(self):
        rewards = [1.0, 0.5, 0.0]
        values = [0.8, 0.4, 0.1]
        advantages = self.ppo.compute_advantages(rewards, values)
        assert len(advantages) == 3

    def test_clip_ratio_within(self):
        ratio = self.ppo.clip_ratio(1.1, 1.0, epsilon=0.2)
        assert ratio == 1.1

    def test_clip_ratio_exceeds(self):
        ratio = self.ppo.clip_ratio(1.5, 1.0, epsilon=0.2)
        assert ratio == 1.2

    def test_compute_ppo_loss(self):
        loss = self.ppo.compute_ppo_loss([0.5, -0.3], [1.1, 0.9])
        assert "total_loss" in loss
        assert "clipped_loss" in loss

    def test_train_step(self):
        model = {"name": "test"}
        batch = [{"reward": 1.0, "value": 0.5, "old_prob": 0.5}]
        result = self.ppo.train_step(model, batch)
        assert "loss" in result


class TestDPOTrainer:
    def setup_method(self):
        self.dpo = DPOTrainer()

    def test_compute_dpo_loss(self):
        loss = self.dpo.compute_dpo_loss([-0.5, -0.3], [-1.0, -0.8], beta=0.1)
        assert "loss" in loss
        assert "accuracies" in loss

    def test_compute_dpo_loss_empty(self):
        loss = self.dpo.compute_dpo_loss([], [])
        assert loss["loss"] == 0.0

    def test_compute_logprobs(self):
        model = {"name": "test"}
        pairs = [PreferencePair(prompt="Hi", preferred="Hello", rejected="Hey")]
        pref, rej = self.dpo.compute_logprobs(model, pairs)
        assert len(pref) == 1

    def test_train_step(self):
        model = {"name": "test"}
        prefs = [PreferencePair(prompt="Q", preferred="A1", rejected="A2")]
        result = self.dpo.train_step(model, prefs)
        assert "loss" in result


class TestORPOTrainer:
    def setup_method(self):
        self.orpo = ORPOTrainer()

    def test_compute_odds_ratio(self):
        ratio = self.orpo.compute_odds_ratio(0.8, 0.2)
        assert ratio > 1.0

    def test_compute_odds_ratio_zero_rejected(self):
        ratio = self.orpo.compute_odds_ratio(0.5, 0.0)
        assert ratio == 1.0

    def test_compute_orpo_loss(self):
        result = self.orpo.compute_orpo_loss(1.0, 2.0, lamda=0.1)
        assert "total_loss" in result
        assert "sft_loss" in result

    def test_train_step(self):
        model = {"name": "test"}
        prefs = [PreferencePair(prompt="Q", preferred="A1", rejected="A2")]
        result = self.orpo.train_step(model, prefs)
        assert "loss" in result

    def test_combine_sft_orpo(self):
        total = self.orpo.combine_sft_orpo(1.0, 0.5, lamda=0.1)
        assert total == 1.05


class TestRewardModel:
    def setup_method(self):
        self.rm = RewardModel()

    def test_score(self):
        score = self.rm.score("Hello", "Hi there!")
        assert 0 <= score <= 1

    def test_batch_score(self):
        scores = self.rm.batch_score(["Hi", "Bye"], ["Hello", "Goodbye"])
        assert len(scores) == 2

    def test_train_on_preferences(self):
        prefs = [PreferencePair(prompt="Q", preferred="Good", rejected="Bad",
                                preferred_score=1.0, rejected_score=0.0)]
        result = self.rm.train_on_preferences(prefs)
        assert "loss" in result
        assert "accuracy" in result

    def test_detect_reward_hacking(self):
        scores = [0.99] * 100
        result = self.rm.detect_reward_hacking(scores)
        assert result["hacking"] is True


# === Subsystem 6: Evaluation ===
from src.evaluation import (
    FineTuningEvaluator, SafetyTester, BenchmarkRunner, RegressionDetector,
    EvalMetrics, SafetyCategory,
)


class TestFineTuningEvaluator:
    def setup_method(self):
        self.evaluator = FineTuningEvaluator()
        self.model = {"name": "ft_model"}
        self.test_data = [{"input": "test", "expected": "output"}]

    def test_evaluate(self):
        metrics = self.evaluator.evaluate(self.model, self.test_data)
        assert 0 <= metrics.accuracy <= 1
        assert metrics.perplexity > 0

    def test_compare_with_base(self):
        base = {"name": "base"}
        improvements = self.evaluator.compare_with_base(self.model, base, self.test_data)
        assert "accuracy" in improvements

    def test_compute_improvement(self):
        base = {"metrics": {"accuracy": 0.7, "perplexity": 10.0}}
        ft = {"metrics": {"accuracy": 0.85, "perplexity": 5.0}}
        improvements = self.evaluator.compute_improvement(base, ft)
        assert improvements["accuracy"] > 0

    def test_get_perplexity(self):
        ppl = self.evaluator.get_perplexity(self.model, ["test data"])
        assert ppl > 0


class TestSafetyTester:
    def setup_method(self):
        self.tester = SafetyTester()
        self.model = {"name": "test"}

    def test_run_safety_tests(self):
        results = self.tester.run_safety_tests(self.model)
        assert len(results) > 0
        assert all(r.passed for r in results)

    def test_check_toxicity(self):
        prompts = ["normal text", "harmful content about violence"]
        result = self.tester.check_toxicity(self.model, prompts)
        assert "toxicity_rate" in result

    def test_test_bias(self):
        result = self.tester.test_bias(self.model, ["group_a", "group_b", "group_c"])
        assert "scores" in result
        assert "max_disparity" in result

    def test_get_safety_score(self):
        from src.evaluation import SafetyResult
        results = [
            SafetyResult(category=SafetyCategory.TOXICITY, score=0.9, passed=True),
            SafetyResult(category=SafetyCategory.BIAS, score=0.85, passed=True),
        ]
        score = self.tester.get_safety_score(results)
        assert 0 <= score <= 1


class TestBenchmarkRunner:
    def setup_method(self):
        self.runner = BenchmarkRunner()

    def test_run_benchmark(self):
        model = {"name": "test"}
        result = self.runner.run_benchmark(model, "hellaswag")
        assert "accuracy" in result.scores

    def test_get_results(self):
        model = {"name": "test"}
        self.runner.run_benchmark(model, "mmlu")
        result = self.runner.get_results("mmlu")
        assert result is not None

    def test_compare_models(self):
        models = [{"name": "model_a"}, {"name": "model_b"}]
        comparisons = self.runner.compare_models(models, "hellaswag")
        assert len(comparisons) == 2

    def test_get_leaderboard(self):
        model = {"name": "test"}
        self.runner.run_benchmark(model, "mmlu")
        lb = self.runner.get_leaderboard("mmlu")
        assert len(lb) > 0


class TestRegressionDetector:
    def setup_method(self):
        self.detector = RegressionDetector()

    def test_set_baseline_and_check(self):
        self.detector.set_baseline({"accuracy": 0.9, "perplexity": 3.0})
        result = self.detector.check_regression({"accuracy": 0.85, "perplexity": 3.5})
        assert "regressed" in result

    def test_check_no_regression(self):
        self.detector.set_baseline({"accuracy": 0.9})
        result = self.detector.check_regression({"accuracy": 0.95})
        assert result["regressed"] is False

    def test_get_regression_report(self):
        report = self.detector.get_regression_report(
            {"accuracy": 0.9, "perplexity": 3.0},
            {"accuracy": 0.85, "perplexity": 2.5}
        )
        assert "improvements" in report
        assert "regressions" in report

    def test_suggest_fixes(self):
        fixes = self.detector.suggest_fixes({"perplexity": {"baseline": 3.0, "current": 5.0}})
        assert len(fixes) > 0


# === Subsystem 7: Deployment ===
from src.deployment import (
    ModelServer, InferenceOptimizer, AdapterServer, AutoScaler,
)


class TestModelServer:
    def setup_method(self):
        self.server = ModelServer()

    def test_load_model(self):
        model = self.server.load_model("/path/to/model", "cpu")
        assert model["loaded"] is True

    def test_predict_without_model(self):
        result = self.server.predict("test")
        assert "error" in result

    def test_predict(self):
        self.server.load_model("/path/to/model")
        result = self.server.predict("Hello world")
        assert "output" in result
        assert "latency_ms" in result

    def test_batch_predict(self):
        self.server.load_model("/path/to/model")
        results = self.server.batch_predict(["Hello", "World"])
        assert len(results) == 2

    def test_get_serving_stats(self):
        self.server.load_model("/path/to/model")
        self.server.predict("test")
        stats = self.server.get_serving_stats()
        assert stats.total_requests == 1


class TestInferenceOptimizer:
    def setup_method(self):
        self.optimizer = InferenceOptimizer()

    def test_enable_speculative_decoding(self):
        model = {"name": "target"}
        draft = {"name": "draft"}
        result = self.optimizer.enable_speculative_decoding(model, draft)
        assert result["speculative_decoding"] is True

    def test_setup_continuous_batching(self):
        config = {"max_batch_size": 64, "scheduling": "priority"}
        result = self.optimizer.setup_continuous_batching(config)
        assert result["continuous_batching"] is True

    def test_apply_kv_cache(self):
        model = {"context_length": 4096}
        result = self.optimizer.apply_kv_cache(model)
        assert result["kv_cache_enabled"] is True

    def test_estimate_throughput(self):
        model = {"latency_ms": 50.0}
        result = self.optimizer.estimate_throughput(model, [1, 4, 16, 32])
        assert len(result) == 4


class TestAdapterServer:
    def setup_method(self):
        self.server = AdapterServer()

    def test_load_adapter(self):
        adapter = self.server.load_adapter("lora_v1", "/adapters/lora_v1")
        assert adapter["loaded"] is True

    def test_switch_adapter(self):
        self.server.load_adapter("lora_v1", "/a1")
        self.server.load_adapter("lora_v2", "/a2")
        assert self.server.switch_adapter("lora_v2") is True
        assert self.server.switch_adapter("nonexistent") is False

    def test_batch_serve(self):
        self.server.load_adapter("a1", "/a1")
        requests = [{"id": "1", "text": "hello"}]
        results = self.server.batch_serve(requests, {"1": "a1"})
        assert len(results) == 1

    def test_get_adapter_stats(self):
        self.server.load_adapter("a1", "/a1")
        stats = self.server.get_adapter_stats()
        assert stats["total_adapters"] == 1


class TestAutoScaler:
    def setup_method(self):
        self.scaler = AutoScaler()

    def test_set_scaling_policy(self):
        policy = self.scaler.set_scaling_policy(2, 20)
        assert policy.min_replicas == 2
        assert policy.max_replicas == 20

    def test_get_current_load(self):
        load = self.scaler.get_current_load()
        assert "cpu" in load

    def test_predict_capacity(self):
        pred = self.scaler.predict_capacity(30)
        assert pred.current_rps > 0
        assert pred.recommended_replicas > 0

    def test_compute_scaling_decision_scale_up(self):
        self.scaler.set_scaling_policy(1, 10)
        decision = self.scaler.compute_scaling_decision({"cpu": 0.9, "latency": 150.0})
        assert decision["action"] == "scale_up"

    def test_compute_scaling_decision_maintain(self):
        self.scaler.set_scaling_policy(1, 10)
        decision = self.scaler.compute_scaling_decision({"cpu": 0.5, "latency": 50.0})
        assert decision["action"] == "maintain"


# === Subsystem 8: Monitoring ===
from src.monitoring import (
    DriftDetector, HallucinationMonitor, CostMonitor, FeedbackLoop,
    AlertLevel,
)


class TestDriftDetector:
    def setup_method(self):
        self.detector = DriftDetector()

    def test_baseline_and_check(self):
        self.detector.baseline([0.1, 0.2, 0.3])
        report = self.detector.check_drift([0.1, 0.2, 0.3])
        assert report.drift_score < 0.1
        assert report.detected is False

    def test_drift_detected(self):
        self.detector.baseline([0.1, 0.2, 0.3])
        report = self.detector.check_drift([-0.5, -0.6, -0.7])
        assert report.drift_score > 0

    def test_get_drift_report(self):
        self.detector.baseline([0.1, 0.2])
        self.detector.check_drift([0.15, 0.25])
        report = self.detector.get_drift_report()
        assert report["measurements"] == 1

    def test_suggest_retraining_none(self):
        result = self.detector.suggest_retraining(0.1)
        assert result["retrain"] is False

    def test_suggest_retraining_critical(self):
        result = self.detector.suggest_retraining(0.7)
        assert result["urgency"] == "high"


class TestHallucinationMonitor:
    def setup_method(self):
        self.monitor = HallucinationMonitor()

    def test_check_response_clean(self):
        result = self.monitor.check_response("The cat sat", "The cat sat on the mat")
        assert result["hallucination_score"] == 0.0

    def test_check_response_hallucinated(self):
        result = self.monitor.check_response("quantum physics dark matter", "cats and dogs")
        assert result["hallucination_score"] > 0

    def test_get_hallucination_rate(self):
        self.monitor.check_response("test", "test context")
        rate = self.monitor.get_hallucination_rate()
        assert 0 <= rate <= 1

    def test_alert_if_above(self):
        self.monitor.check_response("totally unrelated content here", "context")
        alert = self.monitor.alert_if_above(0.01)
        assert alert is not None


class TestCostMonitor:
    def setup_method(self):
        self.monitor = CostMonitor()

    def test_track_request(self):
        entry = self.monitor.track_request(1000, "gpt-4")
        assert entry.cost_usd > 0

    def test_get_cost_report(self):
        self.monitor.track_request(1000, "gpt-4")
        report = self.monitor.get_cost_report(24)
        assert report["total_cost"] > 0

    def test_estimate_monthly_cost(self):
        for _ in range(100):
            self.monitor.track_request(500, "gpt-3.5")
        cost = self.monitor.estimate_monthly_cost()
        assert cost > 0

    def test_suggest_cost_optimizations(self):
        self.monitor.track_request(2000, "gpt-4")
        suggestions = self.monitor.suggest_cost_optimizations()
        assert len(suggestions) > 0


class TestFeedbackLoop:
    def setup_method(self):
        self.loop = FeedbackLoop()

    def test_collect_feedback(self):
        entry = self.loop.collect_feedback("resp_1", 4.5)
        assert entry.rating == 4.5

    def test_analyze_feedback(self):
        self.loop.collect_feedback("r1", 5.0)
        self.loop.collect_feedback("r2", 3.0)
        analysis = self.loop.analyze_feedback(24)
        assert analysis["count"] == 2
        assert analysis["avg_rating"] > 0

    def test_should_trigger_retraining_no(self):
        self.loop.collect_feedback("r1", 5.0)
        self.loop.collect_feedback("r2", 4.0)
        assert self.loop.should_trigger_retraining() is False

    def test_should_trigger_retraining_yes(self):
        assert self.loop.should_trigger_retraining({"negative_ratio": 0.5, "avg_rating": 2.0}) is True

    def test_get_feedback_stats(self):
        self.loop.collect_feedback("r1", 5.0)
        self.loop.collect_feedback("r2", 3.0)
        stats = self.loop.get_feedback_stats()
        assert stats["total"] == 2


# === Subsystem 9: Multimodal ===
from src.multimodal import (
    VisionLanguageAdapter, VisualTokenizer, MultimodalTrainer,
    VisualHallucinationDetector, MultimodalBatch,
)


class TestVisionLanguageAdapter:
    def setup_method(self):
        self.vlm = VisionLanguageAdapter()

    def test_setup_vlm(self):
        config = self.vlm.setup_vlm({"name": "llama"}, "clip")
        assert config["initialized"] is True

    def test_tokenize_image(self):
        image = [[0.1] * 32 for _ in range(32)]
        result = self.vlm.tokenize_image(image)
        assert result["tokens"] > 0

    def test_align_embeddings(self):
        result = self.vlm.align_embeddings([0.5, 0.5], [0.5, 0.5])
        assert result["similarity"] == 1.0

    def test_get_vlm_config(self):
        config = self.vlm.get_vlm_config()
        assert config["model_type"] == "vlm"


class TestVisualTokenizer:
    def setup_method(self):
        self.tokenizer = VisualTokenizer()

    def test_encode_image(self):
        result = self.tokenizer.encode_image([[0.1] * 224 for _ in range(224)])
        assert result["encoded"] is True

    def test_extract_patches(self):
        image = [[float(j) for j in range(32)] for i in range(32)]
        patches = self.tokenizer.extract_patches(image, patch_size=16)
        assert len(patches) > 0

    def test_compute_visual_tokens(self):
        embeddings = [[0.1] * 768 for _ in range(10)]
        tokens = self.tokenizer.compute_visual_tokens(embeddings)
        assert len(tokens) == 10

    def test_decode_tokens(self):
        from src.multimodal import VisualToken
        tokens = [VisualToken(id=0, embedding=[0.5] * 768, position=(0, 0)),
                  VisualToken(id=1, embedding=[0.3] * 768, position=(0, 1))]
        grid = self.tokenizer.decode_tokens(tokens)
        assert len(grid) == 1
        assert len(grid[0]) == 2


class TestMultimodalTrainer:
    def setup_method(self):
        self.trainer = MultimodalTrainer()

    def test_prepare_batch(self):
        batch = self.trainer.prepare_batch([[[0.1]]], ["text"])
        assert len(batch.texts) == 1

    def test_compute_loss(self):
        preds = {"logits": [0.5, 0.3, 0.1]}
        targets = {"logits": [0.5, 0.3, 0.1]}
        loss = self.trainer.compute_loss(preds, targets)
        assert loss["total_loss"] == 0.0

    def test_train_step(self):
        model = {"name": "test"}
        batch = MultimodalBatch(images=[[[0.1]]], texts=["hello"])
        result = self.trainer.train_step(model, batch)
        assert "total_loss" in result

    def test_evaluate_multimodal(self):
        model = {"name": "test"}
        batch = MultimodalBatch(images=[[[0.1]]], texts=["test"])
        result = self.trainer.evaluate_multimodal(model, [batch])
        assert "avg_loss" in result


class TestVisualHallucinationDetector:
    def setup_method(self):
        self.detector = VisualHallucinationDetector()

    def test_detect(self):
        result = self.detector.detect("cat on a table", {"objects": ["cat", "table"]})
        assert "hallucination_score" in result

    def test_compute_hallucination_rate(self):
        results = [{"flagged": True}, {"flagged": False}, {"flagged": True}]
        rate = self.detector.compute_hallucination_rate(results)
        assert rate == pytest.approx(2/3, abs=0.01)

    def test_check_object_grounding(self):
        results = self.detector.check_object_grounding(
            "a cat on a table", ["cat", "table", "dog"]
        )
        grounded = [r for r in results if r.grounded]
        assert len(grounded) == 2

    def test_get_report(self):
        results = [
            {"hallucination_score": 0.2, "flagged": False},
            {"hallucination_score": 0.5, "flagged": True},
        ]
        report = self.detector.get_report(results)
        assert report["total"] == 2
        assert report["flagged_count"] == 1


# === Subsystem 10: Advanced ===
from src.advanced import (
    ModelMerger, ContinualLearner, MechanisticAnalyzer, ScalingPredictor, TaskData,
)


class TestModelMerger:
    def setup_method(self):
        self.merger = ModelMerger()
        self.model_a = {"w1": 1.0, "w2": 0.5}
        self.model_b = {"w1": 0.0, "w2": 1.0}

    def test_merge_slerp(self):
        result = self.merger.merge_slerp(self.model_a, self.model_b, t=0.5)
        assert len(result) == 2
        assert "w1" in result

    def test_merge_slerp_endpoints(self):
        result_a = self.merger.merge_slerp(self.model_a, self.model_b, t=0.0)
        assert abs(result_a["w1"] - 1.0) < 0.01

    def test_merge_ties(self):
        models = [self.model_a, self.model_b]
        result = self.merger.merge_ties(models, [0.5, 0.5])
        assert len(result) == 2

    def test_merge_dare(self):
        models = [self.model_a, self.model_b]
        result = self.merger.merge_dare(models, drop_rate=0.5)
        assert len(result) == 2

    def test_evaluate_merged(self):
        result = self.merger.evaluate_merged({"w1": 0.5})
        assert "accuracy" in result
        assert "perplexity" in result


class TestContinualLearner:
    def setup_method(self):
        self.learner = ContinualLearner()

    def test_add_task(self):
        task = TaskData(name="task1", data=[{"x": 1}])
        self.learner.add_task(task)
        assert len(self.learner._tasks) == 1

    def test_compute_elasticity(self):
        model = {"w1": 10.0, "w2": 0.1}
        elasticity = self.learner.compute_elasticity(model)
        assert elasticity["w2"] > elasticity["w1"]

    def test_apply_ewc(self):
        model = {"w1": 0.5}
        fisher = {"w1": 2.0}
        result = self.learner.apply_ewc(model, fisher)
        assert "w1" in result

    def test_detect_forgetting(self):
        old = {"accuracy": 0.9, "perplexity": 3.0}
        new = {"accuracy": 0.8, "perplexity": 4.0}
        result = self.learner.detect_forgetting(old, new)
        assert result["has_forgetting"] is True

    def test_detect_no_forgetting(self):
        old = {"accuracy": 0.9, "perplexity": 3.0}
        new = {"accuracy": 0.95, "perplexity": 2.5}
        result = self.learner.detect_forgetting(old, new)
        assert result["has_forgetting"] is False


class TestMechanisticAnalyzer:
    def setup_method(self):
        self.analyzer = MechanisticAnalyzer()

    def test_extract_activations(self):
        model = {"layers": ["layer_0", "layer_1"]}
        activations = self.analyzer.extract_activations(model, "input")
        assert len(activations) == 2

    def test_compute_attention_patterns(self):
        model = {"num_heads": 4, "seq_len": 8}
        patterns = self.analyzer.compute_attention_patterns(model, "input")
        assert len(patterns) == 4

    def test_identify_circuits(self):
        model = {"layers": ["layer_0", "layer_1", "layer_2"]}
        circuits = self.analyzer.identify_circuits(model)
        assert len(circuits) == 2

    def test_probe_layer(self):
        model = {"layers": ["layer_0"]}
        probe_data = [{"x": 1}, {"x": 2}]
        result = self.analyzer.probe_layer(model, "layer_0", probe_data)
        assert "accuracy" in result


class TestScalingPredictor:
    def setup_method(self):
        self.predictor = ScalingPredictor()

    def test_predict_loss(self):
        loss = self.predictor.predict_loss(1_000_000, 10_000_000)
        assert loss > 1.5

    def test_predict_loss_larger_better(self):
        small = self.predictor.predict_loss(1_000_000, 1_000_000)
        large = self.predictor.predict_loss(100_000_000, 1_000_000_000)
        assert large < small

    def test_compute_chinchilla_optimal(self):
        result = self.predictor.compute_chinchilla_optimal(3.0)
        assert "optimal_params" in result
        assert "optimal_tokens" in result

    def test_estimate_compute_budget(self):
        flops = self.predictor.estimate_compute_budget(1_000_000, 1_000_000)
        assert flops == 6_000_000_000_000

    def test_get_efficiency_frontier(self):
        frontier = self.predictor.get_efficiency_frontier()
        assert len(frontier) == 5
        assert all("predicted_loss" in f for f in frontier)


# Import pytest for approx
import pytest
