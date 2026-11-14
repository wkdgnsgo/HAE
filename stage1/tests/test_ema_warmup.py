"""EMA warmup: schedule, removal of initial-weight bias, engine wiring, and resume."""
from pathlib import Path
import unittest

import torch
from torch import nn

from hae.checkpoints import seed_all
from hae.config import load_config, validate_config
from hae.ema import ModelEMA
from hae.training import TrainingEngine
from test_training import TinyAutoencoder, config_for

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
STAGE_CONFIGS = ("stage1_alignment", "stage2_gan")


def constant_model(value):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


class ScheduleTests(unittest.TestCase):
    def test_schedule_ramps_then_caps_at_configured_decay(self):
        ema = ModelEMA(constant_model(0.0), decay=0.999, warmup=True)
        self.assertAlmostEqual(ema.decay_at(0), 0.1)
        self.assertAlmostEqual(ema.decay_at(90), 0.91)
        self.assertLess(ema.decay_at(8000), 0.999)
        self.assertEqual(ema.decay_at(8991), 0.999)
        self.assertEqual(ema.decay_at(10**6), 0.999)

    def test_warmup_never_exceeds_a_lower_configured_decay(self):
        ema = ModelEMA(constant_model(0.0), decay=0.5, warmup=True)
        self.assertAlmostEqual(ema.decay_at(0), 0.1)
        self.assertEqual(ema.decay_at(100), 0.5)

    def test_disabled_warmup_keeps_constant_decay_without_a_step(self):
        ema = ModelEMA(constant_model(0.0), decay=0.9)
        self.assertEqual(ema.decay_at(), 0.9)
        self.assertEqual(ema.decay_at(0), 0.9)

    def test_warmup_rejects_a_missing_or_negative_step(self):
        model = constant_model(0.0)
        ema = ModelEMA(model, decay=0.999, warmup=True)
        with self.assertRaises(ValueError):
            ema.update(model)
        with self.assertRaises(ValueError):
            ema.update(model, step=-1)

    def run_short_stage(self, updates, warmup):
        """EMA starts at 0 while the model sits at 1, so the shadow's gap is the initial weight share."""
        model = constant_model(0.0)
        ema = ModelEMA(model, decay=0.999, warmup=warmup)
        with torch.no_grad():
            model.weight.fill_(1.0)
        for step in range(updates):
            ema.update(model, step=step)
        return 1 - ema.shadow["weight"].item()

    def test_short_stage_keeps_initial_weights_only_without_warmup(self):
        self.assertAlmostEqual(self.run_short_stage(500, warmup=False), 0.999 ** 500, delta=1e-3)
        self.assertLess(self.run_short_stage(500, warmup=True), 1e-5)

    def test_reloading_state_continues_the_schedule_with_the_same_schema(self):
        def step_model(model, step):
            with torch.no_grad():
                model.weight.fill_(float(step + 1))

        model = constant_model(0.0)
        continuous = ModelEMA(model, decay=0.999, warmup=True)
        for step in range(20):
            step_model(model, step)
            continuous.update(model, step=step)

        model = constant_model(0.0)
        first = ModelEMA(model, decay=0.999, warmup=True)
        for step in range(10):
            step_model(model, step)
            first.update(model, step=step)
        state = first.state_dict()
        self.assertEqual(set(state), {"format_version", "decay", "shadow", "buffers"})
        resumed = ModelEMA(model, decay=0.999, warmup=True)
        resumed.load_state_dict(state)
        for step in range(10, 20):
            step_model(model, step)
            resumed.update(model, step=step)
        torch.testing.assert_close(resumed.shadow["weight"], continuous.shadow["weight"])


class EngineTests(unittest.TestCase):
    def setUp(self):
        seed_all(2026)

    def make_engine(self, ema_config):
        config = config_for(stage=3)
        config["ema"] = ema_config
        return TrainingEngine(TinyAutoencoder(), config, "cpu")

    def test_engine_applies_warmup_using_its_own_update_count(self):
        engine = self.make_engine({"decay": 0.999, "warmup": True})
        images = torch.rand(2, 3, 4, 4)
        name = "decoder.body.weight"
        for step in range(3):
            before = engine.ema.shadow[name].clone()
            engine.train_update(iter([images, images]), 2)
            raw = dict(engine.model.named_parameters())[name].detach()
            expected = before.lerp(raw, 1 - (1 + step) / (10 + step))
            torch.testing.assert_close(engine.ema.shadow[name], expected)
        self.assertEqual(engine.progress["stage_update"], 3)

    def test_engine_without_the_key_keeps_constant_decay(self):
        engine = self.make_engine({"decay": 0.9})
        self.assertFalse(engine.ema.warmup)
        images = torch.rand(2, 3, 4, 4)
        name = "decoder.body.weight"
        before = engine.ema.shadow[name].clone()
        engine.train_update(iter([images, images]), 2)
        raw = dict(engine.model.named_parameters())[name].detach()
        torch.testing.assert_close(engine.ema.shadow[name], before.lerp(raw, 1 - 0.9))


class ConfigTests(unittest.TestCase):
    def test_every_stage_enables_warmup_by_default(self):
        for name in STAGE_CONFIGS:
            with self.subTest(config=name):
                config = load_config(CONFIGS / f"{name}.yaml")
                validate_config(config, 2)
                self.assertIs(config["ema"]["warmup"], True)

    def test_warmup_can_be_disabled_and_must_be_boolean(self):
        path = CONFIGS / "stage1_alignment.yaml"
        self.assertIs(load_config(path, ["ema.warmup=false"])["ema"]["warmup"], False)
        with self.assertRaises(ValueError):
            validate_config(load_config(path, ["ema.warmup=1"]), 2)


if __name__ == "__main__":
    unittest.main()
