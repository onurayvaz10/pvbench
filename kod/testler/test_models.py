"""Tests for pvbench.models: every architecture builds, trains and can be parameter-matched."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pvbench import models as Z  # noqa: E402

LOOKBACK, FEATURES = 24, 10


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.x = rng.normal(size=(512, LOOKBACK, FEATURES)).astype("float32")
        cls.y = (cls.x[:, -1, 0] + 0.5 * cls.x[:, :, 1].mean(axis=1)).astype("float32")

    def test_every_model_builds_trains_and_predicts(self):
        print()
        for name in Z.MODEL_NAMES:
            Z.keras.utils.set_random_seed(0)
            model = Z.build_model(name, LOOKBACK, FEATURES)
            before = model.evaluate(self.x, self.y, verbose=0)
            model.fit(self.x, self.y, batch_size=64, epochs=3, verbose=0)
            after = model.evaluate(self.x, self.y, verbose=0)
            pred = model.predict(self.x[:8], verbose=0)
            self.assertEqual(pred.shape, (8, 1), name)
            self.assertTrue(np.isfinite(pred).all(), name)
            self.assertLess(after, before, name)
            print(f"  {name:<13} parameters {model.count_params():>7,}   loss {before:.3f} -> {after:.3f}")

    def test_hand_computed_parameter_counts(self):
        # LSTM(64): 4*64*(10+64+1) = 19,200; LSTM(32): 4*32*(64+32+1) = 12,416; Dense(16): 528; out: 17
        self.assertEqual(Z.count_parameters("lstm", LOOKBACK, FEATURES), 19200 + 12416 + 528 + 17)
        # Dense on the flattened window: 240*128+128, 128*64+64, 64*32+32, 32+1
        self.assertEqual(Z.count_parameters("dense", LOOKBACK, FEATURES), 30848 + 8256 + 2080 + 33)

    def test_tcn_last_step_sees_whole_window(self):
        model = Z.build_model("tcn", LOOKBACK, FEATURES, compile_model=False)
        base = model.predict(self.x[:1], verbose=0)
        changed = self.x[:1].copy()
        changed[0, 0, :] += 5.0                      # oldest step is inside the receptive field (1 + 2*2*15 = 61)
        self.assertNotAlmostEqual(float(base[0, 0]), float(model.predict(changed, verbose=0)[0, 0]), places=6)

    def test_parameter_matching(self):
        print()
        for name in Z.MODEL_NAMES:
            width, n = Z.match_width(name, 50_000, LOOKBACK, FEATURES)
            self.assertLess(abs(n - 50_000) / 50_000, 0.05, f"{name}: {n}")
            print(f"  {name:<13} width {width:5.2f} -> {n:>7,} parameters")

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            Z.build_model("resnet", LOOKBACK, FEATURES)


if __name__ == "__main__":
    unittest.main(verbosity=1)
