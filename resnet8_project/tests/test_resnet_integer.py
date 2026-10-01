"""Hardware arithmetic tests with independent Python-integer expectations."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from resnet8.integer import (check_accumulator, conv_accumulate, conv_scalar, multiplier_shift,
                             quantize, quantize_parameters, requantize_wide, same_padding, saturate, IntegerModel)
from resnet8.export import export_model, load_export


class IntegerTests(unittest.TestCase):
    def test_signed_half_ties_and_saturation(self):
        np.testing.assert_array_equal(quantize([-200, -1.5, -.5, 0, .5, 1.5, 200], 1), [-128, -2, -1, 0, 1, 2, 127])
        m, s = multiplier_shift([0.5])
        np.testing.assert_array_equal(requantize_wide(np.arange(-5, 6, dtype=np.int32), m, s),
                                     [-3, -2, -2, -1, -1, 0, 1, 1, 2, 2, 3])
        np.testing.assert_array_equal(saturate([-129, -1, 0, 128], True), [0, 0, 0, 127])

    def test_requantizer_python_integer_oracle(self):
        rng = np.random.default_rng(2608)
        x = rng.integers(-(1 << 31), (1 << 31) - 1, (100, 5), dtype=np.int32)
        x[0] = -(1 << 31)
        x[1] = (1 << 31) - 1
        m, s = multiplier_shift([1e-6, .003, .5, 1., 5.])
        expected = np.empty(x.shape, np.int64)
        for idx in np.ndindex(x.shape):
            product = int(x[idx]) * int(m[idx[1]])
            shift = int(s[idx[1]])
            expected[idx] = (1 if product >= 0 else -1) * ((abs(product) + (1 << (shift - 1))) // (1 << shift))
        np.testing.assert_array_equal(requantize_wide(x, m, s), expected)

    def test_same_padding_is_asymmetric_for_even_stride_two(self):
        self.assertEqual(same_padding(32, 32, 3, 2), ((0, 1, 0, 1), (16, 16)))
        self.assertEqual(same_padding(16, 16, 1, 2), ((0, 0, 0, 0), (8, 8)))

    def test_convolutions_against_scalar_oracle(self):
        rng = np.random.default_rng(8)
        for h, width in ((4, 6), (5, 7)):
            for k in (1, 3):
                for stride in (1, 2):
                    for ci in (3, 26, 52, 104):
                        with self.subTest(h=h, k=k, stride=stride, ci=ci):
                            x = rng.integers(-128, 128, (1, h, width, ci), dtype=np.int8)
                            w = rng.integers(-128, 128, (k, k, ci, 2), dtype=np.int8)
                            b = np.array([123456, -345678], np.int32)
                            np.testing.assert_array_equal(conv_accumulate(x, w, b, stride), conv_scalar(x, w, b, stride))

    def test_int32_overflow_rejected(self):
        with self.assertRaises(OverflowError):
            check_accumulator(np.ones((3, 3, 3, 1), np.int8), np.array([2147483647], np.int32))

    def test_constant_bn_channel_preserves_bias(self):
        params = quantize_parameters(np.full((3, 3, 3, 1), 5e-34, np.float32),
                                     np.array([2.0], np.float32), 0.1, 0.05)
        self.assertLessEqual(params["accumulator_abs_bound"], (1 << 31) - 1)
        np.testing.assert_array_equal(params["weights"], 0)
        result = requantize_wide(params["bias"], params["multiplier"], params["shift"])
        np.testing.assert_array_equal(result, [40])

    def test_residual_does_not_saturate_branches_before_cancellation(self):
        # 100*2 + 100*(-1) = 100; clipping 200 to 127 first would give 27.
        m, s = multiplier_shift([2., 1.])
        layers = [{"name": "negative", "op": "conv", "inputs": ["input"],
                   "weights": -np.eye(3, dtype=np.int8).reshape(1, 1, 3, 3), "bias": np.zeros(3, np.int32),
                   "stride": 1, "relu": False, "multiplier": multiplier_shift([1.]*3)[0],
                   "shift": multiplier_shift([1.]*3)[1]},
                  {"name": "add", "op": "add", "inputs": ["input", "negative"],
                   "multiplier": m, "shift": s, "relu_name": "relu", "relu": True}]
        model = IntegerModel(layers, {"input": 1.})
        result = model.run(np.full((1, 32, 32, 3), 100, np.int8), already_quantized=True)
        np.testing.assert_array_equal(result["relu"], 100)

    def test_average_pool_rounding(self):
        m, s = multiplier_shift([1 / 64])
        np.testing.assert_array_equal(requantize_wide(np.array([-96, -32, 31, 32, 96], np.int32), m, s), [-2, -1, 0, 1, 2])

    def test_binary_layout_and_corruption_detection(self):
        w = np.arange(3*3*3*2, dtype=np.int8).reshape(3, 3, 3, 2)
        m, s = multiplier_shift([0.1, 0.2])
        layer = {"name": "conv0", "op": "conv", "inputs": ["input"], "weights": w,
                 "bias": np.array([100, -200], np.int32), "stride": 2, "relu": True, "multiplier": m, "shift": s}
        model = IntegerModel([layer], {"input": 1., "conv0": 2.})
        with tempfile.TemporaryDirectory() as temp:
            export_model(model, temp)
            p = Path(temp)
            self.assertEqual((p / "weights.bin").read_bytes(), w.transpose(3, 0, 1, 2).tobytes())
            self.assertEqual((p / "biases.bin").read_bytes(), b"\x64\x00\x00\x00\x38\xff\xff\xff")
            restored = load_export(temp)
            x = np.arange(32*32*3, dtype=np.int16).astype(np.int8).reshape(1, 32, 32, 3)
            for name, value in model.run(x, already_quantized=True).items():
                np.testing.assert_array_equal(value, restored.run(x, already_quantized=True)[name])
            (p / "weights.bin").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                load_export(temp)


if __name__ == "__main__":
    unittest.main()
