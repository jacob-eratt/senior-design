"""Measure CPU Conv0 calls with resident operands; never launches RTL tools.

NumPy measures the project's exact INT8/INT32 reference including padding and
its safety checks. Optional TensorFlow measures a compiled FP32 conv+bias on
the same integer-valued operands, including Python call/output materialization.
This is not a tuned CPU INT8 kernel benchmark or an FPGA end-to-end comparison.
"""

import argparse
from datetime import datetime, timezone
import os
import platform
import time

import numpy as np

from resnet8.common import ARTIFACTS, sha256, write_json
from resnet8.export import load_export
from resnet8.integer import conv_accumulate


def cpu_name():
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or platform.machine()


def measure(call, warmup, repeats):
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        begin = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - begin) / 1000.0)
    return {"unit": "microseconds", "warmup": warmup, "repeats": repeats,
            "min": float(np.min(samples)), "median": float(np.median(samples)),
            "p95": float(np.percentile(samples, 95)), "mean": float(np.mean(samples))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--tensorflow", action="store_true")
    parser.add_argument("--threads", type=int, default=1, help="TensorFlow CPU intra-op threads")
    parser.add_argument("--output", default=str(ARTIFACTS / "cpu_conv0_benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("repeats/threads must be positive and warmup must be nonnegative")

    model = load_export(ARTIFACTS / "export")
    layer = model.layers[0]
    assert layer["name"] == "conv0"
    x = np.load(ARTIFACTS / "layer_vectors/conv0/input.npy", allow_pickle=False)
    expected = np.load(ARTIFACTS / "layer_vectors/conv0/expected_acc.npy", allow_pickle=False)
    w, b = layer["weights"], layer["bias"]
    numpy_call = lambda: conv_accumulate(x, w, b, stride=1)
    np.testing.assert_array_equal(numpy_call(), expected)
    results = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "cpu": cpu_name(), "platform": platform.platform(),
               "python": platform.python_version(), "numpy": np.__version__,
               "input_shape": list(x.shape), "kernel_shape_hwio": list(w.shape),
               "output_shape": list(expected.shape), "mac_operations": 718848,
               "fixture_sha256": sha256(ARTIFACTS / "layer_vectors/conv0/expected_acc.npy"),
               "scope": "Resident operands, batch 1, conv+bias only; no file IO, model load, quantization, ReLU, or device transfer in timing.",
               "numpy_int8_int32_reference": measure(numpy_call, args.warmup, args.repeats)}
    if args.tensorflow:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        with tf.device("/CPU:0"):
            tx = tf.constant(x.astype(np.float32))
            tw = tf.constant(w.astype(np.float32))
            tb = tf.constant(b.astype(np.float32))

            @tf.function(autograph=False)
            def conv(inputs, weights, bias):
                with tf.device("/CPU:0"):
                    return tf.nn.bias_add(tf.nn.conv2d(inputs, weights, strides=1, padding="SAME"), bias)

            tensorflow_call = lambda: conv(tx, tw, tb).numpy()
            # For this Conv0 fixture all values/sums are within exact integer
            # representation in FP32; verify rather than assume equivalence.
            np.testing.assert_array_equal(tensorflow_call(), expected.astype(np.float32))
            results["tensorflow_fp32_same_operands"] = measure(tensorflow_call, args.warmup, args.repeats)
        results["tensorflow"] = tf.__version__
        results["tensorflow_threads"] = {"intra_op": args.threads, "inter_op": 1}
        results["tensorflow_note"] = "FP32 arithmetic on the same integer-valued inputs/weights/bias; fixture output equality checked. Includes tf.function dispatch and .numpy(); not an optimized INT8 baseline."
    write_json(args.output, results)
    print(f"CPU: {results['cpu']}")
    for key in ("numpy_int8_int32_reference", "tensorflow_fp32_same_operands"):
        if key in results:
            item = results[key]
            print(f"{key}: median={item['median']:.2f} us, p95={item['p95']:.2f} us, min={item['min']:.2f} us")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
