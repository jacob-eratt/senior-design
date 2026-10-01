"""Generate and verify Phase 0-3 artifacts and future convolution test vectors."""

import argparse
import platform
from pathlib import Path

import numpy as np
from PIL import Image

from resnet8.common import ARTIFACTS, CLASSES, REVISION, ROOT, WEIGHTS, WEIGHTS_REVISION, save_tensors, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--calibration-count", type=int, default=512)
    parser.add_argument("--eval-count", type=int, default=1000, help="First N held-out test images (up to 10000)")
    parser.add_argument("--output", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    if not 1 <= args.calibration_count <= 50000 or not 10 <= args.eval_count <= 10000:
        parser.error("Need 1..50000 calibration images and 10..10000 evaluation images")

    from resnet8.data import cifar10, MD5, URL
    from resnet8.fp32 import Reference
    from resnet8.integer import build_integer, calibrate, conv_accumulate, conv_scalar, requantize_wide, saturate
    from resnet8.export import export_model, load_export, write_mem
    import tensorflow as tf
    import keras

    out = args.output
    reference = Reference()
    (train, _), (test, labels) = cifar10(download=args.download)
    # First occurrence of each ground-truth class, selected without predictions.
    indices = [int(np.flatnonzero(labels == c)[0]) for c in range(10)]
    images = test[indices]
    print(f"Golden test indices (class order): {indices}", flush=True)
    fp = reference.run(images)
    original = tf.keras.models.load_model(str(WEIGHTS), compile=False)
    for a, b in zip(reference.model.get_weights(), original.get_weights(), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(original(images, training=False).numpy(), fp["probabilities"], atol=1e-7, rtol=1e-6)
    folded = reference.folded_run(images)
    fold_errors = {}
    for name, tensor in folded.items():
        # Cancellation near zero requires an absolute tolerance; verify every
        # folded stage, not only final argmax.
        np.testing.assert_allclose(tensor, fp[name], atol=3e-4, rtol=3e-4, err_msg=name)
        fold_errors[name] = float(np.max(np.abs(tensor - fp[name])))
    save_tensors(out / "weights" / "original", {
        f"{l.name}_{i}": w for l in reference.model.layers for i, w in enumerate(l.get_weights())})
    save_tensors(out / "weights" / "folded", {
        f"{n}_{kind}": a for n, pair in reference.folded_weights().items() for kind, a in zip(("kernel", "bias"), pair)})
    write_json(out / "architecture.json", {
        "source_revision": REVISION, "weights_revision": WEIGHTS_REVISION,
        "weights_sha256": sha256(WEIGHTS), "parameters_including_bn_state": reference.model.count_params(),
        "keras_config": reference.model.get_config(), "tensor_shapes": {n: list(a.shape[1:]) for n, a in fp.items()}})
    print(f"Calibrating on first {args.calibration_count} TRAINING images", flush=True)
    scales = calibrate(reference, train[:args.calibration_count])
    integer = build_integer(reference, scales)
    integer.metadata = {"source_revision": REVISION, "weights_revision": WEIGHTS_REVISION,
                        "weights_sha256": sha256(WEIGHTS), "calibration_split": "CIFAR-10 training",
                        "calibration_indices": list(range(args.calibration_count)),
                        "dataset_url": URL, "dataset_md5": MD5, "rounding": "nearest, ties away from zero"}
    manifest = export_model(integer, out / "export")
    restored = load_export(out / "export")
    qi = integer.run(images)
    qr = restored.run(images)
    for name in qi:
        np.testing.assert_array_equal(qi[name], qr[name], err_msg=f"Binary round trip: {name}")
    probabilities = integer.probabilities(qi)
    golden_records = []
    for i, index in enumerate(indices):
        directory = out / "golden" / f"{index:05d}_{CLASSES[i]}"
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(images[i]).save(directory / "input.png")
        save_tensors(directory / "fp32", {"input_uint8": images[i:i+1], **{n: a[i:i+1] for n, a in fp.items()}})
        save_tensors(directory / "integer", {n: a[i:i+1] for n, a in qi.items()})
        record = {"test_index": index, "label": CLASSES[i],
                  "fp32_prediction": CLASSES[int(fp["probabilities"][i].argmax())],
                  "integer_prediction": CLASSES[int(qi["logits"][i].argmax())],
                  "fp32_scores": fp["probabilities"][i].tolist(), "integer_scores": probabilities[i].tolist()}
        write_json(directory / "prediction.json", record)
        golden_records.append(record)
    write_json(out / "golden" / "manifest.json", {"selection": "first test image per true class, no prediction filtering",
               "records": golden_records, "weights_sha256": sha256(WEIGHTS), "export_manifest_sha256": sha256(out / "export" / "manifest.json")})

    # Phase 4 starter fixture: real folded/quantized Conv0 output channel zero.
    layer = integer.layers[0]
    x, w, b = qi["input"][:1], layer["weights"][..., :1].copy(), layer["bias"][:1].copy()
    acc = conv_accumulate(x, w, b)
    np.testing.assert_array_equal(acc, conv_scalar(x, w, b))
    y = saturate(requantize_wide(acc, layer["multiplier"][:1], layer["shift"][:1]), True)
    fixture = out / "conv_fixture"
    save_tensors(fixture, {"input": x, "weights_hwio": w, "bias": b, "expected_acc": acc, "expected_output": y})
    for name, a in {"input": x, "weights": w.transpose(3, 0, 1, 2), "bias": b,
                    "expected_acc": acc, "expected_output": y,
                    "quant": np.stack([layer["multiplier"][:1], layer["shift"][:1]], axis=-1)}.items():
        write_mem(fixture / f"{name}.mem", a)
    write_json(fixture / "config.json", {"input_hwc": [32, 32, 3], "cout": 1, "kernel": 3, "stride": 1,
                "padding_tblr": [1, 1, 1, 1], "relu": True, "input_scale": scales["input"],
                "output_scale": scales["conv0"], "weight_layout": "OHWI", "activation_layout": "NHWC",
                "multiplier": int(layer["multiplier"][0]), "shift": int(layer["shift"][0]),
                "test_index": indices[0], "verified_with": "independent Python integer scalar convolution"})
    # Real programmable-engine vectors for every required convolution.
    for layer in integer.layers:
        if layer["op"] != "conv":
            continue
        name = layer["name"]
        d = out / "layer_vectors" / name
        tensors = {"input": qi[layer["inputs"][0]][:1], "expected_acc": qi[name + "_acc"][:1],
                   "expected_output": qi[name][:1]}
        save_tensors(d, tensors)
        for n, a in tensors.items():
            write_mem(d / f"{n}.mem", a)
        write_json(d / "config.json", next(l for l in manifest["layers"] if l["name"] == name))

    print(f"Evaluating {args.eval_count} held-out TEST images", flush=True)
    fp_correct = q_correct = agreement = 0
    layer_errors = {}
    predictions = []
    for start in range(0, args.eval_count, 16):
        stop = min(start + 16, args.eval_count)
        values = reference.run(test[start:stop])
        ints = integer.run(test[start:stop])
        f_pred, q_pred = values["probabilities"].argmax(-1), ints["logits"].argmax(-1)
        fp_correct += int(np.sum(f_pred == labels[start:stop]))
        q_correct += int(np.sum(q_pred == labels[start:stop]))
        agreement += int(np.sum(f_pred == q_pred))
        predictions.extend({"test_index": j, "label": int(labels[j]), "fp32": int(f), "integer": int(q)}
                           for j, f, q in zip(range(start, stop), f_pred, q_pred))
        for name in scales.keys() & ints.keys() & values.keys():
            error = ints[name].astype(np.float64) * scales[name] - values[name]
            stats = layer_errors.setdefault(name, {"squared_error": 0.0, "count": 0, "max_abs": 0.0, "at_int8_limits": 0})
            stats["squared_error"] += float(np.sum(error * error))
            stats["count"] += int(error.size)
            stats["max_abs"] = max(stats["max_abs"], float(np.abs(error).max()))
            stats["at_int8_limits"] += int(np.sum((ints[name] == -128) | (ints[name] == 127)))
        if stop % 160 == 0 or stop == args.eval_count:
            print(f"  {stop}/{args.eval_count}: FP32 {fp_correct/stop:.3%}, INT8 {q_correct/stop:.3%}", flush=True)
    accuracy = fp_correct / args.eval_count
    quant_accuracy = q_correct / args.eval_count
    for stats in layer_errors.values():
        stats["rmse"] = (stats.pop("squared_error") / stats["count"]) ** 0.5
        stats["fraction_at_int8_limits"] = stats.pop("at_int8_limits") / stats["count"]
    report = {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
              "numpy": np.__version__, "weights_sha256": sha256(WEIGHTS),
              "golden_indices": indices, "calibration_count": args.calibration_count,
              "evaluation_count": args.eval_count, "evaluation_selection": "first N test images",
              "fp32_accuracy": accuracy, "integer_accuracy": quant_accuracy,
              "accuracy_drop_percentage_points": 100 * (accuracy - quant_accuracy),
              "prediction_agreement": agreement / args.eval_count,
              "folded_max_abs_errors": fold_errors, "quantized_layer_errors": layer_errors,
              "h5_vs_instrumented": "pass", "binary_roundtrip_all_golden_tensors": "bit-exact",
              "starter_conv_scalar_oracle": "bit-exact",
              "acceptance": {"min_fp32_accuracy": 0.80, "max_quantization_drop_percentage_points": 3.0},
              "accuracy_gate_pass": accuracy >= 0.80 and accuracy - quant_accuracy <= 0.03,
              "rtl_written": False, "rtl_verified": False}
    write_json(out / "evaluation_predictions.json", predictions)
    write_json(out / "verification.json", report)
    print(f"Report: {out / 'verification.json'}", flush=True)
    if not report["accuracy_gate_pass"]:
        raise SystemExit("Accuracy gate failed; inspect report before proceeding to hardware")


if __name__ == "__main__":
    main()
