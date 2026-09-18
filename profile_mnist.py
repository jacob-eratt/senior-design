"""Evaluate and profile the saved MNIST checkpoint; see PROFILING.md."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MNISTModel


ROOT = Path(__file__).resolve().parent


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--runs", type=positive_int, default=1000)
    parser.add_argument("--profile-runs", type=positive_int, default=30)
    parser.add_argument("--eval-batch-size", type=positive_int, default=256)
    parser.add_argument("--threads", type=positive_int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "mnist_model.pth")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "profiling_results")
    parser.add_argument("--download", action="store_true", help="Download MNIST if missing")
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    selected = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if selected == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu")
    device = torch.device(selected)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    dataset = datasets.MNIST(
        root=args.data_dir, train=False, download=args.download,
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ]),
    )
    if args.batch_size > len(dataset):
        parser.error("batch-size cannot exceed the test dataset size")
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    images, _ = next(iter(DataLoader(dataset, batch_size=args.batch_size)))
    images = images.to(device)
    print(f"Device: {device}; evaluating {len(dataset)} test images...", flush=True)

    with torch.inference_mode():
        correct = 0
        for batch, labels in DataLoader(dataset, batch_size=args.eval_batch_size):
            predicted = model(batch.to(device)).argmax(dim=1).cpu()
            correct += (predicted == labels).sum().item()
        accuracy = correct / len(dataset)
        print(f"Test accuracy: {accuracy:.2%}; measuring baseline...", flush=True)
        for _ in range(args.warmup):
            model(images)
        synchronize()
        start = time.perf_counter()
        for _ in range(args.runs):
            model(images)
        synchronize()
        elapsed = time.perf_counter() - start

    # Gather shapes and arithmetic counts outside both timing passes.
    rows = []
    handles = []

    def capture(name):
        def hook(module, inputs, output):
            macs = 0
            if isinstance(module, nn.Conv2d):
                macs = output[0].numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            elif isinstance(module, nn.Linear):
                macs = output[0].numel() * module.in_features
            rows.append({
                "layer": name, "type": type(module).__name__,
                "input_shape": list(inputs[0].shape), "output_shape": list(output.shape),
                "parameters": sum(p.numel() for p in module.parameters(recurse=False)),
                "parameter_bytes": sum(p.numel() * p.element_size() for p in module.parameters(recurse=False)),
                "output_bytes_per_batch": output.numel() * output.element_size(),
                "macs_per_image": macs,
            })
        return hook

    leaves = [(name, module) for name, module in model.named_modules() if name and not list(module.children())]
    try:
        for name, module in leaves:
            handles.append(module.register_forward_hook(capture(name)))
        with torch.inference_mode():
            model(images)
        synchronize()
    finally:
        for handle in handles:
            handle.remove()

    # Named ranges include each layer's child operators. Never add those child
    # operator totals to layer totals, since that would double-count work.
    handles = []
    ranges = {}

    def enter(name):
        def hook(module, inputs):
            ranges[name] = record_function("layer::" + name)
            ranges[name].__enter__()
        return hook

    def leave(name):
        def hook(module, inputs, output):
            ranges.pop(name).__exit__(None, None, None)
        return hook

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    print("Collecting layer/operator profile...", flush=True)
    try:
        for name, module in leaves:
            handles.append(module.register_forward_pre_hook(enter(name)))
            handles.append(module.register_forward_hook(leave(name)))
        with torch.inference_mode(), profile(
            activities=activities, record_shapes=True, profile_memory=True
        ) as prof:
            for _ in range(args.profile_runs):
                model(images)
                prof.step()
            synchronize()
    finally:
        for handle in handles:
            handle.remove()

    events = {event.key: event for event in prof.key_averages()}
    for row in rows:
        event = events["layer::" + row["layer"]]
        row["cpu_total_us_per_batch"] = event.cpu_time_total / event.count
        row["device_total_us_per_batch"] = event.device_time_total / event.count if device.type == "cuda" else None
    timing_key = "device_total_us_per_batch" if device.type == "cuda" else "cpu_total_us_per_batch"
    layer_total = sum(row[timing_key] for row in rows)
    for row in rows:
        row["percent_of_layer_time"] = 100 * row[timing_key] / layer_total if layer_total else 0
    rows.sort(key=lambda row: row[timing_key], reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "layers.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "test_images": len(dataset), "correct": correct, "test_accuracy": accuracy,
        "batch_size": args.batch_size, "warmup_runs": args.warmup,
        "measured_runs": args.runs, "profile_runs": args.profile_runs,
        "mean_batch_ms": elapsed * 1000 / args.runs,
        "images_per_second": args.runs * args.batch_size / elapsed,
        "timing_scope": "Repeated fixed batch, model only; excludes preprocessing, transfers and argmax. Mean is amortized over the run loop.",
        "layer_timing_scope": "Instrumented inclusive layer times; percentages use the sum of layer times, not baseline wall time.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    (args.output_dir / "operators.txt").write_text(
        prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=100), encoding="utf-8"
    )
    prof.export_chrome_trace(str(args.output_dir / "trace.json"))
    print(f"Baseline: {summary['mean_batch_ms']:.4f} ms/batch; {summary['images_per_second']:.2f} images/s")
    print(f"\n{'Layer':<22} {'Type':<12} {'Time (us)':>12} {'Share':>9} {'MACs/image':>12}")
    for row in rows:
        print(f"{row['layer']:<22} {row['type']:<12} {row[timing_key]:>12.2f} {row['percent_of_layer_time']:>8.1f}% {row['macs_per_image']:>12,}")
    if device.type == "cuda" and not layer_total:
        print("WARNING: CUDA kernel timing was not captured; do not use zero device times to rank layers.")
    print(f"\nReports saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
