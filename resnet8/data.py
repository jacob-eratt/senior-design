"""CIFAR-10 as raw uint8 NCHW arrays.

The accelerator consumes raw 8-bit pixels, so everything downstream (training,
calibration, evaluation, test vectors) starts from the same uint8 tensors and
converts to float with ``pixel / 255`` only where a float model needs it.

The arrays are cached in ``data/cifar10_{train,test}.npz``. Create the cache
once with ``python -m resnet8.data`` (Hugging Face mirror, fast) or
``python -m resnet8.data --source toronto`` (original tarball via torchvision).
"""

import argparse
import io
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"


def cache_path(train, data_dir=DATA_DIR):
    return Path(data_dir) / f"cifar10_{'train' if train else 'test'}.npz"


def load_cifar10(train, data_dir=DATA_DIR):
    """Return (images uint8 [N,3,32,32], labels int64 [N])."""
    path = cache_path(train, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m resnet8.data` first")
    with np.load(path) as cached:
        return cached["images"], cached["labels"]


def to_float(images_u8):
    """uint8 pixels -> float32 network input in [0, 1]."""
    import torch

    return torch.as_tensor(images_u8).float().div_(255.0)


def _fetch_huggingface(train, data_dir):
    import urllib.request

    import pyarrow.parquet as pq
    from PIL import Image

    url = HF_URL.format(split="train" if train else "test")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        table = pq.read_table(io.BytesIO(response.read())).to_pydict()
    images = np.stack([
        np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB")) for item in table["img"]
    ])
    return images.transpose(0, 3, 1, 2), np.asarray(table["label"], dtype=np.int64)


def _fetch_toronto(train, data_dir):
    from torchvision import datasets

    dataset = datasets.CIFAR10(root=data_dir, train=train, download=True)
    return dataset.data.transpose(0, 3, 1, 2), np.asarray(dataset.targets, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Download CIFAR-10 into the uint8 .npz cache.")
    parser.add_argument("--source", choices=("huggingface", "toronto"), default="huggingface")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    fetch = _fetch_huggingface if args.source == "huggingface" else _fetch_toronto
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for train in (True, False):
        images, labels = fetch(train, args.data_dir)
        assert images.shape[1:] == (3, 32, 32) and images.dtype == np.uint8
        np.savez(cache_path(train, args.data_dir), images=np.ascontiguousarray(images), labels=labels)
        print(f"Wrote {cache_path(train, args.data_dir)}: {images.shape}, class counts {np.bincount(labels)}")


if __name__ == "__main__":
    main()
