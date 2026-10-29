"""Read the official CIFAR-10 binary archive without executing pickle data."""

import hashlib
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from .common import ARTIFACTS

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
MD5 = "c32a1d4ab5d03f1284b67883e8d87530"


def cifar10(cache=ARTIFACTS / "data", download=False):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "cifar-10-binary.tar.gz"
    if not archive.exists():
        if not download:
            raise FileNotFoundError(f"Missing {archive}; run prepare_resnet.py --download")
        partial = archive.with_suffix(".download")
        print(f"Downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, partial)
        if hashlib.md5(partial.read_bytes()).hexdigest() != MD5:
            raise ValueError("CIFAR-10 download checksum mismatch")
        partial.replace(archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != MD5:
        raise ValueError("CIFAR-10 archive checksum mismatch")
    with tarfile.open(archive, "r:gz") as tar:
        def batch(name):
            with tar.extractfile(f"cifar-10-batches-bin/{name}.bin") as f:
                data = np.frombuffer(f.read(), np.uint8).reshape(-1, 3073)
            return data[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy(), data[:, 0].copy()
        training = [batch(f"data_batch_{i}") for i in range(1, 6)]
        test = batch("test_batch")
    return (np.concatenate([b[0] for b in training]), np.concatenate([b[1] for b in training])), test
