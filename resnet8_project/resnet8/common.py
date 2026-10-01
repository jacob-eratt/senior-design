import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
WEIGHTS = ROOT / "upstream" / "pretrainedResnet26.h5"
REVISION = "4addd0fa08d216e20637637874e084895f289da4"
WEIGHTS_REVISION = "eb78d0ebaf2c812ce13668f017a22171a38cd051"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_tensors(directory, tensors):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, tensor in tensors.items():
        a = np.asarray(tensor)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"Nonfinite tensor: {name}")
        path = directory / f"{name}.npy"
        np.save(path, a, allow_pickle=False)
        manifest[name] = {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": sha256(path)}
    write_json(directory / "tensors.json", manifest)
    return manifest


def image_input(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (32, 32):
            raise ValueError("Expected a 32x32 image; resize explicitly before inference.")
        return np.asarray(im, dtype=np.uint8)[None]


def scores_text(scores):
    scores = np.asarray(scores).reshape(10)
    return "Class scores:\n" + "\n".join(
        f"{name:10s} = {score:.8f}" for name, score in zip(CLASSES, scores)
    ) + f"\n\nPrediction: {CLASSES[int(scores.argmax())]}"
