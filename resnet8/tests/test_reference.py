"""The functional reference against straightforward loop implementations."""

import numpy as np
import pytest

from resnet8.accel import reference
from resnet8.accel.config import AccelConfig
from resnet8.accel.fixedpoint import rounding_shift_right
from resnet8.accel.qmodel import PoolLayer, QModel, random_qmodel


def naive_conv(x, w, stride, pad):
    n, c, h, wd = x.shape
    o, _, k, _ = w.shape
    oh, ow = (h + 2 * pad - k) // stride + 1, (wd + 2 * pad - k) // stride + 1
    out = np.zeros((n, o, oh, ow), dtype=np.int64)
    for b in range(n):
        for oc in range(o):
            for oy in range(oh):
                for ox in range(ow):
                    total = 0
                    for ic in range(c):
                        for ky in range(k):
                            for kx in range(k):
                                iy, ix = oy * stride - pad + ky, ox * stride - pad + kx
                                if 0 <= iy < h and 0 <= ix < wd:
                                    total += int(x[b, ic, iy, ix]) * int(w[oc, ic, ky, kx])
                    out[b, oc, oy, ox] = total
    return out


@pytest.mark.parametrize("k,stride,pad,h,w", [(3, 1, 1, 6, 5), (3, 2, 1, 7, 6), (1, 2, 0, 5, 5), (1, 1, 0, 3, 4)])
def test_conv_accumulate_matches_loops(k, stride, pad, h, w):
    rng = np.random.default_rng(k * 100 + stride * 10 + h)
    x = rng.integers(0, 256, size=(2, 5, h, w)).astype(np.uint8)
    wt = rng.integers(-127, 128, size=(4, 5, k, k)).astype(np.int8)
    assert np.array_equal(reference.conv_accumulate(x, wt, stride, pad), naive_conv(x, wt, stride, pad))


def test_conv_accumulate_matches_torch_at_full_scale():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    x = rng.integers(0, 256, size=(3, 64, 8, 8)).astype(np.uint8)
    wt = rng.integers(-127, 128, size=(64, 64, 3, 3)).astype(np.int8)
    want = torch.nn.functional.conv2d(
        torch.from_numpy(x.astype(np.float64)), torch.from_numpy(wt.astype(np.float64)), padding=1,
    ).numpy().astype(np.int64)
    assert np.array_equal(reference.conv_accumulate(x, wt, 1, 1), want)


def test_pool_layer():
    rng = np.random.default_rng(8)
    x = rng.integers(0, 256, size=(2, 3, 8, 8)).astype(np.uint8)
    layer = PoolLayer("p", "x", "y", m=1 << 15, shift=21)            # exactly 1/64
    y = reference.run_layer(layer, {"x": x}, AccelConfig())
    want = rounding_shift_right(x.astype(np.int64).sum(axis=(2, 3), keepdims=True), 6)
    assert np.array_equal(y, want)


def test_batch_and_single_image_agree():
    rng = np.random.default_rng(9)
    qmodel = random_qmodel(rng)
    images = rng.integers(0, 256, size=(5, 3, 32, 32)).astype(np.uint8)
    batch = reference.run(qmodel, images)
    singles = np.concatenate([reference.run(qmodel, images[i:i + 1]) for i in range(5)])
    assert np.array_equal(batch, singles)
    assert np.array_equal(reference.predict(qmodel, images, batch_size=2), batch)


def test_qmodel_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(10)
    qmodel = random_qmodel(rng)
    qmodel.scales = {"input": 1 / 255, "logits": 0.01}
    qmodel.save(tmp_path / "q.npz")
    loaded = QModel.load(tmp_path / "q.npz")
    images = rng.integers(0, 256, size=(2, 3, 32, 32)).astype(np.uint8)
    assert np.array_equal(reference.run(qmodel, images), reference.run(loaded, images))
    assert loaded.scales == qmodel.scales and loaded.shapes() == qmodel.shapes()
