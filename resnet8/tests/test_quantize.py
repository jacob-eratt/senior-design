"""Quantisation flow (needs torch): BN folding and float/int agreement."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from resnet8.accel import reference  # noqa: E402
from resnet8.accel.compiler import compile_model  # noqa: E402
from resnet8.model import ResNet8  # noqa: E402
from resnet8.quantize import choose_scale, folded_forward, quantize  # noqa: E402


@pytest.fixture(scope="module")
def model_and_images():
    torch.manual_seed(0)
    model = ResNet8()
    # Give BatchNorm non-trivial statistics and affine parameters.
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.uniform_(0.5, 1.5)
            module.bias.data.uniform_(-0.2, 0.2)
    rng = np.random.default_rng(0)
    # Smooth random "images": low-frequency patterns are closer to real data than white noise.
    base = rng.integers(0, 256, size=(400, 3, 8, 8)).astype(np.float32)
    images = torch.nn.functional.interpolate(torch.from_numpy(base), size=32, mode="bilinear")
    images = images.clamp(0, 255).round().to(torch.uint8).numpy()
    model.train()
    with torch.no_grad():
        for i in range(0, 400, 100):
            model(torch.from_numpy(images[i:i + 100]).float() / 255)
    model.eval()
    return model, images


def test_bn_folding_is_exact(model_and_images):
    model, images = model_and_images
    x = torch.from_numpy(images[:64]).float() / 255
    with torch.inference_mode():
        assert torch.allclose(folded_forward(model, x)["logits"], model(x), atol=1e-4)


def test_integer_model_tracks_float_model(model_and_images):
    model, images = model_and_images
    qmodel = quantize(model, images[:200], "mse")
    compile_model(qmodel)                                  # fits the baseline accelerator
    test = images[200:]
    with torch.inference_mode():
        float_logits = model(torch.from_numpy(test).float() / 255).numpy()
    int_logits = reference.run(qmodel, test) * qmodel.scales["logits"]
    noise = np.mean((float_logits - int_logits) ** 2)
    sqnr = 10 * np.log10(np.mean(float_logits ** 2) / noise)
    assert sqnr > 25, f"logit SQNR only {sqnr:.1f} dB"
    assert np.mean(float_logits.argmax(1) == int_logits.argmax(1)) > 0.9


def test_choose_scale_methods():
    values = np.concatenate([np.linspace(0, 1, 10000), [50.0]])   # one outlier
    assert choose_scale(values, "max") == pytest.approx(50 / 255)
    assert choose_scale(values, "p99") < 1.0 / 255 * 1.01
    assert choose_scale(values, "mse") < 50 / 255                   # clips the outlier
