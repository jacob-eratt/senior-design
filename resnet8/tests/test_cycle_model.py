"""The cycle-level model against the functional reference and the perf model."""

import dataclasses

import numpy as np
import pytest

from resnet8.accel import isa, perf, reference
from resnet8.accel.compiler import compile_model
from resnet8.accel.config import AccelConfig
from resnet8.accel.hw import Core, HardwareError
from resnet8.accel.memfile import pack_word, read_hex
from resnet8.accel.qmodel import random_qmodel

SHAPES = {
    "resnet8": ((3, 32, 32), (16, 16, 32, 64), 10),
    # Partial channel tiles, several output tiles, odd sizes with stride 2.
    "odd": ((5, 13, 11), (20, 20, 24, 40), 7),
}


def build(name, seed=0, cfg=AccelConfig()):
    shape, widths, classes = SHAPES[name]
    rng = np.random.default_rng(seed)
    qmodel = random_qmodel(rng, shape, widths, classes)
    images = rng.integers(0, 256, size=(2, *shape)).astype(np.uint8)
    return qmodel, compile_model(qmodel, cfg), images


@pytest.mark.parametrize("name,image_count", [("resnet8", 1), ("odd", 2)])
def test_bit_exact_every_layer(name, image_count):
    qmodel, compiled, images = build(name)
    core = Core(compiled)
    for image in images[:image_count]:
        result = core.run(image, snapshot=True)
        logits, expected = reference.run(qmodel, image[None], keep_intermediates=True)
        assert set(result.snapshots) == {layer.dst for layer in qmodel.layers}
        for tensor, got in result.snapshots.items():
            assert np.array_equal(got, expected[tensor][0].reshape(got.shape)), tensor
        assert np.array_equal(result.logits, logits[0])


def test_eight_lane_design_point_is_also_exact():
    cfg = AccelConfig(lanes=8, act_bank_depth=2048, weight_depth=2048)
    qmodel, compiled, images = build("odd", seed=1, cfg=cfg)
    result = Core(compiled).run(images[0])
    assert np.array_equal(result.logits, reference.run(qmodel, images[:1], cfg)[0])


@pytest.mark.parametrize("name", ["resnet8", "odd"])
def test_perf_model_matches_cycle_model_exactly(name):
    qmodel, compiled, images = build(name, seed=2)
    result = Core(compiled).run(images[0])
    rows, total = perf.analyze(compiled, qmodel)
    assert total == result.cycles
    for row, layer in zip(rows, result.layers):
        assert row.cycles == layer.cycles
        for key in ("issue", "act_reads", "padding_ops", "weight_reads", "param_reads",
                    "residual_reads", "act_writes"):
            assert getattr(row, key) == layer.counts[key], (row.name, key)


def test_deeper_pipelines_stay_exact_and_predictable():
    cfg = AccelConfig(mac_extra_latency=2, ppu_extra_latency=3)
    qmodel, compiled, images = build("odd", seed=5, cfg=cfg)
    result = Core(compiled).run(images[0], snapshot=True)
    assert np.array_equal(result.logits, reference.run(qmodel, images[:1], cfg)[0])
    assert result.cycles == perf.analyze(compiled, qmodel)[1]
    baseline = compile_model(qmodel)
    conv_layers = sum(d.op == isa.OP_CONV for d in baseline.descriptors)
    assert result.cycles == perf.analyze(baseline, qmodel)[1] + 5 * conv_layers


def test_resnet8_cycle_budget():
    """Pin the baseline design point so schedule changes are deliberate."""
    _, compiled, images = build("resnet8")
    rows, total = perf.analyze(compiled)
    assert total == 56_640
    assert [r.issue for r in rows] == [9216, 9216, 9216, 4608, 9728, 4608, 9728, 256, 4]


def test_write_trace_records_every_activation_write():
    qmodel, compiled, images = build("odd", seed=3)
    result = Core(compiled, trace=True).run(images[0])
    rows, _ = perf.analyze(compiled, qmodel)
    assert len(result.trace) == sum(r.act_writes for r in rows)
    cycles = [entry[0] for entry in result.trace]
    assert cycles == sorted(cycles)


def test_port_conflict_is_detected():
    """Reading the identity residual from the bank the MAC is streaming is a structural hazard."""
    _, compiled, images = build("resnet8")
    identity = next(d for d in compiled.descriptors if d.res_mode == isa.RES_IDENTITY)
    identity.res_bank = identity.in_bank
    with pytest.raises(HardwareError, match="reads in one cycle"):
        Core(compiled).run(images[0])


def test_compiler_rejects_designs_that_do_not_fit():
    qmodel, _, _ = build("resnet8")
    with pytest.raises(RuntimeError, match="no free activation bank"):
        compile_model(qmodel, AccelConfig(num_act_banks=2))
    with pytest.raises(RuntimeError, match="bank depth"):
        compile_model(qmodel, AccelConfig(act_bank_depth=512))
    with pytest.raises(RuntimeError, match="accumulators"):
        compile_model(qmodel, AccelConfig(acc_bits=20))


def test_banks_never_overwrite_live_tensors():
    qmodel, compiled, _ = build("resnet8")
    resident = {compiled.input_bank: qmodel.input_name}
    for layer in qmodel.layers:
        for t in (layer.src, getattr(layer, "res_src", None)):
            if t:
                assert resident[compiled.placements[t].bank] == t, f"{layer.name}: input {t} was overwritten"
        if layer.dst in compiled.placements:
            bank = compiled.placements[layer.dst].bank
            assert bank != compiled.placements[layer.src].bank, f"{layer.name} writes the bank it reads"
            resident[bank] = layer.dst


def test_export_vectors(tmp_path):
    from resnet8.export_vectors import export

    qmodel, compiled, images = build("resnet8", seed=4)
    manifest = export(qmodel, images[0], tmp_path, trace=True)
    cfg = compiled.cfg
    assert read_hex(tmp_path / "program.hex") == compiled.program
    weights = read_hex(tmp_path / "weights.hex")
    assert weights == [pack_word(w, 8) for w in compiled.weight_mem[:compiled.weight_words]]
    assert len((tmp_path / "params.hex").read_text().splitlines()) == compiled.param_words
    assert manifest["expected"]["total_cycles"] == manifest["analytic_total_cycles"] == 56_640
    logits = [int(v) for v in (tmp_path / "expected" / "logits.txt").read_text().split()]
    assert logits == reference.run(qmodel, images[:1])[0].tolist()
    assert len(list((tmp_path / "expected").glob("*.hex"))) == 8
    assert all(len(line) == cfg.act_word_bits // 4 for line in (tmp_path / "input_bank0.hex").read_text().splitlines())
