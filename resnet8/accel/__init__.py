"""Bit-accurate, cycle-level Python model of an int8 ResNet8 inference accelerator.

Layers of the package, from numerics up:

    config      AccelConfig: array size, bit widths, memory depths, clock
    fixedpoint  integer arithmetic primitives shared with the RTL spec
    qmodel      quantized network description (output of resnet8.quantize)
    reference   fast vectorised bit-exact functional model (golden model)
    isa         layer-descriptor ("instruction") format and bit encoding
    compiler    QModel -> program + SRAM images (weights, params, banks)
    hw/         cycle-stepped structural model: SRAMs, sequencer, MAC array,
                accumulators, post-processing unit, pooling unit
    perf        closed-form cycle/utilisation/resource model

Everything in this package depends only on numpy.
"""

from .config import AccelConfig

__all__ = ["AccelConfig"]
