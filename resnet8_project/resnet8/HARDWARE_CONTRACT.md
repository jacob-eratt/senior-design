# Software-to-hardware contract, resnet8-int8-v1

This specifies the implemented integer software and exported memory images.
Phases 4 and 5 remain RTL work. Do not claim their bit-exact RTL exit criteria
until a real simulator has run the supplied vectors.

## Arithmetic

Activations: signed two's-complement INT8 `[-128,127]`, per-tensor positive scale,
zero point 0. Weights: signed INT8, symmetric per-output-channel scale, typically
`[-127,127]`. Numeric value is `integer * scale`. Input scale is `255/127`, so
nonnegative RGB pixels use `[0,127]`; this baseline intentionally sacrifices one
bit of unsigned input precision for a uniform signed datapath.

Calibration uses maximum absolute FP32 values over the recorded training images,
divided by 127. Unobserved/zero ranges have a positive floor. Each residual add
and its following ReLU share the add's scale. Pooling retains its input scale.
Weights and activation scales are offline parameters, never recomputed in RTL.

For Conv + BN, per output channel:

```text
factor = gamma / sqrt(moving_variance + epsilon)
W_folded = W * factor
b_folded = (b - moving_mean) * factor + beta
```

Projection shortcuts have no BN. Folding uses float64 parameter arithmetic then
casts to float32; inference is compared with the original unfused FP32 graph at
every stage using `atol=3e-4, rtol=3e-4`.

For a weighted layer with input scale `Si`, output scale `So`:

```text
Sw[c] = max(max(abs(W_folded[...,c]))/127, 1e-12,
            abs(b_folded[c]) / (Si * (2**30 - 1)))
Wq[c] = clip(round_away(W_folded[c]/Sw[c]), -128, 127)
bq[c] = round_away(b_folded[c]/(Si*Sw[c]))          # INT32
acc[c] = sum(INT8_input * INT8_weight) + bq[c]   # INT32
ratio[c] = Si*Sw[c]/So
wide[c] = round_away(acc[c] * M[c] / 2**R[c])
output[c] = clip(wide[c], relu ? 0 : -128, 127) # INT8
```

`round_away` is nearest integer with **half ties away from zero**. The third term
in `Sw` protects constant-output channels with near-zero kernels and nonzero BN
offset. A static bound `128*sum(abs(Wq[c])) + abs(bq[c]) <= 2**31-1` is checked for
every weighted layer, ensuring partial MAC sums and bias cannot overflow. Overflow
is an error, not wrapping or saturating the accumulator.

`M` is a positive signed INT32 multiplier representing a Q31 significand. For
`ratio=f*2**e`, with `0.5 <= f < 1`, set `M=round(f*2**31)`, `R=31-e`, normalizing
the rare `M=2**31` case. Only `1 <= R <= 62` is supported. RTL executes a signed
64-bit product `p=INT64(acc)*M`, then:

```text
magnitude = (abs(p) + 2**(R-1)) >> R
wide = p < 0 ? -magnitude : magnitude
```

No float multiply occurs in the datapath. There is exactly one rounding step
per requantization. Arithmetic right shift of a negative product with a positive
rounding offset is **not** equivalent to this rule.

Residual add: requantize each INT8 branch separately into the common **output**
scale, retaining wide results. Add them as INT32; apply ReLU and saturate to INT8
only after summation. Do not saturate the branches before they can cancel. The
signed `blockN_add.npy` is clipped for inspection; `blockN_add_aligned_sum.npy`
preserves the true pre-clipping INT32 sum. Pooling sums the 64 elements in INT32,
rounds division by 64 with the same rule, and clips to INT8. Dense follows the
same per-channel MAC, bias and requantization rules, with ReLU disabled. Integer
logits determine argmax; softmax is host-side presentation only.

## Addressing and padding

Activation linear address: `((n*H + y)*W + x)*C + c` (NHWC, channels contiguous).
Kernel address: `((oc*K + ky)*K + kx)*Cin + ic` (**OHWI**). Dense address:
`oc*Cin + ic` (**OI**). Kernel elements are not flipped. Packed files contain
no channel padding, bank interleaving or alignment gaps.

For each dimension, `out=ceil(in/stride)`,
`total_pad=max((out-1)*stride + kernel - in, 0)`.
Leading padding is `floor(total_pad/2)`; trailing padding receives the remainder.
Out-of-bounds samples are integer zero. Thus a 3×3 stride-2 layer at 32 or 16
pixels pads **top/left 0, bottom/right 1**, not one pixel on every side.
A 1×1 stride-2 projection has no padding.

## Binary files

All words are little-endian, two's-complement. Byte offsets refer to the named
file, never element offsets. `export/manifest.json` records sizes, SHA-256,
tensor IDs/shapes/scales, layer descriptions and source/calibration provenance.

| File | Contents |
|---|---|
| `weights.bin` | Concatenated INT8 OHWI convolutions and OI dense weights |
| `biases.bin` | Concatenated INT32 bias vectors, output-channel order |
| `quant_params.bin` | Interleaved `(M,R)` INT32 pairs; per channel for conv/dense, two branch pairs for add, one for pool |
| `layer_config.bin` | Fourteen operation records, each 32 INT32 words (128 bytes) |

The config stream follows graph execution order: conv0, block1's two convs, add,
block2's two convs and projection, add, block3's two convs and projection, add,
pool, dense. Operation codes: conv=1, add=2, pool=3, dense=4. Tensor 0 is input;
other IDs are recorded in the manifest. `-1` means no second input / ReLU output.

| Word index | Meaning |
|---|---|
| 0–3 | op, output tensor ID, input0 ID, input1 ID |
| 4–7 | input height, width, Cin, Cout |
| 8–11 | kernel, stride, output height, output width |
| 12–15 | padding top, bottom, left, right |
| 16 | ReLU enable |
| 17–18 | weights byte offset, number of INT8 weights |
| 19–20 | bias byte offset, number of INT32 biases |
| 21–22 | quant-parameter byte offset, number of `(M,R)` pairs |
| 23 | add's post-ReLU output tensor ID, otherwise -1 |
| 24–31 | reserved zero |

The explicit field list is also embedded in the manifest. Per-layer `.mem`
files contain one 2-digit INT8 or 8-digit INT32 hex word per line, in the same
logical order as the binary stream. A future SystemVerilog memory can use
`$readmemh`; signedness must be declared in RTL. The binary loader verifies hashes,
decodes the streams and uses the decoded integer parameters for inference.
Float scales in JSON are for input conversion, debugging and displayed scores;
MAC/requantization consumes binary integer parameters.

## Phase 4 starter convolution

`artifacts/conv_fixture/` contains a real golden input and output channel 0 of
folded, quantized Conv0. H=W=32, Cin=3, Cout=1, kernel=3, stride=1, SAME padding,
ReLU enabled. `.npy`, `.mem`, config, expected INT32 accumulator and expected INT8
output are provided. The software result is checked against an independent scalar
Python-integer convolution. This is the first fixture a future RTL testbench
should consume; test both accumulator and final output bit-for-bit.

Suggested functional interface, without selecting a bus or cycle schedule:
start/busy/done, input/weight/bias memory read ports, output memory write port,
and either fixed config for the first fixture or a latched config record. Start
must latch all parameters; done must follow the final output write. Future RTL
must separately define reset, memory latency, backpressure and cycle behavior.
No AXI or board integration is required for this gate.

## Phase 5 programmable engine

The software model and exported real layer fixtures cover kernels 1×1/3×3,
strides 1/2, Cin 3/26/52/104, Cout 26/52/104, dimensions 32/16/8, SAME padding,
and ReLU on/off. Each convolution is described by one config record and per-channel
integer parameters; one physical engine can execute them in sequence. Tests also
check small odd/even images and signed extremes to expose address and rounding bugs.

`artifacts/layer_vectors/<layer>/` has a known input, accumulator and final output
for every real convolution. Weights/bias/quant parameters are in `export/`.
For full-network execution, the scheduler must retain each residual input until
both branches have consumed it. Tensor IDs express those dependencies; buffer
allocation, RAM banking, multiplier parallelism and cycle scheduling are future
microarchitecture decisions. Do not overwrite residual inputs early.

Future exit criteria: first the single-convolution RTL matches every fixture
element, then all nine programmable configurations match, then residual/pool/dense
RTL matches the integer reference, and finally every network intermediate and
prediction matches. FP32 accuracy alone does not prove any RTL gate.
