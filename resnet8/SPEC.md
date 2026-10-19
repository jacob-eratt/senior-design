# ResNet8 Inference Accelerator — Microarchitecture Specification

Version 0.1. The executable form of this document is the Python package
[`resnet8/accel`](accel/); where the two disagree, the code is authoritative,
since it is what the tests verify. Every number in this document comes from
the model's output.

## 1. Scope

| | |
|---|---|
| Workload | MLPerf Tiny image classification: ResNet8 on CIFAR-10 (32×32×3 → 10 classes), 77,946 parameters, 12.50 M MACs/inference |
| Numerics | uint8 activations, int8 weights, int32 accumulation, fixed-point requantization — no floating point in hardware |
| Baseline design point | 16×16 MAC array, all weights and activations on chip, 200 MHz target |
| Baseline performance | **56,640 cycles/inference → 283 µs, 3,531 inferences/s @ 200 MHz, 86.2% array utilization** |
| Accuracy | **85.66% int8 accelerator vs 85.63% float** on all 10,000 CIFAR-10 test images (MLPerf Tiny target 85%), §10 |

The RTL is correct when, for any input image, it produces bit-identical
activation-bank contents after every layer and bit-identical int32 logits
to the Python model. Its cycle count should also match the model. When the
RTL pipeline depth differs from the baseline, set `mac_extra_latency` and
`ppu_extra_latency` in `AccelConfig` (§7) so the model matches it.

## 2. Block diagram

```
                 host: preload input bank, start, poll done, read logits
                   │
  ┌────────────────┼──────────────────────────────────────────────────────────┐
  │  IMEM 32×128b ─┴► SEQUENCER / AGU ─────── act addr ──► ACT BANK 0 ┐ 1024×128b each
  │   (descriptors)     │  loop counters          (1R1W) ► ACT BANK 1 ├─ 16 ch × u8 per word
  │                     │                                ► ACT BANK 2 ┘
  │                     ├── weight addr ──► WEIGHT SRAM 512 × 2048b (16×16 × s8 per word)
  │                     ▼                        │                 │ act word
  │              micro-op token          weight word               ▼
  │                     └──────────► ┌──────────────────────────────────┐
  │                                  │ MAC ARRAY 16×16  (u8 × s8)       │
  │                                  │ 16 adder trees (one per out lane)│
  │                                  └───────────────┬──────────────────┘
  │                                                  ▼
  │                                  ACCUMULATORS 2 banks × 16 × int32
  │                                  (bank 0 main kernel, bank 1 1×1 shortcut)
  │                                                  │ one token per output pixel
  │   PARAM SRAM 64 × 1664b ────────────────► POST-PROCESSING UNIT (16 lanes)
  │   ACT BANK (identity residual) ─────────►  bias, ×m, +residual×m_res,
  │                                            round, >>shift, clamp[0,255]
  │                                                  │
  │                         ACT BANK write ◄─────────┴─────► LOGIT REGS 64 × int32
  │   POOLING UNIT (16 lanes, global avg) ──► ACT BANK write
  └───────────────────────────────────────────────────────────────────────────┘
```

Model code by block: sequencer → [`hw/sequencer.py`](accel/hw/sequencer.py);
MAC array, accumulators, PPU and pooling unit →
[`hw/units.py`](accel/hw/units.py); SRAMs → [`hw/sram.py`](accel/hw/sram.py);
pipeline and top level → [`hw/core.py`](accel/hw/core.py).

## 3. Numerics

All tensors are quantized with a real-valued scale `s` and zero point 0:
`real = s × q`.

| Quantity | Type | Notes |
|---|---|---|
| activations | uint8 | Every tensor is post-ReLU (≥ 0), so no zero point or sign bit is needed. The input image is the raw pixel, scale 1/255 (the network is trained on `pixel/255`). |
| weights | int8 in [−127, 127] | Symmetric. Per output channel for convolutions; per tensor for the classifier, so argmax over raw logits is valid. |
| bias | int32 | Scale `s_in × s_w[k]`; added in the PPU, not preloaded into the accumulator. |
| accumulator | int32, wraps | Worst case over *all possible* inputs needs ≤ 24 bits for the trained ResNet8 (`compiler._conv_widths` proves this per layer). |
| multiplier `m` | uint16 | `m / 2^shift ≈ s_in × s_w[k] / s_out` |
| shift | uint6, ≥ 1 | Per output channel. Shared by `m` and `m_res`. |

BatchNorm is folded into the preceding convolution at compile time.

### 3.1 Convolution layer

For output channel `k` at output pixel `(oy, ox)`:

```
acc      = Σ_{c,ky,kx} x[c, oy·S−P+ky, ox·S−P+kx] · w[k,c,ky,kx]     (out-of-range taps contribute 0)
main     = wrap32(acc + bias[k])
operand  = 0                                     res_mode NONE
         = r[k, oy, ox]            (uint8)       res_mode IDENTITY   (block input)
         = wrap32(acc_sc + bias_sc[k])           res_mode PROJECTION (1×1/stride-2 conv of block input)
t        = main · m[k] + operand · m_res[k]      (worst case ≤ 40 bits for ResNet8; model uses int64)
y        = clamp( (t + 2^(shift−1)) >> shift , 0, 255 )   arithmetic shift: round half toward +∞
```

ReLU is the lower clamp at 0, so the residual add and its ReLU are fused into
one PPU pass. The pre-add tensor is never stored or rounded separately.

Classifier (`OUT_S32`): `y = wrap32(acc + bias)`, written to the logit
registers. The host computes argmax directly on these.

### 3.2 Global average pool

`y[c] = clamp( (Σ_{h,w} x[c,h,w] · pool_m + 2^(pool_shift−1)) >> pool_shift, 0, 255 )`.
For 8×8 inputs, `pool_m = 2^15` and `pool_shift = 21`, which is exactly a
rounded divide by 64. The output keeps the input's scale. The sum needs
14 bits; the product needs 29 bits.

## 4. Memories

| Memory | Words × bits | Used (ResNet8) | Ports | Content |
|---|---|---|---|---|
| ACT bank 0..2 | 1024 × 128 | ≤ 1024 words | 1R + 1W, read-first | one pixel × 16 channels (u8) per word |
| WEIGHT | 512 × 2048 | 311 words (622 Kbit) | 1R | 16 in-ch × 16 out-ch × s8 per word |
| PARAM | 64 × 1664 | 16 words | 1R | per lane: bias_main, bias_res, m_main, m_res, shift |
| IMEM | 32 × 128 | 10 descriptors | 1R | §5 |
| LOGIT regs | 64 × int32 | 10 | host-readable | |

SRAM read latency is one cycle: the address is presented in cycle *t* and the
data is registered for cycle *t*+1. Writes become visible in the next cycle.

### 4.1 Word layouts and addressing (L = 16 lanes)

```
activation  addr = (y·W + x)·CT + ct            lane c  = channel ct·L + c
            word bits [8c+7 : 8c]                 (lanes ≥ C hold 0)
weight      addr = w_base + ((ot·K + ky)·K + kx)·IT + it
            element [c][k] = w[ot·L+k, it·L+c, ky, kx] at bits [8(c·L+k)+7 : 8(c·L+k)]
1×1 shortcut addr = res_w_base + ot·RT + rt      (same element order)
param       addr = p_base + ot                  lane k = output channel ot·L + k
            lane bits (104/lane, LSB first): bias_main[31:0] bias_res[63:32]
                                             m_main[79:64] m_res[95:80] shift[103:96]
```

Unused lanes have zero weights, `m = 0` and `shift = 1`, so they produce 0.
Hex packing for `$readmemh` is in [`accel/memfile.py`](accel/memfile.py).

### 4.2 Bank allocation for ResNet8

Each tensor starts at address 0 of its bank. The compiler allocates banks by
liveness. Three banks suffice because a residual block keeps only its input
and one intermediate tensor alive while writing its output.

| Layer | reads | residual from | writes | output |
|---|---|---|---|---|
| stem | bank 0 (input, host-preloaded) | | bank 1 | 32×32×16 |
| s1c1 | bank 1 | | bank 0 | 32×32×16 |
| s1c2 | bank 0 | bank 1 (identity) | bank 2 | 32×32×16 |
| s2c1 | bank 2 | | bank 0 | 16×16×32 |
| s2c2 | bank 0 | bank 2 (1×1/2 projection) | bank 1 | 16×16×32 |
| s3c1 | bank 1 | | bank 0 | 8×8×64 |
| s3c2 | bank 0 | bank 1 (1×1/2 projection) | bank 2 | 8×8×64 |
| pool | bank 2 | | bank 0 | 1×1×64 |
| fc | bank 0 | | logit regs | 10 × int32 |

Weight map (word addresses): stem 0, s1c1 9, s1c2 18, s2c1 27, s2c2 45
(+ shortcut 81), s3c1 83, s3c2 155 (+ shortcut 299), fc 307. Total 311.

## 5. Instruction set (layer descriptors)

There is one 128-bit descriptor per layer, terminated by `op = END`. Fields
are packed LSB first ([`accel/isa.py`](accel/isa.py)):

| Field | Bits | Width | Meaning |
|---|---|---|---|
| `op` | [1:0] | 2 | 0 END, 1 CONV, 2 GAP |
| `in_bank` | [3:2] | 2 | input activation bank |
| `out_bank` | [5:4] | 2 | output bank (OUT_U8) |
| `res_bank` | [7:6] | 2 | shortcut-source bank |
| `res_mode` | [9:8] | 2 | 0 none, 1 identity, 2 projection |
| `out_mode` | [10] | 1 | 0 u8 → bank, 1 int32 → logit regs |
| `in_h`, `in_w`, `in_ct` | [18:11], [26:19], [30:27] | 8, 8, 4 | input height, width, channel tiles |
| `out_h`, `out_w`, `out_ct` | [38:31], [46:39], [50:47] | 8, 8, 4 | output height, width, channel tiles |
| `kernel`, `stride`, `pad` | [52:51], [54:53], [56:55] | 2, 2, 2 | square kernel |
| `res_w`, `res_ct`, `res_stride` | [64:57], [68:65], [70:69] | 8, 4, 2 | shortcut-source width, tiles, 1×1 stride |
| `w_base`, `res_w_base` | [82:71], [94:83] | 12, 12 | weight-SRAM word addresses |
| `p_base` | [102:95] | 8 | param-SRAM word address |
| `pool_m`, `pool_shift` | [118:103], [124:119] | 16, 6 | GAP scaling |

Bits [127:125] are reserved (0).

## 6. Sequencer

The FSM has three states:

- **WAIT:** when the datapath is empty (`busy` low), fetch and decode the
  next descriptor (1 cycle). Go to **RUN**, or to **DONE** on END.
- **RUN:** issue exactly one micro-op per cycle. After the last one, go to
  **WAIT**.
- **DONE:** assert `done`.

The CONV loop nest (outermost first) issues one micro-op per innermost
iteration:

```
for ot in 0..out_ct-1                    output-channel tile (selects param word)
  for oy in 0..out_h-1
    for ox in 0..out_w-1                 one output pixel × 16 channels
      for ky, kx in kernel×kernel        main phase → accumulator bank 0
        for it in 0..in_ct-1               iy = oy·stride − pad + ky, ix likewise
      for rt in 0..res_ct-1              projection phase → accumulator bank 1
                                           (only res_mode = PROJECTION; iy = oy·res_stride)
```

Each micro-op carries:
- the act bank and address, or **padding** (read suppressed, zeros fed to the
  array) when `(iy, ix)` is outside the input;
- the weight address and accumulator bank;
- `clear`, set on the first op of an accumulation, which loads the psum
  instead of adding it;
- `emit`, set on the last op of a pixel, which sends both accumulator banks to
  the PPU;
- the output word address and the param address.

The GAP loop nest is `for ct: for p in 0..in_h·in_w−1`, reading
`addr = p·in_ct + ct` and emitting once per channel tile.

The projection shortcut is fused into the second convolution of each block.
It runs as extra micro-ops in the same pixel loop, so the shortcut tensor is
never written to SRAM and is never separately rounded.

## 7. Pipeline and timing

| Stage | Work | Memory ports used |
|---|---|---|
| ISSUE | sequencer emits op; SRAM addresses presented | act `in_bank` (or `res_bank` in projection phase) R; weight R |
| MUL | 256 products `u8 × s8` (17-bit) | |
| ADD (+`mac_extra_latency`) | 16 adder trees × 16 inputs → psum (21-bit) | |
| ACC | `acc[bank] = clear ? psum : acc + psum` (wrap 32); on `emit` → PPU | |
| PPU_RD | read param word; read identity residual | param R; act `res_bank` R (identity) |
| PPU_MAC (+`ppu_extra_latency`) | `t = wrap(acc+b)·m + operand·m_res` | |
| PPU_WB | round, shift, clamp; write | act `out_bank` W, or logit regs |
| POOL_ACC / POOL_WB | GAP: sum 16 lanes / scale + write | act `out_bank` W |

Timing of the last op of one output pixel (baseline latencies):

```
cycle        t      t+1     t+2     t+3       t+4      t+5      t+6
ISSUE      op_n   op_n+1  op_n+2   ...
MUL               op_n    op_n+1   ...
ADD                       op_n     ...
ACC                                op_n→emit
PPU_RD                                      px
PPU_MAC                                              px
PPU_WB                                                        px  (write commits end of t+6)
```

The design has no stalls. The PPU and pooling unit are fully pipelined, and
a pixel takes at least one issue cycle, so the PPU can always accept the next
token. No memory exceeds its ports in any cycle:
- `in_bank`, `res_bank` and `out_bank` are pairwise distinct (liveness
  allocation).
- The projection phase reads `res_bank` from ISSUE, but in that mode PPU_RD
  does no residual read.

The model checks every port on every cycle and raises `HardwareError` on a
conflict. `test_port_conflict_is_detected` deliberately creates one.

Between layers, the sequencer waits for the pipeline to drain, so a layer
never reads a bank before the previous layer's last write commits. This
costs `conv_drain = 7 + mac_extra_latency + ppu_extra_latency` cycles per CONV
layer and 3 per GAP layer.

The per-layer cycle count is exact:

```
CONV:  out_ct · out_h · out_w · (kernel² · in_ct + [res_ct if projection]) + conv_drain
GAP:   in_ct · in_h · in_w + 3
total: Σ layers + 1 (END fetch)
```

## 8. Performance (baseline, 16×16 @ 200 MHz)

| Layer | Shape out | Cycles | MACs | Util. | Act reads | Padding ops | Weight reads |
|---|---|---|---|---|---|---|---|
| stem | 32×32×16 | 9,223 | 442,368 | 18.7% | 8,836 | 380 | 9,216 |
| s1c1 | 32×32×16 | 9,223 | 2,359,296 | 99.9% | 8,836 | 380 | 9,216 |
| s1c2 (+id) | 32×32×16 | 9,223 | 2,359,296 | 99.9% | 8,836 (+1,024 residual) | 380 | 9,216 |
| s2c1 | 16×16×32 | 4,615 | 1,179,648 | 99.8% | 4,418 | 190 | 4,608 |
| s2c2 (+proj) | 16×16×32 | 9,735 | 2,490,368 | 99.9% | 8,976 | 752 | 9,728 |
| s3c1 | 8×8×64 | 4,615 | 1,179,648 | 99.8% | 4,232 | 376 | 4,608 |
| s3c2 (+proj) | 8×8×64 | 9,735 | 2,490,368 | 99.9% | 8,256 | 1,472 | 9,728 |
| pool | 1×1×64 | 259 | — | — | 256 | 0 | 0 |
| fc | 10 | 11 | 640 | 22.7% | 4 | 0 | 4 |
| **total** | | **56,640** | **12,501,632** | **86.2%** | | | |

Latency is 283.2 µs, giving 3,531 inferences/s and 88.3 effective GOPS
(peak 102.4).

The total is the sum of the rows plus 1 for the final END fetch.

### 8.1 Resource estimate

These are first-order FPGA estimates, not synthesis results:

- **MAC array:** 256 8×8 multipliers, about 128 DSP48E2 with INT8 packing (one
  activation is shared by two weights in a row).
- **PPU:** 32 multipliers of 32 b × 16 b.
- **Activation SRAM:** 384 Kbit, about 12 BRAM36.
- **Weight SRAM:** 622 Kbit used of 1 Mbit. The 2048-bit word sets the count
  at 29 BRAM36, by width rather than capacity.
- **Param SRAM:** 26 Kbit; LUTRAM is a better fit.

### 8.2 Design-space notes (from `perf.explore`)

| Array | Multipliers | Cycles | Latency @200 MHz | Utilization |
|---|---|---|---|---|
| 8×8 | 64 | 207,436 | 1.04 ms | 94.2% |
| **16×16** | **256** | **56,640** | **283 µs** | **86.2%** |
| 32×32 | 1,024 | 36,286 | 181 µs | 33.6% |

A 32×32 array wastes most of its lanes on the 16-channel layers. 16×16
matches ResNet8's channel widths.

Known improvements, not modeled:

1. **Stem packing.** The stem uses 3 of 16 input lanes (18.7%) yet takes
   16% of runtime. Packing (kx, c) into lanes (9 of 16 used) cuts the stem
   from 9,216 to 3,072 issue cycles, about 11% of the total.
2. **Overlapping inter-layer drain** saves at most 59 cycles (8 CONV layers
   × 7, plus 3 for GAP), about 0.1%; not worth the hazard logic.
3. **Weight-stationary reuse.** Every MAC cycle reads a 2048-bit weight word;
   caching one tap's weights across pixels would cut weight-SRAM energy, which
   is likely the dominant dynamic-power term.
4. **24-bit accumulators** suffice for the trained ResNet8 (worst case over
   all inputs, proved per layer by the compiler). This saves 25% of
   accumulator and PPU-input width; keep 32 bits if the weights may change.

## 9. Verification flow

1. `python -m resnet8.export_vectors --image-index N --trace` writes:
   - `program.hex`, `weights.hex`, `params.hex` and `input_bank0.hex`
     (`$readmemh`);
   - one expected bank dump per layer (`expected/NN_<layer>.hex`);
   - `expected/logits.txt`;
   - `write_trace.csv`, every activation-bank write with its cycle;
   - `manifest.json`, with field slices, the memory map and expected cycle
     counts.
2. The RTL testbench preloads the memories, starts the core, and compares
   each layer's output bank with the dump at the layer boundary (or compares
   the write stream with the trace), then compares the logits and total cycles.
3. Python-side guarantees (`python -m pytest resnet8/tests`):
   - the cycle model matches the functional reference on every intermediate
     tensor, for ResNet8, an odd-shaped network (partial tiles, odd sizes,
     several output tiles), an 8-lane design point and deeper pipelines;
   - the closed-form performance model matches the cycle model's cycle count
     and every SRAM access counter;
   - the fixed-point primitives match exact rational arithmetic;
   - port conflicts and memory overflows are detected;
   - activation banks are poisoned before each run, so a read of an unwritten
     word cannot pass silently.

The cycle model does not cover:
- host/DMA timing (the input is preloaded through a backdoor; streaming
  32×32 words at 128 b/cycle would add about 1,024 cycles unless
  double-buffered);
- physical SRAM banking of the 2048-bit weight word;
- clock-domain crossings;
- power and timing closure.

## 10. Accuracy

Results for the checkpoint `resnet8.pth`: 40 epochs, SGD with OneCycle,
crop+flip augmentation. Calibration used 2,000 training images.

| Model | CIFAR-10 test accuracy |
|---|---|
| float32 (PyTorch) | 85.63% |
| **int8 accelerator (bit-exact model)** | **85.66%** |
| prediction agreement int8 vs float | 99.07% |

Per-tensor SQNR (float vs dequantized integer) is 28–36 dB; logits are at
30.7 dB. Fewer than 0.005% of activations saturate.

Sensitivity, measured on the full test set (informational; the defaults were
chosen before seeing these numbers):

| Activation calibration | `mult_bits` | int8 accuracy |
|---|---|---|
| max | 16 | 85.56% |
| 99.99th percentile | 16 | 85.66% |
| 99.999th percentile | 16 | 85.68% |
| **MSE-optimal clip (default)** | **16** | **85.66%** |
| MSE-optimal clip | 24 | 85.67% |

A 16-bit requantization multiplier costs nothing measurable compared with
24 bits. Post-training quantization is enough; quantization-aware training
is not needed for this network.
