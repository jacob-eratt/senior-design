# Step 04 – Where everything lives on chip

Decisions this step builds on: D-002 (activations stay on chip between
layers), D-003 (200 MHz target), D-004 (two accumulators per PE).

## 1. Concept

Off-chip DDR traffic per inference should be only the **image in** (3,072 B)
and the **logits out** (10 B). Everything else lives in on-chip RAM:

```text
                 boot-time DMA (once)               per-inference DMA
DDR ──► weight / bias / quant / descriptor RAMs     DDR ──► IN buffer ──┐
                     │                                                  │
                     ▼                                                  ▼
   ┌──────────── Layer sequencer (descriptor table) ────────────────────────┐
   │                                                                        │
   │  Weight RAM (URAM) ──26 B/clk──► [row skew] ──► 26 × C systolic array  │
   │  Act buffers A/B/C (BRAM) ─► im2col feeder ─C B/clk─► [col skew] ─┘    │
   │                                              │ drain: 1 oc row/clk     │
   │  Bias/quant RAM ──────────────► output pipeline: requant → (+residual  │
   │  Residual read (A/B/C) ───────►   requant, add) → ReLU → sat INT8      │
   │                                              │                         │
   │                          write back to A/B/C ◄┘   → pool/dense → logits │
   └────────────────────────────────────────────────────────────────────────┘
```

## 2. Current design (for contrast)

The Conv0 core stores **32 full replicas of the input image** so each of the 32
pixel lanes can read independently, and compiles weights/quant into logic. That
does not scale past Conv0 and is a large part of the CLB pressure. A systolic
array removes the need: only the C top-edge inputs need fresh data per cycle.

## 3. Activation buffers: three ping-pong buffers

Every activation tensor is ≤ 26,624 B (32×32×26 = 16×16×52·2 = 8×8×104·4).
Three equal buffers **A, B, C** of 26,624 B plus a small **IN** buffer suffice.
Residual adds are **fused** into the second conv of each block, so the conv2
result never needs its own buffer.

| Step | Operation | Feeder reads | Residual reads | Writes |
|---:|---|---|---|---|
| 1 | conv0 | IN | – | A |
| 2 | block1_conv1 | A | – | B |
| 3 | block1_conv2 + add | B | A (identity) | C |
| 4 | block2_conv1 (s2) | C | – | A |
| 5 | block2_shortcut (1×1 s2) | C | – | B |
| 6 | block2_conv2 + add | A | B | C |
| 7 | block3_conv1 (s2) | C | – | A |
| 8 | block3_shortcut (1×1 s2) | C | – | B |
| 9 | block3_conv2 + add (+ pool sums) | A | B | C |
| 10 | avgpool + dense | C (or pool accumulators) | – | logits registers |

Properties:

- **No layer ever reads and writes the same buffer**, so there are no
  read-after-write hazards and every residual input survives until both
  branches consume it (HARDWARE_CONTRACT requirement).
- Pattern from block2 on is a fixed rotation: `C → A`, `C → B`, `A + B → C`.
  The sequencer only needs buffer IDs in each descriptor.
- A separate IN buffer (optionally double-buffered) lets the CPU DMA the
  **next** image while the current one runs.

Fused add, per HARDWARE_CONTRACT: conv2's accumulator is requantized to INT8
exactly as a standalone conv would; that INT8 value and the residual INT8 value
are each requantized with their add `(M,R)` pair to wide INT32, summed, ReLU'd,
then saturated to INT8. Same bits as running the add as a separate pass.

## 4. Activation layout: channel-planar, 16-pixel words

Proposed on-chip layout (differs from the exported NHWC files; the host or
loader repacks the 3 KB input once per image):

```text
word = X[c][p0 .. p0+15]         16 INT8 pixels of ONE channel, 128 bits
p    = y*W + x                   flattened pixel index
addr = base + c*(P/16) + p/16
```

| Tensor width W | One 16-pixel word holds |
|---:|---|
| 32 | half an image row |
| 16 | one image row |
| 8 | two image rows |

Why this layout:

- **It matches the drain.** With one requant unit per column (C = 16), the
  array drains one output-channel row per cycle: 16 pixels of one channel =
  exactly one word. No transpose, one write per cycle.
- **It matches the feeder.** For a given tap `(ky, ic)`, the 16 pixels of a tile
  need 16 neighbouring inputs from one or two input rows of one channel = one
  or two words plus a shift.

### 4.1 Feeder read rate (derived)

Reorder the reduction as `for ky, for ic, for kx` (kx innermost). The feeder
reads a row segment once and reuses it for the three kx taps by shifting.
Integer sums are order-independent, so the weight RAM simply stores weights in
the same `(ky, ic, kx)` order.

| Layer type | Words needed | Cycles they serve | Reads/cycle |
|---|---:|---:|---:|
| 3×3 s1, W=32 (half row + 1 halo pixel) | 2 | 3 | 0.67 |
| 3×3 s1, W=16 (one row, halos are padding) | 1 | 3 | 0.33 |
| 3×3 s1, W=8 (two rows per tile) | ≤ 2 | 3 | ≤ 0.67 |
| 3×3 s2 (pads top/left 0, bottom/right 1) | 2 | 3 | 0.67 |
| 1×1 s2 shortcut (no kx reuse) | 2 | 1 | **2** |

Buffers are true dual-port BRAM. Because no buffer is read and written in the
same layer, **both ports are free for reading**. The 1×1 shortcut (2 reads per
cycle) runs at full rate.

## 5. Weights, constants, descriptors

| Store | Contents | Size | Organization | Physical (derived) |
|---|---|---:|---|---|
| Weight RAM | All conv weights, `[layer][oc_tile][k]`, 26 B per word (one per array row) | 7,775 words × 26 B = 202,150 B | 1 word read per cycle during compute | 6 URAM (3 wide × 2 deep = 8,192 × 216 b) |
| Dense weights | 104 → 10 | 1,040 B | Small RAM or LUTRAM | < 1 BRAM |
| Per-channel params | `{bias, M, R}` × 556 channels | ~6.7 KB | 1 record per drain cycle (drain row = oc) | 2 BRAM36 |
| Add/pool params | 3 adds × 2 pairs + 1 pool pair | 56 B | Registers | – |
| Descriptor table | 14 records × 128 B (existing `layer_config.bin`) | 1,792 B | Read by sequencer per layer | 1 BRAM36 or regs |

Weights are re-streamed once per pixel tile (K words per tile); with all
203 KB resident there is no off-chip weight traffic per inference.

## 6. Budget

| Item | BRAM36 | URAM |
|---|---:|---:|
| A, B, C (1,664 × 128 b each → 2 wide × 4 deep) | 24 | – |
| IN, double-buffered (192 × 128 b each) | 4 | – |
| Per-channel params, descriptors, dense | 3–4 | – |
| Weights | – | 6 |
| **Total (of 144 BRAM36 / 64 URAM)** | **~32 (22%)** | **6 (9%)** |

Logical estimate only; to be confirmed by synthesis. Plenty of margin remains
for 26×32 or extra banking if needed.

## 7. Open questions

1. Repack input NHWC → channel-planar on the CPU (simple) or in hardware?
2. Write block3 output to C (debuggable) or only accumulate pool sums (saves
   one pass of writes)? Recommend writing it at first.
3. Dense layer: run on the array (pad OC 10 → 26, P = 1, 104 cycles), on a tiny
   dedicated MAC, or on the CPU?
4. Weight load at boot: DMA into URAM through the same AXI-Stream as images, or
   a separate AXI path?

## 8. Decisions

| ID | Decision |
|---|---|
| D-002 | Activations stay on chip; three 26,624 B buffers A/B/C + IN, rotation per §3; residual add fused into conv2. |
