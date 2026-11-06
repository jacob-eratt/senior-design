# Step 03 – Systolic array mapping: 26×32 vs 26×16

Team direction (D-001): a general convolution engine built on a systolic array,
26 rows × 32 or 16 columns.

## 1. Concept

Using step 02's matrix view `Y[oc][p] = Σ_k W[oc][k] · X̂[k][p]`:

- **Rows = 26 output channels**, **columns = C output pixels**, reduction `k`
  runs in time. Each PE owns one accumulator: **output-stationary (OS)**.
- Weights enter at the left edge (one per row per cycle) and move right one PE
  per cycle. Activations enter at the top (one per column per cycle) and move
  down one PE per cycle. Inputs are skewed so `W[oc][k]` meets `X̂[k][p]` in
  PE(oc, p).

```text
            X̂[k][p0]  X̂[k][p1] ... X̂[k][pC-1]     (skewed by column)
               ↓          ↓              ↓
W[oc0][k] →  PE ──────→  PE ── ... ──→  PE
               ↓          ↓              ↓
W[oc1][k] →  PE ──────→  PE ── ... ──→  PE         (skewed by row)
   ...
W[oc25][k]→  PE ──────→  PE ── ... ──→  PE
```

Same dataflow as today (OS, oc × pixel), but operands travel **neighbor to
neighbor** instead of being **broadcast**. That swaps a 26/32-way fanout net for
short local wires, the main benefit for timing and routing (the current design
is at 97.7% CLB at 100 MHz).

OS fits ResNet-8 because K varies 27 → 936: K is just "how many cycles to
stream", so no lanes go idle when K changes. A weight-stationary (TPU-style)
array would spatially unroll K and need tiling/partial-sum handling for every K.

### 1.1 The alternative: weight-stationary ("TPU style")

Google's TPU v1 uses a 256×256 **weight-stationary (WS)** systolic array:

```text
             oc0      oc1          ocN-1          each PE holds ONE weight W[k][oc]
X̂[k0][p] →  PE  ───→  PE  ─ ... ─→  PE           activations flow left → right
              ↓ psum   ↓             ↓            partial sums flow top → bottom
X̂[k1][p] →  PE  ───→  PE  ─ ... ─→  PE
   ...                                            rows = reduction k (a K-tile)
              ↓        ↓             ↓
          Y[oc0][p] Y[oc1][p] ...  (partial if K > rows → accumulator SRAM)
```

1. Load a K-tile × OC-tile block of weights into the PEs (one-time cost).
2. Stream pixels through: one new pixel `p` enters per cycle; every PE does one
   MAC and passes the partial sum down. A finished (partial) dot product for
   one pixel exits the bottom of each column every cycle.
3. If `K` is larger than the number of rows, outputs are only partial sums;
   they go to an **accumulator memory** and the next K-tile's results are added.

| | Output-stationary (ours) | Weight-stationary (TPU) |
|---|---|---|
| What stays in the PE | Partial sum (accumulator) | Weight |
| What moves | Weights → and activations ↓ | Activations → and partial sums ↓ |
| Array dimensions | OC × pixels | K × OC |
| Variable K (27–936) | Free: K = number of cycles | Tile K by rows; psum SRAM + adders when K > rows |
| Weight reuse | Each weight used by C pixels per pass | Each weight used by every pixel streamed (high) |
| Cost of few pixels (block3: 64) | None | Weight reload (≈ rows cycles) amortized over only 64 pixels |
| Extra storage | 2 accumulators per PE | Partial-sum memory outside array |
| Best when | K varies, pixel count small/moderate | Big matrices, large batches (TPU serves batches) |

For ResNet-8 at batch 1 with K from 27 to 936 and late layers of only 64
pixels, OS is the better fit; WS shines with large batches, which we don't have.

### 1.2 How tiling works in a weight-stationary array

The layer's weight matrix is `K × OC` (e.g. conv0: 27 × 26; block3_conv2:
936 × 104). An `R × Cc` array can hold one `R × Cc` block of it at a time, so
the matrix is cut into blocks and **every block is a new weight load**:

```text
for oc_t in 0 .. ceil(OC/Cc)-1:          # column tiling: split output channels
  for k_t in 0 .. ceil(K/R)-1:           # row tiling: split the reduction
    load W[k_t*R : (k_t+1)*R][oc_t*Cc : (oc_t+1)*Cc] into the PEs
    for p in 0 .. P-1:                   # stream every pixel through
      psum[p][oc_t] += column outputs    # read-modify-write accumulator RAM
  requant psum[:, oc_t] → INT8           # after the last k_t
```

| Tiling direction | Splits | Consequence |
|---|---|---|
| Columns (`Cc`) | Output channels | Independent results; re-stream all pixels once per column tile |
| Rows (`R`) | Reduction `K` | Results are **partial**; need psum RAM sized `P × Cc` INT32 with a read-modify-write every cycle |
| Pixels (optional) | `P` into chunks of `Pt` | Shrinks psum RAM to `Pt × Cc`, but weights reload `P/Pt` times more often |

TPU v1 double-buffers the weights in each PE: the next block shifts in while
the current block computes, so the weight load (≈ R cycles) is hidden as long
as each block streams at least R pixels.

Worked example, `R × Cc = 27 × 26` (702 PEs; conv0 fits in exactly one block).
Ideal cycles = tiles × P (weight loads hidden, skew ignored):

| Layer | K | K-tiles | OC-tiles | Tiles | P | Cycles |
|---|---:|---:|---:|---:|---:|---:|
| conv0 | 27 | 1 | 1 | 1 | 1,024 | 1,024 |
| block1_conv1/conv2 | 234 | 9 (last 18/27 rows used) | 1 | 9 each | 1,024 | 18,432 |
| block2_conv1 | 234 | 9 | 2 | 18 | 256 | 4,608 |
| block2_conv2 | 468 | 18 | 2 | 36 | 256 | 9,216 |
| block2_shortcut | 26 | 1 | 2 | 2 | 256 | 512 |
| block3_conv1 | 468 | 18 | 4 | 72 | 64 | 4,608 |
| block3_conv2 | 936 | 35 | 4 | 140 | 64 | 8,960 |
| block3_shortcut | 52 | 2 | 4 | 8 | 64 | 512 |
| **Total** | | | | **295** | | **47,872** |

That is 97% utilization (32.56M / (702 × 47,872)), 239 µs at 200 MHz. The cost
is a psum RAM of 1,024 × 26 × 4 B = 106 KB, read and written at 26 × 32 bits per
cycle, plus a feeder that delivers 27 different `k` elements of one pixel per
cycle. The OS array keeps those sums inside the PEs instead.

### 1.3 Pixel tiling and the cost of weight reloads

With one weight register per PE, **every tile in any direction is a weight
reload**. Tiling choices only change *how many* reloads happen and *how big* the
partial-sum (psum) RAM is:

```text
weight loads per layer = (P / Pt) · ceil(K / R) · ceil(OC / Cc)
psum RAM               = Pt · Cc · 4 B          (only needed if K > R)
reload stall           = loads · R              (single-buffered)
                       = loads · max(0, R − Pt) (double-buffered)
```

Pixel tiling = stream only `Pt` pixels per weight block, then move to the next
K-tile, and revisit the next pixel chunk later:

```text
for p_t in 0 .. P/Pt-1:                  # pixel chunk
  for k_t in 0 .. ceil(K/R)-1:           # reload weights each time
    load W block k_t
    stream pixels p_t*Pt .. p_t*Pt+Pt-1, psum[0..Pt-1] += outputs
  requant psum → INT8 for this chunk
```

Example: `R × Cc = 16 × 26`, conv0 (K = 27 → 2 K-tiles of 16 + 11 rows, 84%
row utilization; OC = 26 → 1 column tile; P = 1,024):

| Pixel tile `Pt` | Weight loads | Psum RAM | Stall if single-buffered (16 cyc/load) | Stall if double-buffered |
|---:|---:|---:|---:|---:|
| 1,024 (none) | 2 | 106,496 B | 32 cycles | 0 |
| 256 | 8 | 26,624 B | 128 | 0 |
| 32 (one row) | 64 | 3,328 B | 1,024 (+50%) | 0 |
| 8 | 256 | 832 B | 4,096 (+200%) | 2,048 (Pt < R: 256 × (16 − 8)) |

Double-buffering = **two weight registers per PE** (active + shadow): the next
block shifts into the shadow registers while the active block computes. This is
the WS analogue of our two accumulators per PE. Holding even more weights per PE
(a small per-PE RAM with all K-tiles) removes reloads entirely; each pixel then
sweeps all K locally, which is effectively becoming output-stationary.

In our OS array weights are never "loaded": a new weight enters every row every
cycle (26 B/clk from URAM), so changing tiles costs nothing beyond the drain,
which the second accumulator hides.

Choice of `R` matters: K values are 27, 234, 468, 936, 26, 52. `R = 26` fits
every layer except conv0 (2 tiles, second uses 1 row); `R = 27` fits conv0 but
wastes ~4% elsewhere; `R = 9` fits every 3×3 layer but not the shortcuts.

## 2. Numbers (derived, ideal = K cycles per tile)

Tile = 26 output channels × C output pixels. Pixels are flattened `(oy, ox)`.

| Layer | K | Tiles 26×32 | Cycles 26×32 | Tiles 26×16 | Cycles 26×16 |
|---|---:|---:|---:|---:|---:|
| conv0 | 27 | 32 | 864 | 64 | 1,728 |
| block1_conv1 | 234 | 32 | 7,488 | 64 | 14,976 |
| block1_conv2 | 234 | 32 | 7,488 | 64 | 14,976 |
| block2_conv1 | 234 | 16 | 3,744 | 32 | 7,488 |
| block2_conv2 | 468 | 16 | 7,488 | 32 | 14,976 |
| block2_shortcut | 26 | 16 | 416 | 32 | 832 |
| block3_conv1 | 468 | 8 | 3,744 | 16 | 7,488 |
| block3_conv2 | 936 | 8 | 7,488 | 16 | 14,976 |
| block3_shortcut | 52 | 8 | 416 | 16 | 832 |
| **Total** | | **168** | **39,136** | **336** | **78,272** |

Both shapes reach **100% lane utilization** on every conv layer, because every
OC is a multiple of 26 and every P a multiple of 64. That is a good property of
the 26-row choice.

| Shape | PEs / DSPs (1 MAC/DSP) | 100 MHz | 200 MHz | 250 MHz |
|---|---:|---:|---:|---:|
| 26×32 | 832 (67%) | 391 µs | 196 µs | 157 µs |
| 26×16 | 416 (33%) | 783 µs | 391 µs | 313 µs |

**26×16 at 200 MHz = 26×32 at 100 MHz** in throughput, with half the DSPs.
A systolic array is exactly the structure that makes 200+ MHz realistic.

### 2.1 Overheads the ideal number hides

| Overhead | 26×32 | 26×16 | Mitigation |
|---|---|---|---|
| Skew fill/flush per tile (R+C−2) | 56 cycles × 168 = 9,408 (+24%) | 40 × 336 = 13,440 (+17%) | Stream tiles back-to-back; pay skew once per layer |
| Result drain (26 rows out per column) | 26 cycles/tile | 26 cycles/tile | Shadow accumulator + shift-out chain, overlapped with next tile |
| Layer switch (descriptor load, weight prefetch) | per layer | per layer | Prefetch next layer's weights during current layer |

Drain overlap works for every layer because the drain (26 cycles with one
requant unit per column) is ≤ K for every layer (min K = 26). Without
back-to-back tiles and drain overlap, real time is ~1.2–1.3× the ideal.

### 2.2 Edge bandwidth into the array (per cycle)

| Shape | Activation bytes | Weight bytes | Total on-chip read |
|---|---:|---:|---:|
| 26×32 | 32 | 26 | 58 B/cycle |
| 26×16 | 16 | 26 | 42 B/cycle |

Bandwidth is easy for BRAM/URAM; the hard part is **generating 16/32 different
im2col addresses per cycle** (step 04).

## 3. 26×32 vs 26×16 – assessment

| Criterion | 26×32 | 26×16 |
|---|---|---|
| Throughput at equal clock | 2× | 1× |
| DSPs | 832 (67%) | 416 (33%) |
| Pixel tile vs layer widths 32/16/8 | 1 row / 2 rows / 4 rows | ½ row / 1 row / 2 rows |
| im2col feeder complexity | Tiles span 2–4 output rows in later layers | Tiles span ≤ 2 rows; matches 16×16 exactly |
| Routing/timing risk | Higher | Lower |
| Headroom for requant, residual, buffers, DMA | Less | Much more |
| Upgrade path | — | Second array, DSP INT8 packing, or widen to 32 later |

Recommendation (opinion, to be debated): **start with 26×16, target 200 MHz**,
and parameterize columns (`C`) so 26×32 is a rebuild, not a redesign. It
matches the current design's throughput with half the DSPs, a simpler feeder
and much lower routing risk. If 200 MHz is not met, 26×32 at 100 MHz remains
the fallback.

## 4. What "general engine" really requires (beyond the array)

The array is the easy, regular part. The work that makes it general:

1. **Activation buffer + im2col feeder**: 3×3/s1, 3×3/s2, 1×1/s2, zero padding,
   IC = 3/26/52. Delivers C operands per cycle. (Step 04)
2. **Weight buffer**: all 203 KB resident on chip (fits in URAM), streamed 26
   bytes/cycle in `[oc_tile][k]` order. (Step 04)
3. **Accumulator width**: K=936 INT8×INT8 sums need ≥ 26 bits plus bias; a DSP's
   48-bit P register is ample. (Step 05)
4. **Output pipeline**: per-channel requant → optional residual add → ReLU →
   INT8 write-back to activation buffer. Quant constants become runtime data.
   (Step 06)
5. **Layer sequencer**: descriptor table (OC, P, K, stride, pad, buffer
   addresses, quant pointers), no per-layer RTL. (Step 08)
6. **Avgpool + dense** (1,040 MACs): tiny; accumulate in the output pipeline,
   or leave to the CPU. (Step 08)

## 5. Open questions

1. Are activations kept **on chip between layers** (≈80 KB, fits) or written to
   DDR after every layer? On-chip is strongly preferred for ResNet-8.
2. Clock target: is the team willing to aim at 200 MHz (needs pipelined PEs,
   registered BRAM outputs, DSP internal registers)?
3. Flattened pixel tiles (spanning multiple output rows) vs row-aligned tiles:
   flattening gives 100% utilization but complicates the feeder.
4. Do we want shortcut 1×1 convs to run on the same array (simple, 2% of MACs)
   or fused with the main path?

## 6. Decisions

| ID | Decision |
|---|---|
| D-001 | Output-stationary systolic array, rows = 26 output channels, columns = output pixels, K in time. Column count 16 vs 32 still open. |
| D-003 | Target 200 MHz; PEs, RAM outputs and DSPs fully registered. |
| D-004 | Two accumulators per PE: one accumulates the current tile while the other drains; tiles stream back-to-back. |
