# Conv0 accelerator: summary of the three proposed architectures

This document summarizes the three architectures in this folder and shows how
each one computes the **Conv0** output.

| # | Folder | Style | Array | Status |
|---|---|---|---|---|
| 1 | [arch1_broadcast_832](arch1_broadcast_832/ARCHITECTURE.md) | Broadcast MAC grid (not systolic) | 32 pixels × 26 channels = 832 MACs | **Implemented** for Conv0: RTL, XSim regression, routed on KV260. Not yet run on the board |
| 2 | [arch2_os_systolic_26x32](arch2_os_systolic_26x32/ARCHITECTURE.md) | Output-stationary systolic | 26 channels × 32 pixels = 832 PEs | Proposal only (paper design) |
| 3 | [arch3_ws_tpu_27x26](arch3_ws_tpu_27x26/ARCHITECTURE.md) | Weight-stationary, TPU-style systolic | 27 reduction taps × 26 channels = 702 PEs | Proposal, with partial RTL sketch in `rtl/` (`ws_pe`, `ws_column`, `ws_array`, `ws_skew`, `ws_mxu`) |

---

## 0. The common problem: Conv0 as a matrix multiply

Conv0 maps a 32×32×3 INT8 image to a 32×32×26 INT8 feature map. It uses a
3×3 kernel, stride 1 and zero padding 1. BatchNorm is folded into the weights
and bias. Each output is computed as follows:

```text
Y[oc][p] = bias[oc] + Σ_k  W[oc][k] · X̂[k][p]            then requant → ReLU → sat INT8

  oc ∈ 0..25          output channel     (26)
  p  = (oy, ox)       output pixel       (1,024)
  k  = (ky, kx, ic)   reduction index    (27 = 3 × 3 × 3)
  X̂[k][p] = X[oy+ky−1][ox+kx−1][ic]      (im2col view; zero if outside the image)
```

```text
            K = 27                      P = 1,024                      P = 1,024
        ┌───────────┐               ┌─────────────────┐            ┌─────────────────┐
  OC=26 │     W     │   ×    K = 27 │       X̂         │  =   OC=26 │        Y        │
        │ (702 B)   │               │ (im2col of img) │            │  (26,624 B)     │
        └───────────┘               └─────────────────┘            └─────────────────┘
```

All three designs compute the same 26 × 1,024 × 27 = **718,848 MACs**. They
differ in three ways: which matrix dimension is laid out in space, which
operand stays fixed in each processing element (PE), and where partial sums and
results are stored.

| | Spatial dim #1 | Spatial dim #2 | Runs in time | What stays put in a PE |
|---|---|---|---|---|
| Arch 1 | 32 pixels (ox) | 26 channels | k (27 clk), then rows oy | Accumulator |
| Arch 2 | 26 channels | 32 pixels (ox) | k (27 clk), then rows oy | Accumulator (DSP P reg) |
| Arch 3 | 27 k-taps | 26 channels | pixels p (1,024 clk) | **Weight** |

---

## 1. Architecture 1: broadcast 832-MAC grid (implemented)

### 1.1 Idea

Each tile is one output image row `oy`. A grid of 32 × 26 MACs computes all 32
pixels × 26 channels of that row at the same time. In each clock, the
controller picks one tap `t = (ky, kx, ic)` from the 27 taps. **The same
operand is wired to a whole row or column of MACs at once.** Operands do not
pass from one MAC to the next.

- Each image sample is sent to the 26 MACs in its pixel row. Each of those
  MACs uses a different channel weight.
- Each weight is sent to the 32 MACs in its channel column. Each of those MACs
  uses a different pixel.

After 27 taps, the grid holds 832 finished INT32 sums. The grid drains those
sums, then the same hardware processes the next row. The 32 rows are processed
one after another.

### 1.2 Array structure: what feeds in, where sums live

```text
                         26 weight register banks (27 INT8 each), 1 read per clk
                         wq[c] = W[c][t]      (t = ky·9 + kx·3 + ic, OHWI layout)
                           ch 0        ch 1        ch 2              ch 25
                             │           │           │                 │
                             ▼           ▼           ▼                 ▼   ◄─ weight broadcast DOWN column
 image replica 0  ──s[0]──►[MAC]──────[MAC]──────[MAC]─── ... ─────[MAC]
  (BRAM, 4 banks)            │           │           │                 │
 image replica 1  ──s[1]──►[MAC]──────[MAC]──────[MAC]─── ... ─────[MAC]
                             │           │           │                 │
 image replica 2  ──s[2]──►[MAC]──────[MAC]──────[MAC]─── ... ─────[MAC]
       ⋮                     ⋮           ⋮           ⋮                 ⋮
 image replica 31 ─s[31]──►[MAC]──────[MAC]──────[MAC]─── ... ─────[MAC]
                    ▲
                    └─ sample broadcast ACROSS row
   s[p] = X[oy+ky−1][p+kx−1][ic]   (0 if padding)

   Every line is a broadcast wire. There are no neighbour-to-neighbour pipeline registers.
```

Each of the 32 image replicas holds its own copy of the full 3,072-byte image.
This gives 32 independent read addresses per clock, one per pixel lane.

### 1.3 Inside one MAC (where intermediate values live)

```text
  s[p] (INT8) ──┐
                ├─► DSP48 multiply ─► [INT16 product reg] ─► fabric adder ─► [INT32 accumulator]
  W[c][t] (INT8)┘                                               ▲                    │
                                                                └────────────────────┘
                                         INIT cycle: accumulator ← bias[c]
```

- **Intermediate partial sums:** the 832 INT32 LUT/FF accumulators, one per
  MAC. Nothing is spilled to RAM.
- Multiply uses 1 DSP. Accumulate uses fabric. Result: 832 DSPs, about 95.7k
  LUTs and 97.7% of CLBs.

### 1.4 Per-row schedule (38 cycles/row × 32 rows ≈ 1,239 cycles ≈ 12.4 µs @ 100 MHz)

```text
 cycle:  0     1 ............. 27   28  29   30 ............... 37
       [INIT][ISSUE t=0 ... t=26 ][DRAIN ][EMIT beat0 ... beat7  ]   → next row oy+1
        acc←bias  27 taps, all 832      pipe   8 × 104 sums
                  MACs fire per clk     flush  (4 pixels × 26 ch each)
```

### 1.5 Output path: where results are stored

```text
 832 accumulators ──mux by beat──► 104 × INT32 ──► 104 requant lanes ──► 104 B ──► 2 KB result FIFO
 (accum[p·26+c])    beat b = pixels             (×M, round, >>R,                       │
                    4b..4b+3, all 26 ch          ReLU, sat INT8)                        ▼
                                                                         256-bit AXIS (32 B/beat)
                                                                                        │
                                                                AXI DMA S2MM ──► DDR, HWC layout
                                                                   26,624 B = 832 beats, TLAST on last
```

### 1.6 Full system and data placement

```text
 ┌──────── PS ────────┐                ┌──────────────────────── PL ─────────────────────────────┐
 │ A53 CPU ──AXI-Lite─┼──────────────► │ Control regs (LOAD_KIND, START, STATUS, cycle counters) │
 │                    │                │                                                         │
 │ DDR: image, W, b,  │  AXI DMA MM2S  │ 4-beat RX FIFO ─► 16 B/clk loader ─┬► 32 image replicas │
 │      quant ────────┼──────────────► │                                    ├► 26 weight banks   │
 │                    │  256-bit AXIS  │                                    ├► 26 bias regs      │
 │                    │                │                                    └► quant check only  │
 │ DDR: output ◄──────┼─────────────── │ 832-MAC grid ─► 104 requant ─► 2 KB FIFO                │
 └────────────────────┘  AXI DMA S2MM  └─────────────────────────────────────────────────────────┘
```

| Data | Lives in (FPGA) | How it's used |
|---|---|---|
| Image (3 KB) | 32 BRAM replicas × 4 banks of 192×32 b (~96 KB logical) | One sample per pixel lane per clk |
| Weights (702 B) | 26 register banks × 27 B | One weight per channel per clk, broadcast to 32 MACs |
| Bias (104 B) | 26 INT32 regs | Accumulator initial value |
| Quant M/R | Compile-time constants (`conv0_quant_pkg.sv`) | Runtime packet is only checked against them |
| Partial sums | 832 INT32 accumulators in the MACs | Never leave the array until finished |
| Output | 2 KB FIFO, then DDR via DMA | Streamed back to the CPU |

**Strengths and limits:** this is the simplest control scheme and is already
verified in simulation and timing-closed. Its limits are the very high fabric
use (~98% of CLBs), the 32× replicated image, the 100 MHz clock, and the fact
that it handles Conv0 only. Global broadcast wires do not scale well.

---

## 2. Architecture 2: output-stationary systolic array, 26 × 32 (proposal)

### 2.1 Idea

This design uses the same tile as Arch 1: one output row, 26 channels × 32
pixels, and 27 taps over time. The difference is that it is a **true systolic
array**. Weights enter from the left edge and move one PE to the right each
clock. Activations enter from the top edge and move one PE down each clock.
The inputs are skewed so that `W[oc][k]` and `X̂[k][p]` meet in PE(oc, p) at
cycle `k + oc + p`. Each PE keeps its own output sum, which is why this is
called output-stationary. The design is meant to run the whole network at
200 MHz. Conv0 is its first layer, at about 954 cycles ≈ 4.8 µs.

The reduction order is `k = (ky, ic, kx)`. This order lets the feeder read one
input-row word and reuse it for all three `kx` taps.

### 2.2 Array structure

```text
                              ACTIVATIONS enter top edge (im2col feeder, 32 B/clk, column-skewed)
                  col 0          col 1           col 2                 col 31   (= output pixel ox)
               X̂[k][ox=0]   X̂[k−1][ox=1]   X̂[k−2][ox=2]   ...   X̂[k−31][ox=31]
                    │              │               │                     │
WEIGHTS             ▼              ▼               ▼                     ▼
(URAM, 26 B/clk,
 row-skewed)
row 0  W[0][k]  ─►[PE 0,0]──reg─►[PE 0,1]──reg─►[PE 0,2]── ... ──►[PE 0,31]
                    │reg           │reg            │reg                  │
row 1  W[1][k−1]─►[PE 1,0]──reg─►[PE 1,1]──reg─►[PE 1,2]── ... ──►[PE 1,31]
                    │              │               │                     │
   ⋮                ⋮              ⋮               ⋮                     ⋮
row 25 W[25][k−25]►[PE25,0]─reg─►[PE25,1]──reg─►[PE25,2]── ... ──►[PE25,31]
                    │drain         │drain          │drain                │drain
                    ▼              ▼               ▼                     ▼
                 lane 0         lane 1          lane 2               lane 31  ──► de-skew ──► output pipeline
 (row = output channel oc)                                   (each arrow labelled "reg" is a 1-clk pipeline register)
```

**Skew / wavefront.** This example shows which `k` each PE in the top-left
3×3 corner is working on at clock `t`:

```text
            col0   col1   col2
   row0     t      t−1    t−2
   row1     t−1    t−2    t−3          PE(r,c) processes k = t − r − c
   row2     t−2    t−3    t−4          → the work moves through the array as a diagonal wavefront
```

The last PE, PE(25, 31), finishes `25 + 31 = 56` cycles after PE(0, 0).

### 2.3 Inside one PE (where intermediate values live)

```text
          act_in (8b, from above) + first/last flags
              │
              ├──────────────────────► act_out (registered, to PE below)
              ▼
 w_in ──►[reg]─────────────────────► w_out (to PE right)
              │
      ┌───────▼──────── DSP48E2 ───────────────┐
      │ P = first ? A·B : P + A·B              │  ◄── accumulator #1 = DSP P register (25-bit)
      └───────────────┬────────────────────────┘
                      │ on 'last'
 drain_in ──► [drain reg, 25b] ──► drain_out (to PE below)  ◄── accumulator #2 = drain register
```

- **Intermediate sums** stay in the DSP **P register**. No partial-sum RAM is
  needed, because the number of cycles per tile is simply `K`.
- On the last `k`, the finished sum is copied into the **drain register**. The
  DSP starts the next row's tile on the very next clock, so the array does not
  stall.
- The `first`/`last` flags travel down the array with the activations. No
  global enable signal is broadcast.
- Bias is **not** added in the array. The output pipeline adds it.

### 2.4 How the activations are fed (im2col feeder)

The IN buffer for Conv0 is channel-planar. Each 256-bit word holds one image
row of one channel:

```text
IN buffer (96 words × 256 b):   word(ic, y) = X[y][0..31][ic]     (3 channels × 32 rows)

For output row oy, reduction step (ky, ic):
    read ONE word  word(ic, oy+ky−1)   (all zero if the row is padding)
    hold it in the window register, then serve 3 consecutive k's (kx = 0,1,2):

      kx=0:  col p gets X[oy+ky−1][p−1][ic]     ← shifted right, col 0 = pad 0
      kx=1:  col p gets X[oy+ky−1][p  ][ic]
      kx=2:  col p gets X[oy+ky−1][p+1][ic]     ← shifted left, col 31 = pad 0

    → 1 BRAM read every 3 clocks; 32 bytes/clk delivered to the top edge (then column skew)
```

Weights come from URAM, one 26-byte word per `k`. Row `r` receives byte `r`
delayed by `r` clocks. This delay is a skew triangle of 325 byte registers.

### 2.5 Output path: where results are stored

```text
 drain regs shift DOWN each column (oc 25 → 0), 1 value/clk/column, ≤ 26 clk (hidden under next tile's K)
         │ × 32 lanes
         ▼
 de-skew triangle (columns finish 1 clk apart)
         ▼
 32 output lanes:  acc + bias[oc] → × M[oc] → round-shift R[oc] → ReLU → sat INT8
         │           ({bias, M, R} fetched from Param RAM per drain cycle; oc changes every clk)
         ▼
 one 256-bit word = (one oc, 32 pixels of row oy) ──► activation buffer A (channel-planar)
                                                       → feeds block1_conv1 directly, no DDR round trip
```

Conv0 timeline: 32 tiles × 27 = 864 issue cycles. Fill adds 56 cycles, the
final drain adds 26 cycles, and RAM/requant latency adds about 8 cycles. Total
**≈ 954 cycles ≈ 4.8 µs @ 200 MHz**.

| Data | Lives in | Notes |
|---|---|---|
| Image | IN buffer, 2 × 3 KB (double-buffered), channel-planar | 1 read / 3 clk for Conv0 |
| Weights | URAM (~202 KB for whole net; Conv0 = 27 words × 26 B) | 1 word/clk, row skew |
| Partial sums | DSP P registers (832) | Stationary |
| Finished sums | Drain registers (832 × 25 b) | Shifted out during next tile |
| Bias, M, R | Param RAM | Applied in the 32-lane output pipeline |
| Output | Activation buffer A (BRAM, 832 × 256 b) | Kept on-chip for the next layer |

---

## 3. Architecture 3: weight-stationary "TPU-style" array, 27 × 26 (proposal)

### 3.1 Idea

This design swaps which operand stays in place. Each of the 702 PEs holds
**one weight** for an entire pass. **Rows are reduction taps `k`** (27 rows,
which is exactly Conv0's 3×3×3). **Columns are output channels** (26). Pixels
stream in from the left edge, one new pixel per clock. Partial sums flow
**down** each column through the DSP PCIN/PCOUT cascade. The bottom of column
`c` emits the finished dot product for `(p, oc=c)`.

Conv0 fits the array exactly. All 702 weights are loaded **once**, and then all
1,024 pixels stream through. For Conv0 the partial-sum RAM is not used. That
takes about 1,085 cycles ≈ 5.4 µs @ 200 MHz.

### 3.2 Array structure

```text
                     WEIGHTS: loaded once (27 clk) down the BCIN cascade into B1, then swapped into B2
                     ┌── col 0 ──┐   ┌── col 1 ──┐               ┌── col 25 ──┐   (= output channel oc)
                          │               │                            │
ACTIVATIONS (left edge, row-skewed by r)  ▼                            ▼
row 0  X̂[k=0 ][p]     ─►[W[0][0] ]──reg─►[W[1][0] ]── ... ────────►[W[25][0] ]
                          │ psum          │ psum                       │ psum
row 1  X̂[k=1 ][p−1]   ─►[W[0][1] ]──reg─►[W[1][1] ]── ... ────────►[W[25][1] ]
                          │               │                            │
row 2  X̂[k=2 ][p−2]   ─►[W[0][2] ]──reg─►[W[1][2] ]── ... ────────►[W[25][2] ]
   ⋮                      ⋮               ⋮                            ⋮
row 26 X̂[k=26][p−26]  ─►[W[0][26]]──reg─►[W[1][26]]── ... ────────►[W[25][26]]
                          │               │                            │
                          ▼               ▼                            ▼
                    Y[p][oc 0]      Y[p][oc 1]         ...       Y[p][oc 25]   (column-skewed by c)

  PE(r,c) holds W[oc=c][k=r].   Activations move RIGHT (1 PE/clk, registered).
  Partial sums move DOWN through PCIN→PCOUT: psum_out = psum_in + X̂·W.
```

The activations move to the right along a row. Each pixel's 27-byte window
therefore reaches all 26 channels, and each column applies a different filter
to it.

### 3.3 Inside one PE (where intermediate values live)

```text
            next weight (from PE above via BCIN, during load)
                 │
                 ▼
           ┌─────────── DSP48E2 ─────────────┐
 act_in ──►│ A  = activation (8b)            ├──► act_out (registered, to PE right)
           │ B1 = next weight   (loading)    │
           │ B2 = active weight (stationary) │◄── 'swap' flag travels with activations
 psum_in ─►│ P  = PCIN + A·B2                ├──► psum_out (PCOUT, to PE below)
 (PCIN)    └─────────────────────────────────┘
```

- **Weights** live inside the DSP B2 register. B1 holds the next pass's
  weights, so a new set loads while the current set is in use.
- **Intermediate partial sums** are never stored in a PE. They move through
  the dedicated DSP cascade, one row per clock. In layers deeper than Conv0
  (`K > 27`), they are accumulated across passes in a **partial-sum RAM**
  (26 banks × 1,024 × 32 b). Conv0 skips this RAM.

### 3.4 How the activations are fed (Conv0 window generator)

Each clock, the array needs the full 27-byte window `X̂[0..26][p]` for one
pixel. A line-buffer window generator supplies it:

```text
 IN buffer (raster, 3 B/pixel)
        │ 1 pixel/clk
        ▼
 ┌─────────────────┐   ┌─────────────────┐
 │ line buffer y−1 │──►│ line buffer y   │──►  (32 px × 3 B each)
 └────────┬────────┘   └────────┬────────┘
          │                     │                   incoming row y+1
          ▼                     ▼                         ▼
     ┌─────────────────────────────────────────────────────────┐
     │  3×3 register window of 3-byte pixels (27 bytes)        │  slides 1 pixel/clk, raster order,
     │   [y−1,x−1][y−1,x][y−1,x+1]                             │  zero-substitutes padding at edges
     │   [y  ,x−1][y  ,x][y  ,x+1]                             │
     │   [y+1,x−1][y+1,x][y+1,x+1]                             │
     └──────────────────────┬──────────────────────────────────┘
                            │ 27 bytes = X̂[0..26][p]
                            ▼
              row skew triangle (row r delayed r clk) ──► left edge of array
```

An alternative is for the CPU to pre-expand the image to 1,024 × 27 B. That
means a 27.6 KB DMA instead of 3 KB.

### 3.5 Output path: where results are stored

```text
 bottom of 26 columns: one complete 27-term sum per column per clk (column c lags by c clk)
         │   (for deeper layers: → partial-sum RAM read-modify-write until the last pass)
         ▼
 26 output lanes: psum + bias[c] → × M[c] → round-shift R[c] → ReLU → sat INT8
         │          ({M, R} constant per lane for the whole tile; no per-clk param fetch)
         ▼
 byte de-skew triangle
         ▼
 one 26-channel word per pixel (NHWC, padded to 32 B) ──► activation buffer A
 1 pixel/clk × 1,024 pixels                                (matches the exported HWC layout)
```

Conv0 timeline: weight preload takes 27 cycles and can be prefetched. Streaming
1,024 pixels takes 1,024 cycles. Skew adds 51 cycles and latency adds about 10.
Total **≈ 1,085–1,112 cycles ≈ 5.4–5.6 µs @ 200 MHz**.

| Data | Lives in | Notes |
|---|---|---|
| Image | IN buffer, 2 × 3 KB, raster | Read 1 pixel/clk into the window generator |
| Weights | URAM (~210 KB whole net), then DSP B1/B2 | Conv0: loaded once, 702 values stationary |
| Partial sums | DSP PCIN/PCOUT cascade (in flight) | Conv0: complete at the bottom of each column |
| Multi-pass partial sums | Partial-sum RAM, 26 BRAM36 | Unused in Conv0; needed for K > 27 |
| Bias, M, R | Param RAM | Loaded per lane once per oc tile |
| Output | Activation buffer A, NHWC 26-ch words | On-chip for the next layer |

---

## 4. Side-by-side comparison

```text
            ARCH 1 (broadcast)          ARCH 2 (output-stationary)     ARCH 3 (weight-stationary)
          W  W  W  W                    W→ ┌──┬──┬──┐                 X̂→ ┌──┬──┬──┐
          ↓  ↓  ↓  ↓                    W→ ├──┼──┼──┤  sums stay     X̂→ ├──┼──┼──┤  weights stay
   X ─────●──●──●──●                    W→ └──┴──┴──┘  in PEs        X̂→ └──┴──┴──┘  in PEs
   X ─────●──●──●──●                        ↑  ↑  ↑                      ↓  ↓  ↓
   X ─────●──●──●──●                        X̂  X̂  X̂ (from top)          psums exit bottom
   wires fan out, no hops            operands hop 1 PE/clk           X̂ hops right, psum hops down
```

| | Arch 1: Broadcast 832 | Arch 2: OS systolic 26×32 | Arch 3: WS systolic 27×26 |
|---|---|---|---|
| Physical MACs/PEs | 832 | 832 | 702 |
| Rows × cols | 32 pixels × 26 oc | 26 oc × 32 pixels | 27 k × 26 oc |
| Stationary in PE | Accumulator (fabric INT32) | Accumulator (DSP P, 25 b) | Weight (DSP B2) |
| Operand movement | Global broadcast | X̂ down, W right | X̂ right, psum down (PCIN cascade) |
| Conv0 loop in time | 32 rows × 27 taps | 32 rows × 27 taps | 1,024 pixels |
| Image feed | 32 BRAM replicas, 1 B/lane/clk | Channel-planar row word, 1 read / 3 clk | Line-buffer 3×3×3 window, 27 B/clk |
| Weight feed | 26 reg banks, 1 B/ch/clk | URAM 26 B/clk + skew | URAM, loaded once (27 clk) |
| Where bias is added | Accumulator init | Output pipeline | Output pipeline (or psum init) |
| Output lanes | 104 (4 px × 26 ch) | 32 (1 oc × 32 px) | 26 (1 px × 26 oc) |
| Output destination | 2 KB FIFO → DMA → DDR | On-chip buffer A (channel-planar) | On-chip buffer A (NHWC) |
| Conv0 latency | ~1,239 clk ≈ 12.4 µs @ 100 MHz (sim) | ~954 clk ≈ 4.8 µs @ 200 MHz (est.) | ~1,090 clk ≈ 5.5 µs @ 200 MHz (est.) |
| Scope | Conv0 only | Whole ResNet-8 (~201 µs est.) | Whole ResNet-8 (~246 µs est.) |
| DSP use | 832 (measured) | ~960–1,020 (est.) | ~800–860 (est.) |
| LUT use | 95.7k / 98% CLB (measured) | ~35–50k (est.) | ~30–45k (est.) |
| Extra memory | 32× image replication | None for psums | 26 BRAM36 partial-sum RAM |
| Main risk | Fabric congestion, cannot grow | 200 MHz closure, DSP budget 77–82% | URAM 98.5% full, psum RAM traffic, dense layer is slow |

**In short:**
- **Arch 1** is the working Conv0 baseline. It is simple and verified, but
  global broadcast and data replication use up most of the fabric.
- **Arch 2** keeps Arch 1's tile shape but makes it systolic. It has the
  highest PE use across *all* ResNet-8 layers because `K` simply becomes
  cycles per tile.
- **Arch 3** matches Conv0 exactly (27 × 26) and fits the DSP cascades
  naturally. It needs pass tiling and a partial-sum RAM for deeper layers, and
  it is about 20% slower than Arch 2 for the full network.
