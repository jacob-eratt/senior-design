# Step 02 – A convolution as a loop nest

Every CNN accelerator (systolic, broadcast, FINN, TPU, Eyeriss) is a choice of
**which loops of this nest run in parallel hardware and which run in time**.

## 1. Concept

```text
for oc in 0..OC-1:                 # output channel      (26 / 52 / 104)
  for oy in 0..OH-1:               # output row          (32 / 16 / 8)
    for ox in 0..OW-1:             # output column       (32 / 16 / 8)
      acc = bias[oc]
      for ky in 0..KH-1:           # kernel row          (3, or 1 for shortcut)
        for kx in 0..KW-1:         # kernel column       (3, or 1)
          for ic in 0..IC-1:       # input channel       (3 / 26 / 52 / 104)
            iy = oy*S + ky - PAD
            ix = ox*S + kx - PAD
            acc += W[oc][ky][kx][ic] * X[iy][ix][ic]   # 0 if (iy,ix) is padding
      Y[oy][ox][oc] = requant(acc)  # + residual, ReLU
```

Two useful groupings:

- **Output space** `(oc, oy, ox)` – independent results. Parallelizing these
  needs no adder trees, only more accumulators.
- **Reduction** `K = KH·KW·IC` – summed into one result. Parallelizing these
  needs adder trees (or a systolic partial-sum chain).

If `(oy, ox)` is flattened into a pixel index `p` and `(ky, kx, ic)` into `k`,
every conv layer becomes a matrix multiply `Y[oc][p] = Σ_k W[oc][k] · X̂[k][p]`
where `X̂` is the "im2col" view of the input. This is why one engine can run
every layer: only `OC`, `P`, `K` and the address generator for `X̂` change.

## 2. ResNet-8 as matrix multiplies

| Layer | OC | P = OH·OW | K = KH·KW·IC | Stride | MACs |
|---|---:|---:|---:|---:|---:|
| conv0 | 26 | 1,024 | 27 | 1 | 718,848 |
| block1_conv1 | 26 | 1,024 | 234 | 1 | 6,230,016 |
| block1_conv2 | 26 | 1,024 | 234 | 1 | 6,230,016 |
| block2_conv1 | 52 | 256 | 234 | 2 | 3,115,008 |
| block2_conv2 | 52 | 256 | 468 | 1 | 6,230,016 |
| block2_shortcut (1×1) | 52 | 256 | 26 | 2 | 346,112 |
| block3_conv1 | 104 | 64 | 468 | 2 | 3,115,008 |
| block3_conv2 | 104 | 64 | 936 | 1 | 6,230,016 |
| block3_shortcut (1×1) | 104 | 64 | 52 | 2 | 346,112 |
| logits (dense) | 10 | 1 | 104 | — | 1,040 |

MACs = OC·P·K, which reproduces the manifest totals in
[ANALYSIS.md](../arch1_broadcast_832/reports/meeting_2026_10_20/ANALYSIS.md).

Observations that drive the architecture:

- `OC` is always a multiple of **26**; `P` is always a multiple of **64**.
- `K` ranges **27 → 936** (35×). A design that unrolls K spatially is
  inefficient on some layers; a design that runs K in time handles any K.
- `P·K` shrinks while `OC·K` grows deeper in the network: early layers are
  activation-heavy, late layers are weight-heavy (block3_conv2 alone is 97 KB
  of weights).

## 3. Current design in this vocabulary

[conv0_832_core.sv](../arch1_broadcast_832/src/conv0_832_core.sv) unrolls
`oc` (26) × `ox` (32) in space, and runs `k` (27 taps) then `oy` (32 rows) in
time. Each MAC keeps its accumulator: an **output-stationary** dataflow.
`weight[c]` is broadcast to all 32 pixel lanes and `sample[p]` (read from one of
32 full image replicas) to all 26 channels: a **broadcast** array, not systolic.

## 4. Decisions

_None needed; this step is vocabulary._
