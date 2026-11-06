# Step 03b – Current design vs OS 26×32 vs WS 27×26

Labels: **[M]** measured (simulation/routing report), **[D]** derived from
arithmetic, **[A]** assumed ballpark, unmeasured. Nothing here is board-measured.

| | Current | OS systolic 26×32 | WS ("TPU") 27×26 |
|---|---|---|---|
| Dataflow | Broadcast, output-stationary | Systolic, output-stationary, 2 accumulators/PE | Systolic, weight-stationary, 2 weight regs/PE |
| MACs | 832 | 832 | 702 |
| Clock | 100 MHz (met, +0.365 ns) [M] | 200 MHz target [A] | 200 MHz target [A] |
| Layers supported | Conv0 only (fixed quant constants) | All convs (descriptor driven) | All convs (descriptor driven) |
| Activations between layers | N/A (output to DDR) | On chip, A/B/C buffers | On chip, A/B/C buffers + psum RAM |

## 1. Conv0

| Metric | Current | OS 26×32 | WS 27×26 |
|---|---:|---:|---:|
| MAC-issue cycles | 864 (+352 init/drain/emit) | 864 | 1,024 |
| Compute, START → last result | 1,239 cyc = **12.4 µs** [M] | ~950 cyc = **4.8 µs** [D] | ~1,085 cyc = **5.4 µs** [D] |
| MAC utilization during conv0 | 70% | ~91% | ~94% (K=27 × OC=26 fits exactly) |
| Standalone offload: input DMA | 4,086 B cold / 3,072 warm @16 B/clk = 2.6 / 1.9 µs [M] | 3,072 B ≈ 1.0 µs [D] | 3,072 B ≈ 1.0 µs [D] |
| Standalone offload: output DMA (26,624 B @ 3.2 GB/s) | 8.3 µs, overlapped [D] | 8.3 µs, now the bottleneck [D] | 8.3 µs, now the bottleneck [D] |
| **Standalone Conv0 hardware latency** | **~15 µs** | **~9.5–13 µs** | **~9.5–13 µs** |
| Conv0 inside full network (output stays on chip) | N/A | **4.8 µs** | **5.4 µs** |

Range for the new designs: 3.2 GB/s nominal vs ~70–80% achievable DMA
efficiency [A]. CPU/driver overhead is extra for all three: ~1–5 µs per DMA
transfer bare-metal, ~20–100 µs under Linux/PYNQ [A]. The Python model's
end-to-end estimate for the current design is 19.47 µs (model, not measured).

Takeaway: as a standalone offload, Conv0 is **output-DMA bound** for any
design that computes it faster than ~8.3 µs. Faster arrays only show their
value when Conv0's output never leaves the chip.

## 2. Full ResNet-8 inference

Ideal cycles from [03](03_systolic_array_mapping.md) (OS: tiles × K;
WS: tiles × P with weight loads hidden) [D]:

| Layer | OS 26×32 cycles | WS 27×26 cycles | Note |
|---|---:|---:|---|
| conv0 | 864 | 1,024 | WS: 1 tile, exact fit |
| block1_conv1 | 7,488 | 9,216 | WS: 234/243 rows used |
| block1_conv2 | 7,488 | 9,216 | |
| block2_conv1 | 3,744 | 4,608 | |
| block2_conv2 | 7,488 | 9,216 | |
| block2_shortcut | 416 | 512 | |
| block3_conv1 | 3,744 | 4,608 | |
| block3_conv2 | 7,488 | 8,960 | |
| block3_shortcut | 416 | 512 | |
| **Total ideal** | **39,136** | **47,872** | |
| PE utilization | 100% | 97% | WS loses ~4% to K-tiles of 27 |
| Per-layer overhead (fill, skew, drain, descriptor), ~9 layers | ~900 | ~800 | [A] |
| Pool (fused) + dense (104 cycles) | ~150 | ~150 | [A] |
| **Total cycles** | **~40,200** | **~48,800** | |
| **Latency @ 200 MHz** | **~201 µs** | **~244 µs** | |
| Latency @ 100 MHz (timing fallback) | ~402 µs | ~488 µs | |
| Throughput, one image at a time | ~5,000 img/s | ~4,100 img/s | HW only |

Note: [arch3_ws_tpu_27x26](../arch3_ws_tpu_27x26/ARCHITECTURE.md) adopts
tap-aligned passes (one tap per pass, keeps the feeder to ≤ 2 words/cycle),
which costs +256 cycles on block3_conv2: 48,128 ideal, ~49,100 total, ~246 µs.

Current design for the full network: Conv0 on FPGA (~15 µs), 26,624 B back to
DDR, then the remaining **31.84M MACs (97.8%) on the A53 CPU**. At an assumed
0.5–2 GMAC/s on the A53 [A], that is **~16–64 ms**. By Amdahl's law,
accelerating 2.2% of the work caps the speedup over an all-CPU run at ~1.02×.

Per-PE efficiency is nearly identical (100% vs 97%). **OS is faster here
mainly because it has 130 more PEs**, not because the dataflow is
fundamentally faster. The real differences are memory, feeding and timing (§3).

For reference: OS 26×16 at 200 MHz ≈ 78,272 + ~1,300 ≈ 398 µs with 416 PEs.

## 3. Resources (ballpark)

| Resource (K26: 1,248 DSP, 117k LUT, 234k FF, 144 BRAM36, 64 URAM) | Current [M, routed, incl. PS/DMA] | OS 26×32 [A] | WS 27×26 [A] |
|---|---|---|---|
| DSP, MAC array | 832 | 832 | 702 |
| DSP, output pipeline (runtime requant + fused add, ~4–6 per lane) | 0 (constant LUT requant) | 32 lanes → ~130–190 | 26 lanes → ~100–160 |
| **DSP total** | **832 (67%)** | **~960–1,020 (77–82%)** | **~800–860 (64–69%)** |
| LUT | 95,688 (82%), **CLB 97.7%** | ~35–50k (30–43%) | ~35–50k (30–43%) |
| FF | 56,684 (24%) | ~50–65k (21–28%): 832 shadow accumulators ≈ 27k + skew/pipeline | ~25–40k (11–17%): weights/psums ride DSP registers and PCIN cascade |
| BRAM36 | 8 + 131 BRAM18 ≈ 74 equiv. (51%), mostly 32 image replicas | ~32–40 (22–28%) | ~32–40 activations, +24 if psum in BRAM |
| URAM | 0 | 6 (weights) | 6 weights + ~12 psum (106 KB) |

Why the new designs use far fewer LUTs: the MAC array lives in DSPs, there are
no 32 image replicas, and requantization moves from 104 constant-optimized LUT
units to 26–32 DSP lanes with runtime constants.

## 4. Data movement

| Traffic | Current | OS 26×32 | WS 27×26 |
|---|---|---|---|
| One-time at boot | none beyond bitstream (quant constants compiled) | ~210 KB weights/bias/quant/descriptors ≈ 66 µs @ 3.2 GB/s | same |
| Per inference, DDR → PL | 3,072–4,086 B | 3,072 B (~1 µs) | 3,072 B (~1 µs) |
| Per inference, PL → DDR | 26,624 B (Conv0 map) | 10 B logits (AXI-Lite read) | 10 B logits |
| On-chip weight read rate | broadcast, fixed in regs | 26 B every cycle | 702 B per reload (26 B/clk while loading) |
| On-chip activation read | 32 replicas | ≤ 2 words (32 px) per cycle | 27 different k elements of 1 pixel per cycle |
| Partial sums | in DSPs | in DSPs (no psum RAM) | psum RAM, 26 × 32 b read + write every cycle |

## 5. Qualitative comparison

| Criterion | OS 26×32 | WS 27×26 |
|---|---|---|
| Fit to ResNet-8 shapes | OC multiples of 26 and P multiples of 64 → 100% | conv0 exact; others 96–99% |
| Variable K (27–936) | Free | K-tiling + psum accumulation |
| Small late layers (P = 64) | No penalty | Reload hidden only because 64 ≥ 27 |
| Feeder complexity | 32 pixels of one channel per cycle (shift/decimate) | 27 k-elements of one pixel; R=27 does not align with IC=26/52/104 words |
| Extra memories | None | ~106 KB psum RAM |
| DSP mapping at 200 MHz | Accumulate in P register; 2nd accumulator in fabric | Very natural: psum down PCIN cascade, weights in B1/B2 |
| Timing risk | Moderate (shadow-acc drain muxes) | Moderate-low in array, higher in feeder/psum RAM |

## 6. Conclusions so far

1. Either systolic design is **~65–320× faster than the current design for a
   full inference** ([A] CPU rate), because the current design accelerates only
   2.2% of the work.
2. For Conv0 alone, all three are output-DMA bound around 9–15 µs; Conv0 alone
   is the wrong benchmark for picking the architecture.
3. OS 26×32 is ~18% faster than WS 27×26, mostly from having more PEs, and avoids
   psum RAM and K-tiling. WS uses ~150 fewer DSPs and fewer FFs.
4. Both fit the device comfortably in LUT/BRAM/URAM, unlike the current design's
   97.7% CLB occupancy.
