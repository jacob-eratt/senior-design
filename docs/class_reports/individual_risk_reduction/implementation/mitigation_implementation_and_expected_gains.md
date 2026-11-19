# Weight Storage Mitigations: Implementation Details and Expected Gains

Risk analysis: [weight_storage_risks.md](../weight_storage_risks.md)
Architecture reference: [Conv0Spec_ChipAgents.md](../../../../resnet8_project/arch_specs/Conv0Spec_ChipAgents.md), [uarch 04_on_chip_memory.md](../../../../resnet8_project/brainstorming/uarch_study/04_on_chip_memory.md)
RTL reference: `resnet8_project/design/rtl/conv0/` (`ws_pkg`, `ws_skew`, `ws_pe`, `ws_column`, `ws_array`, `ws_mxu`)

| Mitigation | Scenario | RTL | Status |
|---|---|---|---|
| M1: load-time weight checksum | All weights resident on FPGA | [ws_wload_checksum.sv](ws_wload_checksum.sv) | Lint clean, not simulated |
| M2: ping-pong weight buffer | Weights streamed from CPU per layer | [ws_wbuf_pingpong.sv](ws_wbuf_pingpong.sv) | Lint clean, not simulated |

Neither module is yet connected to the loader, `c0_axil_regs`, or `c0_weight_store`. Those modules are not written.

Scope note: all cycle counts below are estimates from the spec's own numbers. They are not measured. The validation plan in section 5 defines how to measure them.

---

## 1. M1: load-time weight checksum

### 1.1 Problem addressed

With all weights resident, the only copy of the weights lives in the on-chip Weight RAM after boot. The current design (Conv0 spec §8, REQ-LD-01..03) checks the packet's length and TKEEP but not its contents. A corrupted byte that keeps the length and TKEEP valid is accepted silently and then used in every frame.

M1 detects that corruption at load time, before START is allowed.

### 1.2 Algorithm

Packet bytes `d_0..d_{N-1}` (N = 702 for the weight packet). The checksum is a Fletcher-style pair of 32-bit values, both taken mod 2³²:

```text
s1 = sum(d_i)              i = 0..N-1
s2 = sum((N - i) * d_i)
checksum = {s2, s1}        64 bits
```

Equivalent per-byte recurrence: `s1 += d; s2 += s1`. The recurrence is serial, so the RTL uses an equivalent beat-level form.

For a beat with `n` valid bytes (from TKEEP), with `S0 = sum(d_b)` and `S1 = sum(b · d_b)` over the lanes `b = 0..31`:

```text
s1 <= s1 + S0
s2 <= s2 + n · (s1 + S0) - S1
```

I checked this beat-level form against the per-byte definition in Python for 200 random packets, including full, partial and 702-byte packets. All matched. This checks the math, not the RTL.

### 1.3 Microarchitecture

```text
S_AXIS beat (256 b data, 32 b keep)  ──► per-lane mask (keep ? d : 0)
                                        │                       │
                              lane_d[b] = d_b           lane_w[b] = b · d_b   (constant multiply)
                                        │                       │
                              binary adder tree (5 levels)  binary adder tree (5 levels)
                                        │ S0                     │ S1
                                        ▼                        ▼
   n = popcount(keep) ──► base_s1 = clear ? 0 : s1 ──► beat_s1 = base_s1 + S0
                          base_s2 = clear ? 0 : s2 ──► s2_next = base_s2 + n · beat_s1 − S1
                                                              │
                             registers s1, s2 (32 b each) ◄───┘
                             done  <= beat_valid & beat_last   (1-cycle pulse)
                             match  = ({s2, s1} == expected)   (valid while done)
```

Key points:
- **No serial chain.** Each beat is reduced in a tree of depth 5 (32 lanes).
- **Clear merged with the first beat.** The first beat of a packet arrives in the same cycle as the pkt_start event, so `clear` selects zero as the base value instead of resetting first. Without this, the first beat would be lost.
- **`done` is registered.** `s1` and `s2` are stable on the cycle `done` is high, because no new beat can be accepted before the next packet starts.

### 1.4 Interface

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `clk`, `rst` | in | 1 | Synchronous, active-high reset (REQ-IMP-04) |
| `clear` | in | 1 | Packet start; zero the state |
| `beat_valid`, `beat_last` | in | 1 | Accepted beat and its TLAST |
| `beat_data` | in | 256 | Beat data, byte 0 in bits 7:0 |
| `beat_keep` | in | 32 | TKEEP; contiguous from bit 0 |
| `expected` | in | 64 | Reference checksum written by the host |
| `done` | out | 1 | Pulses the cycle after the last beat |
| `match` | out | 1 | `{s2, s1} == expected`; valid while `done` |

### 1.5 Loader integration (planned, not written)

```text
loaded_flag[kind=1] <= done && match && !err_keep && !err_len     (on done, weight packet only)
ERRORS.bit3 <= done && !match                                       (new error bit)
```

The new `ERRORS.bit3` is a proposal. The spec currently defines only bits 0..2 (§10.2). Adding it requires updating REQ-LD-*, the register map, and the host sequence in §10.3.

`expected` is a new AXI-Lite register (proposed offset 0x40, written by the host before the weight packet). The host computes the same checksum over the packet in DDR. The spec currently defines registers only up to 0x3C.

### 1.6 Resource and timing estimate

| Item | Estimate | Basis |
|---|---|---|
| LUT | ~1,000 (~2.5% of the 40k accelerator LUT budget) | About 62 adders of 9–32 bits, plus a 6×32 multiply for `n · beat_s1` |
| FF | ~70 (two 32-bit state registers, done, control) | Direct count |
| DSP | 0 expected | Constant multiplies by `b` map to shifts and adds; the variable `n · beat_s1` may map to a DSP if synthesis chooses |
| Latency | +1 cycle after the last beat | `done` register |
| Frame latency (RUN_CYCLES) | 0 change | Runs only at load time |

Timing risk: the unpipelined path is lane mux, 5 adder levels, base mux, a 6×32 multiply, and a subtract. At 5 ns (200 MHz) this path is likely to fail timing. Expected fix: pipeline the tree in two stages (levels 1–3 and 4–5), and move `n · beat_s1` into its own stage. The added latency is 2–3 cycles at load time only. Until synthesis is run, this is an expectation, not a measurement.

### 1.7 Detection coverage

| Error type | Detected? | Notes |
|---|---|---|
| Single-byte error | Yes | Changes `s1` by a nonzero amount (|δ| ≤ 255, nonzero mod 2³²) |
| Two bytes swapped | Yes, unless the two bytes are equal | Changes `s2` |
| Random multi-byte corruption | Usually | A Fletcher-style sum is weaker than a CRC for some patterns |
| Corruption after load (RAM upset) | No | Needs the periodic CRC scrub (not implemented) |

A CRC-32 would give stronger guarantees and is the recommended follow-up if the reviewer asks for a standard algorithm. The checksum is a cheap first mitigation.

### 1.8 Expected performance improvement

M1 does not improve speed. Its value is avoiding a silent failure:

| Metric | Without M1 | With M1 | Change |
|---|---|---|---|
| Probability a corrupted packet is accepted | Not detected at all | Single-byte errors always caught; multi-byte errors caught with high probability | Qualitative reduction in silent wrong output |
| RUN_CYCLES per frame (1,117 est.) | 1,117 | 1,117 | 0 |
| Cold-start load time | 129 beats | 129 beats + 1 to 3 cycles | Negligible |
| LUT | Baseline | +~1,000 | ~2.5% |

---

## 2. M2: ping-pong weight buffer

### 2.1 Problem addressed

In the streamed scenario, each layer's weights arrive from the CPU through DMA. Without buffering, the array waits for the transfer before it starts each layer (the weight-load stall). The Conv0 spec already hides the 27-cycle WLOAD inside a fixed budget, but per-layer DMA transfers are much longer.

M2 lets the next layer's weights stream into a standby bank while the current layer computes from the active bank.

### 2.2 Microarchitecture

```text
            wr_en, wr_last, wr_row, wr_data ──► [ bank 0 ]  ──┐
                            (writes go to standby = ~active)   ├─► rd_data (active bank, combinational)
                                               [ bank 1 ]  ──┘
                                                                   ▲ rd_row
  staged  : set when a standby bank is complete (wr_last); blocks writes
  swap    : taken only when staged; flips active; clears staged
  wr_ready = ~staged
```

Bank selection:
- `active` selects which bank the array reads.
- `bank_mem[~active]` receives writes.
- Both banks are 27 × 26 bytes, the same layout as `c0_weight_store` (§8.1).

Control rules:
- Writes are ignored while `staged = 1`. A completed bank cannot be overwritten.
- `swap` has no effect unless `staged = 1`. A swap with no staged data is ignored.
- Write and swap cannot occur in the same cycle. `staged` must be 1 for swap to act, and writes require `staged = 0`.
- Memory is not reset. Only `active` and `staged` are reset (REQ-IMP-04).

### 2.3 Interface

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `clk`, `rst` | in | 1 | Synchronous reset |
| `wr_en` | in | 1 | Write one row to the standby bank |
| `wr_last` | in | 1 | Final row of this bank (sets `staged`) |
| `wr_row` | in | 5 | Row index 0..26 (assumed in range) |
| `wr_data` | in | 26 × 8 | One row: 26 weight bytes (`W[c][r]` for c = 0..25) |
| `swap` | in | 1 | Flip banks at a layer boundary |
| `rd_row` | in | 5 | Row read by the controller's WLOAD |
| `wr_ready` | out | 1 | Standby bank can accept writes |
| `staged` | out | 1 | Standby bank complete, ready to swap |
| `rd_data` | out | 26 × 8 | Active bank row, combinational |

The read interface matches `c0_weight_store` (§14.3). The controller can use it unchanged; only the source of the rows changes.

### 2.4 Resource estimate

| Item | Estimate | Basis |
|---|---|---|
| Storage | 2 × 702 B = 1,404 B (vs. 702 B) | Doubles the Conv0 store |
| Conv0 LUTRAM | ~200 LUT (estimate) | 1,404 B ≈ 11,232 bits, one LUT6 holds 64 bits |
| Full-network Weight RAM | 202 KB → 404 KB | Would move from 6 URAM to about 12 URAM (K26 has 64) |
| Mux | 26 × 8-bit 2:1 read mux plus write decode | Small |

For a full ResNet-8 streamed design, the ping-pong buffer only needs to hold the largest layer's weights, not all layers. The 404 KB figure assumes both banks sized for the whole network. A realistic sizing is one bank sized for the largest layer.

### 2.5 Expected performance improvement

The weight-load stall per layer transition depends on the transfer time and on the compute time of the preceding layer.

```text
T_w(L)     = weight_bytes(L) / BW_eff  +  DMA_setup
stall_noPP = T_w(L)
stall_PP   = max(0, T_w(L) - C(L-1))        (transfer overlaps the previous layer's compute)
```

Assumptions (all estimates):
- `BW_eff` = 16 B/cycle at 200 MHz. The ideal AXI-Stream rate is 32 B/cycle (256 b). The 50% derating covers DMA setup, DDR contention and CPU descriptor handling.
- DMA setup is ignored in the table below. It only makes the no-buffer case worse.
- `C(L)` = compute cycles for layer L at one pixel per cycle per output-channel tile, with 26 output channels per tile. This uses `pixels × ceil(OC / 26)`.

Illustrative transitions (values are examples to show scale, not measured layers):

| Transition into layer | Weight bytes | T_w (cycles) | Preceding layer compute C(L−1) | Stall, no buffer | Stall, ping-pong |
|---|---:|---:|---:|---:|---:|
| Conv0-style 3→26 @32×32 (Conv0 frame reload) | 702 | 44 | 1,024 (Conv0 frame) | 44 | 0 |
| 32→32 3×3 @16×16 | 9,216 | 576 | 1,024 (Conv0-style layer at 32×32) | 576 | 0 |
| 64→64 3×3 @8×8 | 36,864 | 2,304 | 512 (32→32 @16×16, 2 tiles) | 2,304 | 1,792 |

Reading the table:
- For the 702 B Conv0 reload, the buffer hides the entire transfer. The DMA transfer for frame n+1 runs during frame n's 1,117-cycle compute.
- For mid-size layers, the buffer hides the transfer completely.
- For the largest layer, the buffer still leaves a stall of about 1,792 cycles. Compute for an 8×8 layer is too short to hide a 37 KB transfer. The fix is to keep this layer's weights resident (the hybrid option from the risk analysis), not to rely on ping-pong alone.

Summary of expected gain:
- Conv0-only streamed design: 44 cycles of weight stall per frame removed (about 4% of 1,117), with RUN_CYCLES reduced by the same amount if the reload was on the critical path.
- Multi-layer streamed design: most of the per-layer stall is removed, except for layers whose transfer is longer than the previous layer's compute.

### 2.6 Timing and integration notes

- The read mux is combinational from `rd_row` to `rd_data`, matching `c0_weight_store`. The WLOAD path does not get longer.
- The `swap` must be asserted at the layer boundary, before the WLOAD of the next layer reads `rd_row`. This is the controller's responsibility.
- The write side (loader or DMA) is not written yet. It must respect `wr_ready`, which is the same back-pressure rule as REQ-LD-04.

---

## 3. Combined expected effect

| Scenario | Mitigation | Speed change | Reliability change | Area change (est.) |
|---|---|---|---|---|
| All-resident (uarch study) | M1 checksum | None | Load-time corruption detected | +~1,000 LUT |
| Streamed per layer (Conv0 spec style or per-layer DMA) | M2 ping-pong | Up to 44 cycles per frame for Conv0; most per-layer stall removed for mid-size layers | None directly | +~200 LUT (Conv0), 2× weight storage |

Neither mitigation changes the other's scenario. The two are alternatives, not a combined design, unless the hybrid option is chosen.

---

## 4. Open items

1. Connect M1 to the loader and add the `expected` register and ERRORS.bit3. Update the spec's register map and REQ-LD-* to match.
2. Pipeline M1 if synthesis shows the 200 MHz path fails.
3. Connect M2 to the loader write path and the controller's `swap` signal.
4. Decide whether the largest layer in the streamed design stays resident. The estimates suggest it should.
5. Replace the Fletcher checksum with CRC-32 if the course or reviewers need a standard algorithm.

---

## 5. Validation plan

No simulation has run yet. The following tests are proposed in the style of the Conv0 spec's §13.

| ID | Module | Stimulus | Pass criterion |
|---|---|---|---|
| T-CS-01 | `ws_wload_checksum` | Random 702-byte packets, TKEEP pattern of the weight kind (30-byte final beat), 1,000 seeds | `match` equals the Python reference for every seed |
| T-CS-02 | `ws_wload_checksum` | Single-byte flip at each of the 702 positions | `match = 0` for every flip |
| T-CS-03 | `ws_wload_checksum` | Two bytes swapped (unequal values) at random positions, 1,000 trials | `match = 0` for every trial |
| T-CS-04 | `ws_wload_checksum` | `clear` in the same cycle as the first beat | First beat included in checksum (match with reference) |
| T-CS-05 | `ws_wload_checksum` | Beat gaps (`beat_valid` low for random cycles) | Checksum unchanged by gaps |
| T-PP-01 | `ws_wbuf_pingpong` | Write bank A, assert `staged`, read active bank | Reads return bank-A data only after `swap` |
| T-PP-02 | `ws_wbuf_pingpong` | Attempt writes while `staged = 1` | Writes ignored; staged bank unchanged |
| T-PP-03 | `ws_wbuf_pingpong` | `swap` with `staged = 0` | No change to `active` |
| T-PP-04 | `ws_wbuf_pingpong` | 100 back-to-back layers with random contents | Each layer's reads match what was written to its bank |
| T-PERF-01 | System | Cold start with M1 enabled | Load time = 129 beats + checksum latency; RUN_CYCLES unchanged |
| T-PERF-02 | System | Streamed design with M2 vs. without, per-layer weight transfer | Measured per-transition stall matches the table in section 2.5 within 10% |

Measurement method for the performance claims: the spec's RUN_CYCLES register (0x14) and the proposed per-layer stall counter. The table in section 2.5 is an estimate until T-PERF-02 runs.

---

## 6. Validation status

| Check | Result |
|---|---|
| Lint (slang, elaborate, warnings on), with `ws_pkg.sv` | Pass, no diagnostics |
| Python check of checksum beat formula vs per-byte definition | Pass, 200 random packets |
| Simulation | Not run (no simulator installed in the environment) |
| Synthesis and timing | Not run |
