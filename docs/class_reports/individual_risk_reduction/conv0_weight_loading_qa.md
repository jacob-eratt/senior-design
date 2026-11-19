# Conv0 Weight Loading: Q&A Notes

Source: [Conv0Spec_ChipAgents.md](../resnet8_project/arch_specs/Conv0Spec_ChipAgents.md)

## 1. How the weights are loaded from the CPU into the FPGA

Weights travel CPU → DDR → AXI DMA → loader → on-chip weight store, then into the PEs on each START (§4.2, §8, §10).

1. **Host stages the packet.** The 702-byte OHWI weight packet sits in DDR4. Byte `a = oc·27 + ky·9 + kx·3 + ic` (§3.1, A5).
2. **Host selects the packet.** Write `LOAD_KIND = 1` to register `0x0C`. Accepted only when idle and the loader is empty; otherwise SLVERR (§10.2).
3. **DMA streams the packet.** AXI DMA (MM2S, simple mode, base `0xA001_0000`) sends 22 beats of 256 bits. The final beat has `TKEEP = 0x3FFF_FFFF` (30 bytes) and `TLAST`.
4. **Loader routes to `c0_weight_store`.** Checks TKEEP contiguity and length. Byte `a = 32·beat + lane` goes to `wstore[k][oc]` with `k = a mod 27`, `oc = a div 27` (REQ-WS-02). Errors set sticky `ERRORS` bits.
5. **Bias and quant follow the same path.** Kind 2 (bias, 4 beats) and kind 3 (M/R, 7 beats) go to `c0_param_regs`. Bad M or R sets ERRORS bit1.
6. **Host polls `STATUS[11:8]`** until all four loaded flags are set. START is refused until then (REQ-REG-03).
7. **Persistence.** Weights stay on chip across PREPARE (warm runs). Only SOFT_RESET clears the loaded flags (REQ-WS-04).
8. **Preload into the array on each START.** WLOAD runs 27 shifts: `wload_row = 26 − s` drives `ws_mxu.weight_load_in[0..25]` with `weight_load_enable = 1`. Row 26 goes first, row 0 last. Column skew is applied inside `ws_mxu` (§8.2, REQ-MXU-03). After the 27th shift, PE(r,c).w_next = W[c][r]. The first pixel must come at least 27 cycles after WLOAD starts (REQ-MXU-04).

## 2. How the FSM works for loading weights

There are two FSMs.

**Loader packet FSM (before START).** The spec does not name its states; these are inferred from §10.1.

| State (inferred) | Behavior |
|---|---|
| IDLE | Waits for the first beat. `TREADY = 1` unless `block` is high. `pkt_start` clears the weight loaded flag. |
| RECV | Checks TKEEP per beat (bit0 error on violation). Writes each beat to `c0_weight_store` with `wr_beat` 0..21. Counts bytes. |
| DONE / ERR | On TLAST, compares byte count to 702. Match: set loaded flag. Mismatch: set ERRORS bit2, leave flag clear. |

ERRORS are sticky until PREPARE or SOFT_RESET (REQ-LD-05). A LOAD_KIND write during a packet returns SLVERR.

**Controller FSM `c0_ctrl` (on START, §11.1).**

| State | Outputs |
|---|---|
| IDLE | `wload_en = 0`, `issue = 0`. Loader may run. |
| WLOAD | `wload_en = 1`, `wload_row = 26 − s`, s = 0..26 (5-bit counter). |
| STREAM | `issue = credit_ok`, pixel counter advances. |
| DRAIN / FLUSH / DONE | Wait on counters; `issue = 0`. |

Transitions:
- IDLE → WLOAD: START accepted. Requires not running, loader idle, flags = `4'b1111`, ERRORS = 0, output-done clear (REQ-REG-03).
- WLOAD → STREAM: after the 27th shift.
- Any state → IDLE: SOFT_RESET.

The loader FSM runs once per cold start and produces data in the store. The controller FSM runs on every START and reads the store out to the array. The only link between them is the loaded flag and ERRORS.

## 3. Preloading the following layer's weights

The Conv0 spec does not cover this. It is Conv0-only (A1). Each START reloads Conv0 weights from `c0_weight_store` (A11, §8.2).

Hooks in the spec that a later layer would use:
- **Runtime M/R and weight packets** via LOAD_KIND (A4, "required for future reuse in later layers").
- **Weights persist across PREPARE** (REQ-WS-04).
- **Mid-frame weight shift** in the array (T-MXU-03): a second weight set is shifted in while frame A runs. This is a unit-level test only.
- **Ping-pong option** `IMG_BUFS = 2` (A10, R-07), defined for the image store, not weights.

Open item: a concrete next-layer weight plan is not in this spec. Candidate sources are `resnet8_project/brainstorming/uarch_study/04_on_chip_memory.md` and related files.

### 3.1 Findings from `uarch_study/04_on_chip_memory.md`

Source: [04_on_chip_memory.md](../resnet8_project/brainstorming/uarch_study/04_on_chip_memory.md)

- The study covers the full ResNet-8 network, not just Conv0. Activations stay on chip (D-002).
- All conv weights for every layer are resident in one **Weight RAM** (§5): 7,775 words × 26 B = 202,150 B, mapped to 6 URAM. Each word is one array row, read at 1 word per cycle during compute.
- Weights are re-streamed once per pixel tile from the on-chip RAM. There is no off-chip weight traffic per inference.
- Weights are loaded **once at boot** by DMA (§1 diagram, "boot-time DMA (once)"). Whether boot load shares the image AXI-Stream or uses a separate AXI path is an open question (§7, Q4).
- Layer switching is driven by a **layer sequencer** that reads a 14-record descriptor table (`layer_config.bin`, 128 B per record) and picks weight-RAM addresses per layer (`[layer][oc_tile][k]`).
- Consequence: the design does not preload the next layer's weights at all. Every layer's weights are already on chip, so the "following layer preload" question is answered by the sequencer's address selection, not by a per-layer load.
- This differs from the Conv0 spec, which reloads from `c0_weight_store` on every START (A11). The two documents are not yet reconciled.
