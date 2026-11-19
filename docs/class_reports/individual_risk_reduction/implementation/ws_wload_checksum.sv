`timescale 1ns/1ps
// Mitigation 1 (all-on-FPGA weights): load-time integrity check of the weight packet.
// Fletcher-style 64-bit checksum over the accepted packet bytes, d_i for i = 0..N-1:
//   s1 = sum(d_i)            mod 2^32
//   s2 = sum((N - i) * d_i)  mod 2^32
// Per accepted beat with n valid bytes, S0 = sum(d_b), S1 = sum(b * d_b), b = 0..BEAT_BYTES-1:
//   s1 <= s1 + S0
//   s2 <= s2 + n * (s1 + S0) - S1
// Beat reduction is a binary adder tree (no serial accumulation chain).
// clear zeroes the state and is merged with the first beat of a packet (same cycle is legal).
// done pulses the cycle after the last beat; match is valid while done is high.
// Loader use: set the weight loaded-flag only when done && match && no packet error.
// The host computes the same checksum over the packet in DDR and writes it to a register.
module ws_wload_checksum #(
  parameter int BEAT_BYTES = 32
) (
  // Inputs
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    clear,
  input  logic                    beat_valid,
  input  logic                    beat_last,
  input  logic [8*BEAT_BYTES-1:0] beat_data,
  input  logic [BEAT_BYTES-1:0]   beat_keep,
  input  logic [63:0]             expected,

  // Outputs
  output logic                    done,
  output logic                    match
);

  logic [31:0] s1;
  logic [31:0] s2;
  logic [31:0] base_s1;
  logic [31:0] base_s2;
  logic [31:0] beat_s1;
  logic [31:0] lane_d [BEAT_BYTES];
  logic [31:0] lane_w [BEAT_BYTES];
  logic [31:0] sum_d;
  logic [31:0] sum_w;
  logic [5:0]  nbytes;

  function automatic logic [31:0] tree_sum(input logic [31:0] v_in [BEAT_BYTES]);
    logic [31:0] v [BEAT_BYTES];
    v = v_in;
    for (int s = 1; s < BEAT_BYTES; s *= 2)
      for (int i = 0; i < BEAT_BYTES; i += 2 * s)
        v[i] = v[i] + v[i + s];
    return v[0];
  endfunction

  for (genvar b = 0; b < BEAT_BYTES; b++) begin : g_lane
    assign lane_d[b] = beat_keep[b] ? 32'(beat_data[8*b +: 8]) : 32'd0;
    assign lane_w[b] = lane_d[b] * unsigned'(b);
  end

  assign sum_d   = tree_sum(lane_d);
  assign sum_w   = tree_sum(lane_w);
  assign nbytes  = 6'(unsigned'($countones(beat_keep)));
  assign base_s1 = clear ? 32'd0 : s1;
  assign base_s2 = clear ? 32'd0 : s2;
  assign beat_s1 = base_s1 + sum_d;

  always_ff @(posedge clk) begin
    if (rst) begin
      s1   <= '0;
      s2   <= '0;
      done <= 1'b0;
    end else begin
      done <= beat_valid && beat_last;
      if (beat_valid) begin
        s1 <= beat_s1;
        s2 <= base_s2 + 32'(nbytes) * beat_s1 - sum_w;
      end else if (clear) begin
        s1 <= '0;
        s2 <= '0;
      end
    end
  end

  assign match = ({s2, s1} == expected);

endmodule
