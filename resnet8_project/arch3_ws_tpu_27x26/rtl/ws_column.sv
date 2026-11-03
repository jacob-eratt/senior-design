`timescale 1ns/1ps
// One column of the weight-stationary array: ROWS ws_pe instances stacked
// top -> bottom. Partial sums cascade down, so the bottom output is
// partial_sum_in + sum_r pixel[r] * w[r] for one pixel, ROWS cycles after the
// pixel enters row 0. Row r must receive that pixel's value (and its
// new_pass_start flag) r cycles after row 0 (input skew) so it meets the
// pixel's partial sum. The next pass's weights shift down the w_next chain
// with one shared enable: shift w[ROWS-1] first and w[0] last.
module ws_column #(
  parameter int ROWS          = ws_pkg::ROWS,
  parameter int PIXEL_W       = ws_pkg::ACT_W,
  parameter int WEIGHT_W      = ws_pkg::W_W,
  parameter int PARTIAL_SUM_W = ws_pkg::PSUM_W
) (
  // Inputs
  input  logic                            clk,
  input  logic                            rst,

  input  logic signed [PIXEL_W-1:0]       pixel_in              [ROWS],
  input  logic                            pixel_valid_in        [ROWS],
  input  logic                            new_pass_start_in     [ROWS],

  input  logic                            weight_load_enable,
  input  logic signed [WEIGHT_W-1:0]      weight_load_in,
  input  logic signed [PARTIAL_SUM_W-1:0] partial_sum_in,

  // Outputs
  output logic signed [PIXEL_W-1:0]       pixel_out             [ROWS],
  output logic                            pixel_valid_out       [ROWS],
  output logic                            new_pass_start_out    [ROWS],

  output logic signed [WEIGHT_W-1:0]      weight_load_out,
  output logic signed [PARTIAL_SUM_W-1:0] partial_sum_out,
  output logic                            partial_sum_valid_out
);

  logic signed [WEIGHT_W-1:0]      weight_chain [ROWS+1];
  logic signed [PARTIAL_SUM_W-1:0] psum_chain   [ROWS+1];

  assign weight_chain[0] = weight_load_in;
  assign psum_chain[0]   = partial_sum_in;

  for (genvar r = 0; r < ROWS; r++) begin : g_row
    ws_pe #(
      .PIXEL_W       (PIXEL_W),
      .WEIGHT_W      (WEIGHT_W),
      .PARTIAL_SUM_W (PARTIAL_SUM_W)
    ) PE_r (
      .clk                (clk),
      .rst                (rst),
      .pixel_in           (pixel_in[r]),
      .pixel_valid_in     (pixel_valid_in[r]),
      .new_pass_start_in  (new_pass_start_in[r]),
      .weight_load_enable (weight_load_enable),
      .weight_load_in     (weight_chain[r]),
      .partial_sum_in     (psum_chain[r]),
      .pixel_out          (pixel_out[r]),
      .pixel_valid_out    (pixel_valid_out[r]),
      .new_pass_start_out (new_pass_start_out[r]),
      .weight_load_out    (weight_chain[r+1]),
      .partial_sum_out    (psum_chain[r+1])
    );
  end

  assign weight_load_out       = weight_chain[ROWS];
  assign partial_sum_out       = psum_chain[ROWS];
  assign partial_sum_valid_out = pixel_valid_out[ROWS-1];
endmodule
