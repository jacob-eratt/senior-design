`timescale 1ns/1ps
// ROWS x COLS weight-stationary systolic grid: COLS ws_column instances side
// by side. Pixels enter on the left (row-skewed) and move one column right per
// cycle; partial sums move down; column c computes output channel c.
// The column inputs are already skewed: row r of a pixel arrives r cycles after
// row 0, and column c's weight load runs c cycles after column 0. Outputs are
// column-skewed: column c's result for a pixel appears c cycles after
// column 0's. Top-of-column partial sums are 0 (bias is added after the array).
module ws_array #(
  parameter int ROWS          = ws_pkg::ROWS,
  parameter int COLS          = ws_pkg::COLS,
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

  input  logic                            weight_load_enable    [COLS],
  input  logic signed [WEIGHT_W-1:0]      weight_load_in        [COLS],

  // Outputs
  output logic signed [PARTIAL_SUM_W-1:0] partial_sum_out       [COLS],
  output logic                            partial_sum_valid_out [COLS]
);

  // Horizontal buses: index c is the input of column c.
  logic signed [PIXEL_W-1:0] pixel_bus [COLS][ROWS];
  logic                      valid_bus [COLS][ROWS];
  logic                      swap_bus  [COLS][ROWS];

  assign pixel_bus[0] = pixel_in;
  assign valid_bus[0] = pixel_valid_in;
  assign swap_bus[0]  = new_pass_start_in;

  for (genvar c = 0; c < COLS; c++) begin : g_col
    logic signed [PIXEL_W-1:0] pixel_right [ROWS];
    logic                      valid_right [ROWS];
    logic                      swap_right  [ROWS];

    ws_column #(
      .ROWS          (ROWS),
      .PIXEL_W       (PIXEL_W),
      .WEIGHT_W      (WEIGHT_W),
      .PARTIAL_SUM_W (PARTIAL_SUM_W)
    ) u_col (
      .clk                   (clk),
      .rst                   (rst),
      .pixel_in              (pixel_bus[c]),
      .pixel_valid_in        (valid_bus[c]),
      .new_pass_start_in     (swap_bus[c]),
      .weight_load_enable    (weight_load_enable[c]),
      .weight_load_in        (weight_load_in[c]),
      .partial_sum_in        ('0),
      .pixel_out             (pixel_right),
      .pixel_valid_out       (valid_right),
      .new_pass_start_out    (swap_right),
      .weight_load_out       (),
      .partial_sum_out       (partial_sum_out[c]),
      .partial_sum_valid_out (partial_sum_valid_out[c])
    );

    if (c + 1 < COLS) begin : g_link
      assign pixel_bus[c+1] = pixel_right;
      assign valid_bus[c+1] = valid_right;
      assign swap_bus[c+1]  = swap_right;
    end
  end
endmodule
