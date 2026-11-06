`timescale 1ns/1ps
// Matrix unit: ws_array plus the skew triangles on its inputs.
// Each cycle the caller presents one whole pixel vector x[0..ROWS-1] with one
// valid and one new_pass_start flag. Weight loads present one row of the next
// filter tile per shift, w[r][0..COLS-1], starting with r = ROWS-1.
// Column c of partial_sum_out is the dot product  sum_r x[r] * w[r][c]  for the
// pixel presented at edge t, visible after edge t + ROWS - 1 + c (the outputs
// stay column-skewed; the partial-sum RAM consumes them that way).
module ws_mxu #(
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
  input  logic                            pixel_valid_in,
  input  logic                            new_pass_start_in,

  input  logic                            weight_load_enable,
  input  logic signed [WEIGHT_W-1:0]      weight_load_in        [COLS],

  // Outputs
  output logic signed [PARTIAL_SUM_W-1:0] partial_sum_out       [COLS],
  output logic                            partial_sum_valid_out [COLS]
);

  localparam int ROW_LANE_W = PIXEL_W + 2;
  localparam int COL_LANE_W = WEIGHT_W + 1;

  logic [ROW_LANE_W-1:0]      row_lane_in   [ROWS];
  logic [ROW_LANE_W-1:0]      row_lane_out  [ROWS];
  logic [COL_LANE_W-1:0]      col_lane_in   [COLS];
  logic [COL_LANE_W-1:0]      col_lane_out  [COLS];
  logic signed [PIXEL_W-1:0]  skew_pixel    [ROWS];
  logic                       skew_valid    [ROWS];
  logic                       skew_swap     [ROWS];
  logic                       skew_wld_en   [COLS];
  logic signed [WEIGHT_W-1:0] skew_wld_data [COLS];

  for (genvar r = 0; r < ROWS; r++) begin : g_row_lane
    assign row_lane_in[r] = {new_pass_start_in, pixel_valid_in, pixel_in[r]};
    assign skew_pixel[r]  = $signed(row_lane_out[r][PIXEL_W-1:0]);
    assign skew_valid[r]  = row_lane_out[r][PIXEL_W];
    assign skew_swap[r]   = row_lane_out[r][PIXEL_W+1];
  end

  for (genvar c = 0; c < COLS; c++) begin : g_col_lane
    assign col_lane_in[c]   = {weight_load_enable, weight_load_in[c]};
    assign skew_wld_data[c] = $signed(col_lane_out[c][WEIGHT_W-1:0]);
    assign skew_wld_en[c]   = col_lane_out[c][WEIGHT_W];
  end

  ws_skew #(.LANES(ROWS), .WIDTH(ROW_LANE_W)) u_row_skew (
    .clk      (clk),
    .rst      (rst),
    .data_in  (row_lane_in),
    .data_out (row_lane_out)
  );

  ws_skew #(.LANES(COLS), .WIDTH(COL_LANE_W)) u_col_skew (
    .clk      (clk),
    .rst      (rst),
    .data_in  (col_lane_in),
    .data_out (col_lane_out)
  );

  ws_array #(
    .ROWS          (ROWS),
    .COLS          (COLS),
    .PIXEL_W       (PIXEL_W),
    .WEIGHT_W      (WEIGHT_W),
    .PARTIAL_SUM_W (PARTIAL_SUM_W)
  ) u_array (
    .clk                   (clk),
    .rst                   (rst),
    .pixel_in              (skew_pixel),
    .pixel_valid_in        (skew_valid),
    .new_pass_start_in     (skew_swap),
    .weight_load_enable    (skew_wld_en),
    .weight_load_in        (skew_wld_data),
    .partial_sum_out       (partial_sum_out),
    .partial_sum_valid_out (partial_sum_valid_out)
  );
endmodule
