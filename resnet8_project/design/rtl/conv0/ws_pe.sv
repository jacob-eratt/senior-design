`timescale 1ns/1ps
// Weight-stationary processing element.
// Activations move left -> right, partial sums move top -> bottom, and the
// next pass's weight shifts top -> bottom through w_next (B1) while w_active
// (B2) is used. swap_in marks the first activation of a new pass: that
// activation already uses w_next, and w_next is copied into w_active.
// Request DSP arithmetic at module and product-signal scope. Verify the
// synthesized DSP count; attributes alone do not guarantee B1/B2 or cascades.
(* use_dsp = "yes" *)
module ws_pe #(
  parameter int PIXEL_W       = ws_pkg::ACT_W,
  parameter int WEIGHT_W      = ws_pkg::W_W,
  parameter int PARTIAL_SUM_W = ws_pkg::PSUM_W
) (
  // Inputs
  input  logic                         clk,
  input  logic                         rst,

  input  logic signed [PIXEL_W-1:0]   pixel_in,
  input  logic                         pixel_valid_in,
  input  logic                         new_pass_start_in,

  input  logic                         weight_load_enable,
  input  logic signed [WEIGHT_W-1:0]  weight_load_in,
  input  logic signed [PARTIAL_SUM_W-1:0] partial_sum_in,

  // Outputs
  output logic signed [PIXEL_W-1:0]   pixel_out,
  output logic                         pixel_valid_out,
  output logic                         new_pass_start_out,

  output logic signed [WEIGHT_W-1:0]  weight_load_out,
  output logic signed [PARTIAL_SUM_W-1:0] partial_sum_out
);

  logic signed [WEIGHT_W-1:0]       w_next;
  logic signed [WEIGHT_W-1:0]       w_active;
  logic signed [WEIGHT_W-1:0]       w_sel;
  (* use_dsp = "yes" *) logic signed [PIXEL_W+WEIGHT_W-1:0] product;

  assign w_sel   = new_pass_start_in ? w_next : w_active;
  assign product = pixel_in * w_sel;
  assign weight_load_out = w_next;

  always_ff @(posedge clk) begin
    if (rst) begin
      pixel_out     <= '0;
      pixel_valid_out <= 1'b0;
      new_pass_start_out    <= 1'b0;
      w_next      <= '0;
      w_active    <= '0;
      partial_sum_out    <= '0;
    end else begin
      pixel_out     <= pixel_in;
      pixel_valid_out <= pixel_valid_in;
      new_pass_start_out    <= new_pass_start_in;
      partial_sum_out    <= partial_sum_in + PARTIAL_SUM_W'(product);
      if (new_pass_start_in) w_active <= w_next;
      if (weight_load_enable)  w_next   <= weight_load_in;
    end
  end
endmodule
