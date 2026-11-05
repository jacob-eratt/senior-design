`timescale 1ns/1ps
// Skew triangle: lane i is delayed by i clock cycles (lane 0 is a wire).
// Used on the left edge of the array (row r of a pixel vector reaches row r
// of the array r cycles later) and on the weight-load inputs (column c loads
// c cycles later, following the swap wave).
module ws_skew #(
  parameter int LANES = 27,
  parameter int WIDTH = 8
) (
  input  logic             clk,
  input  logic             rst,
  input  logic [WIDTH-1:0] data_in  [LANES],
  output logic [WIDTH-1:0] data_out [LANES]
);

  assign data_out[0] = data_in[0];

  for (genvar i = 1; i < LANES; i++) begin : g_lane
    logic [WIDTH-1:0] stage [i];

    always_ff @(posedge clk) begin
      if (rst) begin
        for (int s = 0; s < i; s++) stage[s] <= '0;
      end else begin
        stage[0] <= data_in[i];
        for (int s = 1; s < i; s++) stage[s] <= stage[s-1];
      end
    end

    assign data_out[i] = stage[i-1];
  end
endmodule
