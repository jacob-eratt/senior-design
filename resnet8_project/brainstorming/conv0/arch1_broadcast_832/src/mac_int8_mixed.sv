`timescale 1ns/1ps
// Split inference boundaries: one DSP multiplier, fabric INT32 accumulator.
// These are inference requests; inspect post-synthesis DSP/LUT counts.
(* keep_hierarchy = "yes" *)
module int8_dsp_product(input wire clk, input wire en,
    input wire signed [7:0] a, b, output reg signed [15:0] p);
    (* use_dsp = "yes" *) wire signed [15:0] mult = a * b;
    always @(posedge clk) if (en) p <= mult;
endmodule

(* keep_hierarchy = "yes", use_dsp = "no" *)
module int32_fabric_acc(input wire clk, rst, init, en,
    input wire signed [31:0] bias, input wire signed [15:0] product,
    output reg signed [31:0] acc);
    (* use_dsp = "no" *) wire signed [31:0] sum = acc + {{16{product[15]}}, product};
    always @(posedge clk)
        if (rst) acc <= 0;
        else if (init) acc <= bias;
        else if (en) acc <= sum;
endmodule

module mac_int8_mixed(input wire clk, rst, init, mul_en, acc_en,
    input wire signed [7:0] a, b, input wire signed [31:0] bias,
    output wire signed [31:0] acc);
    wire signed [15:0] product;
    int8_dsp_product mult_i(clk, mul_en, a, b, product);
    int32_fabric_acc acc_i(clk, rst, init, acc_en, bias, product, acc);
endmodule
