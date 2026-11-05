`timescale 1ns/1ps
// Fixed pretrained constants allow constant multiplication and shifts in LUTs.
// Three registered stages, II=1, global freeze preserves all lanes on a stall.
(* use_dsp = "no" *)
module conv0_requant(input wire clk, rst,
    input wire s_valid, output wire s_ready, input wire [3327:0] s_data,
    output wire m_valid, input wire m_ready, output wire [831:0] m_data);
    import conv0_quant_pkg::*;
    reg [2:0] valid;
    wire ce = !valid[2] || m_ready;
    assign s_ready=ce;
    assign m_valid=valid[2];
    always @(posedge clk)
        if (rst) valid<=0;
        else if (ce) valid<={valid[1:0],s_valid};
    for (genvar lane=0; lane<104; lane=lane+1) begin: lane_i
        localparam signed [31:0] M=quant_m(lane%26);
        localparam integer SHIFT=quant_s(lane%26);
        reg signed [63:0] product;
        reg [63:0] rounded;
        reg negative;
        reg [7:0] result;
        wire signed [31:0] acc=$signed(s_data[lane*32 +: 32]);
        wire [63:0] magnitude=product[63] ? (~product+64'd1) : product;
        always @(posedge clk) if (ce) begin
            product<=acc*M;
            rounded<=(magnitude+(64'd1 << (SHIFT-1))) >> SHIFT;
            negative<=product[63];
            result<=negative ? 8'd0 : ((rounded>127) ? 8'd127 : rounded[7:0]);
        end
        assign m_data[lane*8 +: 8]=result;
    end
endmodule
