`timescale 1ns/1ps
// 2048-byte ring, 13 x 64-bit input words and 4 x 64-bit output words.
// Sixteen independent banks, so no bank needs more than one write/clock.
module conv0_result_fifo(input wire clk, rst,
    input wire s_valid, output wire s_ready, input wire [831:0] s_data,
    output wire m_valid, input wire m_ready, output wire [255:0] m_data);
    reg [7:0] wr, rd;
    reg [8:0] count;
    wire push=s_valid && s_ready, pop=m_valid && m_ready;
    wire [63:0] bank_q [0:15];
    assign s_ready=count<=243;
    assign m_valid=count>=4;
    for (genvar bank=0; bank<16; bank=bank+1) begin: bank_i
        (* ram_style = "distributed" *) reg [63:0] mem [0:15];
        wire [3:0] wdelta=bank-wr[3:0];
        wire [3:0] rdelta=bank-rd[3:0];
        wire [7:0] wa=wr+wdelta, ra=rd+rdelta;
        always @(posedge clk) if (push && wdelta<13)
            mem[wa[7:4]]<=s_data[wdelta*64 +: 64];
        assign bank_q[bank]=mem[ra[7:4]];
    end
    for (genvar lane=0; lane<4; lane=lane+1) begin: output_word
        wire [3:0] bank=rd[3:0]+lane;
        assign m_data[lane*64 +: 64]=bank_q[bank];
    end
    always @(posedge clk) begin
        if (rst) begin wr<=0; rd<=0; count<=0; end
        else begin
            if (push) wr<=wr+8'd13;
            if (pop) rd<=rd+8'd4;
            case ({push,pop})
                2'b10: count<=count+9'd13;
                2'b01: count<=count-9'd4;
                2'b11: count<=count+9'd9;
                default: count<=count;
            endcase
        end
    end
endmodule
