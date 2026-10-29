`timescale 1ns/1ps
// One entire image row (32 spatial lanes) x 26 output channels.
// Memory layout: input HWC, weights OHWI, output HWC. Padding is numeric zero.
module conv0_832_core(input wire clk, rst, start,
    input wire load_valid, input wire [1:0] load_kind,
    input wire [11:0] load_addr, input wire [127:0] load_data,
    input wire [15:0] load_keep,
    output wire out_valid, input wire out_ready,
    output wire [3327:0] out_data, output reg done, output wire busy);
    localparam IDLE=0, INIT=1, ISSUE=2, DRAIN=3, EMIT=4;
    reg [2:0] state;
    reg [4:0] row, tap;
    reg [2:0] beat;
    reg read_valid, product_valid, read_last, product_last;
    wire init_acc = state == INIT;
    wire issue = state == ISSUE;
    wire signed [7:0] sample [0:31];
    wire signed [7:0] weight [0:25];
    wire signed [31:0] bias [0:25];
    wire signed [31:0] accum [0:831];
    assign busy = state != IDLE;
    assign out_valid = state == EMIT;

    // Four 32-bit banks per spatial replica permit a 16-byte preload/clock.
    // Each replica supplies one independently addressed input sample/clock.
    for (genvar p=0; p<32; p=p+1) begin: image_replica
        integer sy, sx, addr;
        reg inside_q;
        reg [3:0] select_q;
        wire [31:0] bank_word [0:3];
        always @* begin
            sy = row + tap/9 - 1;
            sx = p + (tap/3)%3 - 1;
            addr = (sy*32 + sx)*3 + tap%3;
            if (sy<0 || sy>=32 || sx<0 || sx>=32) addr=0;
        end
        always @(posedge clk) if (issue) begin
            inside_q <= (sy>=0 && sy<32 && sx>=0 && sx<32);
            select_q <= addr%16;
        end
        for (genvar bank=0; bank<4; bank=bank+1) begin: bank_i
            (* ram_style = "block" *) reg [31:0] mem [0:191];
            reg [31:0] q;
            always @(posedge clk) begin
                if (load_valid && load_kind==0 && load_addr<3072)
                    for (integer byte_i=0; byte_i<4; byte_i=byte_i+1)
                        if (load_keep[bank*4+byte_i])
                            mem[load_addr>>4][byte_i*8 +: 8] <= load_data[(bank*4+byte_i)*8 +: 8];
                if (issue) q <= mem[addr>>4];
            end
            assign bank_word[bank]=q;
        end
        assign sample[p] = inside_q ? (bank_word[select_q[3:2]] >> (select_q[1:0]*8)) : 8'sd0;
    end

    for (genvar c=0; c<26; c=c+1) begin: channel
        // Small weights register banks: 26 independent reads per issue clock.
        reg signed [7:0] wmem [0:26];
        reg signed [7:0] wq;
        reg signed [31:0] b;
        for (genvar t=0; t<27; t=t+1) begin: preload_weight
            localparam integer OFFSET=c*27+t;
            always @(posedge clk)
                if (load_valid && load_kind==1 && load_addr==(OFFSET/16)*16 && load_keep[OFFSET%16])
                    wmem[t] <= load_data[(OFFSET%16)*8 +: 8];
        end
        always @(posedge clk) begin
            if (issue) wq <= wmem[tap];
            if (load_valid && load_kind==2 && load_addr==((c*4)/16)*16 && (&load_keep[(c%4)*4 +: 4]))
                b <= load_data[(c%4)*32 +: 32];
        end
        assign weight[c]=wq;
        assign bias[c]=b;
        for (genvar p=0; p<32; p=p+1) begin: spatial
            mac_int8_mixed mac_i(clk, rst, init_acc, read_valid, product_valid,
                sample[p], weight[c], bias[c], accum[p*26+c]);
        end
    end
    // Eight groups of four pixels, each pixel has all 26 channels.
    for (genvar lane=0; lane<104; lane=lane+1) begin: output_lane
        assign out_data[lane*32 +: 32] = accum[beat*104+lane];
    end
    always @(posedge clk) begin
        if (rst) begin
            state<=IDLE; row<=0; tap<=0; beat<=0; done<=0;
            read_valid<=0; product_valid<=0; read_last<=0; product_last<=0;
        end else begin
            done<=0;
            read_valid<=issue; product_valid<=read_valid;
            read_last<=issue && tap==26; product_last<=read_last;
            case (state)
                IDLE: if (start) begin row<=0; state<=INIT; end
                INIT: begin tap<=0; beat<=0; state<=ISSUE; end
                ISSUE: if (tap==26) state<=DRAIN; else tap<=tap+1'b1;
                DRAIN: if (product_valid && product_last) state<=EMIT;
                EMIT: if (out_ready) begin
                    if (beat==7) begin
                        if (row==31) begin state<=IDLE; done<=1; end
                        else begin row<=row+1'b1; state<=INIT; end
                    end else beat<=beat+1'b1;
                end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
