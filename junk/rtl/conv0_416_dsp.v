`timescale 1ns/1ps
`default_nettype none

// Fixed Conv0: 32x32x3 input, 26 filters, 3x3 stride 1, SAME zero padding.
// 16 spatial positions x 26 channels = 416 signed INT8 multiplier/INT32 MAC lanes.
// Standalone Verilog-2001; no dependency on the older MAC/pixel/serial modules.
//
// PRELOAD while load_ready=1: image HWC byte addresses 0..3071, weights OHWI
// byte addresses 0..701, bias INT32 word addresses 0..25. Independent write ports
// may be used simultaneously. A write occurs on a rising edge with *_we and
// load_ready. Invalid addresses are ignored. Load ALL memories before first start.
// Memory contents survive reset; reset clears control, pipeline valids, and sums.
// Start while idle has priority over writes (load_ready=0 when start=1).
// Start/writes while busy are ignored. Pulse start, don't hold it until completion.
//
// One start computes all 26624 raw signed INT32 outputs. out_addr is an HWC WORD
// index for the first result in out_data. Lane j occupies bits j*32 +: 32 and is
// result out_addr+j. OUTPUT_LANES must be a positive divisor of 416 (e.g. 1,2,4,
// 8,13,16,26,32). Every beat is full; there is no partial-beat mask. Data/address
// remain stable until out_valid && out_ready at a rising edge. done pulses after
// the final accepted beat. This is an INTERNAL memory/stream bus, not a pinout.
//
// Image memory is replicated 16 times to provide 16 synchronous reads per cycle
// with straightforward RAM inference: 16 * 3072 bytes = 48 KiB physical payload.
// Writes broadcast to all copies. 26 small weight banks each deliver one byte,
// shared across the spatial lanes. Biases are cached locally. No external reads
// occur during compute. Pipeline: synchronous RAM read -> registered products ->
// accumulators. All 27 taps issue consecutively, then the pipeline drains before
// the 416-result tile is emitted. Compute pauses during tile output.
//
// Weights/bias MUST already include folded BatchNorm. No requantization, INT8
// saturation, ReLU, residual add, or other ResNet layers are implemented here.
// INT32 arithmetic wraps on overflow; exported parameters must satisfy the
// software accumulator bound. DSP/RAM attributes request mapping; actual DSP,
// RAM, LUT use and achievable clock require synthesis/implementation reports.
// Supplied for review/simulation by the team; not compiled or simulated here.
module conv0_416_dsp #(
    parameter integer OUTPUT_LANES = 4
) (
    input wire clk,
    input wire rst,
    input wire start,
    output wire busy,
    output reg done,

    output wire load_ready,
    input wire image_we,
    input wire [11:0] image_addr,
    input wire signed [7:0] image_data,
    input wire weight_we,
    input wire [9:0] weight_addr,
    input wire signed [7:0] weight_data,
    input wire bias_we,
    input wire [4:0] bias_addr,
    input wire signed [31:0] bias_data,

    output wire out_valid,
    input wire out_ready,
    output wire [14:0] out_addr,
    output wire [OUTPUT_LANES*32-1:0] out_data
);
    localparam integer TILE_VALUES = 416;
    localparam integer TILE_BEATS = TILE_VALUES / OUTPUT_LANES;
    localparam [2:0] IDLE = 3'd0, INIT = 3'd1, ISSUE = 3'd2,
                     DRAIN = 3'd3, EMIT = 3'd4;
    reg [2:0] state;
    reg [5:0] tile;
    reg [4:0] tap;
    reg [1:0] ky, kx, ic;
    reg [8:0] emit_beat;
    reg read_valid, product_valid, read_last, product_last;
    wire issue = (state == ISSUE);
    wire image_write = load_ready && image_we && image_addr < 12'd3072;
    wire weight_write = load_ready && weight_we && weight_addr < 10'd702;
    wire bias_write = load_ready && bias_we && bias_addr < 5'd26;
    wire [4:0] weight_write_channel = weight_addr / 10'd27;
    wire [4:0] weight_write_tap = weight_addr % 10'd27;
    wire signed [7:0] samples [0:15];
    wire signed [7:0] weights [0:25];
    reg signed [31:0] biases [0:25];
    wire signed [31:0] sums [0:415];

    assign load_ready = (state == IDLE) && !rst && !start;
    assign busy = (state != IDLE);
    assign out_valid = (state == EMIT);
    assign out_addr = tile * 15'd416 + emit_beat * OUTPUT_LANES;

    always @(posedge clk) begin
        if (bias_write) biases[bias_addr] <= bias_data;
    end

    genvar ch, pos, lane, beat;
    generate
        for (ch = 0; ch < 26; ch = ch + 1) begin : weight_bank
            (* ram_style = "distributed" *) reg signed [7:0] mem [0:26];
            reg signed [7:0] read_q;
            always @(posedge clk) begin
                if (weight_write && weight_write_channel == ch)
                    mem[weight_write_tap] <= weight_data;
                if (issue) read_q <= mem[tap];
            end
            assign weights[ch] = read_q;
        end

        for (pos = 0; pos < 16; pos = pos + 1) begin : spatial_lane
            (* ram_style = "block" *) reg signed [7:0] image_mem [0:3071];
            reg signed [7:0] read_q;
            reg inside_q;
            // Tile t covers linear spatial positions 16*t .. 16*t+15.
            // Each tile is either the left or right half of a 32-pixel row.
            wire signed [31:0] source_y = $signed({1'b0, tile[5:1]})
                                           + $signed({1'b0, ky}) - 32'sd1;
            wire signed [31:0] source_x = (tile[0] ? 32'sd16 : 32'sd0)
                                           + pos + $signed({1'b0, kx}) - 32'sd1;
            wire inside_image = source_y >= 0 && source_y < 32 &&
                                source_x >= 0 && source_x < 32;
            // Safe read address even for padding. Mask the registered result;
            // do not reset RAM or the RAM output register, to permit inference.
            wire [11:0] read_addr = inside_image ?
                ((source_y * 32 + source_x) * 3 + ic) : 12'd0;
            always @(posedge clk) begin
                if (image_write) image_mem[image_addr] <= image_data;
                if (issue) begin
                    read_q <= image_mem[read_addr];
                    inside_q <= inside_image;
                end
            end
            assign samples[pos] = inside_q ? read_q : 8'sd0;

            for (ch = 0; ch < 26; ch = ch + 1) begin : mac_lane
                (* use_dsp = "yes" *) wire signed [15:0] product;
                reg signed [15:0] product_q;
                reg signed [31:0] acc;
                assign product = samples[pos] * weights[ch];
                always @(posedge clk) begin
                    if (read_valid) product_q <= product;
                    if (rst) acc <= 32'sd0;
                    else if (state == INIT) acc <= biases[ch];
                    else if (product_valid)
                        acc <= acc + {{16{product_q[15]}}, product_q};
                end
                assign sums[pos*26+ch] = acc;
            end
        end

        // Bank the readout mux by output lane. Each mux sees only TILE_BEATS
        // candidates, rather than all 416 accumulators.
        for (lane = 0; lane < OUTPUT_LANES; lane = lane + 1) begin : output_lane
            wire [31:0] bank [0:TILE_BEATS-1];
            for (beat = 0; beat < TILE_BEATS; beat = beat + 1) begin : result
                assign bank[beat] = sums[beat*OUTPUT_LANES+lane];
            end
            assign out_data[lane*32 +: 32] = bank[emit_beat];
        end
    endgenerate

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            tile <= 0;
            tap <= 0;
            ky <= 0;
            kx <= 0;
            ic <= 0;
            emit_beat <= 0;
            read_valid <= 0;
            product_valid <= 0;
            read_last <= 0;
            product_last <= 0;
            done <= 0;
        end else begin
            done <= 0;
            read_valid <= issue;
            product_valid <= read_valid;
            read_last <= issue && (tap == 5'd26);
            product_last <= read_last;
            case (state)
                IDLE: if (start) begin
                    tile <= 0;
                    emit_beat <= 0;
                    state <= INIT;
                end
                INIT: begin
                    tap <= 0;
                    ky <= 0;
                    kx <= 0;
                    ic <= 0;
                    state <= ISSUE;
                end
                ISSUE: begin
                    if (tap == 5'd26) state <= DRAIN;
                    else begin
                        tap <= tap + 1'b1;
                        if (ic == 2'd2) begin
                            ic <= 0;
                            if (kx == 2'd2) begin
                                kx <= 0;
                                ky <= ky + 1'b1;
                            end else kx <= kx + 1'b1;
                        end else ic <= ic + 1'b1;
                    end
                end
                DRAIN: if (product_valid && product_last) begin
                    // Final accumulator update happens at this same edge.
                    emit_beat <= 0;
                    state <= EMIT;
                end
                EMIT: if (out_ready) begin
                    if (emit_beat == TILE_BEATS-1) begin
                        emit_beat <= 0;
                        if (tile == 6'd63) begin
                            state <= IDLE;
                            done <= 1;
                        end else begin
                            tile <= tile + 1'b1;
                            state <= INIT;
                        end
                    end else emit_beat <= emit_beat + 1'b1;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
