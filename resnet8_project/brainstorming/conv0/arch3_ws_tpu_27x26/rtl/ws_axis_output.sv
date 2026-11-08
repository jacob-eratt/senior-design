`timescale 1ns/1ps
// Output data plane: vector-major INT32 words -> held 256-bit AXI beats.
module ws_axis_output (
    input logic clk, reset, start, send_enable,
    input logic [31:0] run_vectors,
    output ws_axi_pkg::result_read_t result_read,
    input logic [31:0] result_word,
    output ws_axi_pkg::axis_fwd_t stream_out,
    input ws_axi_pkg::axis_back_t stream_back,
    output logic finished,
    output logic [31:0] output_bytes
);
    import ws_axi_pkg::*;
    localparam int VEC_W = $clog2(MAX_VECTORS), COL_W = $clog2(COLS);
    localparam int LANE_W = $clog2(WORDS_PER_BEAT);
    localparam logic [COL_W-1:0] LAST_COL = COL_W'(COLS-1);
    localparam logic [LANE_W-1:0] LAST_LANE = LANE_W'(WORDS_PER_BEAT-1);
    logic [VEC_W-1:0] out_vec, final_vec;
    logic [COL_W-1:0] out_col;
    logic [LANE_W-1:0] pack_lane;
    logic [5:0] beat_bytes;
    logic final_word;
    // run_vectors is 1..MAX_VECTORS; the truncated decrement maps 64 -> 63.
    assign final_vec = VEC_W'(run_vectors - 1);
    assign final_word = out_vec == final_vec && out_col == LAST_COL;
    assign result_read.enable = send_enable && !stream_out.valid;
    assign result_read.vector = out_vec;
    assign result_read.column = out_col;
    // Same-edge event, not a delayed done pulse: preserves counter semantics.
    assign finished = send_enable && stream_out.valid && stream_back.ready && stream_out.last;
    always_ff @(posedge clk) begin
        if (reset || start) begin
            out_vec <= 0; out_col <= 0; pack_lane <= 0; beat_bytes <= 0; output_bytes <= 0;
            stream_out <= '0;
        end else if (send_enable) begin
            if (stream_out.valid) begin
                if (stream_back.ready) begin
                    output_bytes <= output_bytes + 32'(beat_bytes);
                    stream_out <= '0;
                end // Otherwise hold DATA, KEEP, LAST and VALID unchanged.
            end else begin
                stream_out.data[{pack_lane, 5'b00000} +: 32] <= result_word;
                stream_out.keep[{pack_lane, 2'b00} +: 4] <= 4'hf;
                if (out_col == LAST_COL) begin
                    out_col <= 0; out_vec <= out_vec + 1'b1;
                end else out_col <= out_col + 1'b1;
                if (pack_lane == LAST_LANE || final_word) begin
                    pack_lane <= 0; stream_out.valid <= 1; stream_out.last <= final_word;
                    beat_bytes <= {pack_lane, 2'b00} + 6'd4;
                end else pack_lane <= pack_lane + 1'b1;
            end
        end
    end
endmodule
