`timescale 1ns/1ps
// Input data plane: packet checks, one-byte-per-clock writes, weight/input RAM.
module ws_axis_loader (
    input logic clk, reset,
    input ws_axi_pkg::control_t control,
    input logic engine_idle, engine_fault,
    input ws_axi_pkg::axis_fwd_t stream_in,
    output ws_axi_pkg::axis_back_t stream_back,
    output ws_axi_pkg::loader_status_t status,
    input ws_axi_pkg::tile_read_t tile_read,
    output ws_axi_pkg::tile_data_t tile_data
);
    import ws_axi_pkg::*;
    logic [COLS*8-1:0] weight_mem [0:ROWS-1];
    logic [ROWS*8-1:0] input_mem [0:MAX_VECTORS-1];
    logic packet_active, packet_kind, beat_valid, beat_last;
    logic [255:0] beat_data;
    int unsigned packet_bytes, byte_offset, byte_lane, beat_bytes;
    logic [$clog2(MAX_VECTORS)-1:0] wr_row; // Row/column counters replace /, % on byte_offset.
    logic [$clog2(ROWS)-1:0] wr_col;
    localparam logic [$clog2(ROWS)-1:0] LAST_WEIGHT_COL = COLS-1, LAST_INPUT_COL = ROWS-1;
    int unsigned expected_bytes, remaining_bytes, accepted_bytes;
    logic [31:0] expected_keep;
    logic expected_last;

    assign status.busy = packet_active || beat_valid;
    assign stream_back.ready = !reset && engine_idle && !beat_valid &&
                               status.errors == 0 && !engine_fault &&
                               !control.write_pending && !control.start;
    assign status.packet_start = stream_in.valid && stream_back.ready && !packet_active;
    assign status.packet_error = stream_in.valid && stream_back.ready &&
        (stream_in.keep != expected_keep || stream_in.last != expected_last);
    always_comb begin
        expected_bytes = packet_active ? packet_bytes :
            (control.load_kind == 0 ? ROWS*COLS : control.vector_count*ROWS);
        remaining_bytes = expected_bytes - byte_offset;
        accepted_bytes = remaining_bytes < STREAM_BYTES ? remaining_bytes : STREAM_BYTES;
        expected_last = remaining_bytes <= STREAM_BYTES;
        expected_keep = 0;
        for (int b = 0; b < STREAM_BYTES; b++)
            if (b < accepted_bytes) expected_keep[b] = 1'b1;
        tile_data = '0;
        if (tile_read.weight_en) tile_data.weights = weight_mem[tile_read.weight_addr];
        if (tile_read.input_en) tile_data.inputs = input_mem[tile_read.input_addr];
    end

    // Memory is not reset in bulk. Ready flags prevent use of uninitialized data.
    always_ff @(posedge clk) begin
        if (reset) begin
            status.weights_ready <= 0; status.inputs_ready <= 0;
            status.loaded_vectors <= 0; status.errors <= 0;
            packet_active <= 0; packet_kind <= 0; packet_bytes <= 0;
            beat_valid <= 0; beat_data <= 0; beat_last <= 0;
            byte_offset <= 0; byte_lane <= 0; beat_bytes <= 0; wr_row <= 0; wr_col <= 0;
        end else begin
            if (control.start) status.inputs_ready <= 0;
            if (stream_in.valid && stream_back.ready) begin
                if (!packet_active) begin
                    packet_kind <= control.load_kind[0]; packet_bytes <= expected_bytes;
                    packet_active <= 1;
                    if (control.load_kind == 0) status.weights_ready <= 0;
                    else begin
                        status.inputs_ready <= 0; status.loaded_vectors <= control.vector_count;
                    end
                end
                if (status.packet_error) begin
                    status.errors[0] <= stream_in.keep != expected_keep;
                    status.errors[1] <= stream_in.last != expected_last;
                end else begin
                    beat_data <= stream_in.data; beat_last <= stream_in.last;
                    beat_bytes <= accepted_bytes; byte_lane <= 0; beat_valid <= 1;
                end
            end
            if (beat_valid) begin
                if (!packet_kind)
                    weight_mem[wr_row][wr_col*8 +: 8] <= beat_data[byte_lane*8 +: 8];
                else
                    input_mem[wr_row][wr_col*8 +: 8] <= beat_data[byte_lane*8 +: 8];
                if (wr_col == (packet_kind ? LAST_INPUT_COL : LAST_WEIGHT_COL)) begin
                    wr_col <= 0; wr_row <= wr_row + 1'b1;
                end else wr_col <= wr_col + 1'b1;
                byte_offset <= byte_offset + 1;
                byte_lane <= byte_lane + 1;
                if (byte_lane + 1 == beat_bytes) begin
                    beat_valid <= 0;
                    if (beat_last) begin
                        packet_active <= 0; byte_offset <= 0; wr_row <= 0; wr_col <= 0;
                        if (!packet_kind) status.weights_ready <= 1;
                        else status.inputs_ready <= 1;
                    end
                end
            end
        end
    end
endmodule
