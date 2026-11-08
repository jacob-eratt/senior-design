`timescale 1ns/1ps
// Fixed compute schedule + existing MXU + independently written result banks.
module ws_tile_engine (
    input logic clk, reset,
    input ws_axi_pkg::control_t control,
    input ws_axi_pkg::loader_status_t loader,
    output ws_axi_pkg::engine_status_t status,
    output ws_axi_pkg::tile_read_t tile_read,
    input ws_axi_pkg::tile_data_t tile_data,
    output logic send_enable,
    output logic [31:0] run_vectors,
    input logic output_finished,
    input ws_axi_pkg::result_read_t result_read,
    output logic [31:0] result_word
);
    import ws_axi_pkg::*;
    typedef enum logic [2:0] {IDLE, LOAD_WEIGHTS, SETTLE, FEED,
                             CAPTURE, SEND, FAULT} state_t;
    state_t state;
    logic signed [31:0] result_mem [0:COLS-1][0:MAX_VECTORS-1];
    int unsigned result_count [0:COLS-1];
    int unsigned weight_index, feed_index, settle_count;
    logic signed [7:0] pixel_vector [ROWS], weight_row [COLS];
    logic signed [31:0] column_results [COLS];
    logic column_valid [COLS];
    logic all_results, capture_overflow, pixel_valid, first_vector, core_reset;

    assign status.idle = state == IDLE;
    assign send_enable = state == SEND;
    assign core_reset = reset || state == IDLE || state == FAULT;
    assign pixel_valid = state == FEED;
    assign first_vector = pixel_valid && feed_index == 0;
    always_comb begin
        tile_read = '0;
        tile_read.weight_en = state == LOAD_WEIGHTS;
        tile_read.weight_addr = ROWS-1-weight_index;
        tile_read.input_en = pixel_valid;
        tile_read.input_addr = feed_index;
        for (int r = 0; r < ROWS; r++)
            pixel_vector[r] = $signed(tile_data.inputs[r*8 +: 8]);
        for (int c = 0; c < COLS; c++)
            weight_row[c] = $signed(tile_data.weights[c*8 +: 8]);
        all_results = 1;
        capture_overflow = 0;
        for (int c = 0; c < COLS; c++) begin
            if (result_count[c] != run_vectors) all_results = 0;
            if (column_valid[c] && result_count[c] >= run_vectors) capture_overflow = 1;
        end
        result_word = 0;
        if (result_read.enable)
            result_word = result_mem[result_read.column][result_read.vector];
    end
    ws_mxu #(.ROWS(ROWS), .COLS(COLS)) u_mxu (
        .clk(clk), .rst(core_reset), .pixel_in(pixel_vector),
        .pixel_valid_in(pixel_valid), .new_pass_start_in(first_vector),
        .weight_load_enable(tile_read.weight_en), .weight_load_in(weight_row),
        .partial_sum_out(column_results), .partial_sum_valid_out(column_valid)
    );

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE; run_vectors <= 0;
            status.compute_done <= 0; status.output_done <= 0; status.errors <= 0;
            status.elapsed_cycles <= 0; status.compute_cycles <= 0;
            weight_index <= 0; feed_index <= 0; settle_count <= 0;
            for (int c = 0; c < COLS; c++) result_count[c] <= 0;
        end else begin
            if (state != IDLE && state != FAULT)
                status.elapsed_cycles <= status.elapsed_cycles + 1;
            if (state == LOAD_WEIGHTS || state == SETTLE || state == FEED || state == CAPTURE)
                status.compute_cycles <= status.compute_cycles + 1;
            if (loader.packet_start) begin status.compute_done <= 0; status.output_done <= 0; end
            // Capture can begin before the final input vector has been fed.
            if (state == FEED || state == CAPTURE) begin
                for (int c = 0; c < COLS; c++) begin
                    if (column_valid[c] && result_count[c] < run_vectors) begin
                        result_mem[c][result_count[c]] <= column_results[c];
                        result_count[c] <= result_count[c] + 1;
                    end
                end
            end
            case (state)
                IDLE: if (control.start) begin
                    state <= LOAD_WEIGHTS; run_vectors <= control.vector_count;
                    status.compute_done <= 0; status.output_done <= 0;
                    weight_index <= 0; feed_index <= 0; settle_count <= 0;
                    status.elapsed_cycles <= 0; status.compute_cycles <= 0;
                    for (int c = 0; c < COLS; c++) result_count[c] <= 0;
                end
                LOAD_WEIGHTS: begin
                    if (weight_index == ROWS-1) begin state <= SETTLE; settle_count <= 0; end
                    else weight_index <= weight_index + 1;
                end
                SETTLE: begin // Match Python schedule: two empty edges after weight load.
                    if (settle_count == 1) state <= FEED;
                    else settle_count <= settle_count + 1;
                end
                FEED: begin
                    if (feed_index + 1 == run_vectors) state <= CAPTURE;
                    else feed_index <= feed_index + 1;
                end
                CAPTURE: if (all_results) begin state <= SEND; status.compute_done <= 1; end
                SEND: if (output_finished) begin state <= IDLE; status.output_done <= 1; end
                default: begin end
            endcase
            if (loader.packet_error) state <= FAULT;
            if ((state == FEED || state == CAPTURE) && capture_overflow) begin
                status.errors[2] <= 1; state <= FAULT;
            end
            if ((state == FEED || state == CAPTURE) &&
                status.compute_cycles > 2*ROWS + COLS + MAX_VECTORS + 32) begin
                status.errors[3] <= 1; state <= FAULT;
            end
        end
    end
endmodule
