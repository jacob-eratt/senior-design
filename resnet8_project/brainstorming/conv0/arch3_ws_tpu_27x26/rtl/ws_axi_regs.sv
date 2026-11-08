`timescale 1ns/1ps
// CPU control plane: address decoding, configuration and held AXI responses.
module ws_axi_regs (
    input logic clk, aresetn,
    input ws_axi_pkg::axil_req_t bus_req,
    output ws_axi_pkg::axil_rsp_t bus_rsp,
    input ws_axi_pkg::loader_status_t loader,
    input ws_axi_pkg::engine_status_t engine,
    input logic [31:0] output_bytes,
    output ws_axi_pkg::control_t control
);
    import ws_axi_pkg::*;
    logic aw_pending, w_pending;
    logic [7:0] awaddr_q;
    logic [31:0] wdata_q;
    logic [3:0] wstrb_q;
    logic [31:0] masked_write, merged_kind, merged_count, status_word, errors;
    logic can_configure, can_start;

    function automatic logic [31:0] merge_bytes(
        input logic [31:0] previous, value, input logic [3:0] strobe);
        logic [31:0] merged;
        merged = previous;
        for (int b = 0; b < 4; b++)
            if (strobe[b]) merged[b*8 +: 8] = value[b*8 +: 8];
        return merged;
    endfunction

    assign masked_write = merge_bytes(32'b0, wdata_q, wstrb_q);
    assign merged_kind = merge_bytes(control.load_kind, wdata_q, wstrb_q);
    assign merged_count = merge_bytes(control.vector_count, wdata_q, wstrb_q);
    assign errors = engine.errors | {30'b0, loader.errors};
    assign can_configure = engine.idle && !loader.busy &&
                           !control.start && !control.soft_reset;
    assign can_start = can_configure && loader.weights_ready && loader.inputs_ready &&
                       loader.loaded_vectors == control.vector_count && errors == 0;
    assign control.write_pending = aw_pending || w_pending;
    assign bus_rsp.awready = aresetn && !aw_pending && !bus_rsp.b.valid;
    assign bus_rsp.wready = aresetn && !w_pending && !bus_rsp.b.valid;
    assign bus_rsp.arready = aresetn && !bus_rsp.r.valid;

    always_comb begin
        status_word = '0;
        status_word[0] = !engine.idle || loader.busy || control.start;
        status_word[1] = loader.weights_ready;
        status_word[2] = loader.inputs_ready && loader.loaded_vectors == control.vector_count;
        status_word[3] = engine.compute_done;
        status_word[4] = engine.output_done;
        status_word[5] = errors != 0;
        status_word[6] = loader.busy;
    end

    // AW and W can arrive in either order. A soft reset must not cancel BVALID.
    always_ff @(posedge clk) begin
        if (!aresetn) begin
            aw_pending <= 0; w_pending <= 0;
            awaddr_q <= 0; wdata_q <= 0; wstrb_q <= 0;
            bus_rsp.b <= '0; bus_rsp.r <= '0;
            control.load_kind <= 0; control.vector_count <= MAX_VECTORS;
            control.start <= 0; control.soft_reset <= 0;
        end else begin
            control.start <= 0; control.soft_reset <= 0;
            if (bus_req.aw.valid && bus_rsp.awready) begin
                awaddr_q <= bus_req.aw.addr; aw_pending <= 1;
            end
            if (bus_req.w.valid && bus_rsp.wready) begin
                wdata_q <= bus_req.w.data; wstrb_q <= bus_req.w.strb; w_pending <= 1;
            end
            if (bus_rsp.b.valid && bus_req.bready) bus_rsp.b.valid <= 0;
            if (aw_pending && w_pending && !bus_rsp.b.valid) begin
                aw_pending <= 0; w_pending <= 0;
                bus_rsp.b.valid <= 1; bus_rsp.b.resp <= 2'b00;
                case (awaddr_q)
                    8'h00: begin
                        if (masked_write == 32'h2) control.soft_reset <= 1;
                        else if (masked_write == 32'h1 && can_start) control.start <= 1;
                        else if (masked_write != 0) bus_rsp.b.resp <= 2'b10;
                    end
                    8'h08: begin
                        if (can_configure && merged_kind <= 1) control.load_kind <= merged_kind;
                        else bus_rsp.b.resp <= 2'b10;
                    end
                    8'h0c: begin
                        if (can_configure && merged_count >= 1 && merged_count <= MAX_VECTORS)
                            control.vector_count <= merged_count;
                        else bus_rsp.b.resp <= 2'b10;
                    end
                    default: bus_rsp.b.resp <= 2'b10;
                endcase
            end
            if (bus_rsp.r.valid && bus_req.rready) bus_rsp.r.valid <= 0;
            if (bus_req.ar.valid && bus_rsp.arready) begin
                bus_rsp.r.valid <= 1; bus_rsp.r.resp <= 2'b00;
                case (bus_req.ar.addr)
                    8'h00: bus_rsp.r.data <= 0;
                    8'h04: bus_rsp.r.data <= status_word;
                    8'h08: bus_rsp.r.data <= control.load_kind;
                    8'h0c: bus_rsp.r.data <= control.vector_count;
                    8'h10: bus_rsp.r.data <= errors;
                    8'h14: bus_rsp.r.data <= engine.elapsed_cycles;
                    8'h18: bus_rsp.r.data <= engine.compute_cycles;
                    8'h1c: bus_rsp.r.data <= output_bytes;
                    8'h20: bus_rsp.r.data <= 32'h57530001;
                    8'h24: bus_rsp.r.data <= MAX_VECTORS;
                    default: begin bus_rsp.r.data <= 0; bus_rsp.r.resp <= 2'b10; end
                endcase
            end
        end
    end
    // bus_req.aw.prot / ar.prot intentionally do not perform access filtering.
endmodule
