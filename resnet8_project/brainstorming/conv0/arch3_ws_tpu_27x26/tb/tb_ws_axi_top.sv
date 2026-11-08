`timescale 1ns/1ps
// Self-contained directed integration tests. Written, not yet run in HDL tools.
module tb_ws_axi_top;
    logic aclk = 0;
    always #5 aclk = ~aclk;
    logic aresetn = 0;
    logic [7:0] s_axi_awaddr = 0, s_axi_araddr = 0;
    logic [2:0] s_axi_awprot = 0, s_axi_arprot = 0;
    logic s_axi_awvalid = 0, s_axi_awready;
    logic [31:0] s_axi_wdata = 0;
    logic [3:0] s_axi_wstrb = 0;
    logic s_axi_wvalid = 0, s_axi_wready;
    logic [1:0] s_axi_bresp, s_axi_rresp;
    logic s_axi_bvalid, s_axi_bready = 0;
    logic s_axi_arvalid = 0, s_axi_arready;
    logic [31:0] s_axi_rdata;
    logic s_axi_rvalid, s_axi_rready = 0;
    logic [255:0] s_axis_tdata = 0, m_axis_tdata;
    logic [31:0] s_axis_tkeep = 0, m_axis_tkeep;
    logic s_axis_tlast = 0, s_axis_tvalid = 0, s_axis_tready;
    logic m_axis_tlast, m_axis_tvalid, m_axis_tready = 0;
    ws_axi_top dut (.*);

    logic checking = 0, force_stall = 0;
    logic [31:0] lfsr = 32'h12345678;
    int expected_vectors = 0, received = 0, total_checked = 0;
    logic stalled = 0;
    logic [288:0] held_beat;

    function automatic int weight_value(input int r, c);
        return ((r*13+c*7)%255)-127;
    endfunction
    function automatic int input_value(input int p, r);
        return ((p*11+r*3)%255)-127;
    endfunction
    function automatic int golden(input int p, c);
        int acc;
        acc = 0;
        for (int r = 0; r < 27; r++) acc += input_value(p,r)*weight_value(r,c);
        return acc;
    endfunction

    always @(negedge aclk) begin
        lfsr = {lfsr[30:0], lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
        m_axis_tready = aresetn && !force_stall && (lfsr[0] || lfsr[3]);
    end
    always @(posedge aclk) begin : output_scoreboard
        int words;
        logic [31:0] want_keep;
        if (!aresetn || !checking) begin
            received = 0; stalled = 0;
        end else begin
            if (stalled && (!m_axis_tvalid ||
                {m_axis_tdata,m_axis_tkeep,m_axis_tlast} !== held_beat))
                $fatal(1, "Output changed under backpressure");
            stalled = m_axis_tvalid && !m_axis_tready;
            held_beat = {m_axis_tdata,m_axis_tkeep,m_axis_tlast};
            if (m_axis_tvalid && m_axis_tready) begin
                words = expected_vectors*26-received;
                if (words > 8) words = 8;
                if (words <= 0) $fatal(1, "Extra result beat");
                want_keep = 0;
                for (int w = 0; w < words; w++) want_keep[w*4 +: 4] = 4'hf;
                if (m_axis_tkeep !== want_keep) $fatal(1, "Bad output KEEP");
                if (m_axis_tlast !== (received+words == expected_vectors*26))
                    $fatal(1, "Bad output LAST");
                for (int w = 0; w < words; w++) begin
                    if ($signed(m_axis_tdata[w*32 +: 32]) !== golden(received/26,received%26))
                        $fatal(1, "Result %0d mismatch", received);
                    received++; total_checked++;
                end
            end
        end
    end

    // Different AW/W delays test both arrival orders; response is held for 3 clocks.
    task automatic write_reg(input logic [7:0] addr, input logic [31:0] data,
                             input int aw_delay, w_delay,
                             input logic [1:0] expected_resp = 0,
                             input logic [3:0] strobes = 4'hf);
        fork
            begin
                repeat (aw_delay+1) @(negedge aclk);
                s_axi_awaddr = addr; s_axi_awvalid = 1;
                do @(posedge aclk); while (!s_axi_awready);
                @(negedge aclk); s_axi_awvalid = 0;
            end
            begin
                repeat (w_delay+1) @(negedge aclk);
                s_axi_wdata = data; s_axi_wstrb = strobes; s_axi_wvalid = 1;
                do @(posedge aclk); while (!s_axi_wready);
                @(negedge aclk); s_axi_wvalid = 0;
            end
        join
        wait (s_axi_bvalid);
        repeat (3) begin
            @(negedge aclk);
            if (!s_axi_bvalid || s_axi_bresp !== expected_resp)
                $fatal(1, "Bad/unstable write response for %h", addr);
        end
        s_axi_bready = 1;
        @(posedge aclk); @(negedge aclk); s_axi_bready = 0;
    endtask

    task automatic read_reg(input logic [7:0] addr, output logic [31:0] value,
                            input logic [1:0] expected_resp = 0);
        @(negedge aclk); s_axi_araddr = addr; s_axi_arvalid = 1;
        do @(posedge aclk); while (!s_axi_arready);
        @(negedge aclk); s_axi_arvalid = 0;
        wait (s_axi_rvalid);
        value = s_axi_rdata;
        repeat (3) begin
            @(negedge aclk);
            if (!s_axi_rvalid || s_axi_rdata !== value || s_axi_rresp !== expected_resp)
                $fatal(1, "Bad/unstable read response");
        end
        s_axi_rready = 1;
        @(posedge aclk); @(negedge aclk); s_axi_rready = 0;
    endtask

    task automatic wait_status(input int bit_number);
        logic [31:0] value;
        value = 0;
        for (int tries = 0; tries < 5000; tries++) begin
            read_reg(8'h04, value);
            if (value[5]) $fatal(1, "Unexpected accelerator error: %h", value);
            if (value[bit_number]) return;
        end
        $fatal(1, "Status bit %0d timeout", bit_number);
    endtask

    task automatic send_packet(input bit kind, input int n,
                               input bit bad_keep = 0, bad_last = 0);
        int length, count, value;
        length = kind ? n*27 : 702;
        for (int offset = 0; offset < length; offset += 32) begin
            repeat (1+(offset%3)) @(negedge aclk);
            count = length-offset; if (count > 32) count = 32;
            s_axis_tdata = 0; s_axis_tkeep = 0;
            for (int b = 0; b < count; b++) begin
                value = kind ? input_value((offset+b)/27,(offset+b)%27) :
                               weight_value((offset+b)/26,(offset+b)%26);
                s_axis_tdata[b*8 +: 8] = value[7:0];
                s_axis_tkeep[b] = 1;
            end
            s_axis_tlast = offset+count == length;
            if (bad_keep) s_axis_tkeep[0] = 0;
            if (bad_last) s_axis_tlast = !s_axis_tlast;
            s_axis_tvalid = 1;
            do @(posedge aclk); while (!s_axis_tready);
            @(negedge aclk); s_axis_tvalid = 0;
            if (bad_keep || bad_last) return;
        end
    endtask

    task automatic run_tile(input int n, input bit load_weights);
        logic [31:0] value;
        write_reg(8'h0c, n, 0, 4, 0, 4'b0001);
        if (load_weights) begin
            write_reg(8'h08, 0, 4, 0);
            send_packet(0,n);
            wait_status(1);
        end
        write_reg(8'h08, 1, 0, 0);
        send_packet(1,n);
        wait_status(2);
        @(negedge aclk); expected_vectors = n; checking = 1; force_stall = 1;
        write_reg(8'h00, 1, 3, 0);
        // Reject changing dimensions while engine is running.
        write_reg(8'h0c, 2, 0, 0, 2'b10);
        wait_status(3);
        repeat (25) @(negedge aclk);
        force_stall = 0;
        wait_status(4);
        if (received != n*26) $fatal(1, "Missing output values");
        read_reg(8'h1c, value);
        if (value != n*104) $fatal(1, "Output byte counter mismatch");
        @(negedge aclk); checking = 0;
        repeat (2) @(negedge aclk);
    endtask

    initial begin : tests
        logic [31:0] value;
        repeat (5) @(negedge aclk); aresetn = 1;
        read_reg(8'h20, value);
        if (value != 32'h57530001) $fatal(1, "Wrong interface ID");
        write_reg(8'h00, 1, 0, 4, 2'b10); // Start without payloads.
        write_reg(8'h0c, 0, 4, 0, 2'b10);
        write_reg(8'h0c, 65, 0, 0, 2'b10);
        read_reg(8'h03, value, 2'b10);
        run_tile(1,1);   // Partial input and output beats.
        run_tile(64,0);  // Warm run reuses weight memory, exercises full capacity.
        run_tile(3,0);   // Different N and partial final output.
        write_reg(8'h08, 0, 0, 0);
        send_packet(0,1,1,0);
        read_reg(8'h10, value);
        if (!value[0]) $fatal(1, "Bad KEEP was not detected");
        write_reg(8'h00, 2, 0, 3); // Reset must retain its AXI write response.
        read_reg(8'h04, value);
        if (value != 0) $fatal(1, "Soft reset did not clear status");
        send_packet(0,1,0,1);
        read_reg(8'h10, value);
        if (!value[1]) $fatal(1, "Bad LAST was not detected");
        write_reg(8'h00, 2, 3, 0);
        run_tile(2,1);
        $display("PASS tb_ws_axi_top: %0d INT32 comparisons, AXI ordering/stalls/errors", total_checked);
        $finish;
    end
    initial begin
        repeat (200000) @(posedge aclk);
        $fatal(1, "Global test timeout");
    end
endmodule
