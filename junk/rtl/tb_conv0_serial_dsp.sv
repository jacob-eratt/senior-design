`timescale 1ns/1ps
`default_nettype none

// Behavioral, self-checking testbench. No vendor primitives or UVM required.
// Default: three complete synthetic frames plus reset/abort tests.
// Optional: +ARTIFACT_DIR=C:/.../resnet8/artifacts (or parameter below)
// enables a fourth frame using the full 26-channel Python Conv0 fixture.
// Run ALL: one frame takes millions of clocks, not 1000 ns.
module tb_conv0_serial_dsp;
    parameter string ARTIFACT_DIR = "";
    parameter integer MAX_FRAME_CYCLES = 25000000;
    parameter logic [31:0] SEED = 32'h2608cafe;

    localparam integer NOUT = 32 * 32 * 26;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, start = 0;
    wire busy, done;
    wire [2:0] req;
    reg [2:0] ready = 0, valid = 0;
    wire [11:0] image_addr;
    wire [9:0] weight_addr;
    wire [4:0] bias_addr;
    wire [11:0] address [0:2];
    assign address[0] = image_addr;
    assign address[1] = {2'b0, weight_addr};
    assign address[2] = {7'b0, bias_addr};
    reg signed [7:0] image_data = 0, weight_data = 0;
    reg signed [31:0] bias_data = 0;
    wire out_valid;
    reg out_ready = 0;
    wire [14:0] out_addr;
    wire signed [31:0] out_data;

    conv0_serial_dsp dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .image_req(req[0]), .image_addr(image_addr),
        .image_ready(ready[0]), .image_valid(valid[0]), .image_data(image_data),
        .weight_req(req[1]), .weight_addr(weight_addr),
        .weight_ready(ready[1]), .weight_valid(valid[1]), .weight_data(weight_data),
        .bias_req(req[2]), .bias_addr(bias_addr),
        .bias_ready(ready[2]), .bias_valid(valid[2]), .bias_data(bias_data),
        .out_valid(out_valid), .out_ready(out_ready),
        .out_addr(out_addr), .out_data(out_data)
    );

    reg signed [7:0] image_mem [0:3071];
    reg signed [7:0] weight_mem [0:701];
    reg signed [31:0] bias_mem [0:25];
    reg signed [31:0] expected [0:NOUT-1];
    reg signed [31:0] fixture_expected [0:NOUT-1];
    bit seen [0:NOUT-1];
    integer channel_counts [0:25];

    // All stimulus/monitoring is sequenced through tick(). This avoids races
    // between independent drivers and scoreboards. Inputs change at negedge;
    // transactions are checked at posedge; registered results after #1.
    integer memory_mode = 0; // 0: immediate, 1: one-cycle, 2: mixed/stalled
    integer pending_port = -1, pending_delay = 0;
    reg signed [31:0] pending_data;
    integer response_latency [0:2];
    bit [2:0] stalled_request = 0;
    reg [11:0] held_address [0:2];
    bit stalled_output = 0;
    reg [14:0] held_out_addr;
    reg signed [31:0] held_out_data;
    bit frame_active = 0;
    integer accepted = 0, frame_cycles = 0, frame_number = 0;
    integer final_stalls = 0;
    integer completed_frames = 0, python_frames = 0;
    longint unsigned total_cycles = 0, comparisons = 0;
    longint unsigned immediate_reads [0:2];
    longint unsigned delayed_reads [0:2];
    longint unsigned request_stalls [0:2];
    longint unsigned output_stalls = 0, busy_start_pulses = 0;
    reg [31:0] rng = SEED;
    string artifact_dir;
    string case_name = "startup";

    // Local deterministic PRNG: independent of simulator-specific $urandom state.
    function automatic reg [31:0] next_random();
        begin
            rng = rng ^ (rng << 13);
            rng = rng ^ (rng >> 17);
            rng = rng ^ (rng << 5);
            next_random = rng;
        end
    endfunction

    function automatic reg signed [31:0] read_memory(input integer port_id,
                                                     input integer addr);
        begin
            case (port_id)
                0: begin
                    if (addr < 0 || addr >= 3072) $fatal(1, "Image address out of range: %0d", addr);
                    read_memory = {{24{image_mem[addr][7]}}, image_mem[addr]};
                end
                1: begin
                    if (addr < 0 || addr >= 702) $fatal(1, "Weight address out of range: %0d", addr);
                    read_memory = {{24{weight_mem[addr][7]}}, weight_mem[addr]};
                end
                2: begin
                    if (addr < 0 || addr >= 26) $fatal(1, "Bias address out of range: %0d", addr);
                    read_memory = bias_mem[addr];
                end
                default: $fatal(1, "Bad testbench memory port");
            endcase
        end
    endfunction

    task automatic drive_data(input integer port_id, input reg signed [31:0] data);
        case (port_id)
            0: image_data = data[7:0];
            1: weight_data = data[7:0];
            2: bias_data = data;
        endcase
    endtask

    task automatic clear_frame();
        integer i;
        begin
            accepted = 0;
            frame_cycles = 0;
            final_stalls = 0;
            for (i = 0; i < NOUT; i = i + 1) seen[i] = 0;
            for (i = 0; i < 26; i = i + 1) channel_counts[i] = 0;
        end
    endtask

    task automatic tick(input bit reset_level, input bit start_level,
                        input bit force_output_stall);
        integer p, requests;
        bit final_transfer, idle_start;
        reg [31:0] random_bits;
        reg signed [31:0] fetched;
        begin
            @(negedge clk);
            rst = reset_level;
            start = start_level;
            ready = 0;
            valid = 0;
            // Poison data outside a valid response to catch premature sampling.
            image_data = 'x;
            weight_data = 'x;
            bias_data = 'x;
            out_ready = 0;
            if (!rst) begin
                random_bits = next_random();
                out_ready = !force_output_stall &&
                            ((memory_mode != 2) || (random_bits[1:0] != 0));
                // Directed stall on the final write, even in the immediate mode.
                if (out_valid && out_addr == NOUT-1 && final_stalls < 6) begin
                    out_ready = 0;
                    final_stalls = final_stalls + 1;
                end
                if (pending_port >= 0) begin
                    pending_delay = pending_delay - 1;
                    if (pending_delay == 0) begin
                        valid[pending_port] = 1;
                        drive_data(pending_port, pending_data);
                    end
                end else begin
                    for (p = 0; p < 3; p = p + 1) begin
                        random_bits = next_random();
                        ready[p] = (memory_mode != 2) || (random_bits[1:0] != 0);
                        response_latency[p] = (memory_mode == 0) ? 0 :
                                              (memory_mode == 1) ? 1 :
                                              int'(random_bits[3:2]);
                        if (req[p] && ready[p]) begin
                            if ($isunknown(address[p])) $fatal(1, "Unknown read address on port %0d", p);
                            fetched = read_memory(p, int'(address[p]));
                            if (response_latency[p] == 0) begin
                                valid[p] = 1;
                                drive_data(p, fetched);
                            end
                        end
                    end
                end
            end

            @(posedge clk);
            total_cycles = total_cycles + 1;
            final_transfer = 0;
            idle_start = 0;
            if (rst) begin
                // Reset aborts a frame and flushes the simulated memories too.
                pending_port = -1;
                stalled_request = 0;
                stalled_output = 0;
                frame_active = 0;
                clear_frame();
            end else begin
                if ($isunknown({req, out_valid, busy, done}))
                    $fatal(1, "Unknown control at cycle %0d", total_cycles);
                idle_start = start && !busy;
                if (idle_start) begin
                    if (frame_active) $fatal(1, "DUT became idle before completing frame");
                    clear_frame();
                    frame_active = 1;
                end else if (start && busy) busy_start_pulses = busy_start_pulses + 1;
                if (frame_active) begin
                    frame_cycles = frame_cycles + 1;
                    if (frame_cycles > MAX_FRAME_CYCLES)
                        $fatal(1, "Timeout in %s: %0d outputs", case_name, accepted);
                end
                requests = int'(req[0]) + int'(req[1]) + int'(req[2]);
                if (requests > 1) $fatal(1, "Multiple simultaneous memory requests");
                if (!frame_active && (requests != 0 || out_valid || busy))
                    $fatal(1, "Unexpected activity while idle");
                if (pending_port >= 0 && requests != 0)
                    $fatal(1, "New request while a response is outstanding");

                for (p = 0; p < 3; p = p + 1) begin
                    if (stalled_request[p] &&
                        (req[p] !== 1'b1 || address[p] !== held_address[p]))
                        $fatal(1, "Request/address changed during stall on port %0d", p);
                    stalled_request[p] = req[p] && !ready[p];
                    held_address[p] = address[p];
                    if (stalled_request[p]) request_stalls[p] = request_stalls[p] + 1;
                end

                if (pending_port >= 0) begin
                    if (valid[pending_port]) pending_port = -1;
                end else begin
                    for (p = 0; p < 3; p = p + 1) begin
                        if (req[p] && ready[p]) begin
                            fetched = read_memory(p, int'(address[p]));
                            if (valid[p]) immediate_reads[p] = immediate_reads[p] + 1;
                            else begin
                                delayed_reads[p] = delayed_reads[p] + 1;
                                pending_port = p;
                                pending_delay = response_latency[p];
                                pending_data = fetched;
                            end
                        end
                    end
                end

                if (stalled_output &&
                    (out_valid !== 1'b1 || out_addr !== held_out_addr || out_data !== held_out_data))
                    $fatal(1, "Output changed while backpressured");
                stalled_output = out_valid && !out_ready;
                held_out_addr = out_addr;
                held_out_data = out_data;
                if (stalled_output) output_stalls = output_stalls + 1;
                if (out_valid) begin
                    if ($isunknown({out_addr, out_data}) || out_addr >= NOUT)
                        $fatal(1, "Unknown/out-of-range output: addr=%h data=%h", out_addr, out_data);
                    if (out_ready) begin
                        if (seen[out_addr]) $fatal(1, "Duplicate output address %0d", out_addr);
                        if (out_data !== expected[out_addr])
                            $fatal(1, "%s: addr=%0d y=%0d x=%0d oc=%0d got=%0d expected=%0d",
                                   case_name, out_addr, out_addr/(32*26), (out_addr/26)%32,
                                   out_addr%26, $signed(out_data), $signed(expected[out_addr]));
                        seen[out_addr] = 1;
                        accepted = accepted + 1;
                        comparisons = comparisons + 1;
                        channel_counts[out_addr%26] = channel_counts[out_addr%26] + 1;
                        final_transfer = (accepted == NOUT);
                        if ((accepted % 4096) == 0)
                            $display("  %s: checked %0d/%0d outputs, cycles=%0d", case_name, accepted, NOUT, frame_cycles);
                    end
                end
            end

            #1; // Observe nonblocking DUT updates.
            if (rst) begin
                if ({busy, done, req, out_valid} !== 6'b0)
                    $fatal(1, "Reset did not clear busy/done/requests/output-valid");
            end else begin
                if (done !== final_transfer)
                    $fatal(1, "done must pulse exactly after the final accepted output");
                if (final_transfer) begin
                    if (busy !== 1'b0 || out_valid !== 1'b0 || req !== 3'b0 || pending_port >= 0)
                        $fatal(1, "Unexpected activity at completion");
                    for (p = 0; p < NOUT; p = p + 1)
                        if (!seen[p]) $fatal(1, "Missing output %0d", p);
                    for (p = 0; p < 26; p = p + 1)
                        if (channel_counts[p] != 1024) $fatal(1, "Wrong count for channel %0d", p);
                    frame_active = 0;
                    completed_frames = completed_frames + 1;
                end else if (frame_active && busy !== 1'b1)
                    $fatal(1, "busy dropped before completion");
            end
        end
    endtask

    // Independent scalar convolution: uses coordinate loops and 64-bit sums,
    // never DUT internals, its counters, or its emission order.
    task automatic compute_reference();
        integer y, x, oc, ky, kx, ic, sy, sx, addr;
        longint signed sum, a, w;
        begin
            for (y = 0; y < 32; y = y + 1)
                for (x = 0; x < 32; x = x + 1)
                    for (oc = 0; oc < 26; oc = oc + 1) begin
                        sum = $signed(bias_mem[oc]);
                        for (ky = 0; ky < 3; ky = ky + 1)
                            for (kx = 0; kx < 3; kx = kx + 1)
                                for (ic = 0; ic < 3; ic = ic + 1) begin
                                    sy = y + ky - 1;
                                    sx = x + kx - 1;
                                    if (sy >= 0 && sy < 32 && sx >= 0 && sx < 32) begin
                                        a = $signed(image_mem[(sy*32+sx)*3+ic]);
                                        w = $signed(weight_mem[((oc*3+ky)*3+kx)*3+ic]);
                                        sum = sum + a*w;
                                    end
                                end
                        if (sum > 64'sd2147483647 || sum < -64'sd2147483648)
                            $fatal(1, "Reference exceeds supported INT32 contract");
                        addr = (y*32+x)*26+oc;
                        expected[addr] = sum[31:0];
                    end
        end
    endtask

    task automatic fill_pattern(input integer pattern);
        integer i, oc, lane;
        reg [31:0] sample;
        begin
            for (i = 0; i < 3072; i = i + 1) begin
                sample = next_random();
                if (pattern == 0) image_mem[i] = 0;
                else if (pattern == 1) begin
                    case (i%5)
                        0: image_mem[i] = -128;
                        1: image_mem[i] = 127;
                        2: image_mem[i] = -1;
                        3: image_mem[i] = 1;
                        4: image_mem[i] = 0;
                    endcase
                end else image_mem[i] = sample[7:0];
            end
            for (oc = 0; oc < 26; oc = oc + 1) begin
                if (pattern == 0)
                    bias_mem[oc] = (oc%2) ? (32'sh7fffffff-oc) : (-32'sd2147483647+oc);
                else bias_mem[oc] = (oc-13)*12345;
                for (lane = 0; lane < 27; lane = lane + 1) begin
                    sample = next_random();
                    if (pattern == 1) begin
                        case ((oc+lane)%5)
                            0: weight_mem[oc*27+lane] = -128;
                            1: weight_mem[oc*27+lane] = 127;
                            2: weight_mem[oc*27+lane] = -1;
                            3: weight_mem[oc*27+lane] = 1;
                            4: weight_mem[oc*27+lane] = 0;
                        endcase
                    end else weight_mem[oc*27+lane] = sample[7:0];
                end
            end
            compute_reference();
        end
    endtask

    task automatic require_file(input string path);
        integer fd;
        begin
            fd = $fopen(path, "r");
            if (fd == 0) $fatal(1, "Cannot open fixture: %s", path);
            $fclose(fd);
        end
    endtask

    task automatic load_python_fixture();
        integer i;
        begin
            require_file({artifact_dir, "/layer_vectors/conv0/input.mem"});
            require_file({artifact_dir, "/export/conv0_weights.mem"});
            require_file({artifact_dir, "/export/conv0_bias.mem"});
            require_file({artifact_dir, "/layer_vectors/conv0/expected_acc.mem"});
            for (i = 0; i < 3072; i = i + 1) image_mem[i] = 'x;
            for (i = 0; i < 702; i = i + 1) weight_mem[i] = 'x;
            for (i = 0; i < 26; i = i + 1) bias_mem[i] = 'x;
            for (i = 0; i < NOUT; i = i + 1) fixture_expected[i] = 'x;
            $readmemh({artifact_dir, "/layer_vectors/conv0/input.mem"}, image_mem);
            $readmemh({artifact_dir, "/export/conv0_weights.mem"}, weight_mem);
            $readmemh({artifact_dir, "/export/conv0_bias.mem"}, bias_mem);
            $readmemh({artifact_dir, "/layer_vectors/conv0/expected_acc.mem"}, fixture_expected);
            for (i = 0; i < 3072; i = i + 1)
                if ($isunknown(image_mem[i])) $fatal(1, "Missing/unknown input word %0d", i);
            for (i = 0; i < 702; i = i + 1)
                if ($isunknown(weight_mem[i])) $fatal(1, "Missing/unknown weight word %0d", i);
            for (i = 0; i < 26; i = i + 1)
                if ($isunknown(bias_mem[i])) $fatal(1, "Missing/unknown bias word %0d", i);
            compute_reference();
            for (i = 0; i < NOUT; i = i + 1) begin
                if ($isunknown(fixture_expected[i]) || fixture_expected[i] !== expected[i])
                    $fatal(1, "Python fixture differs from scalar reference at %0d", i);
                expected[i] = fixture_expected[i];
            end
        end
    endtask

    task automatic run_frame(input string name, input integer mode);
        begin
            case_name = name;
            memory_mode = mode;
            frame_number = frame_number + 1;
            $display("Frame %0d: %s (memory mode %0d)", frame_number, case_name, memory_mode);
            tick(0, 1, 0);
            while (!done) begin
                // start pulses while busy must not restart the engine.
                tick(0, (frame_cycles == 20 || frame_cycles == 200), 0);
            end
            $display("  PASS frame: %0d outputs, %0d clocks", accepted, frame_cycles);
            repeat (4) tick(0, 0, 0); // done is one clock; stay idle, no extra writes.
        end
    endtask

    task automatic abort_during_read(input integer port_id);
        integer guard;
        begin
            case_name = "reset during delayed read";
            memory_mode = 1;
            tick(0, 1, 0);
            guard = 0;
            while (pending_port != port_id) begin
                tick(0, 0, 0);
                guard = guard + 1;
                if (guard > 1000) $fatal(1, "Did not reach read port %0d for reset test", port_id);
            end
            tick(1, 1, 0); // reset has priority over start; flush outstanding read.
            repeat (2) tick(0, 0, 0);
        end
    endtask

    integer init_i;
    initial begin
        artifact_dir = ARTIFACT_DIR;
        if ($value$plusargs("ARTIFACT_DIR=%s", artifact_dir)) begin end
        if ($value$plusargs("SEED=%h", rng)) begin end
        if (rng == 0) $fatal(1, "SEED must be nonzero for xorshift PRNG");
        for (init_i = 0; init_i < 3; init_i = init_i + 1) begin
            immediate_reads[init_i] = 0;
            delayed_reads[init_i] = 0;
            request_stalls[init_i] = 0;
        end
        $display("Starting tb_conv0_serial_dsp: seed=%h; full frames have %0d outputs", rng, NOUT);
        repeat (2) tick(1, 1, 0);
        repeat (2) tick(0, 0, 0);

        fill_pattern(0);
        abort_during_read(2);
        abort_during_read(0);
        abort_during_read(1);
        case_name = "reset during stalled output";
        tick(0, 1, 1);
        while (!out_valid) tick(0, 0, 1);
        repeat (3) tick(0, 0, 1);
        tick(1, 0, 1);
        tick(0, 0, 0);
        $display("PASS reset/abort tests; starting full-frame comparisons");

        run_frame("zero input, extreme signed biases", 0);
        fill_pattern(1);
        run_frame("signed INT8 boundaries and padding", 1);
        fill_pattern(2);
        run_frame("seeded random frame with memory/output stalls", 2);
        if (artifact_dir != "") begin
            load_python_fixture();
            run_frame("Python full Conv0 golden fixture", 2);
            python_frames = 1;
        end else $display("SKIPPED Python fixture: set ARTIFACT_DIR to resnet8/artifacts to enable it.");

        for (init_i = 0; init_i < 3; init_i = init_i + 1) begin
            if (immediate_reads[init_i] == 0 || delayed_reads[init_i] == 0 || request_stalls[init_i] == 0)
                $fatal(1, "Missing memory protocol coverage on port %0d", init_i);
            $display("Port %0d: immediate=%0d delayed=%0d stalled=%0d", init_i,
                     immediate_reads[init_i], delayed_reads[init_i], request_stalls[init_i]);
        end
        if (output_stalls == 0 || busy_start_pulses == 0) $fatal(1, "Missing control/stall coverage");
        tick(1, 0, 0);
        tick(0, 0, 0);
        $display("PASS: tb_conv0_serial_dsp frames=%0d comparisons=%0d cycles=%0d python_fixture=%0d",
                 completed_frames, comparisons, total_cycles, python_frames);
        $finish;
    end

    initial begin
        #2000000000; // 2 simulated seconds = 200 million 100-MHz clocks.
        $fatal(1, "Global simulation watchdog expired");
    end
endmodule

`default_nettype wire
