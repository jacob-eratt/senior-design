`timescale 1ns/1ps
`default_nettype none

// Self-checking behavioral TB for conv0_416_dsp. Supplied, NOT run here.
// Default: five full synthetic frames, plus reset/abort cases.
// Optional parameter or +ARTIFACT_DIR=... adds the full Python Conv0 fixture.
// Select tb_conv0_416_dsp as simulation top and use run all.
module tb_conv0_416_dsp;
    parameter integer OUTPUT_LANES = 4;
    parameter string ARTIFACT_DIR = "";
    parameter logic [31:0] SEED = 32'h4168cafe;
    parameter integer MAX_FRAME_CYCLES = 200000;
    localparam integer NOUT = 26624;

    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, start = 0;
    wire busy, done, load_ready;
    reg image_we = 0, weight_we = 0, bias_we = 0;
    reg [11:0] image_addr = 0;
    reg [9:0] weight_addr = 0;
    reg [4:0] bias_addr = 0;
    reg signed [7:0] image_data = 0, weight_data = 0;
    reg signed [31:0] bias_data = 0;
    wire out_valid;
    reg out_ready = 0;
    wire [14:0] out_addr;
    wire [OUTPUT_LANES*32-1:0] out_data;
    conv0_416_dsp #(.OUTPUT_LANES(OUTPUT_LANES)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .load_ready(load_ready), .image_we(image_we), .image_addr(image_addr),
        .image_data(image_data), .weight_we(weight_we), .weight_addr(weight_addr),
        .weight_data(weight_data), .bias_we(bias_we), .bias_addr(bias_addr),
        .bias_data(bias_data), .out_valid(out_valid), .out_ready(out_ready),
        .out_addr(out_addr), .out_data(out_data)
    );

    reg signed [7:0] image_mem [0:3071];
    reg signed [7:0] weight_mem [0:701];
    reg signed [31:0] bias_mem [0:25];
    reg signed [31:0] expected [0:NOUT-1];
    reg signed [31:0] fixture_expected [0:NOUT-1];
    bit seen [0:NOUT-1];

    // Commands are copied to DUT pins ONLY on falling edges in tick().
    bit cmd_reset = 1, cmd_start = 0, cmd_hold = 0;
    bit cmd_image_we = 0, cmd_weight_we = 0, cmd_bias_we = 0;
    reg [11:0] cmd_image_addr = 0;
    reg [9:0] cmd_weight_addr = 0;
    reg [4:0] cmd_bias_addr = 0;
    reg signed [7:0] cmd_image_data = 0, cmd_weight_data = 0;
    reg signed [31:0] cmd_bias_data = 0;
    bit random_stalls = 0, active_frame = 0, held_output = 0;
    reg [14:0] held_addr;
    reg [OUTPUT_LANES*32-1:0] held_data;
    integer accepted = 0, frame_cycles = 0, final_stalls = 0;
    integer completed_frames = 0, python_frames = 0;
    longint unsigned cycles = 0, comparisons = 0, stalled_cycles = 0;
    longint unsigned ignored_busy_writes = 0, ignored_busy_starts = 0;
    reg [31:0] rng = SEED;
    string artifact_dir, case_name = "startup";

    function automatic reg [31:0] next_random();
        begin
            rng = rng ^ (rng << 13);
            rng = rng ^ (rng >> 17);
            rng = rng ^ (rng << 5);
            next_random = rng;
        end
    endfunction

    task automatic clear_writes();
        begin
            cmd_image_we = 0;
            cmd_weight_we = 0;
            cmd_bias_we = 0;
        end
    endtask

    task automatic clear_frame();
        integer i;
        begin
            accepted = 0;
            frame_cycles = 0;
            final_stalls = 0;
            for (i = 0; i < NOUT; i = i + 1) seen[i] = 0;
        end
    endtask

    task automatic tick();
        integer i, index;
        reg signed [31:0] value;
        reg [31:0] random_bits;
        bit final_transfer;
        begin
            @(negedge clk);
            rst = cmd_reset;
            start = cmd_start;
            image_we = cmd_image_we;
            weight_we = cmd_weight_we;
            bias_we = cmd_bias_we;
            image_addr = cmd_image_addr;
            weight_addr = cmd_weight_addr;
            bias_addr = cmd_bias_addr;
            image_data = cmd_image_data;
            weight_data = cmd_weight_data;
            bias_data = cmd_bias_data;
            random_bits = next_random();
            out_ready = !cmd_hold && (!random_stalls || random_bits[1:0] != 0);
            if (out_valid && out_addr == NOUT-OUTPUT_LANES && final_stalls < 6) begin
                out_ready = 0;
                final_stalls = final_stalls + 1;
            end
            @(posedge clk);
            cycles = cycles + 1;
            final_transfer = 0;
            if (rst) begin
                active_frame = 0;
                held_output = 0;
                clear_frame();
            end else begin
                if ($isunknown({busy, done, out_valid, load_ready}))
                    $fatal(1, "Unknown control in %s", case_name);
                if (load_ready !== (!busy && !start))
                    $fatal(1, "Unexpected load_ready; writes must be blocked during start/busy");
                if (start && !busy) begin
                    if (active_frame) $fatal(1, "Idle before completing previous frame");
                    clear_frame();
                    active_frame = 1;
                end else if (start && busy) ignored_busy_starts = ignored_busy_starts + 1;
                if (busy && (image_we || weight_we || bias_we))
                    ignored_busy_writes = ignored_busy_writes + 1;
                if (active_frame) begin
                    frame_cycles = frame_cycles + 1;
                    if (frame_cycles > MAX_FRAME_CYCLES)
                        $fatal(1, "Frame timeout: %s, checked %0d outputs", case_name, accepted);
                end else if (busy || out_valid) $fatal(1, "Unexpected activity while idle");

                if (held_output &&
                    (out_valid !== 1'b1 || out_addr !== held_addr || out_data !== held_data))
                    $fatal(1, "Output changed during backpressure");
                held_output = out_valid && !out_ready;
                held_addr = out_addr;
                held_data = out_data;
                if (held_output) stalled_cycles = stalled_cycles + 1;
                if (out_valid) begin
                    if ($isunknown({out_addr, out_data}) || out_addr > NOUT-OUTPUT_LANES)
                        $fatal(1, "Unknown/out-of-range output at cycle %0d", cycles);
                    if (out_ready) begin
                        if (out_addr != accepted)
                            $fatal(1, "Output base address %0d, expected %0d", out_addr, accepted);
                        for (i = 0; i < OUTPUT_LANES; i = i + 1) begin
                            index = int'(out_addr) + i;
                            value = $signed(out_data[i*32 +: 32]);
                            if (seen[index]) $fatal(1, "Duplicate output %0d", index);
                            if (value !== expected[index])
                                $fatal(1, "%s: addr=%0d y=%0d x=%0d oc=%0d got=%0d expected=%0d",
                                       case_name, index, index/(32*26), (index/26)%32,
                                       index%26, value, $signed(expected[index]));
                            seen[index] = 1;
                            comparisons = comparisons + 1;
                        end
                        accepted = accepted + OUTPUT_LANES;
                        final_transfer = (accepted == NOUT);
                    end
                end
            end
            #1;
            if (rst) begin
                if ({busy, done, out_valid, load_ready} !== 4'b0)
                    $fatal(1, "Reset failed to clear control outputs");
            end else begin
                if (done !== final_transfer)
                    $fatal(1, "done must pulse exactly after the last accepted beat");
                if (final_transfer) begin
                    if (busy !== 1'b0 || out_valid !== 1'b0)
                        $fatal(1, "Unexpected activity at completion");
                    for (i = 0; i < NOUT; i = i + 1)
                        if (!seen[i]) $fatal(1, "Missing output %0d", i);
                    active_frame = 0;
                    completed_frames = completed_frames + 1;
                end else if (active_frame && busy !== 1'b1)
                    $fatal(1, "busy dropped before final output");
            end
        end
    endtask

    // Scalar 64-bit oracle. Does not inspect DUT hierarchy or reuse its tile
    // counters, memory copies, pipeline, or output mux.
    task automatic compute_reference();
        integer y, x, oc, ky, kx, ic, sy, sx;
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
                        if (sum < -64'sd2147483648 || sum > 64'sd2147483647)
                            $fatal(1, "Reference overflow outside INT32 model contract");
                        expected[(y*32+x)*26+oc] = sum[31:0];
                    end
        end
    endtask

    task automatic fill_pattern(input integer pattern);
        integer i, oc, lane, x, y, c;
        reg [31:0] r;
        begin
            for (i = 0; i < 3072; i = i + 1) begin
                r = next_random();
                case (pattern)
                    0: image_mem[i] = 0;
                    1: case (i%5)
                        0: image_mem[i] = -128;
                        1: image_mem[i] = 127;
                        2: image_mem[i] = -1;
                        3: image_mem[i] = 1;
                        4: image_mem[i] = 0;
                    endcase
                    2: image_mem[i] = r[7:0];
                    3: image_mem[i] = 0;
                endcase
            end
            // Sparse anchors at borders and on both sides of the x=15/16 tile
            // boundary; distinct signs/channels expose lane/sample ordering.
            if (pattern == 3) begin
                for (y = 0; y < 32; y = y + 1)
                    for (x = 0; x < 32; x = x + 1)
                        if ((y == 0 || y == 15 || y == 16 || y == 31) &&
                            (x == 0 || x == 15 || x == 16 || x == 31))
                            for (c = 0; c < 3; c = c + 1)
                                image_mem[(y*32+x)*3+c] = ((x+y+c)%2) ? 127-c : -128+c;
            end
            for (oc = 0; oc < 26; oc = oc + 1) begin
                if (pattern == 0)
                    bias_mem[oc] = (oc%2) ? (32'sh7fffffff-oc) : (32'sh80000000+oc);
                else bias_mem[oc] = (oc-13)*12345;
                for (lane = 0; lane < 27; lane = lane + 1) begin
                    r = next_random();
                    if (pattern == 1) begin
                        case ((lane+oc)%5)
                            0: weight_mem[oc*27+lane] = -128;
                            1: weight_mem[oc*27+lane] = 127;
                            2: weight_mem[oc*27+lane] = -1;
                            3: weight_mem[oc*27+lane] = 1;
                            4: weight_mem[oc*27+lane] = 0;
                        endcase
                    end else weight_mem[oc*27+lane] = r[7:0];
                end
            end
            compute_reference();
        end
    endtask

    task automatic preload();
        integer i;
        begin
            cmd_reset = 0;
            cmd_start = 0;
            cmd_hold = 0;
            for (i = 0; i < 3072; i = i + 1) begin
                cmd_image_we = 1;
                cmd_image_addr = i;
                cmd_image_data = image_mem[i];
                cmd_weight_we = (i < 702);
                if (i < 702) begin
                    cmd_weight_addr = i;
                    cmd_weight_data = weight_mem[i];
                end
                cmd_bias_we = (i < 26);
                if (i < 26) begin
                    cmd_bias_addr = i;
                    cmd_bias_data = bias_mem[i];
                end
                tick();
                if (!load_ready) $fatal(1, "Preload was not accepted");
            end
            // Invalid addresses must not overwrite valid memory locations.
            cmd_image_addr = 4095;
            cmd_weight_we = 1;
            cmd_weight_addr = 1023;
            cmd_bias_we = 1;
            cmd_bias_addr = 31;
            tick();
            clear_writes();
            tick();
        end
    endtask

    task automatic poison_writes();
        begin
            cmd_image_we = 1; cmd_image_addr = 0; cmd_image_data = 77;
            cmd_weight_we = 1; cmd_weight_addr = 0; cmd_weight_data = -37;
            cmd_bias_we = 1; cmd_bias_addr = 0; cmd_bias_data = 32'sd987654;
        end
    endtask

    task automatic run_frame(input string name, input bit with_stalls);
        begin
            case_name = name;
            random_stalls = with_stalls;
            cmd_hold = 0;
            $display("Frame %0d: %s", completed_frames+1, case_name);
            // Start has priority over these writes. They must be ignored.
            poison_writes();
            cmd_start = 1;
            tick();
            cmd_start = 0;
            clear_writes();
            while (!done) begin
                if (frame_cycles == 8 || frame_cycles == 40) begin
                    cmd_start = 1;
                    poison_writes();
                end else begin
                    cmd_start = 0;
                    clear_writes();
                end
                tick();
            end
            $display("  PASS frame: %0d INT32 results, %0d clocks (excludes preload)", accepted, frame_cycles);
            cmd_start = 0;
            clear_writes();
            repeat (4) tick();
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

    task automatic load_python();
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
                if ($isunknown(image_mem[i])) $fatal(1, "Missing/unknown image word %0d", i);
            for (i = 0; i < 702; i = i + 1)
                if ($isunknown(weight_mem[i])) $fatal(1, "Missing/unknown weight word %0d", i);
            for (i = 0; i < 26; i = i + 1)
                if ($isunknown(bias_mem[i])) $fatal(1, "Missing/unknown bias word %0d", i);
            compute_reference();
            for (i = 0; i < NOUT; i = i + 1) begin
                if ($isunknown(fixture_expected[i]) || fixture_expected[i] !== expected[i])
                    $fatal(1, "Python fixture differs from scalar oracle at %0d", i);
                expected[i] = fixture_expected[i];
            end
        end
    endtask

    initial begin
        if (OUTPUT_LANES < 1 || OUTPUT_LANES > 416 || (416%OUTPUT_LANES) != 0)
            $fatal(1, "OUTPUT_LANES must be a positive divisor of 416");
        artifact_dir = ARTIFACT_DIR;
        if ($value$plusargs("ARTIFACT_DIR=%s", artifact_dir)) begin end
        if ($value$plusargs("SEED=%h", rng)) begin end
        if (rng == 0) $fatal(1, "SEED must be nonzero");
        $display("Starting tb_conv0_416_dsp: seed=%h OUTPUT_LANES=%0d", rng, OUTPUT_LANES);
        cmd_start = 1; // Reset wins over start.
        repeat (2) tick();
        cmd_reset = 0;
        cmd_start = 0;
        repeat (2) tick();
        fill_pattern(0);
        preload();

        case_name = "reset during active compute";
        cmd_start = 1;
        tick();
        cmd_start = 0;
        repeat (10) tick();
        cmd_reset = 1;
        cmd_start = 1;
        poison_writes(); // Reset blocks writes, preserving the preload.
        tick();
        cmd_reset = 0;
        cmd_start = 0;
        clear_writes();
        tick();

        case_name = "reset while output stalled";
        cmd_hold = 1;
        cmd_start = 1;
        tick();
        cmd_start = 0;
        while (!out_valid) tick();
        repeat (3) tick();
        cmd_reset = 1;
        tick();
        cmd_reset = 0;
        cmd_hold = 0;
        tick();
        $display("PASS reset/abort tests; preloaded memories retained");

        run_frame("zero input and extreme INT32 biases", 0);
        fill_pattern(1);
        preload();
        run_frame("signed INT8 extremes and padding", 0);
        fill_pattern(2);
        preload();
        run_frame("seeded random frame with output stalls", 1);
        run_frame("restart without reloading, ignored busy writes", 1);
        fill_pattern(3);
        preload();
        run_frame("impulses at image edges and spatial tile boundaries", 1);
        if (artifact_dir != "") begin
            load_python();
            preload();
            run_frame("Python full Conv0 golden fixture", 1);
            python_frames = 1;
        end else $display("SKIPPED Python fixture: set ARTIFACT_DIR to resnet8/artifacts to enable it.");
        if (stalled_cycles == 0 || ignored_busy_writes == 0 || ignored_busy_starts == 0)
            $fatal(1, "Missing stall/busy-control coverage");
        cmd_reset = 1;
        tick();
        cmd_reset = 0;
        tick();
        $display("PASS: tb_conv0_416_dsp frames=%0d comparisons=%0d cycles=%0d python_fixture=%0d",
                 completed_frames, comparisons, cycles, python_frames);
        $finish;
    end

    initial begin
        #100000000; // 100 ms simulated time, not wall-clock time.
        $fatal(1, "Global testbench watchdog expired");
    end
endmodule

`default_nettype wire
