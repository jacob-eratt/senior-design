`timescale 1ns/1ps
`default_nettype none

// Add this file as a SystemVerilog SIMULATION source, not a design source.
// Simulation top: tb_conv_1024. Use Run All (the suite exceeds 1000 ns).
// The module name below must match the declaration INSIDE your top RTL file.
// If you renamed "module conv_1024_pixels" to "module conv_1024", change
// the next define to conv_1024. File names alone do not affect instantiation.
`ifndef CONV_DUT
`define CONV_DUT conv_1024_pixels
`endif

module tb_conv_1024;
    // Optional: put an absolute path with forward slashes here, or pass
    // +FIXTURE_DIR=<path> (XSim: -testplusarg FIXTURE_DIR=<path>).
    // The folder must contain input.mem, weights.mem, bias.mem, expected_acc.mem.
    parameter string FIXTURE_DIR = "";
    parameter integer RANDOM_CASES = 64;

    localparam integer H = 32;
    localparam integer W = 32;
    localparam integer CHANNELS = 3;
    localparam integer MAX_ERROR_MESSAGES = 20;
    localparam longint signed INT32_MIN = -64'sd2147483648;
    localparam longint signed INT32_MAX =  64'sd2147483647;

    reg clk = 1'b0;
    always #5 clk = ~clk; // 100 MHz simulation stimulus; not a timing claim.

    reg rst = 1'b1;
    reg load_bias = 1'b0;
    reg enable = 1'b0;
    reg [24575:0] image_values = '0;
    reg [215:0] weight_values = '0;
    reg signed [31:0] bias = '0;
    wire [32767:0] acc_values;

    `CONV_DUT dut (
        .clk(clk), .rst(rst), .load_bias(load_bias), .enable(enable),
        .image_values(image_values), .weight_values(weight_values),
        .bias(bias), .acc_values(acc_values)
    );

    // Convenient signed signals to add to the Vivado waveform window.
    wire signed [31:0] out_top_left     = acc_values[0*32 +: 32];
    wire signed [31:0] out_top_right    = acc_values[31*32 +: 32];
    wire signed [31:0] out_center       = acc_values[(16*32+16)*32 +: 32];
    wire signed [31:0] out_bottom_left  = acc_values[(31*32)*32 +: 32];
    wire signed [31:0] out_bottom_right = acc_values[1023*32 +: 32];

    // Logical arrays keep the reference independent of DUT window wiring.
    integer image [0:H-1][0:W-1][0:CHANNELS-1];
    integer weights [0:2][0:2][0:CHANNELS-1];
    longint signed expected [0:H-1][0:W-1];
    reg reference_valid = 1'b0;
    integer errors = 0;
    integer checked_cycles = 0;
    integer pixel_comparisons = 0;
    integer random_cases;
    reg [31:0] rng_state = 32'h2608_cafe;
    string test_name = "initialization";
    string fixture_directory;
    bit fixture_ran = 1'b0;

    reg [7:0] fixture_image [0:3071];
    reg [7:0] fixture_weights [0:26];
    reg [31:0] fixture_bias [0:0];
    reg [31:0] fixture_acc [0:1023];

    // Explicit PRNG for repeatable signed stimuli, independent of simulator
    // default seeds. Override with +SEED=<hex> when needed.
    function automatic integer random_int8();
        begin
            rng_state = rng_state ^ (rng_state << 13);
            rng_state = rng_state ^ (rng_state >> 17);
            rng_state = rng_state ^ (rng_state << 5);
            random_int8 = integer'(rng_state[7:0]) - 128;
        end
    endfunction

    task automatic fill_image(input integer value);
        integer y, x, c;
        begin
            for (y = 0; y < H; y = y + 1)
                for (x = 0; x < W; x = x + 1)
                    for (c = 0; c < CHANNELS; c = c + 1)
                        image[y][x][c] = value;
        end
    endtask

    task automatic fill_weights(input integer value);
        integer ky, kx, c;
        begin
            for (ky = 0; ky < 3; ky = ky + 1)
                for (kx = 0; kx < 3; kx = kx + 1)
                    for (c = 0; c < CHANNELS; c = c + 1)
                        weights[ky][kx][c] = value;
        end
    endtask

    task automatic randomize_data();
        integer y, x, c, ky, kx;
        begin
            for (y = 0; y < H; y = y + 1)
                for (x = 0; x < W; x = x + 1)
                    for (c = 0; c < CHANNELS; c = c + 1)
                        image[y][x][c] = random_int8();
            for (ky = 0; ky < 3; ky = ky + 1)
                for (kx = 0; kx < 3; kx = kx + 1)
                    for (c = 0; c < CHANNELS; c = c + 1)
                        weights[ky][kx][c] = random_int8();
        end
    endtask

    task automatic pack_inputs();
        integer y, x, c, ky, kx, index;
        begin
            index = 0;
            for (y = 0; y < H; y = y + 1)
                for (x = 0; x < W; x = x + 1)
                    for (c = 0; c < CHANNELS; c = c + 1) begin
                        if (image[y][x][c] < -128 || image[y][x][c] > 127)
                            $fatal(1, "Testbench input outside signed INT8 range");
                        image_values[index*8 +: 8] = image[y][x][c];
                        index = index + 1;
                    end
            index = 0;
            for (ky = 0; ky < 3; ky = ky + 1)
                for (kx = 0; kx < 3; kx = kx + 1)
                    for (c = 0; c < CHANNELS; c = c + 1) begin
                        if (weights[ky][kx][c] < -128 || weights[ky][kx][c] > 127)
                            $fatal(1, "Testbench weight outside signed INT8 range");
                        weight_values[index*8 +: 8] = weights[ky][kx][c];
                        index = index + 1;
                    end
        end
    endtask

    // Scalar 64-bit software convolution, not a copy of the generated RTL.
    // We deliberately stay within the design's INT32 no-overflow contract.
    function automatic longint signed dot_product(input integer y, input integer x);
        integer dy, dx, c, iy, ix;
        longint signed sum, sample_value, weight_value;
        begin
            sum = 0;
            for (dy = -1; dy <= 1; dy = dy + 1)
                for (dx = -1; dx <= 1; dx = dx + 1) begin
                    iy = y + dy;
                    ix = x + dx;
                    if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                        for (c = 0; c < CHANNELS; c = c + 1) begin
                            sample_value = image[iy][ix][c];
                            weight_value = weights[dy+1][dx+1][c];
                            sum = sum + sample_value * weight_value;
                        end
                end
            dot_product = sum;
        end
    endfunction

    task automatic compare_outputs(input string label_text);
        integer y, x;
        reg signed [31:0] actual;
        reg signed [31:0] wanted;
        begin
            for (y = 0; y < H; y = y + 1)
                for (x = 0; x < W; x = x + 1) begin
                    actual = acc_values[(y*W+x)*32 +: 32];
                    wanted = expected[y][x];
                    pixel_comparisons = pixel_comparisons + 1;
                    // Case inequality also catches any X/Z in an output.
                    if (actual !== wanted) begin
                        errors = errors + 1;
                        if (errors <= MAX_ERROR_MESSAGES)
                            $display("ERROR %s t=%0t pixel=(%0d,%0d) expected=%0d actual=%0d hex=%08h",
                                     label_text, $time, y, x, wanted, actual, actual);
                    end
                end
        end
    endtask

    // Drive only on falling edges. Check old state before the rising edge
    // (including reset assertion), then wait for nonblocking assignments and
    // all combinational adder-tree delta cycles before checking new state.
    task automatic step(
        input bit reset_value, input bit load_value, input bit enable_value,
        input integer bias_value, input string label_text
    );
        integer y, x;
        begin
            @(negedge clk);
            test_name = label_text;
            rst = reset_value;
            load_bias = load_value;
            enable = enable_value;
            bias = bias_value;
            pack_inputs();
            #1;
            if (reference_valid)
                compare_outputs({label_text, " BEFORE rising edge"});
            @(posedge clk);
            for (y = 0; y < H; y = y + 1)
                for (x = 0; x < W; x = x + 1) begin
                    if (reset_value)
                        expected[y][x] = 0;
                    else if (load_value)
                        expected[y][x] = bias_value;
                    else if (enable_value)
                        expected[y][x] = expected[y][x] + dot_product(y, x);
                    if (expected[y][x] < INT32_MIN || expected[y][x] > INT32_MAX)
                        $fatal(1, "Testbench generated out-of-contract overflow in %s", label_text);
                end
            #1;
            reference_valid = 1'b1;
            checked_cycles = checked_cycles + 1;
            compare_outputs(label_text);
        end
    endtask

    task automatic fresh_convolution(input integer bias_value, input string label_text);
        begin
            step(0, 1, 0, bias_value, {label_text, " / load bias"});
            step(0, 0, 1, bias_value, {label_text, " / convolution"});
        end
    endtask

    // Independent hand-calculated anchors for padding and signed extremes.
    task automatic check_anchor(input integer y, input integer x, input integer wanted);
        reg signed [31:0] actual;
        begin
            actual = acc_values[(y*W+x)*32 +: 32];
            pixel_comparisons = pixel_comparisons + 1;
            if (actual !== wanted) begin
                errors = errors + 1;
                if (errors <= MAX_ERROR_MESSAGES)
                    $display("ERROR anchor %s pixel=(%0d,%0d) expected=%0d actual=%0d",
                             test_name, y, x, wanted, actual);
            end
        end
    endtask

    task automatic require_file(input string filename);
        integer handle;
        begin
            handle = $fopen(filename, "r");
            if (handle == 0)
                $fatal(1, "Cannot open fixture file: %s", filename);
            $fclose(handle);
        end
    endtask

    task automatic run_python_fixture(input string directory);
        integer y, x, c, ky, kx, i;
        begin
            require_file({directory, "/input.mem"});
            require_file({directory, "/weights.mem"});
            require_file({directory, "/bias.mem"});
            require_file({directory, "/expected_acc.mem"});
            $readmemh({directory, "/input.mem"}, fixture_image);
            $readmemh({directory, "/weights.mem"}, fixture_weights);
            $readmemh({directory, "/bias.mem"}, fixture_bias);
            $readmemh({directory, "/expected_acc.mem"}, fixture_acc);
            // Uninitialized/truncated/X-valued fixtures must not pass silently.
            for (i = 0; i < 3072; i = i + 1)
                if ((^fixture_image[i]) === 1'bx) $fatal(1, "Invalid input.mem word %0d", i);
            for (i = 0; i < 27; i = i + 1)
                if ((^fixture_weights[i]) === 1'bx) $fatal(1, "Invalid weights.mem word %0d", i);
            for (i = 0; i < 1024; i = i + 1)
                if ((^fixture_acc[i]) === 1'bx) $fatal(1, "Invalid expected_acc.mem word %0d", i);
            if ((^fixture_bias[0]) === 1'bx) $fatal(1, "Invalid bias.mem");

            i = 0;
            for (y = 0; y < H; y = y + 1)
                for (x = 0; x < W; x = x + 1)
                    for (c = 0; c < CHANNELS; c = c + 1) begin
                        image[y][x][c] = $signed(fixture_image[i]);
                        i = i + 1;
                    end
            i = 0;
            for (ky = 0; ky < 3; ky = ky + 1)
                for (kx = 0; kx < 3; kx = kx + 1)
                    for (c = 0; c < CHANNELS; c = c + 1) begin
                        weights[ky][kx][c] = $signed(fixture_weights[i]);
                        i = i + 1;
                    end
            fresh_convolution($signed(fixture_bias[0]), "Python golden fixture");
            // Check the SV reference against Python expectations as well as
            // checking the DUT: a wrong model must not bless a wrong DUT.
            for (y = 0; y < H; y = y + 1)
                for (x = 0; x < W; x = x + 1) begin
                    if (expected[y][x] != $signed(fixture_acc[y*W+x]))
                        $fatal(1, "SV/Python reference disagreement at (%0d,%0d)", y, x);
                    check_anchor(y, x, $signed(fixture_acc[y*W+x]));
                end
            fixture_ran = 1'b1;
            $display("Completed Python fixture comparison: %s", directory);
        end
    endtask

    function automatic integer extreme(input integer index);
        case (index)
            0: extreme = -128;
            1: extreme = -1;
            2: extreme = 0;
            3: extreme = 1;
            default: extreme = 127;
        endcase
    endfunction

    integer y, x, c, ky, kx, lane, a, b, trial, iy, ix, random_bias;
    initial begin
        random_cases = RANDOM_CASES;
        fixture_directory = FIXTURE_DIR;
        if ($value$plusargs("RANDOM_CASES=%d", random_cases)) begin end
        if ($value$plusargs("SEED=%h", rng_state)) begin end
        if ($value$plusargs("FIXTURE_DIR=%s", fixture_directory)) begin end
        if (random_cases < 0 || random_cases > 1000)
            $fatal(1, "RANDOM_CASES must be 0..1000");
        if (rng_state == 0) $fatal(1, "SEED must be nonzero");
        $display("Starting tb_conv_1024: random_cases=%0d seed=%08h", random_cases, rng_state);

        fill_image(0);
        fill_weights(0);
        step(1, 0, 0, 0, "initial synchronous reset");

        $display("Group 1: reset, load, hold, priority, bias counted once");
        fresh_convolution(0, "all zero");
        fresh_convolution(1234567, "positive bias only");
        fresh_convolution(-7654321, "negative bias only");
        randomize_data();
        step(0, 0, 0, 999, "disabled with changed inputs/weights/bias");
        step(0, 1, 1, -12345, "load beats enable");
        step(0, 0, 1, 99999, "bias port ignored while accumulating");
        step(1, 1, 1, 555, "reset beats load and enable");
        step(1, 0, 1, -777, "reset held during enable");
        step(0, 0, 0, 888, "hold zero after reset release");
        step(0, 0, 1, 888, "accumulate directly after reset, no implicit bias");

        $display("Group 2: padding anchors and signed INT8 boundary products");
        fill_image(1);
        fill_weights(1);
        fresh_convolution(7, "ones / padding");
        check_anchor(0, 0, 19);   // 4 spatial samples * 3 channels + 7
        check_anchor(0, 31, 19);
        check_anchor(31, 0, 19);
        check_anchor(31, 31, 19);
        check_anchor(0, 16, 25);  // 6 spatial samples * 3 channels + 7
        check_anchor(16, 0, 25);
        check_anchor(31, 16, 25);
        check_anchor(16, 31, 25);
        check_anchor(16, 16, 34); // 9 spatial samples * 3 channels + 7
        for (a = 0; a < 5; a = a + 1)
            for (b = 0; b < 5; b = b + 1) begin
                fill_image(extreme(a));
                fill_weights(extreme(b));
                fresh_convolution(-19, $sformatf("signed boundaries input=%0d weight=%0d", extreme(a), extreme(b)));
                check_anchor(16, 16, -19 + 27 * extreme(a) * extreme(b));
            end

        $display("Group 3: all 27 weight lanes independently, spatial/channel ordering");
        for (y = 0; y < H; y = y + 1)
            for (x = 0; x < W; x = x + 1)
                for (c = 0; c < CHANNELS; c = c + 1)
                    image[y][x][c] = ((y*37 + x*17 + c*83) % 256) - 128;
        for (lane = 0; lane < 27; lane = lane + 1) begin
            fill_weights(0);
            ky = lane / 9;
            kx = (lane / 3) % 3;
            c = lane % 3;
            weights[ky][kx][c] = (lane % 2 == 0) ? 1 : -3;
            fresh_convolution(101, $sformatf("isolated kernel lane %0d", lane));
        end

        $display("Group 4: impulse images at corners, edge midpoints, center, every channel");
        for (ky = 0; ky < 3; ky = ky + 1)
            for (kx = 0; kx < 3; kx = kx + 1)
                for (c = 0; c < CHANNELS; c = c + 1) begin
                    lane = (ky*3+kx)*3+c;
                    weights[ky][kx][c] = (lane % 2 == 0) ? lane+1 : -(lane+1);
                end
        for (a = 0; a < 3; a = a + 1)
            for (b = 0; b < 3; b = b + 1)
                for (c = 0; c < CHANNELS; c = c + 1) begin
                    iy = (a == 0) ? 0 : ((a == 1) ? 16 : 31);
                    ix = (b == 0) ? 0 : ((b == 1) ? 16 : 31);
                    fill_image(0);
                    image[iy][ix][c] = (c == 1) ? -128 : 127;
                    fresh_convolution(-37, $sformatf("impulse y=%0d x=%0d channel=%0d", iy, ix, c));
                end

        $display("Group 5: large INT32 biases, repeated accumulation, restart");
        fill_image(127);
        fill_weights(127);
        fresh_convolution(2146983647, "near INT32 maximum without overflow");
        fill_image(-128);
        fill_weights(127);
        fresh_convolution(-2146983648, "near INT32 minimum without overflow");
        fill_image(1);
        fill_weights(-1);
        fresh_convolution(100, "repeated accumulation first edge");
        step(0, 0, 1, -999, "repeated accumulation second edge");
        check_anchor(16, 16, 46); // 100 - 27 - 27; bias added only once
        step(0, 0, 1, 999, "repeated accumulation third edge");
        check_anchor(16, 16, 19);
        step(0, 1, 0, -50, "reload clears previous lane accumulations");
        step(0, 0, 1, 999, "new result after reload");
        check_anchor(16, 16, -77);

        $display("Group 6: seeded random frames, stalls, consecutive enabled edges");
        for (trial = 0; trial < random_cases; trial = trial + 1) begin
            randomize_data();
            random_bias = random_int8() * 1000 + random_int8();
            fresh_convolution(random_bias, $sformatf("random trial %0d", trial));
            randomize_data();
            step(0, 0, 1, random_bias + 123, $sformatf("random trial %0d accumulate changed data", trial));
            randomize_data();
            step(0, 0, 0, -random_bias, $sformatf("random trial %0d disabled hold", trial));
        end

        if (fixture_directory != "")
            run_python_fixture(fixture_directory);
        else
            $display("SKIPPED Python fixture: set FIXTURE_DIR to enable it. Synthetic tests still run.");

        step(1, 1, 1, 123, "final reset clears every pixel");
        step(0, 0, 0, 456, "final disabled hold");
        if (errors != 0)
            $fatal(1, "FAIL: %0d mismatches; first %0d shown. cycles=%0d comparisons=%0d",
                   errors, MAX_ERROR_MESSAGES, checked_cycles, pixel_comparisons);
        $display("PASS: all 1024 pixels matched. cycles=%0d comparisons=%0d random_cases=%0d python_fixture=%0d",
                 checked_cycles, pixel_comparisons, random_cases, fixture_ran);
        $finish;
    end

    initial begin
        #1000000; // 1 ms simulated-time watchdog, well above the default suite.
        $fatal(1, "TIMEOUT: testbench did not finish; last test=%s", test_name);
    end
endmodule

`default_nettype wire
