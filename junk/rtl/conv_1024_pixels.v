`default_nettype none

// Fully parallel 32x32 RGB convolution, ONE output channel.
// Kernel: 3x3; stride: 1; SAME padding: one zero on every edge.
// Instantiates 1024 pixel modules, containing 27,648 mac_int8 instances.
// All output positions operate on the same clock edges.
//
// Input image: 3072 signed INT8 values packed in HWC order.
//   byte_index = ((y * 32) + x) * 3 + channel
//   image_values[byte_index*8 +: 8] = image[y][x][channel]
//   y,x = 0..31; channel = 0..2. Byte 0 is in bits [7:0].
//
// Shared filter: 27 signed INT8 weights, packed in kernel-row,
// kernel-column, input-channel order (same ordering as pixel.v).
//   lane = ((kernel_y * 3) + kernel_x) * 3 + channel
//   weight_values[lane*8 +: 8] = weight[kernel_y][kernel_x][channel]
//
// Outputs: 1024 signed INT32 accumulators packed in row-major order.
//   pixel_index = y * 32 + x
//   acc_values[pixel_index*32 +: 32] = output[y][x]
// Extracted slices must be interpreted as signed INT32 by the caller.
//
// Operation:
//   1. Assert load_bias for one rising edge to initialize all pixels.
//      Each pixel receives the shared bias exactly once.
//   2. Deassert load_bias, present the entire image and filter, and assert
//      enable for ONE rising edge. All 1024 windows compute in parallel.
//   3. After the MAC registers and pixel adder trees settle, acc_values
//      contains bias + convolution for every output position.
//   4. Deassert enable to hold. Load bias again before a new image/filter.
//
// Holding enable high accumulates another image/filter contribution on
// every rising edge; it does not automatically reset for a new result.
// Control priority: synchronous active-high rst > load_bias > enable.
// No additional input/output registers, pipeline stages, or valid signal.
// Image and weights must meet setup/hold timing at the enabled clock edge.
// Window selection and padding below are static wiring, not RAM accesses.
// Caller must ensure accumulator values fit INT32.
// Requantization, ReLU, and output saturation are separate operations.
// This is a fully unrolled structure; FPGA resource/timing fit is not established.
module conv_1024_pixels (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     load_bias,
    input  wire                     enable,
    input  wire [24575:0]            image_values,
    input  wire [215:0]              weight_values,
    input  wire signed [31:0]        bias,
    output wire [32767:0]            acc_values
);

    genvar output_y;
    genvar output_x;
    genvar kernel_y;
    genvar kernel_x;
    genvar channel;

    generate
        for (output_y = 0; output_y < 32; output_y = output_y + 1) begin : gen_rows
            for (output_x = 0; output_x < 32; output_x = output_x + 1) begin : gen_columns
                wire [215:0] window_values;
                wire signed [31:0] pixel_acc;

                for (kernel_y = 0; kernel_y < 3; kernel_y = kernel_y + 1) begin : gen_kernel_rows
                    for (kernel_x = 0; kernel_x < 3; kernel_x = kernel_x + 1) begin : gen_kernel_columns
                        localparam integer SOURCE_Y = output_y + kernel_y - 1;
                        localparam integer SOURCE_X = output_x + kernel_x - 1;

                        for (channel = 0; channel < 3; channel = channel + 1) begin : gen_channels
                            localparam integer LANE = (kernel_y * 3 + kernel_x) * 3 + channel;

                            // Generate-time bounds selection avoids creating
                            // out-of-range image slices for padding locations.
                            if ((SOURCE_Y >= 0) && (SOURCE_Y < 32) &&
                                (SOURCE_X >= 0) && (SOURCE_X < 32)) begin : gen_image_value
                                localparam integer BYTE_INDEX =
                                    (SOURCE_Y * 32 + SOURCE_X) * 3 + channel;
                                assign window_values[LANE*8 +: 8] =
                                    image_values[BYTE_INDEX*8 +: 8];
                            end else begin : gen_padding_value
                                assign window_values[LANE*8 +: 8] = 8'sd0;
                            end
                        end
                    end
                end

                pixel u_pixel (
                    .clk           (clk),
                    .rst           (rst),
                    .load_bias     (load_bias),
                    .enable        (enable),
                    .input_values  (window_values),
                    .weight_values (weight_values),
                    .bias          (bias),
                    .acc           (pixel_acc)
                );

                assign acc_values[(output_y*32 + output_x)*32 +: 32] = pixel_acc;
            end
        end
    endgenerate

endmodule

`default_nettype wire
