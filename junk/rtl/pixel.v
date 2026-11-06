`default_nettype none

// Parallel 3x3 RGB dot product for ONE output pixel and ONE output channel.
// Instantiates 27 mac_int8 blocks, one per kernel-row/kernel-column/channel.
//
// Packed data layout (each byte is a signed INT8 value):
//   lane = ((kernel_y * 3) + kernel_x) * 3 + channel
//   input_values [lane*8 +: 8] = input for that lane
//   weight_values[lane*8 +: 8] = weight for that lane
// Lane 0 occupies bits [7:0]; lane 26 occupies bits [215:208].
// The caller supplies the 3x3 window and zeros for out-of-bounds pixels.
//
// Operation, on rising clock edges:
//   1. Assert load_bias: lane 0 loads bias; all other lanes load zero.
//   2. Deassert load_bias, present all 27 pairs, assert enable for ONE edge.
//   3. After the MAC registers and combinational adder tree settle:
//        acc = bias + sum(input_values[i] * weight_values[i]), i = 0..26.
//   4. Deassert enable to hold the result. Load bias again for a new pixel.
//
// Holding enable high for multiple edges ACCUMULATES additional sets of 27
// products; it does not automatically start a new pixel each clock cycle.
// Control priority is rst > load_bias > enable, as in mac_int8.
// rst is synchronous and active high. load_bias suppresses multiplication.
//
// The bias is counted ONCE, not once per MAC. The tree is combinational,
// with five addition levels and no extra output register or valid signal.
// Requantization, ReLU, saturation, and image addressing are separate blocks.
// Caller must keep each lane accumulator and final result within INT32.
// Out-of-range final results wrap to 32 bits, matching the MAC's behavior.
module pixel (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     load_bias,
    input  wire                     enable,
    input  wire [215:0]              input_values,
    input  wire [215:0]              weight_values,
    input  wire signed [31:0]        bias,
    output wire signed [31:0]        acc
);

    wire signed [31:0] lane_acc [0:26];

    // A 37-bit sum can represent the sum of all 27 signed INT32 lanes.
    // Pad to 32 leaves to form a balanced binary tree:
    //   node 1: root; nodes 2..31: internal; nodes 32..63: leaves.
    wire signed [36:0] sum_tree [1:63];

    genvar lane;
    genvar padding;
    genvar node;
    generate
        for (lane = 0; lane < 27; lane = lane + 1) begin : gen_macs
            wire signed [7:0] lane_input;
            wire signed [7:0] lane_weight;
            wire signed [31:0] lane_bias;

            // Declare signed wires explicitly: slices of packed buses are
            // otherwise unsigned expressions in Verilog.
            assign lane_input = input_values[lane*8 +: 8];
            assign lane_weight = weight_values[lane*8 +: 8];
            assign lane_bias = (lane == 0) ? bias : 32'sd0;

            mac_int8 u_mac (
                .clk          (clk),
                .rst          (rst),
                .load_bias    (load_bias),
                .enable       (enable),
                .input_value  (lane_input),
                .weight_value (lane_weight),
                .bias         (lane_bias),
                .acc          (lane_acc[lane])
            );

            assign sum_tree[32 + lane] =
                {{5{lane_acc[lane][31]}}, lane_acc[lane]};
        end

        for (padding = 27; padding < 32; padding = padding + 1) begin : gen_zero_leaves
            assign sum_tree[32 + padding] = 37'sd0;
        end

        for (node = 1; node < 32; node = node + 1) begin : gen_adders
            assign sum_tree[node] = sum_tree[2*node] + sum_tree[2*node + 1];
        end
    endgenerate

    assign acc = sum_tree[1][31:0];

endmodule

`default_nettype wire
