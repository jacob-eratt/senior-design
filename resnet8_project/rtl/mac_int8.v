`default_nettype none

// Signed INT8 multiply-accumulate with a signed INT32 accumulator.
// All state changes occur on the rising clock edge.
//
// Control priority:
//   1. rst       : clear acc to zero (synchronous, active high).
//   2. load_bias : start a new dot product by loading bias into acc.
//   3. enable    : acc <= acc + input_value * weight_value.
//   4. Otherwise: hold acc.
//
// Example for one 3x3 filter with three input channels:
//   - Assert load_bias for one clock edge.
//   - Present the 27 input/weight pairs, with enable high on each MAC edge.
//   - After the 27th MAC edge, acc contains bias + all 27 products.
//   - For padded image positions, present input_value = 0.
//
// load_bias and enable together load ONLY the bias; no product is added.
// There is no internal pipeline or done signal. The caller counts MAC edges.
// Requantization, ReLU, and INT8 saturation belong to a later block.
// INT32 overflow wraps; the caller must ensure the sum fits signed INT32.
// The frozen software model checks this bound for its exported parameters.
module mac_int8 (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     load_bias,
    input  wire                     enable,
    input  wire signed [7:0]        input_value,
    input  wire signed [7:0]        weight_value,
    input  wire signed [31:0]       bias,
    output reg  signed [31:0]       acc
);

    // A signed 8-bit x 8-bit product fits exactly in signed 16 bits.
    wire signed [15:0] product;
    wire signed [31:0] product_extended;

    assign product = input_value * weight_value;
    assign product_extended = {{16{product[15]}}, product};

    always @(posedge clk) begin
        if (rst) begin
            acc <= 32'sd0;
        end else if (load_bias) begin
            acc <= bias;
        end else if (enable) begin
            acc <= acc + product_extended;
        end
    end

endmodule

`default_nettype wire
