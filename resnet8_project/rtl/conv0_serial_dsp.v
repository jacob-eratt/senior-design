`timescale 1ns/1ps
`default_nettype none

// Resource-reused Conv0 for the frozen ResNet-8 model.
//
// Computes the complete Conv0 output tensor for one 32x32 RGB image:
//   input:  HWC INT8, 32x32x3 (3072 bytes)
//   weight: OHWI INT8, 26x3x3x3 (702 bytes)
//   bias:   INT32, 26 values
//   output: HWC INT32 accumulators, 32x32x26 (26624 words)
//
// Uses ONE signed 8x8 multiplier and ONE INT32 accumulator, reused over every
// kernel tap, spatial position and output channel. Expected MACs per image:
//   32*32*26*3*3*3 = 718848
// This is sequential hardware: it deliberately trades throughput for area.
//
// Memory request protocol (image/weight/bias): assert *_req with stable address
// until *_ready is high on a rising edge. Wait for *_valid, then capture *_data.
// The memory returns a one-cycle valid pulse for each accepted request; it may
// return valid on the same edge as ready or on a later edge. Only one read is
// outstanding at a time. Image and weight addresses are independent byte/word
// indices starting at zero. This interface permits BRAM controllers with
// registered read latency and prevents a 24576-bit image input port.
//
// Output protocol: out_valid/data/address remain stable until out_ready is high
// on a rising edge. Output address is HWC order: ((y*32+x)*26 + output_channel).
//
// Start is accepted only while idle and begins all 26 output channels. Bias is
// fetched once per channel and loaded into the MAC at the start of each pixel.
// SAME padding: out-of-image taps are supplied as integer zero and need no read.
// The module produces CONVOLUTION + BIAS in INT32. BatchNorm is expected folded
// into weights/bias. Requantization, INT8 saturation and ReLU are intentionally
// separate and are NOT performed here. This raw accumulator is not directly
// consumable by the following INT8 layer until that post-processing is added.
//
// DSP inference: USE_DSP requests that Vivado implement the signed multiplier
// in a DSP slice. Check the new synthesis utilization report for DSP=1. There is
// one multiplier in this module, unlike conv_1024_pixels (27648 multipliers).
// DSP slices perform multiply/MAC arithmetic; ReLU and max-pooling use ordinary
// compare/select logic and should not consume DSP slices.
module conv0_serial_dsp (
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,
    output wire                    busy,
    output reg                     done,

    output wire                    image_req,
    output wire [11:0]             image_addr,
    input  wire                    image_ready,
    input  wire                    image_valid,
    input  wire signed [7:0]       image_data,

    output wire                    weight_req,
    output wire [9:0]              weight_addr,
    input  wire                    weight_ready,
    input  wire                    weight_valid,
    input  wire signed [7:0]       weight_data,

    output wire                    bias_req,
    output wire [4:0]              bias_addr,
    input  wire                    bias_ready,
    input  wire                    bias_valid,
    input  wire signed [31:0]      bias_data,

    output wire                    out_valid,
    input  wire                    out_ready,
    output wire [14:0]             out_addr,
    output wire signed [31:0]      out_data
);

    localparam [3:0] ST_IDLE        = 4'd0;
    localparam [3:0] ST_BIAS_REQ    = 4'd1;
    localparam [3:0] ST_BIAS_WAIT   = 4'd2;
    localparam [3:0] ST_LOAD_BIAS   = 4'd3;
    localparam [3:0] ST_TAP_SETUP   = 4'd4;
    localparam [3:0] ST_IMAGE_REQ   = 4'd5;
    localparam [3:0] ST_IMAGE_WAIT  = 4'd6;
    localparam [3:0] ST_WEIGHT_REQ  = 4'd7;
    localparam [3:0] ST_WEIGHT_WAIT = 4'd8;
    localparam [3:0] ST_MAC         = 4'd9;
    localparam [3:0] ST_TAP_NEXT    = 4'd10;
    localparam [3:0] ST_WRITE_OUT   = 4'd11;

    reg [3:0] state;
    reg [4:0] output_y;
    reg [4:0] output_x;
    reg [4:0] output_channel;
    reg [1:0] kernel_y;
    reg [1:0] kernel_x;
    reg [1:0] input_channel;
    reg signed [31:0] channel_bias;
    reg signed [7:0] sample_reg;
    reg signed [7:0] weight_reg;
    reg signed [31:0] accumulator;

    wire signed [6:0] source_y;
    wire signed [6:0] source_x;
    wire signed [31:0] source_y_wide;
    wire signed [31:0] source_x_wide;
    wire tap_inside_image;
    (* use_dsp = "yes" *) wire signed [15:0] product;
    wire signed [31:0] product_extended;
    wire signed [7:0] image_operand;
    wire signed [7:0] weight_operand;

    assign source_y = $signed({2'b00, output_y})
                    + $signed({2'b00, kernel_y}) - 7'sd1;
    assign source_x = $signed({2'b00, output_x})
                    + $signed({2'b00, kernel_x}) - 7'sd1;
    assign source_y_wide = source_y;
    assign source_x_wide = source_x;
    assign tap_inside_image = (source_y >= 0) && (source_y < 32)
                           && (source_x >= 0) && (source_x < 32);

    // HWC input address: ((y*32)+x)*3+channel.
    assign image_addr = ((source_y_wide * 32'sd32 + source_x_wide) * 32'sd3)
                      + input_channel;

    // OHWI weight address for K=3,Cin=3:
    // output_channel*27 + kernel_y*9 + kernel_x*3 + input_channel.
    assign weight_addr = output_channel * 10'd27
                       + kernel_y * 10'd9
                       + kernel_x * 10'd3
                       + input_channel;

    assign bias_addr = output_channel;

    assign image_req = (state == ST_IMAGE_REQ);
    assign weight_req = (state == ST_WEIGHT_REQ);
    assign bias_req = (state == ST_BIAS_REQ);
    assign out_valid = (state == ST_WRITE_OUT);
    assign busy = (state != ST_IDLE);

    // HWC output address: ((y*32)+x)*26 + output_channel.
    assign out_addr = ((output_y * 15'd32 + output_x) * 15'd26)
                    + output_channel;
    assign out_data = accumulator;

    // Target only the product signal so Vivado infers the multiplier into a
    // DSP, while the FSM and address logic remain in regular fabric.
    assign product = image_operand * weight_operand;

    // Operand mux: the MAC consumes captured RAM values; for out-of-bounds
    // taps, ST_TAP_SETUP sets sample_reg and weight_reg to zero.
    assign image_operand = sample_reg;
    assign weight_operand = weight_reg;
    assign product_extended = {{16{product[15]}}, product};

    always @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            output_y <= 5'd0;
            output_x <= 5'd0;
            output_channel <= 5'd0;
            kernel_y <= 2'd0;
            kernel_x <= 2'd0;
            input_channel <= 2'd0;
            channel_bias <= 32'sd0;
            sample_reg <= 8'sd0;
            weight_reg <= 8'sd0;
            accumulator <= 32'sd0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (start) begin
                        output_y <= 5'd0;
                        output_x <= 5'd0;
                        output_channel <= 5'd0;
                        state <= ST_BIAS_REQ;
                    end
                end

                ST_BIAS_REQ: begin
                    if (bias_ready) begin
                        if (bias_valid) begin
                            channel_bias <= bias_data;
                            state <= ST_LOAD_BIAS;
                        end else begin
                            state <= ST_BIAS_WAIT;
                        end
                    end
                end

                ST_BIAS_WAIT: begin
                    if (bias_valid) begin
                        channel_bias <= bias_data;
                        state <= ST_LOAD_BIAS;
                    end
                end

                ST_LOAD_BIAS: begin
                    accumulator <= channel_bias;
                    kernel_y <= 2'd0;
                    kernel_x <= 2'd0;
                    input_channel <= 2'd0;
                    state <= ST_TAP_SETUP;
                end

                ST_TAP_SETUP: begin
                    if (tap_inside_image) begin
                        state <= ST_IMAGE_REQ;
                    end else begin
                        sample_reg <= 8'sd0;
                        weight_reg <= 8'sd0;
                        state <= ST_MAC;
                    end
                end

                ST_IMAGE_REQ: begin
                    if (image_ready) begin
                        if (image_valid) begin
                            sample_reg <= image_data;
                            state <= ST_WEIGHT_REQ;
                        end else begin
                            state <= ST_IMAGE_WAIT;
                        end
                    end
                end

                ST_IMAGE_WAIT: begin
                    if (image_valid) begin
                        sample_reg <= image_data;
                        state <= ST_WEIGHT_REQ;
                    end
                end

                ST_WEIGHT_REQ: begin
                    if (weight_ready) begin
                        if (weight_valid) begin
                            weight_reg <= weight_data;
                            state <= ST_MAC;
                        end else begin
                            state <= ST_WEIGHT_WAIT;
                        end
                    end
                end

                ST_WEIGHT_WAIT: begin
                    if (weight_valid) begin
                        weight_reg <= weight_data;
                        state <= ST_MAC;
                    end
                end

                ST_MAC: begin
                    accumulator <= accumulator + product_extended;
                    state <= ST_TAP_NEXT;
                end

                ST_TAP_NEXT: begin
                    if (input_channel != 2'd2) begin
                        input_channel <= input_channel + 1'b1;
                        state <= ST_TAP_SETUP;
                    end else begin
                        input_channel <= 2'd0;
                        if (kernel_x != 2'd2) begin
                            kernel_x <= kernel_x + 1'b1;
                            state <= ST_TAP_SETUP;
                        end else begin
                            kernel_x <= 2'd0;
                            if (kernel_y != 2'd2) begin
                                kernel_y <= kernel_y + 1'b1;
                                state <= ST_TAP_SETUP;
                            end else begin
                                state <= ST_WRITE_OUT;
                            end
                        end
                    end
                end

                ST_WRITE_OUT: begin
                    if (out_ready) begin
                        if (output_x != 5'd31) begin
                            output_x <= output_x + 1'b1;
                            state <= ST_LOAD_BIAS;
                        end else begin
                            output_x <= 5'd0;
                            if (output_y != 5'd31) begin
                                output_y <= output_y + 1'b1;
                                state <= ST_LOAD_BIAS;
                            end else begin
                                output_y <= 5'd0;
                                if (output_channel != 5'd25) begin
                                    output_channel <= output_channel + 1'b1;
                                    state <= ST_BIAS_REQ;
                                end else begin
                                    state <= ST_IDLE;
                                    done <= 1'b1;
                                end
                            end
                        end
                    end
                end

                default: begin
                    state <= ST_IDLE;
                    done <= 1'b0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
