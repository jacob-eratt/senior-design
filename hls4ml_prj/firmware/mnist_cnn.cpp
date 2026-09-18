#include <iostream>

#include "mnist_cnn.h"
#include "parameters.h"


void mnist_cnn(
    input_t x[28*28*1],
    result_t layer11_out[10]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x,layer11_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<features_0_weight_t, 288>(w2, "w2.txt");
        nnet::load_weights_from_txt<features_0_bias_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<features_3_weight_t, 18432>(w5, "w5.txt");
        nnet::load_weights_from_txt<features_3_bias_t, 64>(b5, "b5.txt");
        nnet::load_weights_from_txt<classifier_1_weight_t, 401408>(w9, "w9.txt");
        nnet::load_weights_from_txt<classifier_1_bias_t, 128>(b9, "b9.txt");
        nnet::load_weights_from_txt<classifier_4_weight_t, 1280>(w11, "w11.txt");
        nnet::load_weights_from_txt<classifier_4_bias_t, 10>(b11, "b11.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    features_0_result_t layer2_out[28*28*32];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer3_t layer3_out[28*28*32];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    layer4_t layer4_out[14*14*32];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    features_3_result_t layer5_out[14*14*64];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer6_t layer6_out[14*14*64];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    layer7_t layer7_out[7*7*64];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    auto& layer8_out = layer7_out;
    classifier_1_result_t layer9_out[128];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    layer10_t layer10_out[128];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    nnet::conv_2d_cl<input_t, features_0_result_t, config2>(x, layer2_out, w2, b2); // features_0

    nnet::relu<features_0_result_t, layer3_t, relu_config3>(layer2_out, layer3_out); // features_1

    nnet::pooling2d_cl<layer3_t, layer4_t, config4>(layer3_out, layer4_out); // features_2

    nnet::conv_2d_cl<layer4_t, features_3_result_t, config5>(layer4_out, layer5_out, w5, b5); // features_3

    nnet::relu<features_3_result_t, layer6_t, relu_config6>(layer5_out, layer6_out); // features_4

    nnet::pooling2d_cl<layer6_t, layer7_t, config7>(layer6_out, layer7_out); // features_5

    nnet::dense<layer7_t, classifier_1_result_t, config9>(layer8_out, layer9_out, w9, b9); // classifier_1

    nnet::relu<classifier_1_result_t, layer10_t, relu_config10>(layer9_out, layer10_out); // classifier_2

    nnet::dense<layer10_t, result_t, config11>(layer10_out, layer11_out, w11, b11); // classifier_4

}

