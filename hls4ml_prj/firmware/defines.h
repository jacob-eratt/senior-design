#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<37,17> features_0_accum_t;
typedef ap_fixed<37,17> features_0_result_t;
typedef ap_fixed<16,6> features_0_weight_t;
typedef ap_fixed<16,6> features_0_bias_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<18,8> features_1_table_t;
typedef ap_fixed<16,6> features_2_accum_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<42,22> features_3_accum_t;
typedef ap_fixed<42,22> features_3_result_t;
typedef ap_fixed<16,6> features_3_weight_t;
typedef ap_fixed<16,6> features_3_bias_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<18,8> features_4_table_t;
typedef ap_fixed<16,6> features_5_accum_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<45,25> classifier_1_accum_t;
typedef ap_fixed<45,25> classifier_1_result_t;
typedef ap_fixed<16,6> classifier_1_weight_t;
typedef ap_fixed<16,6> classifier_1_bias_t;
typedef ap_uint<1> layer9_index;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<18,8> classifier_2_table_t;
typedef ap_fixed<40,20> classifier_4_accum_t;
typedef ap_fixed<40,20> result_t;
typedef ap_fixed<16,6> classifier_4_weight_t;
typedef ap_fixed<16,6> classifier_4_bias_t;
typedef ap_uint<1> layer11_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
