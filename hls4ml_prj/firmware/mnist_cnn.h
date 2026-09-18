#ifndef MNIST_CNN_H_
#define MNIST_CNN_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void mnist_cnn(
    input_t x[28*28*1],
    result_t layer11_out[10]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
