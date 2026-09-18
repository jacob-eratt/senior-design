Python 3.12.3
Name: hls4ml
Version: 1.3.0
Summary: Machine learning in FPGAs using HLS
Home-page: https://fastmachinelearning.org/hls4ml
Author: hls4ml Team
Author-email:
License: Apache-2.0
Location: /home/henrytudor2005/venvs/mnist-hls/lib/python3.12/site-packages
Requires: h5py, numpy, pydigitalwavetools, pyyaml, quantizers
Required-by:


Model Hash
7f233b203215e13f61aaaa7df1e8a6322fb34ca4e31cd2bfd1558a47b66b0498  mnist_model.pth

c++ validation results:

Validation results
  Samples:              10000
  PyTorch accuracy:     99.11%
  HLS accuracy:         99.06%
  Prediction agreement: 99.95%
  Mean absolute error:  0.480840
  Max absolute error:   15.457331
