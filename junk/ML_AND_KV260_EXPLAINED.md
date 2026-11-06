# Machine learning and the KV260, from the beginning

This guide assumes you understand ordinary programming but have little ML or SoC experience. Numerical examples below are explanations, not measured board performance. Return to the [roadmap](RESNET50_ROADMAP.md) for scope and estimates.

## 1. An image is an array of numbers

A grayscale MNIST image has 28 rows and 28 columns: 784 brightness values. A color image has a red, green, and blue value at each position. A 224 x 224 RGB image therefore contains 150,528 values.

An ML model is a recipe that transforms those values into scores. A digit model produces ten scores, one for each digit. The largest score is its prediction. A score is not automatically a probability.

A tensor is simply a multidimensional array. `NCHW` means batch, channels, height, width. For one RGB image, that is `[1, 3, 224, 224]`. `NHWC` stores the same conceptual values in a different order. Hardware cannot guess which order you intended.
https://github.com/Xilinx/PYNQ_Bootcamp.git
## 2. Training and inference are different jobs

During training, software repeatedly changes the model's weights using labeled examples. During inference, the weights are fixed and we calculate a prediction for a new image.

Think of training as finding the recipe and inference as following it. This project builds machinery to follow an existing recipe. Our FPGA does not need to learn new weights, compute gradients, or implement an optimizer.

The `.pth` checkpoint stores trained values. It is not a hardware design. An export script will turn it into plain numeric arrays plus metadata our RTL understands.

## 3. What a convolution actually does

A filter is a small collection of learned weights. For one output position, multiply corresponding input and filter values, add the products, and add a bias.

For this made-up 3 x 3 patch and filter:

```text
Input patch       Filter
1 2 0             1  0 -1
0 1 3             1  0 -1
2 2 1             1  0 -1

Output before bias = 1 - 0 + 0 - 3 + 2 - 1 = -1
```

Move to another position and repeat with the SAME weights. Different filters produce different output channels. Channels after the first layer are learned feature maps, not necessarily red/green/blue.

For multiple input channels, one output sums contributions from every input channel. A 3 x 3 filter over 64 channels requires 576 products per output value. One multiply plus its accumulation is called a MAC.

Padding specifies values outside the image boundary. Stride specifies how far the filter moves between output positions. Most ML libraries implement the operation commonly called convolution as cross-correlation: do not reverse the exported kernel unless the reference explicitly does.

The direct MAC count is:

```text
output_height * output_width * output_channels
    * input_channels * kernel_height * kernel_width
```

One surprising fact: a 1 x 1 convolution still does real work. At every pixel, it mixes all input channels into new output channels. It is a dot product across channels, not a no-op.

## 4. The other operations

| Operation | Plain-language behavior | Likely RTL implementation |
|---|---|---|
| Bias | Add one learned offset per output channel | Adder or initial accumulator value |
| ReLU | Replace negative values with zero | Sign test and selection, adjusted for numeric representation |
| Max pool | Keep the largest value in a small window | Comparators and buffering |
| Average pool | Average a region | Sum, then specified division/rounding |
| Fully connected layer | Form weighted sums of all input features | The same kind of dot-product engine |
| Flatten | Choose an ordering for a multidimensional array | Address/layout handling |
| Residual add | Add two equally shaped feature tensors | Read both, align scales, add |
| Softmax | Convert scores into normalized probabilities | Optional CPU postprocessing; unnecessary for ranking |

Dropout randomly suppresses values during training. It is inactive in evaluation mode and has no runtime role in our MNIST accelerator.

## 5. How true is “MNIST is hello world”?

It is a fair analogy for learning the ML workflow: load a manageable dataset, train a small network, and check predictions. MNIST provides 60,000 training and 10,000 test examples of handwritten digits. [Original dataset description](https://yann.lecun.org/exdb/mnist/index.html)

The analogy stops being useful when it implies custom hardware should take an afternoon. A library already contains the arithmetic, storage management, and execution machinery. In our RTL project, we are implementing much of that machinery ourselves.

Also, MNIST is a dataset, not a particular neural-network architecture. A tiny linear classifier and your two-convolution model both classify MNIST, but require different hardware.

Our counts below are calculated directly from model.py, with one input image and direct dense convolution. Bias additions, pooling, ReLU, and memory movement are excluded from MAC counts.

| Layer | Weight values | Bias values | MACs per image |
|---|---:|---:|---:|
| Conv 1: 1 to 32 channels, 3 x 3 | 288 | 32 | 225,792 |
| Conv 2: 32 to 64 channels, 3 x 3 | 18,432 | 64 | 3,612,672 |
| Dense: 3,136 to 128 | 401,408 | 128 | 401,408 |
| Dense: 128 to 10 | 1,280 | 10 | 1,280 |
| Total | 421,408 | 234 | 4,241,152 |

The second convolution dominates arithmetic; the first dense layer dominates weight storage. Optimizing the largest weight array and optimizing the most expensive computation are different tasks.

## 6. What makes ResNet-50 different?

A residual block calculates a transformation `F(x)` and adds a shortcut:

```text
x ------------------------> add --> ReLU
 \--> learned operations --> /
```

Conceptually, the branch can learn a correction to an existing representation. This residual formulation is the central idea of ResNet. [Original ResNet paper](https://arxiv.org/abs/1512.03385)

For hardware, the key consequence is a lifetime problem: keep the shortcut input until the other branch has finished. If shapes change, a learned projection transforms the shortcut too.

For the proposed torchvision ResNet-50 v1.5 target, the graph has a 7 x 7 stem, four stages with 3/4/6/3 bottleneck blocks, global average pooling, and a classifier. A bottleneck uses 1 x 1, 3 x 3, and 1 x 1 convolutions. Downsampling stride in torchvision is on the 3 x 3 convolution; the original variant differs. Freeze the implementation and checkpoint together. [Torchvision implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html)

| Point in graph | Tensor shape, excluding batch | INT8 storage for this tensor |
|---|---|---:|
| Input | 3 x 224 x 224 | 147 KiB |
| Stem convolution | 64 x 112 x 112 | 784 KiB |
| Stem max pool | 64 x 56 x 56 | 196 KiB |
| Stage 1 output | 256 x 56 x 56 | 784 KiB |
| Stage 2 output | 512 x 28 x 28 | 392 KiB |
| Stage 3 output | 1,024 x 14 x 14 | 196 KiB |
| Stage 4 output | 2,048 x 7 x 7 | 98 KiB |
| Global average pool | 2,048 | 2 KiB |
| Classifier | 1,000 scores | Depends on chosen score precision |

Shapes describe the selected standard graph; storage sizes are our arithmetic at one byte per value. Accumulators, scratch buffers, and simultaneous live tensors need additional memory.

The name “50” counts the conventional main-path learned layers: one stem convolution, 16 bottlenecks with three convolutions each, and one classifier. It does not mean 50 identical hardware blocks. Projection branches add work, and pooling/addition/activation also need execution. We can reuse one hardware engine across these operations over time rather than physically instantiating the whole network at once.

Torchvision lists 25,557,032 parameters for ResNet-50. Its pretrained weight variants also have different preprocessing recipes. Pin one named variant and use its associated transform. [Model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)

Our direct convolution/dense count for that v1.5 graph is 4,089,184,256 MACs per 224 x 224 image, including projection convolutions and excluding pooling/BN/ReLU. That is approximately **964 times the MACs of our MNIST model**. The parameter ratio is about **61 times**. Arithmetic derivation by stage:

| Region | Direct MAC count |
|---|---:|
| Stem | 118,013,952 |
| Stage 1 | 667,942,912 |
| Stage 2 | 1,027,604,480 |
| Stage 3 | 1,464,336,384 |
| Stage 4 | 809,238,528 |
| Classifier | 2,048,000 |

Some references count a multiply and add as two operations and others report a combined MAC as one. Compare operation counts only after aligning conventions. Here 4.089 billion MACs corresponds to roughly 8.178 billion multiply/add operations.

## 7. Quantization: smaller numbers with an agreed meaning

A real value can be represented approximately by an integer and a scale:

```text
real_value ~= scale * (stored_integer - zero_point)
```

For signed symmetric quantization, choose zero point 0. With scale 0.02, integer 18 represents 0.36. The real number 0.37 falls between levels and must round. Values outside the chosen range clip. Smaller numbers reduce storage and can simplify arithmetic, at the cost of approximation.

Start by evaluating signed 8-bit weights and activations, with wider bias and accumulation. Prefer weight scales per output channel if a single scale loses too much accuracy. Calibration estimates useful activation ranges using representative inputs. This is consistent with established quantization practice, but the exact arithmetic contract remains our design. [PyTorch quantization overview](https://pytorch.org/blog/quantization-in-practice/)

Our proposed runtime calculation is:

```text
A[o] = bias_integer[o] + sum(input_integer[i] * weight_integer[o, i])
output_integer[o] = saturate(round(A[o] * multiplier[o] / 2**shift[o]))
```

The multiplier and shift approximate `(input_scale * weight_scale[o]) / output_scale`. A bias is expressed in the accumulator's scale. Intermediate products used during rescaling can need more than 32 bits even if the accumulator fits in 32 bits.

Example: input scale 0.02 and weight scale 0.01 give product scale 0.0002. An accumulator of 5,000 represents 1.0. If the next layer's activation scale is 0.05, the output integer is 20. We need not use floating-point division in RTL: export an integer multiplier/shift and implement its exact rounding rule.

Two branches cannot be blindly added when their scales differ. Integer 10 at scale 0.1 means 1.0; integer 10 at scale 0.01 means 0.1. Their integer sum of 20 has no useful interpretation without choosing a common scale. A residual-add unit needs explicit rescaling and overflow handling.

For full-range signed 8-bit operands, each product magnitude is bounded conservatively by 16,384. If a dot product has K terms, a conservative accumulator magnitude bound is `K * 16,384 + abs(integer_bias)`. Choose a signed width that covers that bound, then include any further scaling operations separately. Do not assume every generated layer has safe 32-bit bias/accumulation without checking.

The software reference must define how negatives round, how ties round, and whether overflows saturate or wrap. Python's rounding and SystemVerilog shifts/casts can behave differently. “Both are INT8” is not a correctness specification.

## 8. Batch normalization can disappear from the runtime datapath

In evaluation mode, batch normalization uses fixed stored statistics. For a convolution followed immediately by BN, the transformation can be absorbed into new weights and biases:

```text
alpha[o] = gamma[o] / sqrt(running_variance[o] + epsilon)
new_weight[o] = alpha[o] * weight[o]
new_bias[o] = beta[o] + alpha[o] * (bias[o] - running_mean[o])
```

Use bias zero if the original convolution has no bias. Perform this offline in floating point, verify the folded model against the original, and then quantize. No runtime square-root or BN unit is needed for these folded pairs. PyTorch provides an evaluation-mode fusion utility. [Conv/BN fusion documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.fusion.fuse_conv_bn_eval.html)

## 9. What the KV260 actually contains

The KV260 starter kit has a K26 system-on-module and a carrier board. Its chip combines a **processing system (PS)** with **programmable logic (PL)**.

The processing system includes Arm Cortex-A53 application processors and Cortex-R5F real-time processors. Our Linux/Python application uses the A53 side; the R5F is not needed for the proposed first design. [K26 processing-system specification](https://docs.amd.com/r/en-US/ds987-k26-som/Processing-System)

The programmable logic contains configurable logic, registers, arithmetic DSP blocks, and internal memories. Your RTL describes circuits that synthesis maps into those resources. The board has 4 GB DDR memory, 144 block-RAM blocks, and 64 UltraRAM blocks. [KV260 specifications](https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit.html)

| Resource | Useful analogy | Role in this design |
|---|---|---|
| ARM CPU | Supervisor | Run Linux, preprocessing, launch work, check results |
| DDR | Large storage room | Keep weights and feature maps too large for local buffers |
| BRAM/UltraRAM | Workbench storage | Hold tiles near arithmetic units |
| DSP slices | Arithmetic stations | Implement multipliers and related arithmetic |
| LUTs and registers | Control circuits and small storage | Addresses, state machines, routing, comparisons, pipelines |
| AXI interconnect | Roads | Carry requests and data between components |
| DMA | Delivery mechanism | Transfer blocks without a CPU instruction per value |

A bitstream configures the PL circuit. A Python script does not automatically become an FPGA circuit. The bootcamp's DPU is an accelerator circuit loaded into PL; it is not the ARM processor, and it is not hardware we must keep when loading our custom design.

## 10. The memory distinction that matters most

**4 GB of DDR is not 4 GB of local FPGA memory.** It has capacity for model weights, but the PL must request and transfer that data over interfaces.

Using 36-Kib block-RAM blocks and 288-Kib UltraRAM blocks, the theoretical combined BRAM/URAM bit capacity is:

```text
144 * 36 Kib / 8 =   648 KiB
 64 * 288 Kib / 8 = 2304 KiB
Total             = 2952 KiB = 2.883 MiB
```

The block capacities come from [AMD's memory overview](https://www.amd.com/en/products/adaptive-socs-and-fpgas/technologies/memory.html). This sum is an upper bound, not freely usable byte-array capacity: aspect ratios, parity/ECC bits, banking, FIFOs, and implementation overhead affect usable storage.

ResNet has roughly 24.3 MiB of one-byte parameter storage before wider biases/scales and graph-specific folding adjustments. Consequently, the complete weight set cannot live in these local buffers at once. Load portions from DDR and reuse them. Even MNIST's roughly 412 KiB of 8-bit weights does not solve bandwidth or memory-port constraints automatically.

## 11. What to understand before writing the first CNN RTL

You should be able to explain one output pixel as a dot product, say where its inputs and weights live, identify when its result is complete, and calculate the next layer's expected integer representation. Then explain how the result gets back to Python.

If one of those answers is missing, that is the next learning task. The [architecture guide](RTL_ARCHITECTURE_AND_DATAFLOW.md) makes each part concrete.
