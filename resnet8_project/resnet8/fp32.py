"""Instrument the unchanged upstream graph; never substitute a lookalike model."""

import numpy as np
import tensorflow as tf

from .common import ROOT, WEIGHTS, sha256
from .upstream.keras_model import resnet_v1_eembc

SOURCE_HASH = "52a7564de25828abba7aa4aa785f1355513ad86079bad1ef9f26167073791f45"
WEIGHT_HASH = "7e469415849ea0be2d5a6cb988a0840e2ffc4184ee1bdff3442a2f1ae4774fdf"
# (name, input, Cin, Cout, kernel, stride, batchnorm, relu)
CONVS = [
    ("conv0", "input", 3, 26, 3, 1, True, True),
    ("block1_conv1", "conv0", 26, 26, 3, 1, True, True),
    ("block1_conv2", "block1_conv1", 26, 26, 3, 1, True, False),
    ("block2_conv1", "block1_relu", 26, 52, 3, 2, True, True),
    ("block2_conv2", "block2_conv1", 52, 52, 3, 1, True, False),
    ("block2_shortcut", "block1_relu", 26, 52, 1, 2, False, False),
    ("block3_conv1", "block2_relu", 52, 104, 3, 2, True, True),
    ("block3_conv2", "block3_conv1", 104, 104, 3, 1, True, False),
    ("block3_shortcut", "block2_relu", 52, 104, 1, 2, False, False),
]


def producer(tensor):
    return tensor._keras_history.operation


class Reference:
    def __init__(self, weights=WEIGHTS):
        if sha256(ROOT / "upstream" / "keras_model.py") != SOURCE_HASH:
            raise ValueError("Frozen upstream architecture source changed")
        if sha256(weights) != WEIGHT_HASH:
            raise ValueError("Weights differ from the frozen MLPerf checkpoint")
        tf.keras.backend.clear_session()
        self.model = resnet_v1_eembc(conv_filters=26)
        self.model.load_weights(str(weights))
        convs = [l for l in self.model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        # Graph order places a projection alongside the main path. Identify by
        # output channels/kernel, then main-path order, rather than Keras names.
        ordered = []
        for channels in (26, 52, 104):
            ordered.extend(l for l in convs if l.filters == channels and l.kernel_size == (3, 3))
            ordered.extend(l for l in convs if l.filters == channels and l.kernel_size == (1, 1))
        assert len(ordered) == 9
        self.layers = {}
        self.bn = {}
        tensors = {"input": self.model.input}

        def child(layer, kind):
            matches = [l for l in self.model.layers if isinstance(l, kind)
                       and not isinstance(l.input, list) and producer(l.input) is layer]
            assert len(matches) == 1, (layer.name, kind, matches)
            return matches[0]

        for spec, layer in zip(CONVS, ordered):
            name, _, ci, co, k, s, bn, relu = spec
            assert tuple(layer.kernel.shape) == (k, k, ci, co)
            assert layer.strides == (s, s) and layer.padding == "same" and layer.use_bias
            assert layer.activation == tf.keras.activations.linear
            self.layers[name] = layer
            tensors[name + "_raw"] = layer.output
            end = layer
            if bn:
                end = child(layer, tf.keras.layers.BatchNormalization)
                assert end.epsilon == 0.001
                self.bn[name] = end
                tensors[name + "_bn"] = end.output
            if relu:
                end = child(end, tf.keras.layers.Activation)
                assert end.activation == tf.keras.activations.relu
            tensors[name] = end.output
        adds = [l for l in self.model.layers if isinstance(l, tf.keras.layers.Add)]
        assert len(adds) == 3
        for i, layer in enumerate(adds, 1):
            tensors[f"block{i}_add"] = layer.output
            tensors[f"block{i}_relu"] = child(layer, tf.keras.layers.Activation).output
        for name, source, *_ in CONVS:
            assert producer(self.layers[name].input) is producer(tensors[source])
        for i, layer in enumerate(adds, 1):
            shortcut = "conv0" if i == 1 else f"block{i}_shortcut"
            assert {producer(t) for t in layer.input} == {
                producer(tensors[shortcut]), producer(tensors[f"block{i}_conv2"])}
        pool = next(l for l in self.model.layers if isinstance(l, tf.keras.layers.AveragePooling2D))
        dense = next(l for l in self.model.layers if isinstance(l, tf.keras.layers.Dense))
        assert pool.pool_size == (8, 8) and pool.padding == "valid"
        assert tuple(dense.kernel.shape) == (104, 10) and dense.use_bias
        assert dense.activation == tf.keras.activations.softmax
        assert self.model.input_shape == (None, 32, 32, 3)
        assert self.model.output_shape == (None, 10)
        self.layers["dense"] = dense
        tensors["avgpool"] = pool.output
        tensors["flatten"] = dense.input
        tensors["probabilities"] = dense.output
        self.names = list(tensors)
        self.probe = tf.keras.Model(self.model.input, list(tensors.values()))
        self.weights_path = weights

    def run(self, images):
        images = np.asarray(images, np.float32)
        if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ValueError("Expected NHWC [N,32,32,3]")
        if not np.all(np.isfinite(images)) or images.min() < 0 or images.max() > 255:
            raise ValueError("Expected finite RGB pixel values in [0,255]")
        values = self.probe(images, training=False)
        result = {n: v.numpy() for n, v in zip(self.names, values)}
        dense = self.layers["dense"]
        result["logits"] = (tf.matmul(result["flatten"], dense.kernel) + dense.bias).numpy()
        np.testing.assert_allclose(tf.nn.softmax(result["logits"]).numpy(),
                                   result["probabilities"], atol=2e-6, rtol=2e-5)
        return result

    def folded_weights(self):
        result = {}
        for name, layer in self.layers.items():
            w, b = [a.astype(np.float64) for a in layer.get_weights()]
            if name in self.bn:
                bn = self.bn[name]
                gamma, beta, mean, var = [a.astype(np.float64) for a in bn.get_weights()]
                factor = gamma / np.sqrt(var + bn.epsilon)
                w = w * factor
                b = (b - mean) * factor + beta
            result[name] = (w.astype(np.float32), b.astype(np.float32))
        return result

    def folded_run(self, images):
        weights = self.folded_weights()
        values = {"input": tf.convert_to_tensor(images, tf.float32)}
        for name, source, _, _, _, stride, _, relu in CONVS:
            w, b = weights[name]
            y = tf.nn.conv2d(values[source], w, strides=[1, stride, stride, 1], padding="SAME") + b
            values[name] = tf.nn.relu(y) if relu else y
            if name == "block1_conv2" or name.endswith("shortcut"):
                i = int(name[5])
                residual = values["conv0" if i == 1 else f"block{i}_shortcut"]
                values[f"block{i}_add"] = values[f"block{i}_conv2"] + residual
                values[f"block{i}_relu"] = tf.nn.relu(values[f"block{i}_add"])
        values["avgpool"] = tf.nn.avg_pool2d(values["block3_relu"], 8, 8, "VALID")
        w, b = weights["dense"]
        values["logits"] = tf.reshape(values["avgpool"], [-1, 104]) @ w + b
        values["probabilities"] = tf.nn.softmax(values["logits"])
        return {n: v.numpy() for n, v in values.items()}
