"""Frozen MLPerf Tiny ResNet-8 software and integer hardware reference."""

import os

# Set before importing TensorFlow. CPU is the reproducible reference backend.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
