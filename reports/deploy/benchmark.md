# Inference latency: PyTorch vs ONNX Runtime

Batch 1, fp32, model forward pass only (no image decoding/preprocessing); 20 warm-up runs, then mean / p95 over 100 runs.

| backend                  |   cnn mean (ms) |   cnn p95 (ms) |   vit mean (ms) |   vit p95 (ms) |
|:-------------------------|----------------:|---------------:|----------------:|---------------:|
| PyTorch CPU (5 thr)      |          103.05 |         104.94 |           13.79 |          14.05 |
| ONNX Runtime CPU (5 thr) |            5.86 |           6.31 |           24.03 |          24.52 |
| ONNX Runtime CoreML      |            1.06 |           1.18 |           49.93 |          56.01 |
| PyTorch MPS              |            5.05 |           5.25 |            6.71 |           6.87 |

Environment: macOS-26.5.2-arm64-arm-64bit-Mach-O, Python 3.14.5, torch 2.15.0, onnxruntime 1.31.1, 5 CPU threads, torch oneDNN available: False.
