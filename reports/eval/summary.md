# Evaluation summary

Test split: 2000 images, 20 classes.

## Comparison

| model                        | top1   | top5   |   macro_f1 |   errors | params_total   | params_inference   |   mps_mean_ms |   mps_p95_ms |   cpu_mean_ms |   cpu_p95_ms |
|:-----------------------------|:-------|:-------|-----------:|---------:|:---------------|:-------------------|--------------:|-------------:|--------------:|-------------:|
| EfficientNet-B0 (fine-tuned) | 90.45% | 98.30% |     0.9047 |      190 | 4.0M           | 4.0M               |           5   |          5.2 |         102.5 |        103.6 |
| ViT-S/16 (fine-tuned)        | 92.05% | 99.25% |     0.92   |      159 | 21.7M          | 21.7M              |           6.7 |          6.9 |          13.7 |         14   |
| CLIP ViT-B/32 (zero-shot)    | 92.70% | 99.65% |     0.9271 |      146 | 151.3M         | 87.8M              |           9   |          9.2 |          21   |         21.6 |

Latency: batch 1, fp32, model forward pass only (no image decoding/resizing), mean / p95 over 50 runs after 10 warm-up runs.

Environment: torch 2.15.0, 5 CPU threads, oneDNN/MKLDNN available: False.

> Without oneDNN, PyTorch's CPU path runs depthwise convolutions through a slow generic fallback (`aten::_slow_conv2d_forward`, one call per channel group), which penalises EfficientNet far more than ViTs or ResNets. CPU numbers for depthwise-conv models reflect this build, not the architecture's FLOPs.

## Pairwise significance (exact McNemar)

| model_a   | model_b   |   top1_a |   top1_b |   only_a_correct |   only_b_correct |   p_value |
|:----------|:----------|---------:|---------:|-----------------:|-----------------:|----------:|
| cnn       | vit       |   0.905  |   0.9205 |               67 |               98 |    0.0192 |
| cnn       | clip      |   0.905  |   0.927  |               78 |              122 |    0.0023 |
| vit       | clip      |   0.9205 |   0.927  |               81 |               94 |    0.3644 |

## Wrong for every model: 49 images

## Per model

### EfficientNet-B0 (fine-tuned)

Lowest-F1 classes:

| class_name     |   precision |   recall |    f1 |   support |
|:---------------|------------:|---------:|------:|----------:|
| steak          |       0.739 |     0.68 | 0.708 |       100 |
| filet_mignon   |       0.724 |     0.76 | 0.741 |       100 |
| ice_cream      |       0.899 |     0.8  | 0.847 |       100 |
| chicken_curry  |       0.882 |     0.9  | 0.891 |       100 |
| chocolate_cake |       0.907 |     0.88 | 0.893 |       100 |

Most-confused pairs:

| class_a        | class_b      |   a_as_b |   b_as_a |   total |
|:---------------|:-------------|---------:|---------:|--------:|
| filet_mignon   | steak        |       18 |       25 |      43 |
| frozen_yogurt  | ice_cream    |        3 |        7 |      10 |
| chocolate_cake | ice_cream    |        3 |        4 |       7 |
| samosa         | spring_rolls |        1 |        5 |       6 |
| caesar_salad   | samosa       |        2 |        3 |       5 |

### ViT-S/16 (fine-tuned)

Lowest-F1 classes:

| class_name   |   precision |   recall |    f1 |   support |
|:-------------|------------:|---------:|------:|----------:|
| steak        |       0.697 |     0.69 | 0.693 |       100 |
| filet_mignon |       0.735 |     0.75 | 0.743 |       100 |
| ice_cream    |       0.973 |     0.72 | 0.828 |       100 |
| omelette     |       0.876 |     0.92 | 0.898 |       100 |
| samosa       |       0.895 |     0.94 | 0.917 |       100 |

Most-confused pairs:

| class_a       | class_b      |   a_as_b |   b_as_a |   total |
|:--------------|:-------------|---------:|---------:|--------:|
| filet_mignon  | steak        |       18 |       24 |      42 |
| frozen_yogurt | ice_cream    |        2 |       12 |      14 |
| ice_cream     | waffles      |        7 |        0 |       7 |
| samosa        | spring_rolls |        3 |        4 |       7 |
| chicken_curry | steak        |        2 |        2 |       4 |

### CLIP ViT-B/32 (zero-shot)

Lowest-F1 classes:

| class_name    |   precision |   recall |    f1 |   support |
|:--------------|------------:|---------:|------:|----------:|
| ice_cream     |       0.769 |     0.7  | 0.733 |       100 |
| filet_mignon  |       0.816 |     0.71 | 0.759 |       100 |
| steak         |       0.722 |     0.83 | 0.772 |       100 |
| frozen_yogurt |       0.773 |     0.92 | 0.84  |       100 |
| samosa        |       0.939 |     0.93 | 0.935 |       100 |

Most-confused pairs:

| class_a        | class_b      |   a_as_b |   b_as_a |   total |
|:---------------|:-------------|---------:|---------:|--------:|
| filet_mignon   | steak        |       27 |       15 |      42 |
| frozen_yogurt  | ice_cream    |        7 |       25 |      32 |
| chocolate_cake | ice_cream    |        7 |        1 |       8 |
| samosa         | spring_rolls |        1 |        5 |       6 |
| ice_cream      | waffles      |        1 |        3 |       4 |
