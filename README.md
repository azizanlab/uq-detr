<div align="center">

# Uncertainty Quantification in Detection Transformers

**A lightweight Python toolkit for object-level calibration and image-level reliability evaluation**

[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://github.com/young-j-park/uq-detr)
[![License](https://img.shields.io/github/license/young-j-park/uq-detr)](https://github.com/young-j-park/uq-detr/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2412.01782-b31b1b.svg)](https://arxiv.org/abs/2412.01782)
[![TPAMI](https://img.shields.io/badge/TPAMI-2026-blue)](https://arxiv.org/abs/2412.01782)

[📘 Documentation](https://young-j-park.github.io/uq-detr) |
[📄 Paper (TPAMI 2026)](https://arxiv.org/abs/2412.01782) |
[🛠️ Installation](#installation) |
[🚀 Quick Start](#quick-start) |
[🤔 Issues](https://github.com/young-j-park/uq-detr/issues)

[Young-Jin Park](https://young-j-park.github.io/), [Carson Sobolewski](https://csobolew.github.io/), and [Navid Azizan](https://azizan.mit.edu/) (MIT)

</div>

## Highlight Results

### Why OCE? Existing calibration metrics have a structural pitfall

D-ECE and LA-ECE achieve their optima at thresholds near 0 or 1 — since they **do not penalize missed detections (false negatives)**, they can reach near-zero error by retaining only a few highly confident predictions. In contrast, **OCE** exhibits a bell-shaped curve with its optimum around a practical threshold of ~0.3, aligning with common deployment choices.

<div align="center">
<img src="assets/fig1_threshold_vs_metrics.svg" width="500" alt="Metrics vs confidence threshold on COCO (Cal-DETR)">

*Impact of confidence threshold on metrics (Cal-DETR on COCO). OCE identifies the practical sweet spot.*
</div>

### Image-level reliability

Pearson correlation between contrastive confidence and image-level reliability (mAP per image). Positive contrast strongly correlates with reliability; negative contrast is anti-correlated — validating the contrastive signal.

<div align="center">
<img src="assets/imreli_comparison.png" width="500" alt="Image-level reliability comparison">
</div>

## Installation

**Requirements:** Python >= 3.9

```bash
pip install uq-detr
```

**Core dependencies:** `numpy >= 1.20`, `scipy >= 1.7` only. No PyTorch required.

To run the [tutorials](tutorials/) (HuggingFace DETR inference on COCO), install the optional tutorial dependencies:

```bash
pip install uq-detr[tutorials]
```

This additionally installs `torch >= 1.10`, `transformers >= 4.30`, `datasets >= 2.14`, `timm >= 0.9`, and `matplotlib >= 3.5`.

## Quick Start

```python
import uq_detr
from uq_detr import Detections, GroundTruth

# Collect predictions and ground truths across the dataset
all_detections = []
all_ground_truths = []

for image, annotation in dataset:  # your dataset loop
    pred_boxes, pred_scores = model(image)  # your model inference

    all_detections.append(Detections(
        boxes=pred_boxes,    # (N, 4) xyxy absolute pixels
        scores=pred_scores,  # (N, C) class probabilities
    ))
    all_ground_truths.append(GroundTruth(
        boxes=annotation["boxes"],   # (M, 4) xyxy absolute pixels
        labels=annotation["labels"], # (M,)
    ))

# Evaluate calibration over the entire dataset
print("OCE:   ", uq_detr.oce(all_detections, all_ground_truths).score)
print("D-ECE: ", uq_detr.dece(all_detections, all_ground_truths, tp_criterion="greedy").score)
print("LA-ECE:", uq_detr.laece(all_detections, all_ground_truths, tp_criterion="greedy").score)
print("LRP:   ", uq_detr.lrp(all_detections, all_ground_truths).score)
```

## Metrics

| Metric | Function | What it measures |
|--------|----------|------------------|
| **OCE** | `uq_detr.oce()` | Object-level Calibration Error --- Brier score per GT object. Evaluates model + post-processing jointly. |
| **D-ECE** | `uq_detr.dece()` | Detection ECE --- gap between confidence and precision. |
| **LA-ECE** | `uq_detr.laece()` | Label-Aware ECE --- per-class ECE with IoU-weighted accuracy. |
| **LRP** | `uq_detr.lrp()` | Localization Recall Precision --- combines FP, FN, and localization error. |
| **ContrastiveConf** | `uq_detr.contrastive_conf()` | Image-level reliability via positive/negative confidence contrast. |

## Working with DETR Outputs

For DETR models, pass all queries (before post-processing) and use `select()` to choose a post-processing strategy. This enables OCE's key feature: evaluating how well post-processing recovers the calibrated predictions.

```python
from uq_detr import select

# all_queries: Detections with all 900 DETR queries for one image
# Try different post-processing strategies
for thr in [0.1, 0.3, 0.5, 0.7]:
    filtered = select(all_queries, method="threshold", param=thr)
    score = uq_detr.oce([filtered], [gt]).score
    print(f"  threshold={thr} -> OCE={score:.4f}")
```

### Box Formats

`Detections` and `GroundTruth` expect **xyxy boxes in absolute pixel coordinates**. Use the built-in constructors for other formats:

```python
# From DETR output (normalized cxcywh)
det = Detections.from_cxcywh(pred_boxes, scores, image_size=(H, W))
gt = GroundTruth.from_cxcywh(gt_boxes, gt_labels, image_size=(H, W))

# From COCO-format annotations (absolute xywh)
gt = GroundTruth.from_xywh(coco_boxes, labels)

# Or convert manually
from uq_detr import box_convert
boxes_xyxy = box_convert(pred_boxes, "cxcywh", "xyxy", image_size=(H, W))
```

Supported formats: `"xyxy"`, `"xywh"`, `"cxcywh"`. Pass `image_size=(H, W)` to denormalize [0, 1] coordinates.

### Hungarian Matching

The package includes a numpy reimplementation of the Hungarian matcher used in DETR variants (e.g., Deformable-DETR):

```python
from uq_detr import hungarian_match

pred_idx, gt_idx = hungarian_match(
    pred_logits,   # (Q, C) raw logits
    pred_boxes,    # (Q, 4) cxcywh normalized
    gt_labels,     # (N,)
    gt_boxes,      # (N, 4) cxcywh normalized
)
```

## Flexible Input: Three Ways to Create Detections

`Detections` accepts different combinations of `scores` and `labels`:

```python
from uq_detr import Detections

# 1. Full class distributions (N, C) --- labels inferred via argmax
det = Detections(boxes=boxes, scores=class_probs)  # labels auto-computed

# 2. Full class distributions + explicit labels
det = Detections(boxes=boxes, scores=class_probs, labels=pred_labels)

# 3. Max-confidence (N,) + labels --- e.g., from supervision or COCO JSON
det = Detections(boxes=boxes, scores=max_confidences, labels=pred_labels)
```

Mode 1 and 2 give exact OCE (multi-class Brier score). Mode 3 uses a binary Brier approximation for OCE --- useful for outputs from frameworks like [supervision](https://github.com/roboflow/supervision) where only max-confidence is available. D-ECE, LA-ECE, and LRP work identically in all modes.

## Citation

```bibtex
@article{park2024uqdetr,
  title={Uncertainty Quantification in Detection Transformers: Object-Level Calibration and Image-Level Reliability},
  author={Park, Young-Jin and Sobolewski, Carson and Azizan, Navid},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026}
}
```

## License

MIT
