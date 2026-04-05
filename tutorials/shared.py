"""Shared utilities for tutorials: load HF COCO dataset + run HF DETR inference."""

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection

import uq_detr
from uq_detr import Detections, GroundTruth


# ── COCO class ID mapping ──
# The original COCO annotations use sparse category IDs (1-90 with gaps).
# DETR models output logits over these 91 indices (0-90).
# The HuggingFace `detection-datasets/coco` remaps to contiguous 0-79.
# This table converts HF's contiguous IDs back to the original sparse IDs.
# (Same mapping used in torchvision, Detectron2, mmdetection.)
COCO_CONTIGUOUS_TO_SPARSE = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def load_coco_val(n_images=200):
    """Load a subset of COCO val from HuggingFace."""
    ds = load_dataset("detection-datasets/coco", split="val", streaming=True)
    samples = []
    for i, sample in enumerate(ds):
        if i >= n_images:
            break
        samples.append(sample)
    print(f"Loaded {len(samples)} COCO val images.")
    return samples


def run_inference(model_name, samples, device=None):
    """Run a HF DETR model on COCO samples.

    Returns:
        all_queries: list[Detections] — full query set per image
        ground_truths: list[GroundTruth] — GT with labels in model's class space
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    model.to(device).eval()

    num_logit_classes = None  # determined from first forward pass
    all_queries = []
    ground_truths = []

    for i, sample in enumerate(samples):
        image = sample["image"].convert("RGB")
        H, W = sample["height"], sample["width"]

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.squeeze(0).cpu().numpy()
        boxes = outputs.pred_boxes.squeeze(0).cpu().numpy()

        if num_logit_classes is None:
            num_logit_classes = logits.shape[-1]
            is_softmax = (num_logit_classes == model.config.num_labels + 1)
            print(f"  {model_name}: Q={logits.shape[0]}, C={num_logit_classes}, "
                  f"activation={'softmax' if is_softmax else 'sigmoid'}")

        # Activation
        if is_softmax:
            from scipy.special import softmax
            probs = softmax(logits, axis=-1)[:, :-1]
        else:
            probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

        all_queries.append(
            Detections.from_cxcywh(boxes, probs, image_size=(H, W))
        )

        # Ground truth: map HF contiguous IDs → sparse COCO IDs
        gt_boxes = np.array(sample["objects"]["bbox"], dtype=np.float64)
        gt_cats = sample["objects"]["category"]

        if num_logit_classes >= 91:
            gt_labels = np.array(
                [COCO_CONTIGUOUS_TO_SPARSE[c] for c in gt_cats], dtype=np.int64
            )
        else:
            gt_labels = np.array(gt_cats, dtype=np.int64)

        if len(gt_boxes) > 0:
            ground_truths.append(GroundTruth(boxes=gt_boxes, labels=gt_labels))
        else:
            ground_truths.append(GroundTruth(
                boxes=np.zeros((0, 4)), labels=np.zeros(0, dtype=np.int64)
            ))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)} images...")

    print(f"  Done: {model_name} on {len(samples)} images.")
    return all_queries, ground_truths
