"""Tutorial 2: Ranking Models by Calibration Quality Using OCE

Compares calibration across multiple DETR variants using OCE, D-ECE,
LA-ECE, and LRP. Demonstrates how OCE provides a more informative
ranking than prediction-level metrics.
"""

import numpy as np
import matplotlib.pyplot as plt

import uq_detr
from uq_detr import select
from shared import load_coco_val, run_inference

# ── Config ──
MODELS = [
    "facebook/detr-resnet-50",
    "SenseTime/deformable-detr",
    "microsoft/conditional-detr-resnet-50",
]
MODEL_SHORT = ["DETR", "D-DETR", "Cond-DETR"]
N_IMAGES = 200
THRESHOLD = 0.3  # common threshold for comparison

# ── Load data ──
samples = load_coco_val(N_IMAGES)

# ── Run inference for each model ──
model_data = {}
for name, short in zip(MODELS, MODEL_SHORT):
    print(f"\n--- {short} ---")
    queries, gts = run_inference(name, samples)
    model_data[short] = (queries, gts)

# ── 1. Compare at a fixed threshold ──
print("\n" + "=" * 70)
print(f"Model Comparison at threshold={THRESHOLD}")
print("=" * 70)

print(f"{'Model':<12} {'OCE':>8} {'D-ECE':>8} {'LA-ECE':>8} {'LRP':>8} {'Avg#Det':>8}")
print("-" * 60)

oce_scores = []
for short in MODEL_SHORT:
    queries, gts = model_data[short]
    filtered = [select(q, method="threshold", param=THRESHOLD) for q in queries]
    avg_n = np.mean([d.num_detections for d in filtered])

    oce = uq_detr.oce(filtered, gts).score
    dece = uq_detr.dece(filtered, gts, tp_criterion="independent").score
    laece = uq_detr.laece(filtered, gts, tp_criterion="independent").score
    lrp = uq_detr.lrp(filtered, gts).score

    oce_scores.append(oce)
    print(f"{short:<12} {oce:8.4f} {dece:8.4f} {laece:8.4f} {lrp:8.4f} {avg_n:8.1f}")

# ── 2. OCE on optimal positive (oracle) ──
print("\n" + "=" * 70)
print("Oracle (Optimal Positive) Calibration — requires Hungarian matching")
print("=" * 70)

print(f"{'Model':<12} {'OCE*':>8} {'D-ECE*':>8} {'LA-ECE*':>8}")
print("-" * 40)

for short in MODEL_SHORT:
    queries, gts = model_data[short]

    # Compute Hungarian matching for each image
    oracle_dets = []
    for q, gt in zip(queries, gts):
        if gt.num_objects == 0 or q.num_detections == 0:
            oracle_dets.append(uq_detr.Detections(
                boxes=np.zeros((0, 4)),
                scores=np.zeros((0, q.scores.shape[1])) if q.has_class_distribution else np.zeros(0),
                labels=np.zeros(0, dtype=int),
            ))
            continue

        # Use Hungarian matching to find optimal positives
        # Need raw cxcywh normalized boxes — reconstruct from xyxy
        from uq_detr import box_convert
        pred_boxes_cx = box_convert(q.boxes, "xyxy", "cxcywh")
        gt_boxes_cx = box_convert(gt.boxes, "xyxy", "cxcywh")

        # Normalize (approx — use max coords as image size proxy)
        img_h = max(q.boxes[:, 3].max(), gt.boxes[:, 3].max()) * 1.05
        img_w = max(q.boxes[:, 2].max(), gt.boxes[:, 2].max()) * 1.05
        pred_norm = pred_boxes_cx / np.array([img_w, img_h, img_w, img_h])
        gt_norm = gt_boxes_cx / np.array([img_w, img_h, img_w, img_h])

        # Reconstruct logits from probs (inverse sigmoid)
        probs = q.scores
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs_clipped / (1 - probs_clipped))

        pred_idx, gt_idx = uq_detr.hungarian_match(logits, pred_norm, gt.labels, gt_norm)
        if len(pred_idx) > 0:
            oracle_dets.append(uq_detr.Detections(
                boxes=q.boxes[pred_idx], scores=q.scores[pred_idx], labels=q.labels[pred_idx]
            ))
        else:
            oracle_dets.append(uq_detr.Detections(
                boxes=np.zeros((0, 4)),
                scores=np.zeros((0, q.scores.shape[1])) if q.has_class_distribution else np.zeros(0),
                labels=np.zeros(0, dtype=int),
            ))

    oce = uq_detr.oce(oracle_dets, gts).score
    dece = uq_detr.dece(oracle_dets, gts, tp_criterion="independent").score
    laece = uq_detr.laece(oracle_dets, gts, tp_criterion="independent").score
    print(f"{short:<12} {oce:8.4f} {dece:8.4f} {laece:8.4f}")

# ── 3. Per-model threshold sweep plot ──
print("\n" + "=" * 70)
print("Per-Model Threshold Sweep")
print("=" * 70)

fig, axes = plt.subplots(1, len(MODEL_SHORT), figsize=(6 * len(MODEL_SHORT), 5), sharey=True)

thresholds = np.arange(0.05, 0.90, 0.05)

for ax, short in zip(axes, MODEL_SHORT):
    queries, gts = model_data[short]

    oce_vals, dece_vals = [], []
    for thr in thresholds:
        filtered = [select(q, method="threshold", param=thr) for q in queries]
        oce_vals.append(uq_detr.oce(filtered, gts).score)
        dece_vals.append(uq_detr.dece(filtered, gts, tp_criterion="independent").score)

    ax.plot(thresholds, oce_vals, "o-", label="OCE", linewidth=2, markersize=4)
    ax.plot(thresholds, dece_vals, "s--", label="D-ECE", linewidth=2, markersize=4)

    best_thr = thresholds[np.argmin(oce_vals)]
    ax.axvline(best_thr, color="tab:blue", alpha=0.3, linestyle=":")

    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_title(f"{short}\n(best OCE at thr={best_thr:.2f})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    print(f"  {short}: best OCE at thr={best_thr:.2f} (OCE={min(oce_vals):.4f})")

axes[0].set_ylabel("Score", fontsize=12)
plt.tight_layout()
plt.savefig("tutorial2_ranking_models.png", dpi=150, bbox_inches="tight")
print("\nSaved: tutorial2_ranking_models.png")
