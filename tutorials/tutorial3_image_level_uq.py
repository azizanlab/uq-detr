"""Tutorial 3: Image-Level Uncertainty Quantification with ContrastiveConf

Demonstrates how to quantify per-image reliability using ContrastiveConf,
including fitting the lambda parameter on a validation split and evaluating
on a test split.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import uq_detr
from uq_detr import select
from shared import load_coco_val, run_inference

# ── Config ──
MODEL = "SenseTime/deformable-detr"
N_IMAGES = 400  # use more images for meaningful correlation
THRESHOLD = 0.3

# ── Load data and run inference ──
samples = load_coco_val(N_IMAGES)
all_queries, ground_truths = run_inference(MODEL, samples)

# ── Split into val / test ──
n_val = N_IMAGES // 2
val_queries, test_queries = all_queries[:n_val], all_queries[n_val:]
val_gts, test_gts = ground_truths[:n_val], ground_truths[n_val:]
print(f"\nVal: {n_val} images, Test: {N_IMAGES - n_val} images")

# ── Compute per-image OCE as a proxy for reliability ──
# (In practice you'd use per-image AP from pycocotools; we use -OCE here
#  since lower OCE = more reliable, and it requires no extra dependencies.)
print("\nComputing per-image OCE as reliability proxy...")

def compute_per_image_reliability(queries, gts, thr):
    """Negated per-image OCE: higher = more reliable."""
    reliabilities = []
    for q, gt in zip(queries, gts):
        filtered = select(q, method="threshold", param=thr)
        oce = uq_detr.oce([filtered], [gt]).score
        reliabilities.append(-oce)  # negate so higher = better
    return np.array(reliabilities)

val_reliability = compute_per_image_reliability(val_queries, val_gts, THRESHOLD)
test_reliability = compute_per_image_reliability(test_queries, test_gts, THRESHOLD)

# ── Step 1: Fit lambda on validation set ──
print("\n" + "=" * 70)
print("Step 1: Fitting lambda on validation set")
print("=" * 70)

best_lam, best_pcc = uq_detr.fit_lambda(
    val_queries, val_reliability,
    method="threshold", param=THRESHOLD,
)
print(f"  Best lambda: {best_lam} (PCC={best_pcc:.4f})")

# Also show the full sweep
lambdas = [0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
print(f"\n  {'Lambda':>8} {'PCC (val)':>12}")
print(f"  {'-'*22}")
for lam in lambdas:
    scores = uq_detr.contrastive_conf(val_queries, method="threshold", param=THRESHOLD, lambda_=lam)
    valid = ~np.isnan(val_reliability)
    pcc = pearsonr(scores[valid], val_reliability[valid])[0]
    marker = " <-- best" if lam == best_lam else ""
    print(f"  {lam:8.2f} {pcc:12.4f}{marker}")

# ── Step 2: Evaluate on test set ──
print("\n" + "=" * 70)
print("Step 2: Evaluating on test set")
print("=" * 70)

# Conf+ only (lambda=0)
test_conf_pos = uq_detr.contrastive_conf(
    test_queries, method="threshold", param=THRESHOLD, lambda_=0.0
)
# ContrastiveConf with fitted lambda
test_contrast = uq_detr.contrastive_conf(
    test_queries, method="threshold", param=THRESHOLD, lambda_=best_lam
)

valid = ~np.isnan(test_reliability)
pcc_pos = pearsonr(test_conf_pos[valid], test_reliability[valid])[0]
pcc_contrast = pearsonr(test_contrast[valid], test_reliability[valid])[0]

print(f"  Conf+ only (lambda=0):         PCC = {pcc_pos:.4f}")
print(f"  ContrastiveConf (lambda={best_lam}):  PCC = {pcc_contrast:.4f}")
print(f"  Improvement: {pcc_contrast - pcc_pos:+.4f}")

# ── Step 3: Identify most/least reliable images ──
print("\n" + "=" * 70)
print("Step 3: Most and Least Reliable Images (by ContrastiveConf)")
print("=" * 70)

order = np.argsort(test_contrast)
print("\n  Least reliable (lowest ContrastiveConf):")
for idx in order[:5]:
    n_gt = test_gts[idx].num_objects
    n_det = select(test_queries[idx], method="threshold", param=THRESHOLD).num_detections
    print(f"    Image {idx}: ContrastiveConf={test_contrast[idx]:.4f}, "
          f"reliability={test_reliability[idx]:.4f}, #GT={n_gt}, #det={n_det}")

print("\n  Most reliable (highest ContrastiveConf):")
for idx in order[-5:]:
    n_gt = test_gts[idx].num_objects
    n_det = select(test_queries[idx], method="threshold", param=THRESHOLD).num_detections
    print(f"    Image {idx}: ContrastiveConf={test_contrast[idx]:.4f}, "
          f"reliability={test_reliability[idx]:.4f}, #GT={n_gt}, #det={n_det}")

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Lambda sweep
ax = axes[0]
pccs = []
for lam in lambdas:
    scores = uq_detr.contrastive_conf(val_queries, method="threshold", param=THRESHOLD, lambda_=lam)
    valid = ~np.isnan(val_reliability)
    pccs.append(pearsonr(scores[valid], val_reliability[valid])[0])
ax.plot(lambdas, pccs, "o-", linewidth=2, markersize=5)
ax.axvline(best_lam, color="tab:red", alpha=0.5, linestyle="--", label=f"best={best_lam}")
ax.set_xlabel("Lambda", fontsize=12)
ax.set_ylabel("PCC with reliability", fontsize=12)
ax.set_title("Lambda Sweep (validation)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Conf+ scatter
ax = axes[1]
ax.scatter(test_conf_pos[valid], test_reliability[valid], alpha=0.3, s=15, color="tab:orange")
ax.set_xlabel("Conf+ (positive confidence)", fontsize=12)
ax.set_ylabel("Image reliability (-OCE)", fontsize=12)
ax.set_title(f"Conf+ vs Reliability (PCC={pcc_pos:.3f})", fontsize=13)
ax.grid(alpha=0.3)

# ContrastiveConf scatter
ax = axes[2]
ax.scatter(test_contrast[valid], test_reliability[valid], alpha=0.3, s=15, color="tab:blue")
ax.set_xlabel(f"ContrastiveConf (lambda={best_lam})", fontsize=12)
ax.set_ylabel("Image reliability (-OCE)", fontsize=12)
ax.set_title(f"ContrastiveConf vs Reliability (PCC={pcc_contrast:.3f})", fontsize=13)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("tutorial3_image_level_uq.png", dpi=150, bbox_inches="tight")
print("\nSaved: tutorial3_image_level_uq.png")
