"""Tutorial 1: Finding the Best Post-Processing Scheme Using OCE

Demonstrates OCE's key advantage: jointly evaluating model calibration
and post-processing effectiveness. We sweep confidence thresholds,
showing that OCE identifies a practical sweet spot while D-ECE
misleadingly favors discarding almost everything.
"""

import numpy as np
import matplotlib.pyplot as plt

import uq_detr
from uq_detr import select
from shared import load_coco_val, run_inference

# ── Config ──
MODEL = "SenseTime/deformable-detr"
N_IMAGES = 200

# ── Load data and run inference ──
samples = load_coco_val(N_IMAGES)
all_queries, ground_truths = run_inference(MODEL, samples)

# ── Threshold sweep ──
print("\n" + "=" * 70)
print("Threshold Sweep: OCE vs D-ECE vs LA-ECE vs LRP")
print("=" * 70)

thresholds = np.arange(0.05, 0.95, 0.05)
results = {"thr": [], "oce": [], "dece": [], "laece": [], "lrp": [], "n_kept": []}

for thr in thresholds:
    filtered = [select(q, method="threshold", param=thr) for q in all_queries]
    avg_kept = np.mean([d.num_detections for d in filtered])

    results["thr"].append(thr)
    results["oce"].append(uq_detr.oce(filtered, ground_truths).score)
    results["dece"].append(uq_detr.dece(filtered, ground_truths, tp_criterion="independent").score)
    results["laece"].append(uq_detr.laece(filtered, ground_truths, tp_criterion="independent").score)
    results["lrp"].append(uq_detr.lrp(filtered, ground_truths).score)
    results["n_kept"].append(avg_kept)

    print(f"  thr={thr:.2f}  kept={avg_kept:6.1f}  OCE={results['oce'][-1]:.4f}  "
          f"D-ECE={results['dece'][-1]:.4f}  LRP={results['lrp'][-1]:.4f}")

best_oce_idx = int(np.argmin(results["oce"]))
best_dece_idx = int(np.argmin(results["dece"]))

print(f"\nBest OCE:   thr={results['thr'][best_oce_idx]:.2f} -> OCE={results['oce'][best_oce_idx]:.4f}")
print(f"Best D-ECE: thr={results['thr'][best_dece_idx]:.2f} -> D-ECE={results['dece'][best_dece_idx]:.4f}")
print(">>> D-ECE favors high thresholds (discards almost everything) — the pitfall!")
print(f">>> OCE finds a practical sweet spot at thr={results['thr'][best_oce_idx]:.2f}.")

# ── Top-k comparison ──
print("\n" + "=" * 70)
print("Top-k Comparison")
print("=" * 70)

ks = [5, 10, 20, 50, 100, 200, 300]
for k in ks:
    filtered = [select(q, method="topk", param=k) for q in all_queries]
    oce = uq_detr.oce(filtered, ground_truths).score
    dece = uq_detr.dece(filtered, ground_truths, tp_criterion="independent").score
    print(f"  top-{k:<4d}  OCE={oce:.4f}  D-ECE={dece:.4f}")

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(results["thr"], results["oce"], "o-", label="OCE", linewidth=2)
ax.plot(results["thr"], results["dece"], "s--", label="D-ECE", linewidth=2)
ax.plot(results["thr"], results["laece"], "^--", label="LA-ECE", linewidth=2)
ax.plot(results["thr"], results["lrp"], "d--", label="LRP", linewidth=2)
ax.axvline(results["thr"][best_oce_idx], color="tab:blue", alpha=0.3, linestyle=":")
ax.set_xlabel("Confidence Threshold", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title(f"Tutorial 1: Threshold Sweep ({MODEL.split('/')[-1]})", fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(results["thr"], results["n_kept"], "k-o", linewidth=2)
ax.set_xlabel("Confidence Threshold", fontsize=12)
ax.set_ylabel("Avg # Detections Kept", fontsize=12)
ax.set_title("Detections Retained", fontsize=14)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("tutorial1_best_postprocessing.png", dpi=150, bbox_inches="tight")
print("\nSaved: tutorial1_best_postprocessing.png")
