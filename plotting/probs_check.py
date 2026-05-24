import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ==========================================================
# CONFIG
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROBS_PATH = PROJECT_ROOT / "inference" / "st-mem" / "physionet_gn" / "physionet_gn_Sev2" / "probs21.npy"

# OPTIONAL, if it exists in the same folder
LABELS_PATH = PROBS_PATH.with_name("labels21.npy")

SAVE_DIR = PROJECT_ROOT / "Plots" / "prob_analysis"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD
# ==========================================================

probs = np.load(PROBS_PATH)
print(f"Loaded probs: {probs.shape}")

labels = None
if os.path.exists(LABELS_PATH):
    labels = np.load(LABELS_PATH)
    print(f"Loaded labels: {labels.shape}")
else:
    print("⚠️ No labels found -> skipping pos/neg analysis")

# ==========================================================
# 1. GLOBAL STATS
# ==========================================================

print("\n=== GLOBAL STATS ===")
print("Mean:", probs.mean())
print("Std :", probs.std())
print("Min :", probs.min())
print("Max :", probs.max())

plt.figure()
plt.hist(probs.flatten(), bins=100)
plt.title("Global Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Count")
plt.savefig(SAVE_DIR / "global_hist.png")
plt.close()

# ==========================================================
# 2. PER-CLASS STATS
# ==========================================================

print("\n=== PER CLASS STATS ===")

mean_per_class = probs.mean(axis=0)
std_per_class  = probs.std(axis=0)
max_per_class  = probs.max(axis=0)

for i in range(probs.shape[1]):
    print(f"Class {i:02d} | mean={mean_per_class[i]:.4f} | std={std_per_class[i]:.4f} | max={max_per_class[i]:.4f}")

plt.figure()
plt.bar(range(len(mean_per_class)), mean_per_class)
plt.title("Mean Probability per Class")
plt.xlabel("Class")
plt.ylabel("Mean Probability")
plt.savefig(SAVE_DIR / "per_class_mean.png")
plt.close()

plt.figure()
plt.bar(range(len(max_per_class)), max_per_class)
plt.title("Max Probability per Class")
plt.xlabel("Class")
plt.ylabel("Max Probability")
plt.savefig(SAVE_DIR / "per_class_max.png")
plt.close()

# ==========================================================
# 3. MAX PER RECORD
# ==========================================================

print("\n=== MAX PER RECORD ===")

max_per_record = probs.max(axis=1)

print("Mean max prob:", max_per_record.mean())
print("Std  max prob:", max_per_record.std())

plt.figure()
plt.hist(max_per_record, bins=100)
plt.title("Max Probability per Record")
plt.xlabel("Max Probability")
plt.ylabel("Count")
plt.savefig(SAVE_DIR / "max_per_record_hist.png")
plt.close()

# ==========================================================
# 4. POS vs NEG ANALYSIS (if labels exist)
# ==========================================================

if labels is not None:
    print("\n=== POS vs NEG ===")

    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    print("Pos mean:", pos_probs.mean())
    print("Neg mean:", neg_probs.mean())

    plt.figure()
    plt.hist(pos_probs, bins=100, alpha=0.5, label="Positive")
    plt.hist(neg_probs, bins=100, alpha=0.5, label="Negative")
    plt.legend()
    plt.title("Positive vs Negative Probabilities")
    plt.savefig(SAVE_DIR / "pos_vs_neg.png")
    plt.close()

# ==========================================================
# 5. QUICK THRESHOLD CHECK
# ==========================================================

print("\n=== THRESHOLD CHECK (0.5) ===")

above_thresh = (probs > 0.5).sum()
total = probs.size

print(f"Values > 0.5: {above_thresh} / {total} ({100*above_thresh/total:.4f}%)")

# ==========================================================
# DONE
# ==========================================================

print(f"\nPlots saved to: {SAVE_DIR}")