import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# =========================
# PATHS (EDIT THESE)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "data" / "Benchmark"

paths = {
    "clean": os.path.join(BASE, "physionet_clean", "physionet_clean"),
    "em": os.path.join(BASE, "physionet_em", "physionet_em_Sev2"),
    "ma": os.path.join(BASE, "physionet_ma", "physionet_ma_Sev2"),
}

record_rel = r"chapman_shaoxing\g1\JS00009"

# =========================
# SETTINGS
# =========================
LEAD_IDX = 1        # Lead II
SECONDS = 30
FS = 100
N = SECONDS * FS

# =========================
# LOAD FUNCTION
# =========================
def load_signal(folder):
    path = os.path.join(folder, record_rel)
    sig, _ = wfdb.rdsamp(path)
    sig = sig[:, LEAD_IDX]
    sig = sig[:N]
    return sig

# =========================
# LOAD DATA
# =========================
signals = {
    "Clean": load_signal(paths["clean"]),
    "Electrode Motion": load_signal(paths["em"]),
    "Muscle Artifact": load_signal(paths["ma"]),
}

# =========================
# GLOBAL Y-LIMITS (IMPORTANT)
# =========================
ymin = min(sig.min() for sig in signals.values())
ymax = max(sig.max() for sig in signals.values())

# Add padding (10%)
yrange = ymax - ymin
ymin -= 0.1 * yrange
ymax += 0.1 * yrange

# =========================
# PLOT
# =========================
plt.figure(figsize=(10, 8))

for i, (name, sig) in enumerate(signals.items()):
    plt.subplot(len(signals), 1, i+1)

    t = np.arange(len(sig)) / FS

    plt.plot(t, sig, color='black', linewidth=1)

    # Consistent axes
    plt.xlim(t[0], t[-1])
    plt.ylim(ymin, ymax)
    plt.margins(x=0)

    # Clean labeling inside plot
    plt.text(0.01, 0.85, name,
             transform=plt.gca().transAxes,
             fontsize=11,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.ylabel("mV")
    plt.grid(alpha=0.3)

    if i == len(signals) - 1:
        plt.xlabel("Time (s)")

# Layout fix
plt.tight_layout()

# =========================
# SAVE
# =========================
save_path = PROJECT_ROOT / "Plots" / "artifact_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.show()