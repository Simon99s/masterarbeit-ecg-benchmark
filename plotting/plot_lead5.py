import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# PLOT STYLE
# ==========================================================
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 17,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "figure.titlesize": 15,
})

# ==========================================================
# CONFIG
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = PROJECT_ROOT / "data" / "Benchmark"
OUT_DIR = PROJECT_ROOT / "Plots" / "Bm_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RECORD_REL_PATH = Path(r"chapman_shaoxing\g1\JS00001")

CORRUPTIONS = ["dn", "gn", "in", "ma", "em"]
SEVERITIES = ["Sev1", "Sev2", "Sev3"]

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

MAX_SAMPLES = 5000  # ~10 seconds at 500 Hz


# ==========================================================
# LOAD FUNCTION
# ==========================================================

def load_lead(path_no_ext, lead_index):
    record = wfdb.rdrecord(str(path_no_ext))
    sig = record.p_signal[:MAX_SAMPLES, lead_index]
    fs = record.fs
    unit = record.units[lead_index] if record.units is not None else "mV"
    return sig, fs, unit


# ==========================================================
# PLOT FUNCTION
# ==========================================================

def plot_one_lead(lead_index, lead_name):
    signals = []
    titles = []

    clean_path = BASE_PATH / "physionet_clean" / "physionet_clean" / RECORD_REL_PATH
    print(f"Loading CLEAN {lead_name}: {clean_path}")

    sig, fs, unit = load_lead(clean_path, lead_index)
    signals.append(sig)
    titles.append("Clean")

    for corr in CORRUPTIONS:
        for sev in SEVERITIES:
            path = (
                BASE_PATH
                / f"physionet_{corr}"
                / f"physionet_{corr}_{sev}"
                / RECORD_REL_PATH
            )

            print(f"Loading {lead_name} {corr.upper()} {sev}: {path}")

            try:
                sig, _, _ = load_lead(path, lead_index)
                signals.append(sig)
                titles.append(f"{corr.upper()}-{sev}")
            except Exception as e:
                print(f"❌ Failed for {lead_name} {corr} {sev}: {e}")

    global_min = min(s.min() for s in signals)
    global_max = max(s.max() for s in signals)

    fig, axes = plt.subplots(4, 4, figsize=(17, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    time_axis = np.arange(MAX_SAMPLES) / fs

    for i, (sig, title) in enumerate(zip(signals, titles)):
        axes[i].plot(time_axis[:len(sig)], sig, color="black", linewidth=1)
        axes[i].set_title(title, fontsize=17)
        axes[i].set_ylim(global_min, global_max)
        axes[i].grid(True, linestyle="--", linewidth=0.3)
        axes[i].tick_params(axis="both", labelsize=13)

    for j in range(len(signals), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"ECG Lead {lead_name} – Clean vs Corruptions", fontsize=20)
    fig.supxlabel("Time (s)", fontsize=20)
    fig.supylabel(f"Amplitude ({unit})", fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    safe_lead_name = lead_name.replace("a", "a")
    out_path = OUT_DIR / f"{safe_lead_name}_comparison.pdf"

    plt.savefig(out_path, format="pdf")
    plt.close(fig)

    print(f"✅ Saved: {out_path}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    for lead_index, lead_name in enumerate(LEAD_NAMES):
        plot_one_lead(lead_index, lead_name)

    print("\n✅ Finished plotting all 12 leads.")


if __name__ == "__main__":
    main()