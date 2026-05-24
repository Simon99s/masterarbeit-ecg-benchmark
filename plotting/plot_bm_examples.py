import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

# Lead names (standard 12-lead order)
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# ==========================================================
# LOAD FUNCTION
# ==========================================================

def load_record(path_no_ext):
    """
    Load WFDB record given path without .dat/.hea
    """
    record = wfdb.rdrecord(str(path_no_ext))
    signal = record.p_signal  # shape: (T, 12)
    return signal


# ==========================================================
# PLOTTING
# ==========================================================

def plot_12_leads(signal, title, save_path):
    """
    Plots 12 leads stacked vertically and saves as pdf
    """
    fig, axes = plt.subplots(12, 1, figsize=(12, 10), sharex=True)

    for i in range(12):
        axes[i].plot(signal[:, i], linewidth=0.8)
        axes[i].set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=20, fontsize=8)
        axes[i].grid(True, linestyle="--", linewidth=0.3)

    axes[-1].set_xlabel("Samples")
    fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, format="pdf")
    plt.close(fig)


# ==========================================================
# MAIN
# ==========================================================

def main():
    # ---------------------------
    # CLEAN
    # ---------------------------
    clean_path = BASE_PATH / "physionet_clean" / "physionet_clean" / RECORD_REL_PATH
    print(f"Loading CLEAN: {clean_path}")

    signal = load_record(clean_path)

    out_file = OUT_DIR / "clean.pdf"
    plot_12_leads(signal, "Clean", out_file)

    # ---------------------------
    # CORRUPTIONS
    # ---------------------------
    for corr in CORRUPTIONS:
        for sev in SEVERITIES:
            path = (
                BASE_PATH
                / f"physionet_{corr}"
                / f"physionet_{corr}_{sev}"
                / RECORD_REL_PATH
            )

            print(f"Loading {corr.upper()} {sev}: {path}")

            try:
                signal = load_record(path)

                title = f"{corr.upper()} - {sev}"
                out_file = OUT_DIR / f"{corr}_{sev}.pdf"

                plot_12_leads(signal, title, out_file)

            except Exception as e:
                print(f"❌ Failed for {corr} {sev}: {e}")


if __name__ == "__main__":
    main()