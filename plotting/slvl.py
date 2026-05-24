import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import PchipInterpolator

# --------- user settings ----------
artifact_str = "in"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
plots_dir = PROJECT_ROOT / "Plots" / "Severity_lvls"
plots_dir.mkdir(parents=True, exist_ok=True)

# NEW VALUES
clean_value = 0.526699
random_value = 0.202456

# FIXED severity targets (already computed)
LEVELS = {
    0.75: 0.4456315,
    0.50: 0.364573,
    0.25: 0.2835145,
}

SEVERITY_LABELS = {
    0.75: "Severity Level 1",
    0.50: "Severity Level 2",
    0.25: "Severity Level 3",
}

THRESH_COLORS = {
    0.75: "tab:green",   # Level 1 (mild)
    0.50: "tab:orange",  # Level 2 (medium)
    0.25: "tab:red",     # Level 3 (severe)
}

# measured datapoints (SNR -> score)
data = {
    0.007: 0.447752,
    0.008: 0.437963,
    0.009: 0.428241,
    0.015: 0.381297,
    0.017: 0.368265,
    0.018: 0.362271,
    0.039: 0.285149,
    0.04: 0.283178,
}

# chosen severity SNRs
chosen_levels = {
    0.75: 9,
    0.50: 4,
    0.25: -6,
}

# ----------------------------------

# --------------------------
# Benchmark bins (visual axis only)
# --------------------------
snr_bins = [10, 8, 6, 4, 2, 0, -2]

def snr_to_benchmark_index(snr):
    if snr in snr_bins:
        return 1 + snr_bins.index(snr)

    for i in range(len(snr_bins) - 1):
        hi, lo = snr_bins[i], snr_bins[i + 1]
        if hi >= snr >= lo:
            t = (hi - snr) / (hi - lo)
            return 1 + i + t

    if snr < snr_bins[-1]:
        return 1 + len(snr_bins)

    return None


# --------------------------
# Build arrays
# --------------------------
snr_real = np.array(list(data.keys()))
y = np.array(list(data.values()))

# sort high → low SNR
order = np.argsort(snr_real)
snr_real = snr_real[order]
y = y[order]

x = np.array([snr_to_benchmark_index(s) for s in snr_real])


# --------------------------
# Interpolation
# --------------------------
order_interp = np.argsort(snr_real)
snr_interp = snr_real[order_interp]
y_interp = y[order_interp]

interp_fn = PchipInterpolator(snr_interp, y_interp)

snr_dense = np.linspace(
    snr_interp.min(),
    snr_interp.max(),
    300
)
y_dense = interp_fn(snr_dense)


# --------------------------
# Crossing helper
# --------------------------
def find_crossing_interp(x_dense, y_dense, y_target):
    diff = y_dense - y_target
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(sign_change) == 0:
        return None

    i = sign_change[0]
    x0, x1 = x_dense[i], x_dense[i+1]
    y0, y1 = y_dense[i], y_dense[i+1]

    return x0 + (y_target - y0) * (x1 - x0) / (y1 - y0)


# # --------------------------
# # BENCHMARK AXIS PLOT
# # --------------------------
# plt.figure(figsize=(9, 5))

# plt.plot(x, y, linewidth=2, alpha=0.7)
# plt.scatter(x, y, s=60, zorder=3)

# xticks = [0] + [1 + i for i in range(len(snr_bins))]
# xtick_labels = ["clean"] + [str(s) for s in snr_bins]
# plt.xticks(xticks, xtick_labels)

# plt.xlabel("SNR (dB)", fontsize=16)
# plt.ylabel("Physionet 2021 Challenge Score", fontsize=16)
# plt.title("ECGFounder Performance under Gaussian Noise", fontsize=18)
# plt.grid(True)

# for p, y_target in LEVELS.items():
#     color = THRESH_COLORS[p]

#     snr_cross = find_crossing_interp(snr_dense, y_dense, y_target)
#     x_cross = snr_to_benchmark_index(snr_cross)

#     plt.axhline(
#         y=y_target,
#         color=color,
#         linestyle=":",
#         linewidth=1.5,
#         alpha=0.8
#     )

#     plt.axvline(
#         x=x_cross,
#         color=color,
#         linestyle="--",
#         linewidth=1.8,
#         label=SEVERITY_LABELS[p]
#     )


# plt.legend(
#     loc="lower right",
#     framealpha=0.9
# )
# plt.tight_layout()

# plt.savefig(
#     plots_dir / f"challenge_score_vs_snr_{artifact_str}_benchmark.svg",
#     format="svg",
#     bbox_inches="tight"
# )
# plt.close()


# --------------------------
# INTERPOLATED PLOT (REAL SNR)
# --------------------------
plt.figure(figsize=(9, 5))

plt.plot(snr_dense, y_dense, linewidth=2, label="Interpolated")
plt.scatter(snr_real, y, s=60, label="Measured")

plt.axhline(
    y=clean_value,
    color="black",
    linestyle="-",
    linewidth=1.6,
    label="Clean"
)
plt.axhline(
    y=random_value,
    color="gray",
    linestyle="--",
    linewidth=1.6,
    label="Random"
)

for p, y_target in LEVELS.items():
    color = THRESH_COLORS[p]

    snr_cross = find_crossing_interp(snr_dense, y_dense, y_target)

    plt.axhline(
        y=y_target,
        color=color,
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label=SEVERITY_LABELS[p]
    )

    if snr_cross is not None:
        plt.axvline(
            x=snr_cross,
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8
        )



plt.xlabel("Impulse Probability p", fontsize=16)
plt.ylabel("Physionet 2021 Challenge Score", fontsize=16)
plt.title("ECGFounder Performance under Impulse Noise", fontsize=18)
plt.grid(True)
plt.legend(
    loc="upper right",
    framealpha=0.9
)
plt.tight_layout()
plots_dir = Path(r"C:\Users\simon\Desktop\bm\Plots\Severity_lvls")
plots_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    plots_dir / "challenge_score_vs_p_in_benchmark.svg",
    format="svg",
    bbox_inches="tight"
)
plt.close()