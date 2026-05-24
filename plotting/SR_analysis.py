import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 10,
})

# ==========================================================
# CONFIG
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = PROJECT_ROOT / "Plots" / "sr_analysis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_ORDER = ["clean", "Sev1", "Sev2", "Sev3"]

COLORS = {
    "ST-MEM": "green",
    "JEPA": "orange",
    "xECG": "red"
}

# ==========================================================
# DATA (same as before)
# ==========================================================

data = {
    "ST-MEM": {
        "clean": {"recall": 0.12, "sr": 10329, "total": 187988},
        "Electrode Motion Artifact": {"Sev1": {"recall": 0.56, "sr": 33799, "total": 327952},
               "Sev2": {"recall": 0.70, "sr": 42213, "total": 382906},
               "Sev3": {"recall": 0.81, "sr": 50765, "total": 439862}},
        "Muscle Artifact": {"Sev1": {"recall": 0.40, "sr": 24827, "total": 253222},
               "Sev2": {"recall": 0.55, "sr": 32523, "total": 326892},
               "Sev3": {"recall": 0.65, "sr": 38372, "total": 409888}},
        "Gaussian Noise": {"Sev1": {"recall": 0.05, "sr": 4156, "total": 291258},
               "Sev2": {"recall": 0.01, "sr": 524, "total": 363526},
               "Sev3": {"recall": 0.00, "sr": 3, "total": 368831}},
        "Discretization Noise": {"Sev1": {"recall": 0.20, "sr": 10648, "total": 378055},
               "Sev2": {"recall": 0.06, "sr": 4334, "total": 305103},
               "Sev3": {"recall": 0.00, "sr": 563, "total": 180917}},
        "Impulse Noise": {"Sev1": {"recall": 0.05, "sr": 4776, "total": 356802},
               "Sev2": {"recall": 0.01, "sr": 696, "total": 536131},
               "Sev3": {"recall": 0.00, "sr": 1, "total": 404289}},
    },

    "JEPA": {
        "clean": {"recall": 0.16, "sr": 11519, "total": 187323},
        "Electrode Motion Artifact": {"Sev1": {"recall": 0.80, "sr": 47901, "total": 530179},
               "Sev2": {"recall": 0.61, "sr": 36751, "total": 591815},
               "Sev3": {"recall": 0.17, "sr": 11411, "total": 605237}},
        "Muscle Artifact": {"Sev1": {"recall": 0.72, "sr": 40854, "total": 396431},
               "Sev2": {"recall": 0.84, "sr": 50796, "total": 519807},
               "Sev3": {"recall": 0.52, "sr": 32061, "total": 581708}},
        "Gaussian Noise": {"Sev1": {"recall": 0.61, "sr": 32623, "total": 457289},
               "Sev2": {"recall": 0.30, "sr": 16385, "total": 569779},
               "Sev3": {"recall": 0.02, "sr": 1809, "total": 566724}},
        "Discretization Noise": {"Sev1": {"recall": 0.49, "sr": 23978, "total": 241266},
               "Sev2": {"recall": 0.31, "sr": 15042, "total": 261245},
               "Sev3": {"recall": 0.30, "sr": 13473, "total": 227208}},
        "Impulse Noise": {"Sev1": {"recall": 0.54, "sr": 31845, "total": 468132},
               "Sev2": {"recall": 0.41, "sr": 23276, "total": 622009},
               "Sev3": {"recall": 0.04, "sr": 3151, "total": 590885}},
    },

    "xECG": {
        "clean": {"recall": 0.14, "sr": 12084, "total": 166213},
        "Electrode Motion Artifact": {"Sev1": {"recall": 0.77, "sr": 54004, "total": 381648},
               "Sev2": {"recall": 0.92, "sr": 64572, "total": 471000},
               "Sev3": {"recall": 0.96, "sr": 70471, "total": 542886}},
        "Muscle Artifact": {"Sev1": {"recall": 0.55, "sr": 38440, "total": 257326},
               "Sev2": {"recall": 0.78, "sr": 54908, "total": 343037},
               "Sev3": {"recall": 0.94, "sr": 68583, "total": 448268}},
        "Gaussian Noise": {"Sev1": {"recall": 0.39, "sr": 28611, "total": 222194},
               "Sev2": {"recall": 0.80, "sr": 53844, "total": 342032},
               "Sev3": {"recall": 0.94, "sr": 66407, "total": 431629}},
        "Discretization Noise": {"Sev1": {"recall": 0.51, "sr": 32238, "total": 224813},
               "Sev2": {"recall": 0.66, "sr": 44139, "total": 294223},
               "Sev3": {"recall": 0.19, "sr": 19558, "total": 294114}},
        "Impulse Noise": {"Sev1": {"recall": 0.47, "sr": 33137, "total": 241870},
               "Sev2": {"recall": 0.78, "sr": 52013, "total": 332943},
               "Sev3": {"recall": 0.94, "sr": 66003, "total": 431435}},
    }
}

# ==========================================================
# PLOTTING
# ==========================================================

artifacts = ["Electrode Motion Artifact", "Muscle Artifact", "Gaussian Noise", "Discretization Noise", "Impulse Noise"]

for artifact in artifacts:

    fig, axs = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    for model in ["ST-MEM", "JEPA", "xECG"]:

        color = COLORS[model]

        recall_vals = []
        share_vals = []

        # clean
        clean = data[model]["clean"]
        recall_vals.append(clean["recall"])
        share_vals.append(clean["sr"] / clean["total"])

        # severities
        for sev in ["Sev1", "Sev2", "Sev3"]:
            d = data[model][artifact][sev]
            recall_vals.append(d["recall"])
            share_vals.append(d["sr"] / d["total"])

        axs[0].plot(SEVERITY_ORDER, recall_vals, marker="o", color=color, label=model)
        axs[1].plot(SEVERITY_ORDER, share_vals, marker="o", color=color, label=model)

    axs[0].set_title(f"{artifact} — SR Recall")
    axs[0].set_ylabel("Recall")
    axs[0].grid()
    axs[0].legend()

    axs[1].set_title("SR Prediction Share")
    axs[1].set_ylabel("SR / Total Predictions")
    axs[1].set_xlabel("Severity Level")
    axs[1].grid()

    plt.tight_layout()

    # ✅ SAVE PDF
    plt.savefig(PLOT_DIR / f"{artifact}_sr_analysis.pdf", bbox_inches="tight")

    plt.show()