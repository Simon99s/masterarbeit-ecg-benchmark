import pandas as pd
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

CSV_PATH = PROJECT_ROOT / "precision_recall_all.csv"
PLOT_DIR = PROJECT_ROOT / "Plots" / "precision_recall_analysis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_ORDER = ["clean", "Sev1", "Sev2", "Sev3"]

ARTIFACT_ORDER = ["physionet_em", "physionet_ma", "physionet_gn", "physionet_in", "physionet_dn"]

ARTIFACT_LABELS = {
    "physionet_em": "Electrode Motion Artifact",
    "physionet_ma": "Muscle Artifact",
    "physionet_gn": "Gaussian Noise",
    "physionet_dn": "Discretization Noise",
    "physionet_in": "Impulse Noise",
}

MODEL_LABELS = {
    "jepa": "ECG-JEPA",
    "st_mem": "ST-MEM",
    "st-mem": "ST-MEM",
    "xecg": "xECG",
    "ECGFounder": "ECGFounder",
    "ecgfounder": "ECGFounder"
}


# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(CSV_PATH)

# split clean + corrupted
df_clean = df[df["artifact"] == "physionet_clean"]
df_corr  = df[df["artifact"] != "physionet_clean"]

# ==========================================================
# HELPER: ADD CLEAN ROW
# ==========================================================

def add_clean_row(sub_df, model):
    clean_row = df_clean[df_clean["model"] == model].copy()
    clean_row["severity"] = "clean"
    return pd.concat([clean_row, sub_df], ignore_index=True)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
models = list(df["model"].unique())

MODEL_COLORS = {
    "xecg": "#d62728",
    "xECG": "#d62728",
    "jepa": "#ff7f0e",
    "JEPA": "#ff7f0e",
    "ECG-JEPA": "#ff7f0e",
    "st-mem": "#2ca02c",
    "st_mem": "#2ca02c",
    "ST-MEM": "#2ca02c",
}


# ==========================================================
# 1. MACRO PRECISION / RECALL VS SEVERITY
# ==========================================================

for artifact in ARTIFACT_ORDER:

    pretty_artifact = ARTIFACT_LABELS.get(artifact, artifact)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for model in models:

        sub = df_corr[
            (df_corr["model"] == model) &
            (df_corr["artifact"] == artifact)
        ]

        if sub.empty:
            continue

        sub = add_clean_row(sub, model)
        sub = sub.set_index("severity").reindex(SEVERITY_ORDER)

        color = MODEL_COLORS.get(model, "black")

        ax.plot(
            SEVERITY_ORDER,
            sub["precision_macro"],
            marker="o",
            linestyle="-",
            color=color,
            label=f"{model} Precision"
        )

        ax.plot(
            SEVERITY_ORDER,
            sub["recall_macro"],
            marker="x",
            linestyle="--",
            color=color,
            label=f"{model} Recall"
        )

    ax.set_title(f"{pretty_artifact}: Macro Precision and Recall")
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("Macro Score")
    ax.legend()
    ax.grid()

    fig.savefig(PLOT_DIR / f"{artifact}_macro_precision_recall.png", dpi=300)
    fig.savefig(PLOT_DIR / f"{artifact}_macro_precision_recall.pdf", dpi=300)

    plt.show()


# ==========================================================
# 2. FP / FN CURVES
# ==========================================================

for artifact in ARTIFACT_ORDER:

    pretty_artifact = ARTIFACT_LABELS.get(artifact, artifact)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for model in models:

        sub = df_corr[
            (df_corr["model"] == model) &
            (df_corr["artifact"] == artifact)
        ]

        if sub.empty:
            continue

        sub = add_clean_row(sub, model)
        sub = sub.set_index("severity").reindex(SEVERITY_ORDER)

        color = MODEL_COLORS.get(model, "black")

        ax.plot(
            SEVERITY_ORDER,
            sub["FP"],
            marker="o",
            linestyle="-",
            color=color,
            label=f"{model} FP"
        )

        ax.plot(
            SEVERITY_ORDER,
            sub["FN"],
            marker="x",
            linestyle="--",
            color=color,
            label=f"{model} FN"
        )

    ax.set_title(f"{pretty_artifact}: FP and FN")
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid()

    fig.savefig(PLOT_DIR / f"{artifact}_fp_fn.png", dpi=300)
    fig.savefig(PLOT_DIR / f"{artifact}_fp_fn.pdf", dpi=300)

    plt.show()


# ==========================================================
# 3. MACRO vs MICRO PRECISION GAP
# ==========================================================

df["precision_gap"] = df["precision_macro"] - df["precision_micro"]

df_clean = df[df["artifact"] == "physionet_clean"]
df_corr = df[df["artifact"] != "physionet_clean"]

for artifact in ARTIFACT_ORDER:

    pretty_artifact = ARTIFACT_LABELS.get(artifact, artifact)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for model in models:

        sub = df_corr[
            (df_corr["model"] == model) &
            (df_corr["artifact"] == artifact)
        ]

        if sub.empty:
            continue

        sub = add_clean_row(sub, model)
        sub = sub.set_index("severity").reindex(SEVERITY_ORDER)

        color = MODEL_COLORS.get(model, "black")

        ax.plot(
            SEVERITY_ORDER,
            sub["precision_gap"],
            marker="o",
            color=color,
            label=model
        )

    ax.set_title(f"{pretty_artifact}: Macro-Micro Precision Gap")
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("Macro Precision - Micro Precision")
    ax.legend()
    ax.grid()

    fig.savefig(PLOT_DIR / f"{artifact}_precision_gap.png", dpi=300)
    fig.savefig(PLOT_DIR / f"{artifact}_precision_gap.pdf", dpi=300)

    plt.show()


# ==========================================================
# 4. PRECISION vs RECALL GLOBAL
# ==========================================================

plt.figure(figsize=(6, 6))

for model in models:

    sub = df[df["model"] == model]

    if sub.empty:
        continue

    color = MODEL_COLORS.get(model, "black")

    plt.scatter(
        sub["recall_macro"],
        sub["precision_macro"],
        color=color,
        label=model
    )

plt.xlabel("Macro Recall")
plt.ylabel("Macro Precision")
plt.title("Macro Precision-Recall Trade-off")
plt.legend()
plt.grid()

plt.savefig(PLOT_DIR / "precision_recall_tradeoff.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOT_DIR / "precision_recall_tradeoff.pdf", dpi=300, bbox_inches="tight")

plt.show()