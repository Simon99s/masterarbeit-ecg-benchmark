import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLOT_DIR = PROJECT_ROOT / "Plots" / "analysis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# CONFIG
# ==========================================================

LABEL_ROOT = PROJECT_ROOT / "inference"

CSV_PATH = PROJECT_ROOT / "evaluation" / "benchmark_scores.csv"
METRIC = "challenge_metric"   # main metric

PROB_PATHS = {
    "jepa": PROJECT_ROOT / "inference" / "jepa" / "physionet_clean" / "physionet_clean" / "probs21.npy",
    "st_mem": PROJECT_ROOT / "inference" / "st-mem" / "physionet_clean" / "physionet_clean" / "probs21.npy",
    # "xecg": ... (later)
}

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

MODEL_COLORS = {
    "xecg": "#d62728",
    "xECG": "#d62728",

    "st-mem": "#2ca02c",
    "st_mem": "#2ca02c",
    "ST-MEM": "#2ca02c",

    "jepa": "#ff7f0e",
    "ECG-JEPA": "#ff7f0e",
}

# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(CSV_PATH)

# ----------------------------------------------------------
# Clean / Corrupted split
# ----------------------------------------------------------

df_clean = df[df["artifact"] == "physionet_clean"]
df_corr = df[df["artifact"] != "physionet_clean"]

# ----------------------------------------------------------
# Extract severity level (clean, Sev1, Sev2, Sev3)
# ----------------------------------------------------------

def extract_severity(s):
    if "Sev1" in s:
        return "Sev1"
    elif "Sev2" in s:
        return "Sev2"
    elif "Sev3" in s:
        return "Sev3"
    else:
        return "clean"

df["severity_level"] = df["severity"].apply(extract_severity)

# ==========================================================
# STEP 1 — CLEAN BASELINE
# ==========================================================

clean_scores = df_clean.groupby("model")[METRIC].mean().sort_values(ascending=False)

print("\n=== CLEAN PERFORMANCE ===")
print(clean_scores)

# ==========================================================
# STEP 2 — ABSOLUTE ROBUSTNESS
# ==========================================================

mean_corr = df_corr.groupby("model")[METRIC].mean()

absolute_df = pd.DataFrame({
    "clean": clean_scores,
    "mean_corrupted": mean_corr
})

print("\n=== ABSOLUTE ROBUSTNESS ===")
print(absolute_df)

# ==========================================================
# STEP 2.5 — ADDITIONAL ROBUSTNESS STATS
# ==========================================================

# Variance / stability
std_corr = df_corr.groupby("model")[METRIC].std()

# Worst-case (minimum performance across corruptions)
worst_corr = df_corr.groupby("model")[METRIC].min()

# Best-case (maximum performance across corruptions)
best_corr = df_corr.groupby("model")[METRIC].max()

# Add to main table
absolute_df["std_corrupted"] = std_corr
absolute_df["worst_case"] = worst_corr
absolute_df["best_case"] = best_corr

print("\n=== EXTENDED ROBUSTNESS STATS ===")
print(absolute_df)

# ==========================================================
# STEP 3 — RELATIVE ROBUSTNESS (SAFE VERSION)
# ==========================================================

S_random = clean_scores.get("random", None)

relative_drop = {}

for model in clean_scores.index:

    # skip random itself
    if model == "random":
        continue

    # skip if corrupted value missing
    if model not in mean_corr.index:
        print(f"[SKIP] {model}: no corrupted data")
        continue

    S_clean = clean_scores.get(model, None)
    S_corr = mean_corr.get(model, None)

    # skip if anything missing or nan
    if S_clean is None or S_corr is None or np.isnan(S_corr):
        print(f"[SKIP] {model}: invalid values")
        continue

    if S_random is None:
        print("[WARNING] random baseline missing → skipping relative metric")
        break

    try:
        rel = (S_clean - S_corr) / (S_clean - S_random)
        relative_drop[model] = rel
    except Exception as e:
        print(f"[SKIP] {model}: error {e}")
        continue

relative_df = pd.Series(relative_drop).sort_values()

print("\n=== RELATIVE ROBUSTNESS (LOWER = BETTER) ===")
print(relative_df)

# ==========================================================
# STEP 4 — SEVERITY CURVES
# ==========================================================

plt.figure(figsize=(10, 6))

# --- get random baseline ---
random_value = clean_scores.get("random", None)

for model in df["model"].unique():

    if model == "random":
        continue  # skip normal plotting

    model_df = df[df["model"] == model]

    sev_means = (
        model_df.groupby("severity_level")[METRIC]
        .mean()
        .reindex(["clean", "Sev1", "Sev2", "Sev3"])
    )

    plt.plot(sev_means.index, sev_means.values, marker="o", label=model, color=MODEL_COLORS.get(model, None))

# --- draw random as horizontal line ---
if random_value is not None:
    plt.axhline(
        y=random_value,
        linestyle="--",
        linewidth=2,
        label="Random Classifier"
    )

plt.title("Challenge Score Severity Curves")
plt.ylabel("Challenge Score")
plt.xlabel("Severity Level")
plt.legend()
plt.grid()
plt.savefig(PLOT_DIR / "Severity_Curves.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOT_DIR / "Severity_Curves.pdf", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# STEP 5 — CORRUPTION BREAKDOWN
# ==========================================================

corr_breakdown = (
    df_corr.groupby(["model", "artifact"])[METRIC]
    .mean()
    .unstack()
)

print("\n=== CORRUPTION BREAKDOWN ===")
print(corr_breakdown)

plt.figure(figsize=(12, 6))
sns.heatmap(corr_breakdown, annot=True, fmt=".3f", cmap="viridis")
plt.title("Performance per Corruption Type")
plt.savefig(PLOT_DIR / "Corruption_Breakdown.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOT_DIR / "Corruption_Breakdown.pdf", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# STEP 5.5 — BEST MODEL PER CORRUPTION
# ==========================================================

winner_per_artifact = corr_breakdown.idxmax()

print("\n=== BEST MODEL PER CORRUPTION ===")
print(winner_per_artifact)

# ==========================================================
# STEP 6 — RANKING STABILITY
# ==========================================================

clean_rank = clean_scores.rank(ascending=False)
corr_rank = mean_corr.rank(ascending=False)

ranking_df = pd.DataFrame({
    "clean_rank": clean_rank,
    "corrupted_rank": corr_rank
})

print("\n=== RANKING STABILITY ===")
print(ranking_df.sort_values("clean_rank"))

# ==========================================================
# STEP 7 — CORRELATION ANALYSIS (FULLY SAFE)
# ==========================================================

plt.figure(figsize=(6, 6))

for model in clean_scores.index:

    # skip random
    if model == "random":
        continue

    # skip if not in corrupted
    if model not in mean_corr.index:
        print(f"[SKIP CORR] {model}: no corrupted data")
        continue

    S_clean = clean_scores.get(model, None)
    S_corr = mean_corr.get(model, None)

    # skip invalid
    if S_clean is None or S_corr is None or np.isnan(S_corr):
        print(f"[SKIP CORR] {model}: invalid values")
        continue

    plt.scatter(S_clean, S_corr, label=model)

plt.xlabel("Clean Performance")
plt.ylabel("Corrupted Performance")
plt.title("Clean and Corrupted Performance")
plt.legend()
plt.grid()
plt.savefig(PLOT_DIR / "Clean_vs_Corrupted_Performance.pdf", dpi=300, bbox_inches="tight")
plt.savefig(PLOT_DIR / "Clean_vs_Corrupted_Performance.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# STEP 8 — PER-CORRUPTION SEVERITY CURVES (WITH CLEAN)
# ==========================================================

for artifact in ARTIFACT_ORDER:
    plt.figure(figsize=(8, 5))

    artifact_label = ARTIFACT_LABELS.get(artifact, artifact)

    for model in df["model"].unique():

        if model == "random":
            continue

        model_label = MODEL_LABELS.get(model, model)

        # --- clean value ---
        clean_val = df_clean[
            df_clean["model"] == model
        ][METRIC].mean()

        # --- corrupted values ---
        sub = df[
            (df["model"] == model) &
            (df["artifact"] == artifact)
        ]

        sev_means = (
            sub.groupby("severity_level")[METRIC]
            .mean()
            .reindex(["Sev1", "Sev2", "Sev3"])
        )

        # --- combine clean + corrupted ---
        x = ["Clean", "Sev1", "Sev2", "Sev3"]
        y = [clean_val] + list(sev_means.values)

        plt.plot(x, y, marker="o", label=model_label, color=MODEL_COLORS.get(model, None))

    plt.title(f"{artifact_label} — Severity Curves")
    plt.ylabel("Challenge Score")
    plt.xlabel("Severity level")
    plt.legend()
    plt.grid()

    plt.savefig(PLOT_DIR / f"{artifact}_Severity_Curve.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig(PLOT_DIR / f"{artifact}_Severity_Curve.png",
                dpi=300, bbox_inches="tight")

    plt.show()
# ==========================================================
# STEP 9 — IMAGENET-C STYLE RELATIVE ROBUSTNESS
# ==========================================================

# --- average over ALL corruptions (artifact + severity) ---
mean_corr_all = (
    df_corr.groupby("model")[METRIC]
    .mean()
)

# --- ECGFounder baseline ---
if "ECGFounder" not in mean_corr_all:
    raise ValueError("ECGFounder missing from data!")

baseline = mean_corr_all["ECGFounder"]

# --- relative scores ---
rel_scores = mean_corr_all / baseline

# --- x-axis: clean performance ---
x = clean_scores

# align indices
rel_scores = rel_scores.reindex(x.index)

# ==========================================================
# mean challenge score
# ==========================================================

plt.figure(figsize=(7, 6))

for model in x.index:

    if model not in rel_scores or np.isnan(rel_scores[model]):
        continue

    plt.scatter(
        x[model],
        rel_scores[model],
        s=80,
        label=model
    )

# --- ECGFounder reference line ---
plt.axhline(y=1.0, linestyle="--", linewidth=2, label="ECGFounder baseline")

plt.xlabel("Clean Challenge Score")
plt.ylabel("Relative Corruption Score")
plt.title("Clean and Relative Corrupted Performance")
plt.legend()
plt.grid()

plt.savefig(PLOT_DIR / "relative_robustness_ecgfounder.pdf", dpi=300, bbox_inches="tight")
plt.savefig(PLOT_DIR / "relative_robustness_ecgfounder.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# AUROC
# ==========================================================

METRIC_AUROC = "auroc"

plt.figure(figsize=(10, 6))

for model in df["model"].unique():

    if model == "random":
        continue

    model_df = df[df["model"] == model]

    sev_means = (
        model_df.groupby("severity_level")[METRIC_AUROC]
        .mean()
        .reindex(["clean", "Sev1", "Sev2", "Sev3"])
    )

    plt.plot(sev_means.index, sev_means.values, marker="o", label=model, color=MODEL_COLORS.get(model, None))


plt.title("AUROC Severity Curves")
plt.ylabel("AUROC")
plt.xlabel("Severity Level")
plt.legend()
plt.grid()
plt.savefig(PLOT_DIR / "AUROC_Severity_Curves.pdf", dpi=300, bbox_inches="tight")
plt.savefig(PLOT_DIR / "AUROC_Severity_Curves.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# AUROC PER-CORRUPTION (WITH CLEAN START)
# ==========================================================

METRIC_AUROC = "auroc"

for artifact in df_corr["artifact"].unique():

    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():

        if model == "random":
            continue

        # --- clean value ---
        clean_val = df_clean[
            df_clean["model"] == model
        ][METRIC_AUROC].mean()

        # --- corrupted values ---
        sub = df[
            (df["model"] == model) &
            (df["artifact"] == artifact)
        ]

        sev_means = (
            sub.groupby("severity_level")[METRIC_AUROC]
            .mean()
            .reindex(["Sev1", "Sev2", "Sev3"])
        )

        # --- combine clean + corrupted ---
        x = ["clean", "Sev1", "Sev2", "Sev3"]
        y = [clean_val] + list(sev_means.values)

        plt.plot(x, y, marker="o", label=model, color=MODEL_COLORS.get(model, None))

    pretty_artifact = ARTIFACT_LABELS.get(artifact, artifact)
    plt.title(f"{pretty_artifact} — AUROC Severity Curves")
    plt.ylabel("AUROC")
    plt.xlabel("Severity Level")
    plt.legend()
    plt.grid()

    plt.savefig(PLOT_DIR / f"{artifact}_AUROC_Severity_Curve.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig(PLOT_DIR / f"{artifact}_AUROC_Severity_Curve.png",
                dpi=300, bbox_inches="tight")

    plt.show()

# ==========================================================
# AUPRC
# ==========================================================

METRIC_AUPRC = "auprc"

plt.figure(figsize=(10, 6))

for model in df["model"].unique():

    if model == "random":
        continue

    model_df = df[df["model"] == model]

    sev_means = (
        model_df.groupby("severity_level")[METRIC_AUPRC]
        .mean()
        .reindex(["clean", "Sev1", "Sev2", "Sev3"])
    )

    plt.plot(
        sev_means.index,
        sev_means.values,
        marker="o",
        label=model, 
        color=MODEL_COLORS.get(model, None)
    )

plt.title("AUPRC Severity Curves")
plt.ylabel("AUPRC")
plt.xlabel("Severity Level")
plt.legend()
plt.grid()

plt.savefig(PLOT_DIR / "AUPRC_Severity_Curves.pdf",
            dpi=300, bbox_inches="tight")
plt.savefig(PLOT_DIR / "AUPRC_Severity_Curves.png",
            dpi=300, bbox_inches="tight")

plt.show()

# ==========================================================
# AUPRC PER-CORRUPTION (WITH CLEAN START)
# ==========================================================

METRIC_AUPRC = "auprc"

for artifact in df_corr["artifact"].unique():

    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():

        if model == "random":
            continue

        # --- clean value ---
        clean_val = df_clean[
            df_clean["model"] == model
        ][METRIC_AUPRC].mean()

        # --- corrupted values ---
        sub = df[
            (df["model"] == model) &
            (df["artifact"] == artifact)
        ]

        sev_means = (
            sub.groupby("severity_level")[METRIC_AUPRC]
            .mean()
            .reindex(["Sev1", "Sev2", "Sev3"])
        )

        # --- combine clean + corrupted ---
        x = ["clean", "Sev1", "Sev2", "Sev3"]
        y = [clean_val] + list(sev_means.values)

        plt.plot(x, y, marker="o", label=model, color=MODEL_COLORS.get(model, None))

    pretty_artifact = ARTIFACT_LABELS.get(artifact, artifact)
    plt.title(f"{pretty_artifact} — AUPRC Severity Curves")
    plt.ylabel("AUPRC")
    plt.xlabel("Severity Level")
    plt.legend()
    plt.grid()

    plt.savefig(PLOT_DIR / f"{artifact}_AUPRC_Severity_Curve.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig(PLOT_DIR / f"{artifact}_AUPRC_Severity_Curve.png",
                dpi=300, bbox_inches="tight")

    plt.show()

# ==========================================================
# F1 SCORE
# ==========================================================

METRIC_F1 = "f_measure"   # make sure this matches your CSV column

plt.figure(figsize=(10, 6))

for model in df["model"].unique():

    if model == "random":
        continue

    model_df = df[df["model"] == model]

    sev_means = (
        model_df.groupby("severity_level")[METRIC_F1]
        .mean()
        .reindex(["clean", "Sev1", "Sev2", "Sev3"])
    )

    plt.plot(
        sev_means.index,
        sev_means.values,
        marker="o",
        label=model, 
        color=MODEL_COLORS.get(model, None)
    )

plt.title("F1 Score Severity Curves")
plt.ylabel("F1 Score")
plt.xlabel("Severity Level")
plt.legend()
plt.grid()

plt.savefig(PLOT_DIR / "F1_Severity_Curves.pdf",
            dpi=300, bbox_inches="tight")
plt.savefig(PLOT_DIR / "F1_Severity_Curves.png",
            dpi=300, bbox_inches="tight")

plt.show()

# ==========================================================
# STEP X — F1 PER-CORRUPTION SEVERITY CURVES
# ==========================================================

METRIC_F1 = "f_measure"   # make sure this matches your CSV

for artifact in df_corr["artifact"].unique():

    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():

        if model == "random":
            continue

        # --- clean value ---
        clean_val = df_clean[
            df_clean["model"] == model
        ][METRIC_F1].mean()

        # --- corrupted values ---
        sub = df[
            (df["model"] == model) &
            (df["artifact"] == artifact)
        ]

        sev_means = (
            sub.groupby("severity_level")[METRIC_F1]
            .mean()
            .reindex(["Sev1", "Sev2", "Sev3"])
        )

        # --- combine clean + corrupted ---
        x = ["clean", "Sev1", "Sev2", "Sev3"]
        y = [clean_val] + list(sev_means.values)

        plt.plot(x, y, marker="o", label=model, color=MODEL_COLORS.get(model, None))

    pretty_artifact = ARTIFACT_LABELS.get(artifact, artifact)
    plt.title(f"{pretty_artifact} — F1 Severity Curves")
    plt.ylabel("F1 Score")
    plt.xlabel("Severity Level")
    plt.legend()
    plt.grid()

    plt.savefig(PLOT_DIR / f"{artifact}_F1_Severity_Curve.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig(PLOT_DIR / f"{artifact}_F1_Severity_Curve.png",
                dpi=300, bbox_inches="tight")

    plt.show()

# ==========================================================
# STEP 14.1 — PROBABILITY HISTOGRAMS
# ==========================================================

for model, path in PROB_PATHS.items():

    try:
        probs = np.load(path).flatten()
    except:
        print(f"[SKIP] {model}: probs not found")
        continue

    plt.figure(figsize=(6, 4))
    plt.hist(probs, bins=50)
    plt.title(f"{model} — Probability Distribution (Clean)")
    plt.xlabel("Probability")
    plt.ylabel("Count")

    plt.savefig(PLOT_DIR / f"{model}_prob_hist.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(PLOT_DIR / f"{model}_prob_hist.png", dpi=300, bbox_inches="tight")
    plt.show()

# ==========================================================
# STEP 14.2 — CONFIDENCE BINS
# ==========================================================

def compute_bins(probs):
    return {
        "low (0-0.2)": np.mean((probs >= 0.0) & (probs < 0.2)),
        "mid (0.2-0.8)": np.mean((probs >= 0.2) & (probs <= 0.8)),
        "high (0.8-1)": np.mean((probs > 0.8) & (probs <= 1.0)),
    }

print("\n=== CONFIDENCE BINS ===")

for model, path in PROB_PATHS.items():

    try:
        probs = np.load(path).flatten()
    except:
        print(f"[SKIP] {model}")
        continue

    bins = compute_bins(probs)

    print(f"\n{model}")
    for k, v in bins.items():
        print(f"{k}: {v:.4f}")


import glob
import os
import pandas as pd


# ==========================================================
# STEP X — GLOBAL HEATMAP (MODEL × SEVERITY vs ARTIFACT)
# ==========================================================

# Internal order
ARTIFACT_ORDER = [
    "clean",
    "physionet_em",
    "physionet_ma",
    "physionet_gn",
    "physionet_dn",
    "physionet_in"
]

SEVERITY_ORDER = ["clean", "Sev1", "Sev2", "Sev3"]

MODEL_ORDER = ["ECGFounder", "ECG-JEPA", "ST-MEM", "xECG"]

# Pretty labels
MODEL_LABELS = {
    "jepa": "ECG-JEPA",
    "st_mem": "ST-MEM",
    "st-mem": "ST-MEM",
    "xecg": "xECG",
    "ECGFounder": "ECGFounder",
    "ecgfounder": "ECGFounder"
}

ARTIFACT_LABELS = {
    "clean": "Clean",
    "physionet_em": "EM",
    "physionet_ma": "MA",
    "physionet_gn": "GN",
    "physionet_dn": "DN",
    "physionet_in": "IN"
}

df_plot = df.copy()

# ----------------------------------------------------------
# Remove random baseline
# ----------------------------------------------------------
df_plot = df_plot[df_plot["model"] != "random"]

# ----------------------------------------------------------
# Apply clean naming
# ----------------------------------------------------------
df_plot.loc[df_plot["artifact"] == "physionet_clean", "severity_level"] = "clean"
df_plot.loc[df_plot["artifact"] == "physionet_clean", "artifact"] = "clean"

# ----------------------------------------------------------
# Apply readable model names
# ----------------------------------------------------------
df_plot["model"] = df_plot["model"].replace(MODEL_LABELS)

# ----------------------------------------------------------
# Pivot: rows = model/severity, columns = artifact
# ----------------------------------------------------------
pivot = df_plot.pivot_table(
    index=["model", "severity_level"],
    columns="artifact",
    values=METRIC
)

# ----------------------------------------------------------
# Enforce artifact order
# ----------------------------------------------------------
pivot = pivot.reindex(columns=ARTIFACT_ORDER)

# ----------------------------------------------------------
# Enforce model and severity order
# ----------------------------------------------------------
pivot = pivot.reindex(
    pd.MultiIndex.from_product(
        [MODEL_ORDER, SEVERITY_ORDER],
        names=["Model", "Severity"]
    )
)

# ----------------------------------------------------------
# Rename artifact columns for display
# ----------------------------------------------------------
pivot = pivot.rename(columns=ARTIFACT_LABELS)

print("\n=== GLOBAL HEATMAP TABLE ===")
print(pivot)

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------
plt.figure(figsize=(10, 8))

sns.heatmap(
    pivot,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    cbar=True,
    linewidths=0.5,
    linecolor="white",
    annot_kws={"fontsize": 10}
)

plt.title("Benchmark Performance Across Models, Severities and Artifacts")
plt.ylabel("Model / Severity")
plt.xlabel("Artifact")

plt.savefig(PLOT_DIR / "global_heatmap_model_severity_artifact.png",
            dpi=300, bbox_inches="tight")
plt.savefig(PLOT_DIR / "global_heatmap_model_severity_artifact.pdf",
            dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# ENTROPY PER MODEL / ARTIFACT / SEVERITY
# ==========================================================

def compute_entropy(probs):
    p = np.clip(probs.astype(np.float64), 1e-8, 1 - 1e-8)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    entropy = np.nan_to_num(entropy)
    return entropy


ALL_RESULTS = []

MODELS = ["jepa", "st-mem", "xecg"]

for model in MODELS:

    paths = glob.glob(
        str(LABEL_ROOT / model / "**" / "probs21.npy"),
        recursive=True
    )

    print(f"\nFound {len(paths)} files for {model}")

    for path in paths:

        try:
            probs = np.load(path)
        except:
            print(f"[ERROR] loading {path}")
            continue

        # --------------------------------------------
        # extract folder names
        # --------------------------------------------
        severity_folder = os.path.basename(os.path.dirname(path))
        artifact_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))

        # --------------------------------------------
        # parse severity
        # --------------------------------------------
        if "Sev1" in severity_folder:
            severity = "Sev1"
        elif "Sev2" in severity_folder:
            severity = "Sev2"
        elif "Sev3" in severity_folder:
            severity = "Sev3"
        elif "clean" in severity_folder:
            severity = "clean"
        else:
            continue

        artifact = artifact_folder  # e.g. physionet_gn

        # --------------------------------------------
        # compute entropy
        # --------------------------------------------
        entropy = compute_entropy(probs)
        mean_entropy = np.mean(entropy)

        ALL_RESULTS.append({
            "model": model,
            "artifact": artifact,
            "severity": severity,
            "entropy": mean_entropy
        })


# ==========================================================
# CREATE DATAFRAME
# ==========================================================

df_entropy = pd.DataFrame(ALL_RESULTS)

print("\n=== RAW ENTROPY TABLE ===")
print(df_entropy.head())


# ==========================================================
# NICE TABLE (pivot)
# ==========================================================

pivot = df_entropy.pivot_table(
    index=["model", "artifact"],
    columns="severity",
    values="entropy"
)

print("\n=== ENTROPY TABLE (MODEL x ARTIFACT x SEVERITY) ===")
print(pivot)


# ==========================================================
# OPTIONAL: SAVE
# ==========================================================

pivot.to_csv("entropy_per_model_artifact_severity.csv")

import glob
import os
import numpy as np
import pandas as pd

# ==========================================================
# CONFIG (FIX PATHS!)
# ==========================================================

MODELS = ["jepa", "st-mem", "xecg"]



# ==========================================================
# HELPER
# ==========================================================

def extract_info(path):
    """
    Extract model, artifact, severity from path
    """
    severity_folder = os.path.basename(os.path.dirname(path))
    artifact_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))

    if "Sev1" in severity_folder:
        severity = "Sev1"
    elif "Sev2" in severity_folder:
        severity = "Sev2"
    elif "Sev3" in severity_folder:
        severity = "Sev3"
    elif "clean" in severity_folder:
        severity = "clean"
    else:
        return None, None

    return artifact_folder, severity


# ==========================================================
# MAIN
# ==========================================================

ALL_RESULTS = []

for model in MODELS:

    prob_paths = glob.glob(
        str(LABEL_ROOT / model / "**" / "probs21.npy"),
        recursive=True
    )

    print(f"\nFound {len(prob_paths)} files for {model}")

    for prob_path in prob_paths:

        try:
            probs = np.load(prob_path)
        except:
            print(f"[ERROR] loading probs {prob_path}")
            continue

        # --------------------------------------------------
        # LOAD LABELS (adjust if needed)
        # --------------------------------------------------
        label_path = prob_path.replace("probs21.npy", "labels21.npy")

        if not os.path.exists(label_path):
            print(f"[SKIP] labels missing for {prob_path}")
            continue

        labels = np.load(label_path)

        # --------------------------------------------------
        # EXTRACT META
        # --------------------------------------------------
        artifact, severity = extract_info(prob_path)

        if artifact is None:
            continue

        # --------------------------------------------------
        # PREDICTIONS
        # --------------------------------------------------
        preds = (probs > 0.5).astype(int)

        correct_mask = (preds == labels)
        wrong_mask = (preds != labels)

        # --------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------
        correct_conf = probs[correct_mask]
        wrong_conf = probs[wrong_mask]

        # handle edge cases
        if len(correct_conf) == 0 or len(wrong_conf) == 0:
            continue

        ALL_RESULTS.append({
            "model": model,
            "artifact": artifact,
            "severity": severity,
            "correct_conf": np.mean(correct_conf),
            "wrong_conf": np.mean(wrong_conf),
            "gap": np.mean(correct_conf) - np.mean(wrong_conf)
        })

