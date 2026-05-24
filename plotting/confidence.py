import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

PLOT_DIR = Path(__file__).resolve().parent.parent / "Plots" / "analysis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# CONFIG
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_ROOT = PROJECT_ROOT / "inference"

CSV_PATH = PROJECT_ROOT / "evaluation" / "benchmark_scores.csv"
METRIC = "challenge_metric"   # main metric

PROB_PATHS = {
    "jepa": INFERENCE_ROOT / "jepa" / "physionet_clean" / "physionet_clean" / "probs21.npy",
    "st_mem": INFERENCE_ROOT / "st-mem" / "physionet_clean" / "physionet_clean" / "probs21.npy",
    # "xecg": ... (later)
}

CONF_PLOT_DIR = PROJECT_ROOT / "Plots" / "confidence"
CONF_PLOT_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd

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


import glob
import os
import numpy as np
import pandas as pd

# ==========================================================
# CONFIG (FIX PATHS!)
# ==========================================================

MODELS = ["jepa", "st-mem", "xecg"]

LABEL_ROOT = INFERENCE_ROOT

HEA_ROOT = PROJECT_ROOT / "evaluation" / "evaluation_heas"

# IMPORTANT: use YOUR real mapping here
from mapping_rules import build_groups_from_tasks

tasks_txt = PROJECT_ROOT / "training" / "tasks.txt"

EVAL2021_DIR = PROJECT_ROOT / "evaluation" / "evaluation-2021-main"

def import_official_evaluator(eval_dir: Path):
    import sys, os
    sys.path.insert(0, str(eval_dir))
    old_cwd = os.getcwd()
    os.chdir(eval_dir)
    import evaluate_model
    return evaluate_model, old_cwd

evaluator, old_cwd = import_official_evaluator(EVAL2021_DIR)

weights_file = "weights_21.csv"
classes, weights = evaluator.load_weights(weights_file)

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

LABEL_CACHE = {}

for model in MODELS:

    prob_paths = glob.glob(
        str(INFERENCE_ROOT / model / "**" / "probs21.npy"),
        recursive=True
    )

    print(f"\nFound {len(prob_paths)} files for {model}")

    for prob_path in prob_paths:

        # -------------------------------
        # LOAD PROBS
        # -------------------------------
        try:
            probs = np.load(prob_path)
        except:
            print(f"[ERROR] loading probs {prob_path}")
            continue

        # -------------------------------
        # LOAD RECORD IDS
        # -------------------------------
        record_ids_path = prob_path.replace("probs21.npy", "record_ids.npy")

        if not os.path.exists(record_ids_path):
            print(f"[SKIP] record_ids missing for {prob_path}")
            continue

        record_ids = np.load(record_ids_path, allow_pickle=True).tolist()

        # clean IDs
        clean_ids = []
        for rid in record_ids:
            if isinstance(rid, bytes):
                rid = rid.decode()
            rid = Path(rid).stem
            clean_ids.append(rid)

        record_ids = clean_ids

        # ==================================================
        # LABEL CACHE (ONLY ONCE)
        # ==================================================

        cache_key = "physionet_labels"

        if cache_key not in LABEL_CACHE:

            print("Loading labels ONCE...")

            tmp_out = Path(prob_path).parent / "tmp_outputs"
            tmp_out.mkdir(exist_ok=True)

            for rid in record_ids:
                (tmp_out / f"{rid}.csv").touch()

            label_files, output_files = evaluator.find_challenge_files(
                str(HEA_ROOT),
                str(tmp_out)
            )

            # load labels ONCE
            labels = evaluator.load_labels(label_files, classes)

            # save ordered_ids for later alignment
            ordered_ids = [Path(f).stem for f in label_files]

            LABEL_CACHE[cache_key] = (labels, ordered_ids)

            import shutil
            shutil.rmtree(tmp_out)

        else:
            labels, ordered_ids = LABEL_CACHE[cache_key]


        # --------------------------------------------------
        # ALWAYS recompute indices (IMPORTANT)
        # --------------------------------------------------
        id_to_idx = {rid: i for i, rid in enumerate(record_ids)}

        try:
            indices = [id_to_idx[rid] for rid in ordered_ids]
        except KeyError as e:
            print(f"❌ Missing ID in mapping: {e}")
            continue

        # reorder probs for THIS file
        probs = probs[indices]

        # -------------------------------
        # DEBUG
        # -------------------------------
        # print(f"{model} | {prob_path}")
        # print("probs:", probs.shape)
        # print("labels:", labels.shape)

        if probs.shape != labels.shape:
            print("❌ SHAPE MISMATCH — SKIPPING")
            continue

        # -------------------------------
        # META
        # -------------------------------
        artifact, severity = extract_info(prob_path)

        if artifact is None:
            continue

        # -------------------------------
        # PREDICTIONS
        # -------------------------------
        preds = (probs > 0.5).astype(int)

        # probability of predicted class
        pred_conf = np.where(preds == 1, probs, 1 - probs)

        # correctness per prediction
        correct_mask = (preds == labels)
        wrong_mask   = (preds != labels)

        correct_conf = pred_conf[correct_mask]
        wrong_conf   = pred_conf[wrong_mask]

        if len(correct_conf) == 0 or len(wrong_conf) == 0:
            continue

        # -------------------------------
        # SAVE RESULT
        # -------------------------------
        ALL_RESULTS.append({
            "model": model,
            "artifact": artifact,
            "severity": severity,
            "correct_conf": np.mean(correct_conf),
            "wrong_conf": np.mean(wrong_conf),
            "gap": np.mean(correct_conf) - np.mean(wrong_conf)
        })


# ==========================================================
# DATAFRAME
# ==========================================================

df_conf = pd.DataFrame(ALL_RESULTS)

print("\n=== RAW CONFIDENCE TABLE ===")
print(df_conf.head())


# ==========================================================
# PIVOT TABLE (VERY IMPORTANT)
# ==========================================================
df_conf = df_conf.astype({
    "correct_conf": "float64",
    "wrong_conf": "float64",
    "gap": "float64"
})

pivot_conf = df_conf.pivot_table(
    index=["model", "artifact"],
    columns="severity",
    values=["correct_conf", "wrong_conf", "gap"],
    aggfunc="mean"
)

print("\n=== CONFIDENCE (CORRECT vs WRONG) ===")
print(pivot_conf)


# ==========================================================
# SAVE
# ==========================================================

pivot_conf.to_csv("confidence_correct_vs_wrong.csv")

# ==========================================================
# CONFIDENCE HEATMAPS IN GLOBAL HEATMAP FORMAT
# ==========================================================

ARTIFACT_ORDER_CONF = [
    "clean",
    "physionet_em",
    "physionet_ma",
    "physionet_gn",
    "physionet_dn",
    "physionet_in",
]

SEVERITY_ORDER = ["clean", "Sev1", "Sev2", "Sev3"]

MODEL_ORDER = ["ECGFounder", "ECG-JEPA", "ST-MEM", "xECG"]

MODEL_LABELS = {
    "jepa": "ECG-JEPA",
    "st_mem": "ST-MEM",
    "st-mem": "ST-MEM",
    "xecg": "xECG",
    "ECGFounder": "ECGFounder",
    "ecgfounder": "ECGFounder",
}

ARTIFACT_LABELS_SHORT = {
    "clean": "Clean",
    "physionet_em": "EM",
    "physionet_ma": "MA",
    "physionet_gn": "GN",
    "physionet_dn": "DN",
    "physionet_in": "IN",
}

SEVERITY_LABELS = {
    "clean": "Clean",
    "Sev1": "Sev1",
    "Sev2": "Sev2",
    "Sev3": "Sev3",
}


def make_global_confidence_heatmap(df_conf, value_col, title, filename, vmin=None, vmax=None):
    df_plot = df_conf.copy()

    # readable model names
    df_plot["model"] = df_plot["model"].replace(MODEL_LABELS)

    # clean naming
    df_plot.loc[df_plot["artifact"] == "physionet_clean", "artifact"] = "clean"
    df_plot.loc[df_plot["severity"] == "clean", "severity"] = "clean"

    # pivot: rows = model/severity, columns = artifact
    pivot = df_plot.pivot_table(
        index=["model", "severity"],
        columns="artifact",
        values=value_col,
        aggfunc="mean"
    )

    # enforce artifact order
    pivot = pivot.reindex(columns=ARTIFACT_ORDER_CONF)

    # enforce model/severity order
    pivot = pivot.reindex(
        pd.MultiIndex.from_product(
            [MODEL_ORDER, SEVERITY_ORDER],
            names=["Model", "Severity"]
        )
    )

    # remove models that are completely missing
    pivot = pivot.dropna(how="all")

    # rename columns
    pivot = pivot.rename(columns=ARTIFACT_LABELS_SHORT)

    # make severity label prettier
    pivot.index = pd.MultiIndex.from_tuples(
        [(model, SEVERITY_LABELS.get(sev, sev)) for model, sev in pivot.index],
        names=["Model", "Severity"]
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar=True,
        linewidths=0.5,
        linecolor="white",
        vmin=vmin,
        vmax=vmax,
        annot_kws={"fontsize": 10}
    )

    plt.title(title)
    plt.ylabel("Model / Severity")
    plt.xlabel("Artifact")

    plt.savefig(CONF_PLOT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(CONF_PLOT_DIR / f"{filename}.pdf", dpi=300, bbox_inches="tight")

    plt.show()


# Correct confidence: range is naturally 0 to 1
make_global_confidence_heatmap(
    df_conf=df_conf,
    value_col="correct_conf",
    title="Correct Confidence Across Models, Severities and Artifacts",
    filename="correct_confidence_global_heatmap",
)

# Wrong confidence: also 0 to 1, so use the same color scale
make_global_confidence_heatmap(
    df_conf=df_conf,
    value_col="wrong_conf",
    title="Wrong Confidence Across Models, Severities and Artifacts",
    filename="wrong_confidence_global_heatmap",
)

# Gap can be smaller and may be negative, but still use viridis for visual consistency
make_global_confidence_heatmap(
    df_conf=df_conf,
    value_col="gap",
    title="Confidence Gap Across Models, Severities and Artifacts",
    filename="confidence_gap_global_heatmap"
)

#line plot of confidence gap by severity for each model/artifact
import pandas as pd

df_plot = df_conf.copy()

# enforce order
severity_order = ["clean", "Sev1", "Sev2", "Sev3"]
df_plot["severity"] = pd.Categorical(df_plot["severity"], categories=severity_order, ordered=True)

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df_plot,
    x="severity",
    y="correct_conf",
    hue="model",
    style="artifact",
    markers=True
)

plt.title("Correct Confidence vs Severity")
plt.tight_layout()

plt.savefig(CONF_PLOT_DIR / "correct_conf_trend.png", dpi=300)
plt.savefig(CONF_PLOT_DIR / "correct_conf_trend.pdf")
plt.close()

#gap lineplot
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df_plot,
    x="severity",
    y="gap",
    hue="model",
    style="artifact",
    markers=True
)

plt.axhline(0, linestyle="--")

plt.title("Confidence Gap vs Severity")
plt.tight_layout()

plt.savefig(CONF_PLOT_DIR / "gap_trend.png", dpi=300)
plt.savefig(CONF_PLOT_DIR / "gap_trend.pdf")
plt.close()


# ==========================================================
# SUMMARY TABLE (MODEL LEVEL)
# ==========================================================

# mean performance (from your CSV)
mean_corr = df_corr.groupby("model")[METRIC].mean()
clean_scores = df_clean.groupby("model")[METRIC].mean()

# mean confidence
conf_summary = df_conf.groupby("model")[["correct_conf", "wrong_conf", "gap"]].mean()

summary = pd.concat([
    clean_scores.rename("clean"),
    mean_corr.rename("mean_corrupted"),
    conf_summary
], axis=1)

print("\n=== MODEL SUMMARY ===")
print(summary)

summary.to_csv(CONF_PLOT_DIR / "confidence_summary.csv")


# ==========================================================
# PERFORMANCE vs WRONG CONFIDENCE
# ==========================================================

plt.figure(figsize=(7, 6))

for model in summary.index:
    if model == "random":
        continue

    plt.scatter(
        summary.loc[model, "mean_corrupted"],
        summary.loc[model, "wrong_conf"],
        s=120,
        label=model
    )

plt.xlabel("Mean Corrupted Performance")
plt.ylabel("Wrong Confidence")
plt.title("Performance vs Overconfidence")
plt.legend()
plt.grid()

plt.savefig(CONF_PLOT_DIR / "performance_vs_wrong_conf.png", dpi=300)
plt.savefig(CONF_PLOT_DIR / "performance_vs_wrong_conf.pdf")
plt.close()


# ==========================================================
# PERFORMANCE vs CONFIDENCE GAP
# ==========================================================

plt.figure(figsize=(7, 6))

for model in summary.index:
    if model == "random":
        continue

    plt.scatter(
        summary.loc[model, "mean_corrupted"],
        summary.loc[model, "gap"],
        s=120,
        label=model
    )

plt.axhline(0, linestyle="--")

plt.xlabel("Mean Corrupted Performance")
plt.ylabel("Confidence Gap")
plt.title("Robustness vs Confidence Separation")
plt.legend()
plt.grid()

plt.savefig(CONF_PLOT_DIR / "performance_vs_gap.png", dpi=300)
plt.savefig(CONF_PLOT_DIR / "performance_vs_gap.pdf")
plt.close()



# ==========================================================
# PERFORMANCE + WRONG CONFIDENCE vs SEVERITY
# ==========================================================

for model in df_conf["model"].unique():

    plt.figure(figsize=(8, 5))

    # performance
    perf = (
        df[df["model"] == model]
        .groupby("severity_level")[METRIC]
        .mean()
        .reindex(["clean", "Sev1", "Sev2", "Sev3"])
    )

    # wrong confidence
    conf = (
        df_conf[df_conf["model"] == model]
        .groupby("severity")["wrong_conf"]
        .mean()
        .reindex(["clean", "Sev1", "Sev2", "Sev3"])
    )

    plt.plot(perf.index, perf.values, marker="o", label="performance")
    plt.plot(conf.index, conf.values, marker="o", label="wrong_conf")

    plt.title(f"{model}: Performance vs Overconfidence")
    plt.xlabel("Severity")
    plt.legend()
    plt.grid()

    plt.savefig(CONF_PLOT_DIR / f"{model}_perf_vs_conf.png", dpi=300)
    plt.savefig(CONF_PLOT_DIR / f"{model}_perf_vs_conf.pdf")
    plt.close()