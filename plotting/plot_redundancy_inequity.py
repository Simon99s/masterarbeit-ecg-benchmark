from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


# ============================================================
# CONFIG
# ============================================================

# Script is located directly in bm/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "evaluation" / "feature_collapse_metrics.csv"

OUT_DIR = PROJECT_ROOT / "Plots" / "redundancy_inequity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["ecgfounder", "jepa", "st-mem", "xecg"]

ARTIFACT_ORDER = [
    "physionet_em",
    "physionet_ma",
    "physionet_gn",
    "physionet_in",
    "physionet_dn",
]

ARTIFACT_LABELS = {
    "physionet_em": "EM",
    "physionet_ma": "MA",
    "physionet_gn": "GN",
    "physionet_in": "IN",
    "physionet_dn": "DN",
    "physionet_clean": "Clean",
}

MODEL_LABELS = {
    "ecgfounder": "ECGFounder",
    "jepa": "ECG-JEPA",
    "st-mem": "ST-MEM",
    "xecg": "xECG",
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

SEVERITY_ORDER = ["Clean", "Sev1", "Sev2", "Sev3"]


# ============================================================
# METRICS
# ============================================================

METRICS_FEATURE_SPACE = {
    "feature_redundancy_global": "Global Feature Redundancy",
    "centroid_inequity_global": "Global Centroid Inequity",
}

METRICS_PREDICTION_SPACE = {
    "prediction_inequity_global_norm": "Global Prediction Inequity",
    "mean_probability_mass_global": "Global Mean Probability Mass",
    "prediction_binary_entropy_norm_global": "Global Prediction Binary Entropy",
}

METRICS_COMBINED_5X5 = {
    "feature_redundancy_global": "Global Feature Redundancy",
    "centroid_inequity_global": "Global Centroid Inequity",
    # "prediction_inequity_global_norm": "Global Prediction Inequity",
    # "mean_probability_mass_global": "Global Mean Probability Mass",
    # "prediction_binary_entropy_norm_global": "Global Prediction Binary Entropy",
}


# ============================================================
# DATA PREPARATION
# ============================================================

def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["model"] = df["model"].astype(str).str.lower().str.strip()
    df["artifact"] = df["artifact"].astype(str).str.lower().str.strip()
    df["severity"] = df["severity"].astype(str).str.strip()

    return df


def extract_severity(row) -> str:
    """
    Converts severity values like:
    - physionet_em_Sev1
    - physionet_dn_Sev3
    - Sev1
    - clean / physionet_clean

    into:
    - Sev1, Sev2, Sev3, Clean
    """

    artifact = str(row["artifact"]).lower()
    severity = str(row["severity"]).lower()

    if "clean" in artifact or "clean" in severity:
        return "Clean"

    for sev in ["Sev1", "Sev2", "Sev3"]:
        if sev.lower() in severity:
            return sev

    return str(row["severity"])


def add_clean_rows_to_each_artifact(df: pd.DataFrame) -> pd.DataFrame:
    """
    If clean rows exist only once as physionet_clean, this duplicates them
    into each artifact panel so every curve starts with Clean.
    """

    df = df.copy()

    clean_df = df[
        (df["artifact"].str.contains("clean", case=False, na=False))
        | (df["severity_clean"] == "Clean")
    ].copy()

    non_clean_df = df[df["severity_clean"] != "Clean"].copy()

    if clean_df.empty:
        print("Warning: No clean rows found. Plots will start at Sev1.")
        return non_clean_df

    duplicated_clean_rows = []

    for artifact in ARTIFACT_ORDER:
        temp = clean_df.copy()
        temp["artifact"] = artifact
        temp["severity_clean"] = "Clean"
        duplicated_clean_rows.append(temp)

    clean_expanded = pd.concat(duplicated_clean_rows, ignore_index=True)

    return pd.concat([clean_expanded, non_clean_df], ignore_index=True)


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = normalize_strings(df)

    df["severity_clean"] = df.apply(extract_severity, axis=1)

    df = add_clean_rows_to_each_artifact(df)

    # Keep only relevant artifacts and severity levels
    df = df[df["artifact"].isin(ARTIFACT_ORDER)]
    df = df[df["severity_clean"].isin(SEVERITY_ORDER)]

    all_metrics = {}
    all_metrics.update(METRICS_FEATURE_SPACE)
    all_metrics.update(METRICS_PREDICTION_SPACE)

    missing_cols = [col for col in all_metrics.keys() if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing columns in CSV:\n"
            + "\n".join(missing_cols)
            + "\n\nYou probably need to rerun feature_collapse_metrics.py with FORCE_RECOMPUTE = True."
        )

    # Convert metric columns to numeric
    for col in all_metrics.keys():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use categorical ordering
    df["severity_clean"] = pd.Categorical(
        df["severity_clean"],
        categories=SEVERITY_ORDER,
        ordered=True,
    )

    df["artifact"] = pd.Categorical(
        df["artifact"],
        categories=ARTIFACT_ORDER,
        ordered=True,
    )

    df["model"] = pd.Categorical(
        df["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    df = df.sort_values(["artifact", "model", "severity_clean"])

    return df


# ============================================================
# LINE PLOTS
# ============================================================

def plot_metric_grid(
    df: pd.DataFrame,
    metrics: dict,
    filename: str,
    title: str,
):
    """
    Creates one combined line-plot figure.

    Rows    = metrics
    Columns = artifacts
    Lines   = models
    X-axis  = Clean, Sev1, Sev2, Sev3
    """

    n_rows = len(metrics)
    n_cols = len(ARTIFACT_ORDER)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.2 * n_rows),
        sharex=True,
        squeeze=False,
    )

    for row_idx, (metric_col, metric_label) in enumerate(metrics.items()):
        for col_idx, artifact in enumerate(ARTIFACT_ORDER):
            ax = axes[row_idx][col_idx]

            sub = df[df["artifact"] == artifact]

            for model in MODEL_ORDER:
                model_sub = sub[sub["model"] == model].copy()

                if model_sub.empty:
                    continue

                model_sub = model_sub.sort_values("severity_clean")

                ax.plot(
                    model_sub["severity_clean"].astype(str),
                    model_sub[metric_col],
                    marker="o",
                    linewidth=2,
                    label=MODEL_LABELS.get(model, model),
                    color=MODEL_COLORS.get(model, None),
                )

            if row_idx == 0:
                ax.set_title(ARTIFACT_LABELS.get(artifact, artifact))

            if col_idx == 0:
                ax.set_ylabel(metric_label)

            ax.set_xlabel("Severity")
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(MODEL_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(title, y=1.07, fontsize=16)

    plt.tight_layout()

    out_path_pdf = OUT_DIR / f"{filename}.pdf"
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path_pdf}")


def plot_single_metric_per_figure(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
):
    """
    Creates one separate line-plot figure per metric.

    Columns = artifacts
    Lines   = models
    X-axis  = Clean, Sev1, Sev2, Sev3
    """

    fig, axes = plt.subplots(
        1,
        len(ARTIFACT_ORDER),
        figsize=(4.0 * len(ARTIFACT_ORDER), 3.5),
        sharey=True,
        squeeze=False,
    )

    axes = axes[0]

    for ax, artifact in zip(axes, ARTIFACT_ORDER):
        sub = df[df["artifact"] == artifact]

        for model in MODEL_ORDER:
            model_sub = sub[sub["model"] == model].copy()

            if model_sub.empty:
                continue

            model_sub = model_sub.sort_values("severity_clean")

            ax.plot(
                model_sub["severity_clean"].astype(str),
                model_sub[metric_col],
                marker="o",
                linewidth=2,
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model, None),
            )

        ax.set_title(ARTIFACT_LABELS.get(artifact, artifact))
        ax.set_xlabel("Severity")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(metric_label)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(MODEL_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )

    fig.suptitle(metric_label, y=1.12, fontsize=15)

    plt.tight_layout()

    out_path_pdf = OUT_DIR / f"{metric_col}.pdf"
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path_pdf}")


# ============================================================
# HEATMAPS
# ============================================================

def plot_metric_heatmap_by_model(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
):
    """
    Creates one heatmap figure for one metric.

    Panels  = models
    Rows    = artifacts
    Columns = Clean, Sev1, Sev2, Sev3
    Color   = metric value
    """

    n_models = len(MODEL_ORDER)

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(4.2 * n_models, 4.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    axes = axes[0]

    metric_values = df[metric_col].dropna()

    if metric_values.empty:
        print(f"Warning: No valid values for {metric_col}. Skipping heatmap.")
        plt.close(fig)
        return

    # Shared scale across models for fair comparison
    vmin = metric_values.min()
    vmax = metric_values.max()

    last_im = None

    for ax, model in zip(axes, MODEL_ORDER):
        model_df = df[df["model"] == model].copy()

        heatmap_data = (
            model_df
            .pivot_table(
                index="artifact",
                columns="severity_clean",
                values=metric_col,
                aggfunc="mean",
                observed=False,
            )
            .reindex(index=ARTIFACT_ORDER, columns=SEVERITY_ORDER)
        )

        data = heatmap_data.to_numpy(dtype=float)

        last_im = ax.imshow(
            data,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(MODEL_LABELS.get(model, model))

        ax.set_xticks(range(len(SEVERITY_ORDER)))
        ax.set_xticklabels(SEVERITY_ORDER, rotation=45, ha="right")

        ax.set_yticks(range(len(ARTIFACT_ORDER)))
        ax.set_yticklabels([
            ARTIFACT_LABELS.get(a, a) for a in ARTIFACT_ORDER
        ])

        # Value annotations
        for i in range(len(ARTIFACT_ORDER)):
            for j in range(len(SEVERITY_ORDER)):
                value = data[i, j]

                if pd.isna(value):
                    text = "–"
                else:
                    text = f"{value:.3f}"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    cbar = fig.colorbar(
        last_im,
        ax=axes,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label(metric_label)

    fig.suptitle(metric_label, fontsize=16, y=1.05)

    plt.tight_layout()

    out_path_pdf = OUT_DIR / f"{metric_col}_heatmap.pdf"
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path_pdf}")


# ============================================================
# COMBINED 5 x 5 PLOT
# ============================================================

def plot_combined_5x5_severity_curves(df: pd.DataFrame):

    PLOT_LABELS = {
        "Global Feature Redundancy": "Feature Redundancy",
        "Global Centroid Inequity": "Centroid Inequity",
    }

    metrics = METRICS_COMBINED_5X5

    ARTIFACT_LAYOUT = [
        ("physionet_em", (0, 0)),
        ("physionet_ma", (0, 1)),
        ("physionet_gn", (1, 0)),
        ("physionet_in", (1, 1)),
        ("physionet_dn", (1, 2)),
    ]

    for metric_col, metric_label in metrics.items():

        metric_values = df[metric_col].dropna()
        y_min = metric_values.min()
        y_max = metric_values.max()
        padding = 0.05 * (y_max - y_min)
        if padding == 0:
            padding = 0.01

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(14, 8),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        axes_flat = axes.flatten()

        for artifact, (r, c) in ARTIFACT_LAYOUT:
            ax = axes[r][c]
            sub = df[df["artifact"] == artifact]

            for model in MODEL_ORDER:
                model_sub = sub[sub["model"] == model].copy()

                if model_sub.empty:
                    continue

                model_sub = model_sub.sort_values("severity_clean")

                ax.plot(
                    model_sub["severity_clean"].astype(str),
                    model_sub[metric_col],
                    marker="o",
                    linewidth=2,
                    color=MODEL_COLORS.get(model, None),
                    label=MODEL_LABELS.get(model, model),
                )

            ax.set_title(ARTIFACT_LABELS.get(artifact, artifact))
            ax.set_ylim(y_min - padding, y_max + padding)
            ax.set_xlabel("Severity")
            ax.grid(True, alpha=0.3)

        axes[0][2].axis("off")

        axes[0][0].set_ylabel(PLOT_LABELS.get(metric_label, metric_label), fontsize=14)
        axes[1][0].set_ylabel(PLOT_LABELS.get(metric_label, metric_label), fontsize=14)

        handles, labels = axes_flat[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(MODEL_ORDER),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

        fig.suptitle(
            PLOT_LABELS.get(metric_label, metric_label),
            y=1.06,
            fontsize=16,
        )

        fig.subplots_adjust(
            top=0.84,
            hspace=0.45,
            wspace=0.25
        )

        out_name = metric_col.replace("_global", "")
        out_path_pdf = OUT_DIR / f"{out_name}_2x3.pdf"

        fig.savefig(out_path_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {out_path_pdf}")


# ============================================================
# MAIN
# ============================================================

def main():
    df = prepare_dataframe(CSV_PATH)

    print("Loaded rows:", len(df))
    print()
    print("Available model/artifact/severity combinations:")
    print(df[["model", "artifact", "severity_clean"]].drop_duplicates())

    all_metrics = {}
    all_metrics.update(METRICS_FEATURE_SPACE)
    all_metrics.update(METRICS_PREDICTION_SPACE)

    # --------------------------------------------------------
    # Combined line plots
    # --------------------------------------------------------

    plot_metric_grid(
        df=df,
        metrics=METRICS_FEATURE_SPACE,
        filename="feature_space_global_redundancy_inequity",
        title="Global Feature-Space Redundancy and Inequity under Corruption",
    )

    plot_metric_grid(
        df=df,
        metrics=METRICS_PREDICTION_SPACE,
        filename="prediction_space_global_inequity_mass_entropy",
        title="Global Prediction-Space Inequity, Probability Mass, and Binary Entropy under Corruption",
    )

    # --------------------------------------------------------
    # Individual line plots
    # --------------------------------------------------------

    for metric_col, metric_label in all_metrics.items():
        plot_single_metric_per_figure(
            df=df,
            metric_col=metric_col,
            metric_label=metric_label,
        )

    # --------------------------------------------------------
    # Heatmaps
    # --------------------------------------------------------

    for metric_col, metric_label in all_metrics.items():
        plot_metric_heatmap_by_model(
            df=df,
            metric_col=metric_col,
            metric_label=metric_label,
        )

    # --------------------------------------------------------
    # Combined 5 x 5 severity curve figure
    # --------------------------------------------------------

    plot_combined_5x5_severity_curves(df)


if __name__ == "__main__":
    main()