# analyze_cinc_dataset.py
from __future__ import annotations

import os
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEADER_ROOT = PROJECT_ROOT / "evaluation" / "evaluation_heas"
SIGNAL_ROOT = PROJECT_ROOT / "data" / "Benchmark" / "physionet_clean" / "physionet_clean"
OUT_DIR = PROJECT_ROOT / "Plots" / "dataset_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paste your exact 21-class SNOMED codes here.
# If you leave this empty, the script analyzes all labels found in the headers.
# ============================================================
# CLASS CONFIG
# ============================================================

# Official scored CinC 2021 classes, individually.
SCORED_CLASSES_30 = [
    "164889003",  # AF
    "164890007",  # AFL
    "6374002",    # BBB
    "426627000",  # Brady
    "733534002",  # CLBBB
    "713427006",  # CRBBB
    "270492004",  # IAVB
    "713426002",  # IRBBB
    "39732003",   # LAD
    "445118002",  # LAnFB
    "164909002",  # LBBB
    "251146004",  # LQRSV
    "698252002",  # NSIVCB
    "426783006",  # NSR
    "284470004",  # PAC
    "10370003",   # PR
    "365413008",  # PRWP
    "427172004",  # PVC
    "164947007",  # LPR
    "111975006",  # LQT
    "164917005",  # QAb
    "47665007",   # RAD
    "59118001",   # RBBB
    "427393009",  # SA
    "426177001",  # SB
    "427084000",  # STach
    "63593006",   # SVPB
    "164934002",  # TAb
    "59931005",   # TInv
    "17338001",   # VPB
]

SNOMED_NAMES = {
    "164889003": "AF",
    "164890007": "AFL",
    "6374002": "BBB",
    "426627000": "Brady",
    "733534002": "CLBBB",
    "713427006": "CRBBB",
    "270492004": "IAVB",
    "713426002": "IRBBB",
    "39732003": "LAD",
    "445118002": "LAnFB",
    "164909002": "LBBB",
    "251146004": "LQRSV",
    "698252002": "NSIVCB",
    "426783006": "NSR",
    "284470004": "PAC",
    "10370003": "PR",
    "365413008": "PRWP",
    "427172004": "PVC",
    "164947007": "LPR",
    "111975006": "LQT",
    "164917005": "QAb",
    "47665007": "RAD",
    "59118001": "RBBB",
    "427393009": "SA",
    "426177001": "SB",
    "427084000": "STach",
    "63593006": "SVPB",
    "164934002": "TAb",
    "59931005": "TInv",
    "17338001": "VPB",
}

# Classes to visualize as ECG examples.
# Use codes from your target label space.
EXAMPLE_CLASSES = [
    "426783006",  # NSR
    "164889003",  # AF
    "164890007",  # AFL
    "164909002",  # LBBB
    "59118001",   # RBBB
    "284470004",  # PAC
]

# For example ECG plot
SECONDS_TO_PLOT = 10
LEADS_TO_PLOT = ["I", "II", "V1", "V2", "V5", "V6"]  


# ============================================================
# HEADER PARSING
# ============================================================

def find_header_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.hea"))


def parse_header_fast(header_path: Path, signal_path_map: dict[str, str]) -> dict:
    """
    Parses basic information from a WFDB .hea file without loading signal data.
    Supports both '#Dx:' and '# Dx:' label formats.
    """
    text = header_path.read_text(errors="ignore").splitlines()

    first = text[0].split()
    record_id = first[0]
    fs = float(first[2])
    sig_len = int(first[3])
    duration_sec = sig_len / fs

    labels = []
    for line in text:
        line_clean = line.strip()

        if line_clean.startswith("#Dx:") or line_clean.startswith("# Dx:"):
            labels_str = line_clean.split(":", 1)[1]
            labels = [x.strip() for x in labels_str.split(",") if x.strip()]
            break

    record_path = signal_path_map.get(record_id)

    return {
        "record_path": record_path,
        "record_id": record_id,
        "fs": fs,
        "sig_len": sig_len,
        "duration_sec": duration_sec,
        "labels": labels,
    }


def build_metadata(header_root: Path, signal_root: Path) -> pd.DataFrame:
    header_files = find_header_files(header_root)

    if not header_files:
        raise RuntimeError(f"No .hea files found in {header_root}")

    signal_path_map = build_signal_path_map(signal_root)

    rows = []
    for hp in tqdm(header_files, desc="Reading header files"):
        rows.append(parse_header_fast(hp, signal_path_map))

    df = pd.DataFrame(rows)

    missing_signals = df["record_path"].isna().sum()
    print(f"Loaded metadata for {len(df)} records.")
    print(f"Records without matching signal file: {missing_signals}")

    return df


# ============================================================
# CLASS PROCESSING
# ============================================================
def build_signal_path_map(signal_root: Path) -> dict[str, str]:
    """
    Maps record IDs to full WFDB record paths without file extension.
    Example:
    JS00001 -> C:/.../chapman_shaoxing/g1/JS00001
    """
    signal_paths = {}

    signal_files = list(signal_root.rglob("*.dat")) + list(signal_root.rglob("*.mat"))

    for file_path in tqdm(signal_files, desc="Indexing signal files"):
        record_id = file_path.stem
        signal_paths[record_id] = str(file_path.with_suffix(""))

    print(f"Indexed {len(signal_paths)} signal records.")
    return signal_paths

def class_name(code: str) -> str:
    return SNOMED_NAMES.get(str(code), str(code))


def add_target_label_columns(df: pd.DataFrame, target_classes: list[str]) -> pd.DataFrame:
    target_set = set(target_classes)

    df = df.copy()
    df["target_labels"] = df["labels"].apply(lambda xs: [x for x in xs if x in target_set])
    df["num_target_labels"] = df["target_labels"].apply(len)

    return df


# ============================================================
# PLOT 1: ECG EXAMPLES
# ============================================================

def find_example_record(df: pd.DataFrame, class_code: str) -> str | None:
    """
    Prefer a record where the class occurs and the duration is not extremely long.
    """
    candidates = df[df["target_labels"].apply(lambda xs: class_code in xs)].copy()

    if candidates.empty:
        return None

    # Prefer records around 10-60 seconds if possible.
    preferred = candidates[
        (candidates["duration_sec"] >= 8) &
        (candidates["duration_sec"] <= 60)
    ]

    if not preferred.empty:
        candidates = preferred

    # Pick the shortest suitable record to keep loading fast.
    row = candidates.sort_values("duration_sec").iloc[0]
    return row["record_path"]


def plot_example_ecgs(df: pd.DataFrame, example_classes: list[str]) -> None:
    available_examples = []

    for code in example_classes:
        rec_path = find_example_record(df, code)
        if rec_path is not None:
            available_examples.append((code, rec_path))
        else:
            print(f"[WARN] No example record found for class {code} ({class_name(code)}).")

    if not available_examples:
        print("[WARN] No ECG examples could be plotted.")
        return

    n_rows = len(available_examples)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(12, 2.3 * n_rows),
        sharex=False
    )

    if n_rows == 1:
        axes = [axes]

    for ax, (code, rec_path) in zip(axes, available_examples):
        sig, meta = wfdb.rdsamp(rec_path)
        fs = float(meta["fs"])
        lead_names = meta["sig_name"]

        n_samples = min(sig.shape[0], int(SECONDS_TO_PLOT * fs))
        sig = sig[:n_samples, :]

        t = np.arange(n_samples) / fs

        selected_indices = []
        selected_names = []

        for lead in LEADS_TO_PLOT:
            if lead in lead_names:
                selected_indices.append(lead_names.index(lead))
                selected_names.append(lead)

        if not selected_indices:
            selected_indices = list(range(min(6, sig.shape[1])))
            selected_names = [lead_names[i] for i in selected_indices]

        selected = sig[:, selected_indices]

        # Normalize each lead for visualization only.
        selected = selected - np.nanmean(selected, axis=0, keepdims=True)
        selected = selected / (np.nanstd(selected, axis=0, keepdims=True) + 1e-8)

        offset = 3.0
        for i in range(selected.shape[1]):
            ax.plot(t, selected[:, i] + i * offset, linewidth=0.8)

        ax.set_yticks(np.arange(len(selected_names)) * offset)
        ax.set_yticklabels(selected_names)
        ax.set_title(f"{class_name(code)} ({code})")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Lead")
        ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    out_path = OUT_DIR / "cinc_example_ecgs.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")

def print_record_length_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prints and saves the number of records for each exact recording length.

    Example output:
        10 s: 80000
        5 s: 1000
        30 s: 500
    """
    length_counts = (
        df["duration_sec"]
        .round(3)  # avoids tiny floating point issues
        .value_counts()
        .sort_index()
        .reset_index()
    )

    length_counts.columns = ["duration_sec", "num_records"]

    # Also make a nice display column, e.g. 10 instead of 10.0
    length_counts["length_label"] = length_counts["duration_sec"].apply(
        lambda x: f"{int(x)} s" if float(x).is_integer() else f"{x:.3f} s"
    )

    out_csv = OUT_DIR / "cinc_record_length_counts.csv"
    length_counts.to_csv(out_csv, index=False)

    print("\nRecord length counts:")
    for _, row in length_counts.iterrows():
        print(f"{row['length_label']}: {int(row['num_records'])}")

    print(f"Saved: {out_csv}")

    return length_counts

# ============================================================
# PLOT 2: LENGTH HISTOGRAM + STATS TABLE
# ============================================================

def plot_length_histogram(df: pd.DataFrame) -> None:
    durations = df["duration_sec"].values

    stats = {
        "n_records": len(durations),
        "mean_sec": np.mean(durations),
        "std_sec": np.std(durations),
        "min_sec": np.min(durations),
        "p25_sec": np.percentile(durations, 25),
        "median_sec": np.percentile(durations, 50),
        "p75_sec": np.percentile(durations, 75),
        "p95_sec": np.percentile(durations, 95),
        "max_sec": np.max(durations),
    }

    stats_df = pd.DataFrame([stats])
    stats_path = OUT_DIR / "cinc_length_statistics.csv"
    stats_df.to_csv(stats_path, index=False)

    print("\nRecording length statistics:")
    print(stats_df.to_string(index=False))
    print(f"Saved: {stats_path}")

    # ------------------------------------------------------------
    # Histogram settings
    # ------------------------------------------------------------
    bin_width = 1  # seconds

    # Main region: most records are here.
    main_x_max = 150

    # Long-record region: needed to show the 1800 s records.
    long_x_min = 1700
    long_x_max = 1810

    main_bins = np.arange(0, main_x_max + bin_width, bin_width)
    long_bins = np.arange(long_x_min, long_x_max + bin_width, bin_width)

    counts_main, _ = np.histogram(durations, bins=main_bins)
    counts_long, _ = np.histogram(durations, bins=long_bins)

    max_count = max(counts_main.max(), counts_long.max())

    # Lower y-axis shows ordinary bins.
    y_break_lower_max = 750

    # ------------------------------------------------------------
    # Create 2x2 axes:
    # top-left     = high y, normal x
    # top-right    = high y, long x
    # bottom-left  = low y, normal x
    # bottom-right = low y, long x
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        2,
        sharey="row",
        figsize=(11, 6),
        gridspec_kw={
            "height_ratios": [1, 3],
            "width_ratios": [4, 1],
            "hspace": 0.05,
            "wspace": 0.05,
        },
    )

    ax_top_left = axes[0, 0]
    ax_top_right = axes[0, 1]
    ax_bottom_left = axes[1, 0]
    ax_bottom_right = axes[1, 1]

    # Plot main x-region.
    ax_top_left.hist(durations, bins=main_bins)
    ax_bottom_left.hist(durations, bins=main_bins)

    # Plot long x-region.
    ax_top_right.hist(durations, bins=long_bins)
    ax_bottom_right.hist(durations, bins=long_bins)

    # ------------------------------------------------------------
    # X limits: broken x-axis
    # ------------------------------------------------------------
    ax_top_left.set_xlim(0, main_x_max)
    ax_bottom_left.set_xlim(0, main_x_max)

    ax_top_right.set_xlim(long_x_min, long_x_max)
    ax_bottom_right.set_xlim(long_x_min, long_x_max)

    # ------------------------------------------------------------
    # Y limits: broken y-axis
    # ------------------------------------------------------------
    ax_bottom_left.set_ylim(0, y_break_lower_max)
    ax_bottom_right.set_ylim(0, y_break_lower_max)

    if max_count > y_break_lower_max:
        upper_min = max(y_break_lower_max, max_count * 0.90)
        ax_top_left.set_ylim(upper_min, max_count * 1.05)
        ax_top_right.set_ylim(upper_min, max_count * 1.05)
    else:
        ax_top_left.set_ylim(0, max_count * 1.05)
        ax_top_right.set_ylim(0, max_count * 1.05)

    # ------------------------------------------------------------
    # Hide spines between broken axes
    # ------------------------------------------------------------
    ax_top_left.spines["bottom"].set_visible(False)
    ax_top_right.spines["bottom"].set_visible(False)
    ax_bottom_left.spines["top"].set_visible(False)
    ax_bottom_right.spines["top"].set_visible(False)

    ax_top_left.spines["right"].set_visible(False)
    ax_bottom_left.spines["right"].set_visible(False)
    ax_top_right.spines["left"].set_visible(False)
    ax_bottom_right.spines["left"].set_visible(False)

    ax_top_left.tick_params(labelbottom=False)
    ax_top_right.tick_params(labelbottom=False)
    ax_top_right.tick_params(labelleft=False)
    ax_bottom_right.tick_params(labelleft=False)

    # ------------------------------------------------------------
    # Draw diagonal break marks
    # ------------------------------------------------------------
    d = 0.012

    # y-axis break marks between top and bottom
    kwargs = dict(color="k", clip_on=False, linewidth=1)

    ax_top_left.plot((-d, +d), (-d, +d), transform=ax_top_left.transAxes, **kwargs)
    ax_top_left.plot((1 - d, 1 + d), (-d, +d), transform=ax_top_left.transAxes, **kwargs)

    ax_top_right.plot((-d, +d), (-d, +d), transform=ax_top_right.transAxes, **kwargs)
    ax_top_right.plot((1 - d, 1 + d), (-d, +d), transform=ax_top_right.transAxes, **kwargs)

    ax_bottom_left.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom_left.transAxes, **kwargs)
    ax_bottom_left.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom_left.transAxes, **kwargs)

    ax_bottom_right.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom_right.transAxes, **kwargs)
    ax_bottom_right.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom_right.transAxes, **kwargs)

    # x-axis break marks between left and right
    ax_top_left.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_top_left.transAxes, **kwargs)
    ax_top_left.plot((1 - d, 1 + d), (-d, +d), transform=ax_top_left.transAxes, **kwargs)

    ax_bottom_left.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom_left.transAxes, **kwargs)
    ax_bottom_left.plot((1 - d, 1 + d), (-d, +d), transform=ax_bottom_left.transAxes, **kwargs)

    ax_top_right.plot((-d, +d), (1 - d, 1 + d), transform=ax_top_right.transAxes, **kwargs)
    ax_top_right.plot((-d, +d), (-d, +d), transform=ax_top_right.transAxes, **kwargs)

    ax_bottom_right.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom_right.transAxes, **kwargs)
    ax_bottom_right.plot((-d, +d), (-d, +d), transform=ax_bottom_right.transAxes, **kwargs)


    # ------------------------------------------------------------
    # X ticks
    # ------------------------------------------------------------

    # Main x-axis: show steps of 10 s so the 10 s peak is clearly labeled.
    main_xticks = np.arange(0, main_x_max, 10)
    ax_bottom_left.set_xticks(main_xticks)

    ax_bottom_right.set_xticks([1800])
    ax_bottom_right.set_xticklabels(["1800"])

    # Optional: hide x tick labels on the top row.
    ax_top_left.tick_params(labelbottom=False)
    ax_top_right.tick_params(labelbottom=False)


    # ------------------------------------------------------------
    # Labels and styling
    # ------------------------------------------------------------
    ax_top_left.set_title("Distribution of ECG recording lengths")

    fig.text(0.5, 0.02, "Recording length [s]", ha="center")
    fig.text(0.04, 0.5, "Number of records", va="center", rotation="vertical")

    for ax in [ax_top_left, ax_top_right, ax_bottom_left, ax_bottom_right]:
        ax.grid(True, axis="y", alpha=0.3)

    # Avoid tight_layout warning with broken axes.
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.92)

    out_path = OUT_DIR / "cinc_length_histogram_broken_xy_axis.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


# ============================================================
# PLOT 3: CLASS DISTRIBUTION
# ============================================================

def compute_class_counts(df: pd.DataFrame, target_classes: list[str]) -> pd.DataFrame:
    counts = Counter()

    for labels in df["target_labels"]:
        counts.update(labels)

    rows = []
    for code in target_classes:
        rows.append({
            "code": code,
            "class_name": class_name(code),
            "count": counts.get(code, 0),
        })

    count_df = pd.DataFrame(rows).sort_values("count", ascending=False)
    return count_df


def plot_class_distribution(count_df: pd.DataFrame) -> None:
    out_csv = OUT_DIR / "cinc_scored30_class_distribution.csv"
    count_df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = count_df["class_name"].values
    counts = count_df["count"].values

    ax.bar(np.arange(len(labels)), counts)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Number of records")
    ax.set_title("Class distribution of selected diagnostic labels")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = OUT_DIR / "cinc_scored30_class_distribution.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_path}")


# ============================================================
# PLOT 4: LABEL CO-OCCURRENCE HEATMAP
# ============================================================

def compute_cooccurrence_counts(df: pd.DataFrame, target_classes: list[str]) -> pd.DataFrame:
    class_to_idx = {c: i for i, c in enumerate(target_classes)}
    mat = np.zeros((len(target_classes), len(target_classes)), dtype=int)

    for labels in df["target_labels"]:
        labels = [x for x in labels if x in class_to_idx]

        for a in labels:
            i = class_to_idx[a]
            for b in labels:
                j = class_to_idx[b]
                mat[i, j] += 1

    names = [class_name(c) for c in target_classes]
    return pd.DataFrame(mat, index=names, columns=names)


def compute_cooccurrence_percent(df: pd.DataFrame, target_classes: list[str]) -> pd.DataFrame:
    """
    Directional conditional co-occurrence percentage.

    Cell [A, B] means:
        records with both A and B / records with B * 100

    Example:
        [AV, NSR] = co-occurrence(AV, NSR) / count(NSR)
        [NSR, AV] = co-occurrence(AV, NSR) / count(AV)

    Therefore, the matrix is not necessarily symmetric.
    The diagonal is 100% for classes with at least one record.
    """
    co_counts = compute_cooccurrence_counts(df, target_classes)

    values = co_counts.values.astype(float)

    # The diagonal contains the total number of records for each class.
    class_counts = np.diag(values)

    percent = np.divide(
        values,
        class_counts[np.newaxis, :],
        out=np.zeros_like(values, dtype=float),
        where=class_counts[np.newaxis, :] != 0,
    ) * 100.0

    return pd.DataFrame(percent, index=co_counts.index, columns=co_counts.columns)


def plot_cooccurrence_heatmap(co_percent_df: pd.DataFrame) -> None:
    out_csv = OUT_DIR / "cinc_scored30_label_cooccurrence_percent.csv"
    co_percent_df.to_csv(out_csv)

    values = co_percent_df.values.copy()

    # Ignore the 100% diagonal values when choosing the color scale.
    non_100_values = values[values < 100]

    if len(non_100_values) > 0:
        vmax = np.ceil(non_100_values.max())
    else:
        vmax = 100

    print(f"Co-occurrence heatmap color scale: 0 to {vmax:.0f}%")

    # Mask the 100% values so they can be shown in a separate color.
    values_without_100 = np.ma.masked_where(values >= 100, values)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Main heatmap: scaled only from 0 to highest non-100 value.
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")  # color for 100% cells

    im = ax.imshow(
        values_without_100,
        aspect="auto",
        vmin=0,
        vmax=vmax,
        cmap=cmap,
    )

    fig.colorbar(
        im,
        ax=ax,
        label=f"Co-occurrence [% of column class], scaled 0-{vmax:.0f}%",
    )

    ax.set_xticks(np.arange(co_percent_df.shape[1]))
    ax.set_yticks(np.arange(co_percent_df.shape[0]))
    ax.set_xticklabels(co_percent_df.columns, rotation=60, ha="right")
    ax.set_yticklabels(co_percent_df.index)

    ax.set_title("Directional label co-occurrence")
    ax.set_xlabel("Reference class")
    ax.set_ylabel("Co-occurring class")

    # Add percentage labels to cells.
    for i in range(co_percent_df.shape[0]):
        for j in range(co_percent_df.shape[1]):
            value = co_percent_df.iloc[i, j]

            if value == 0:
                text = ""
            elif value < 1:
                text = f"{value:.1f}"
            else:
                text = f"{value:.0f}"

            # Use black text on the light gray diagonal.
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=5,
                color="black",
            )

    fig.tight_layout()
    out_path = OUT_DIR / "cinc_scored30_label_cooccurrence_percent.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_path}")


# ============================================================
# LATEX TABLE HELPER
# ============================================================

def write_latex_length_table() -> None:
    stats_path = OUT_DIR / "cinc_length_statistics.csv"
    stats_df = pd.read_csv(stats_path)

    row = stats_df.iloc[0]

    latex = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Recording length statistics of the PhysioNet/CinC 2021 training data.}}
\label{{tab:cinc_length_stats}}
\begin{{tabular}}{{rrrrrr}}
\toprule
Mean & Std. & Min. & Median & 95th percentile & Max. \\
\midrule
{row['mean_sec']:.2f} & {row['std_sec']:.2f} & {row['min_sec']:.2f} & {row['median_sec']:.2f} & {row['p95_sec']:.2f} & {row['max_sec']:.2f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip()

    out_path = OUT_DIR / "length_stats_table_latex.txt"
    out_path.write_text(latex, encoding="utf-8")
    print(f"Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = build_metadata(HEADER_ROOT, SIGNAL_ROOT)

    print(df[["record_id", "labels"]].head(10))
    print("Records with labels:", (df["labels"].apply(len) > 0).sum())

    target_classes = SCORED_CLASSES_30
    df = add_target_label_columns(df, target_classes)

    # Keep only records with at least one target label for class-based plots.
    df_target = df[df["num_target_labels"] > 0].copy()

    print(f"Records with at least one selected target label: {len(df_target)}")
    print(f"Number of selected target classes: {len(target_classes)}")

    # Save metadata
    metadata_path = OUT_DIR / "cinc_metadata_summary.csv"
    df.to_csv(metadata_path, index=False)
    print(f"Saved: {metadata_path}")

    # 1. Example ECGs
    plot_example_ecgs(df_target, EXAMPLE_CLASSES)

    # 2. Length counts and histogram
    print_record_length_counts(df)
    plot_length_histogram(df)

    # 3. Class distribution
    count_df = compute_class_counts(df_target, target_classes)
    plot_class_distribution(count_df)

    # 4. Label co-occurrence
    # 4. Label co-occurrence
    co_counts_df = compute_cooccurrence_counts(df_target, target_classes)
    co_counts_path = OUT_DIR / "cinc_scored30_label_cooccurrence_counts.csv"
    co_counts_df.to_csv(co_counts_path)
    print(f"Saved: {co_counts_path}")

    co_percent_df = compute_cooccurrence_percent(df_target, target_classes)
    plot_cooccurrence_heatmap(co_percent_df)

    # Optional LaTeX table for length statistics
    write_latex_length_table()

    print("\nDone.")


if __name__ == "__main__":
    main()