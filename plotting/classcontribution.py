import glob
import os
import numpy as np
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================

MODELS = ["st-mem", "jepa", "xecg"]

ARTIFACTS = ["physionet_em", "physionet_ma", "physionet_gn", "physionet_dn", "physionet_in"]
SEVERITIES = ["Sev1", "Sev2", "Sev3"]

EXPERIMENTS = [("physionet_clean", "physionet_clean")]

for art in ARTIFACTS:
    for sev in SEVERITIES:
        EXPERIMENTS.append((art, sev))

SR_CLASS = 12


PROJECT_ROOT = Path(__file__).resolve().parent.parent

HEA_ROOT = PROJECT_ROOT / "evaluation" / "evaluation_heas"
EVAL2021_DIR = PROJECT_ROOT / "evaluation" / "evaluation-2021-main"

LABEL_CACHE = {}

# ==========================================================
# IMPORT EVALUATOR
# ==========================================================

def import_official_evaluator(eval_dir: Path):
    import sys, os
    sys.path.insert(0, str(eval_dir))
    old_cwd = os.getcwd()
    os.chdir(eval_dir)
    import evaluate_model
    return evaluate_model, old_cwd

evaluator, old_cwd = import_official_evaluator(EVAL2021_DIR)
classes, weights = evaluator.load_weights("weights_21.csv")

# ==========================================================
# LOAD LABELS ONCE
# ==========================================================

def load_labels_once(record_ids, target_path):

    if "labels" in LABEL_CACHE:
        return LABEL_CACHE["labels"]

    print("Loading labels (once)...")

    tmp_out = Path(target_path).parent / "tmp_outputs"
    tmp_out.mkdir(exist_ok=True)

    for rid in record_ids:
        (tmp_out / f"{rid}.csv").touch()

    label_files, output_files = evaluator.find_challenge_files(
        str(HEA_ROOT),
        str(tmp_out)
    )

    labels = evaluator.load_labels(label_files, classes)
    ordered_ids = [Path(f).stem for f in label_files]

    LABEL_CACHE["labels"] = (labels, ordered_ids)

    import shutil
    shutil.rmtree(tmp_out)

    return labels, ordered_ids


# ==========================================================
# MAIN LOOP
# ==========================================================

results = {}

for MODEL in MODELS:
    print(f"\n==============================")
    print(f"MODEL: {MODEL}")
    print(f"==============================")

    results[MODEL] = {}

    for TARGET_ARTIFACT, TARGET_SEVERITY in EXPERIMENTS:

        print(f"\n--- {TARGET_ARTIFACT} | {TARGET_SEVERITY} ---")

        # --------------------------------------------------
        # FIND FILE
        # --------------------------------------------------

        prob_paths = glob.glob(
            str(PROJECT_ROOT/ "inference" / MODEL / TARGET_ARTIFACT / "**" / "probs21.npy"),
            recursive=True
        )

        target_path = None

        for p in prob_paths:
            if TARGET_SEVERITY in p:
                target_path = p
                break

        if target_path is None:
            print("❌ No file found")
            continue

        print("Using:", target_path)

        # --------------------------------------------------
        # LOAD PROBS + IDS
        # --------------------------------------------------

        probs = np.load(target_path).astype(np.float32)

        record_ids = np.load(
            target_path.replace("probs21.npy", "record_ids.npy"),
            allow_pickle=True
        ).tolist()

        record_ids = [
            Path(r.decode() if isinstance(r, bytes) else r).stem
            for r in record_ids
        ]

        # --------------------------------------------------
        # LOAD LABELS
        # --------------------------------------------------

        labels, ordered_ids = load_labels_once(record_ids, target_path)

        # --------------------------------------------------
        # ALIGN
        # --------------------------------------------------

        id_to_idx = {rid: i for i, rid in enumerate(record_ids)}

        try:
            indices = [id_to_idx[rid] for rid in ordered_ids]
        except KeyError as e:
            print(f"❌ Missing ID: {e}")
            continue

        probs_aligned = probs[indices]

        if probs_aligned.shape != labels.shape:
            print("❌ Shape mismatch")
            continue

        # --------------------------------------------------
        # ANALYSIS
        # --------------------------------------------------

        preds = (probs_aligned > 0.5).astype(int)

        # =========================
        # PER CLASS STATS
        # =========================

        tp_per_class = ((preds == 1) & (labels == 1)).sum(axis=0)
        fp_per_class = ((preds == 1) & (labels == 0)).sum(axis=0)
        fn_per_class = ((preds == 0) & (labels == 1)).sum(axis=0)

        # =========================
        # GLOBAL STATS
        # =========================

        total_tp = tp_per_class.sum()
        total_fp = fp_per_class.sum()
        total_pred = preds.sum()  # total positive predictions across all classes

        # =========================
        # SINUS RHYTHM STATS
        # =========================

        sr_tp = tp_per_class[SR_CLASS]
        sr_fp = fp_per_class[SR_CLASS]
        sr_fn = fn_per_class[SR_CLASS]

        sr_total_pred = sr_tp + sr_fp

        # precision = TP / (TP + FP)
        sr_precision = sr_tp / (sr_total_pred + 1e-8)

        # recall = TP / (TP + FN)
        sr_recall = sr_tp / (sr_tp + sr_fn + 1e-8)

        # share based on TP (your old metric)
        sr_share_tp = sr_tp / total_tp if total_tp > 0 else 0

        # share based on predictions (NEW, very important)
        sr_share_pred = sr_total_pred / total_pred if total_pred > 0 else 0

        # =========================
        # PRINT
        # =========================

        print("\n=== SINUS RHYTHM ANALYSIS ===")
        print(f"SR TP            : {sr_tp}")
        print(f"SR FP            : {sr_fp}")
        print(f"SR FN            : {sr_fn}")
        print(f"SR Total Pred    : {sr_total_pred}")
        print(f"Total Predictions: {total_pred}")

        print(f"\nSR Precision     : {sr_precision:.4f}")
        print(f"SR Recall        : {sr_recall:.4f}")

        print(f"\nSR TP Share      : {sr_share_tp:.2%}")
        print(f"SR Pred Share    : {sr_share_pred:.2%}")

        # =========================
        # STORE
        # =========================

        results[MODEL][(TARGET_ARTIFACT, TARGET_SEVERITY)] = {
            "tp_per_class": tp_per_class,
            "fp_per_class": fp_per_class,
            "fn_per_class": fn_per_class,

            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_pred": total_pred,

            "sr_tp": sr_tp,
            "sr_fp": sr_fp,
            "sr_fn": sr_fn,

            "sr_total_pred": sr_total_pred,
            "sr_precision": sr_precision,
            "sr_recall": sr_recall,

            "sr_share_tp": sr_share_tp,
            "sr_share_pred": sr_share_pred,
        }

# ==========================================================
# OPTIONAL: COMPARE CLEAN VS CORRUPTED
# ==========================================================

print("\n\n==============================")
print("DELTA vs CLEAN")
print("==============================")

for MODEL in MODELS:

    if ("physionet_clean", "physionet_clean") not in results[MODEL]:
        continue

    clean_tp = results[MODEL][("physionet_clean", "physionet_clean")]["tp_per_class"]

    print(f"\nMODEL: {MODEL}")

    for key, val in results[MODEL].items():

        if key == ("physionet_clean", "physionet_clean"):
            continue

        delta = val["tp_per_class"] - clean_tp

        print(f"\n--- {key} ---")

        idx = np.argsort(-delta)

        for i in idx[:5]:
            print(f"Class {i:02d} | ΔTP: {delta[i]}")