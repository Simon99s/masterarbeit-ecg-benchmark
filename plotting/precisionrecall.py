import glob
import os
import numpy as np
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================

MODELS = ["ECGFounder", "ST-MEM", "xECG", "JEPA"]

ARTIFACTS = ["physionet_em", "physionet_ma", "physionet_gn", "physionet_dn", "physionet_in"]
SEVERITIES = ["Sev1", "Sev2", "Sev3"]

EXPERIMENTS = [("physionet_clean", "physionet_clean")]

for art in ARTIFACTS:
    for sev in SEVERITIES:
        EXPERIMENTS.append((art, sev))

        
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INFERENCE_ROOT = PROJECT_ROOT / "inference"

HEA_ROOT = PROJECT_ROOT / "data" / "evaluation_data"
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
# LOAD LABELS ONCE (SHARED)
# ==========================================================

def load_labels_once(record_ids, target_path):

    if "labels" in LABEL_CACHE:
        return LABEL_CACHE["labels"]

    print("Loading labels (once)...")

    tmp_out = Path(target_path).parent / "tmp_outputs"
    tmp_out.mkdir(exist_ok=True)

    for rid in record_ids:
        (tmp_out / f"{rid}.csv").touch()

    label_files, _ = evaluator.find_challenge_files(
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
            str(INFERENCE_ROOT / MODEL / TARGET_ARTIFACT / "**" / "probs21.npy"),
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
        # LOAD LABELS (ONCE)
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
        # METRICS
        # --------------------------------------------------

        preds = (probs_aligned > 0.5).astype(int)

        # ==========================================================
        # GLOBAL (MICRO) COUNTS
        # ==========================================================

        TP_total = ((preds == 1) & (labels == 1)).sum()
        FP_total = ((preds == 1) & (labels == 0)).sum()
        FN_total = ((preds == 0) & (labels == 1)).sum()

        precision_micro = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
        recall_micro    = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0

        # ==========================================================
        # MACRO PRECISION / RECALL
        # ==========================================================

        num_classes = labels.shape[1]

        precision_per_class = []
        recall_per_class = []

        for k in range(num_classes):

            y_true = labels[:, k]
            y_pred = preds[:, k]

            TP = ((y_pred == 1) & (y_true == 1)).sum()
            FP = ((y_pred == 1) & (y_true == 0)).sum()
            FN = ((y_pred == 0) & (y_true == 1)).sum()

            precision_k = TP / (TP + FP) if (TP + FP) > 0 else np.nan
            recall_k    = TP / (TP + FN) if (TP + FN) > 0 else np.nan

            precision_per_class.append(precision_k)
            recall_per_class.append(recall_k)

        precision_macro = np.nanmean(precision_per_class)
        recall_macro    = np.nanmean(recall_per_class)

        results[MODEL][(TARGET_ARTIFACT, TARGET_SEVERITY)] = {
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "TP": TP_total,
            "FP": FP_total,
            "FN": FN_total
        }

# ==========================================================
# OPTIONAL: DELTA VS CLEAN
# ==========================================================

print("\n\n==============================")
print("DELTA vs CLEAN")
print("==============================")

for MODEL in MODELS:

    if ("physionet_clean", "physionet_clean") not in results[MODEL]:
        continue

    clean = results[MODEL][("physionet_clean", "physionet_clean")]

    print(f"\nMODEL: {MODEL}")

    for key, val in results[MODEL].items():

        if key == ("physionet_clean", "physionet_clean"):
            continue

        dp = val["precision_macro"] - clean["precision_macro"]
        dr = val["recall_macro"]    - clean["recall_macro"]

        print(f"{key} | ΔPrecision: {dp:.4f} | ΔRecall: {dr:.4f}")


import pandas as pd

rows = []

for model, res in results.items():
    for (artifact, severity), vals in res.items():

        rows.append({
            "model": model,
            "artifact": artifact,
            "severity": severity,
            "precision_macro": vals["precision_macro"],
            "recall_macro": vals["recall_macro"],
            "precision_micro": vals["precision_micro"],
            "recall_micro": vals["recall_micro"],
            "TP": vals["TP"],
            "FP": vals["FP"],
            "FN": vals["FN"]
        })


OUTPUT_PATH = PROJECT_ROOT / "precision_recall_all.csv"

df_out = pd.DataFrame(rows)
df_out = df_out.sort_values(
    ["model", "artifact", "severity"]
)
df_out.to_csv(OUTPUT_PATH, index=False)
print("\nSaved to:", OUTPUT_PATH)

print("\nSaved precision_recall_all.csv")