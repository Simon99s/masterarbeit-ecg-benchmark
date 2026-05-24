# evaluation.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mapping_rules import build_groups_from_tasks
from concurrent.futures import ThreadPoolExecutor

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["ECGFounder", "jepa", "st-mem", "xecg"],
        help="Model name corresponding to the folder in inference/",
    )
    return parser.parse_args()

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

args = parse_args()
MODEL_NAME = args.model

REPO_ROOT = Path(__file__).resolve().parent

PROJECT_ROOT = REPO_ROOT.parent

INFERENCE_ROOT = PROJECT_ROOT / "inference"

MODEL_ROOT = INFERENCE_ROOT / MODEL_NAME

LABELS_DIR = PROJECT_ROOT / "data" / "evaluation_data"

EVAL2021_DIR = REPO_ROOT / "evaluation-2021-main"

WEIGHTS_CSV = EVAL2021_DIR / "weights_21.csv"

TASKS_TXT = REPO_ROOT / "tasks.txt"

SUMMARY_FILE = REPO_ROOT / "benchmark_scores.csv"


# ------------------------------------------------
# PhysioNet classes
# ------------------------------------------------

WHITELIST_21 = [
"164889003","164890007","733534002|164909002","713427006|59118001",
"270492004","713426002","39732003","445118002","47665007",
"251146004","111975006","698252002","426783006",
"284470004|63593006","10370003","427172004|17338001",
"427393009","426177001","427084000","164934002","59931005"
]


# ------------------------------------------------
# Helper functions
# ------------------------------------------------

def load_tasks(tasks_txt: Path) -> List[str]:

    lines = tasks_txt.read_text().splitlines()
    return [x.strip() for x in lines if x.strip()]


def load_weights_classes(weights_csv: Path):

    df = pd.read_csv(weights_csv, header=0, index_col=0)
    return list(df.columns)


def max_pool_probs(probs150: np.ndarray, idxs: List[int]):

    if len(idxs) == 0:
        return np.zeros((probs150.shape[0],), dtype=np.float32)

    return probs150[:, idxs].max(axis=1).astype(np.float32)


def map_probs150_to_probs21(
    probs150: np.ndarray,
    weights_classes21: List[str],
    groups: Dict[str, List[int]],
):

    N = probs150.shape[0]
    probs21 = np.zeros((N, len(weights_classes21)), dtype=np.float32)

    for j, cls in enumerate(weights_classes21):

        idxs = groups.get(cls, [])
        probs21[:, j] = max_pool_probs(probs150, idxs)

    return probs21


# ------------------------------------------------
# CSV writing
# ------------------------------------------------

def _write_one_csv(args):

    rid, p_row, b_row, outputs_dir, classes_line = args

    with open(outputs_dir / f"{rid}.csv", "w") as f:

        f.write(f"{rid}\n")
        f.write(classes_line + "\n")
        f.write(",".join(str(int(x)) for x in b_row) + "\n")
        f.write(",".join(f"{float(v):.6f}" for v in p_row) + "\n")


def write_outputs_physionet2021(
    outputs_dir: Path,
    record_ids: List[str],
    probs21: np.ndarray,
    thr27: np.ndarray,
    class_names27: List[str],
):

    outputs_dir.mkdir(parents=True, exist_ok=True)

    bin_pred = (probs21 >= thr27[None, :]).astype(int)

    classes_line = ",".join(class_names27)

    with ThreadPoolExecutor(max_workers=8) as ex:

        ex.map(
            _write_one_csv,
            (
                (rid, p_row, b_row, outputs_dir, classes_line)
                for rid, p_row, b_row in zip(record_ids, probs21, bin_pred)
            ),
        )


# ------------------------------------------------
# Import official evaluator
# ------------------------------------------------

def import_official_evaluator(eval_dir: Path):

    sys.path.insert(0, str(eval_dir))

    old_cwd = os.getcwd()

    os.chdir(eval_dir)

    import evaluate_model

    return evaluate_model, old_cwd


def run_official_eval(evaluator, labels_dir: Path, outputs_dir: Path): 

    weights_file = "weights_21.csv"

    classes, weights = evaluator.load_weights(weights_file)

    sinus_rhythm = next(c for c in classes if c == {"426783006"})

    label_files, output_files = evaluator.find_challenge_files(
        str(labels_dir),
        str(outputs_dir)
    )

    labels = evaluator.load_labels(label_files, classes)

    binary_outputs, scalar_outputs = load_outputs_21_from_27_csvs(
        output_files,
        WHITELIST_21
    )

    print("\n=== DEBUG: PER-CLASS AUROC ===")

    for j, cls in enumerate(WHITELIST_21):
        try:
            auc = roc_auc_score(labels[:, j], scalar_outputs[:, j])
            print(f"{cls}: {auc:.4f}")
        except Exception as e:
            print(f"{cls}: skipped ({e})")

    print("================================\n")

    auroc, auprc, *_ = evaluator.compute_auc(labels, scalar_outputs)

    accuracy = evaluator.compute_accuracy(labels, binary_outputs)

    f_measure, *_ = evaluator.compute_f_measure(labels, binary_outputs)

    challenge_metric = evaluator.compute_challenge_metric(
        weights,
        labels,
        binary_outputs,
        classes,
        sinus_rhythm
    )

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "accuracy": float(accuracy),
        "f_measure": float(f_measure),
        "challenge_metric": float(challenge_metric),
    }


# ------------------------------------------------
# search exp
# ------------------------------------------------

def find_experiment_dirs(model_root: Path):

    exp_dirs = []

    for artifact_dir in model_root.iterdir():

        if not artifact_dir.is_dir():
            continue

        for sev_dir in artifact_dir.iterdir():

            if not sev_dir.is_dir():
                continue

            probs150 = sev_dir / "probs150.npy"
            probs21  = sev_dir / "probs21.npy"
            rids     = sev_dir / "record_ids.npy"

            if rids.exists() and (probs150.exists() or probs21.exists()):
                exp_dirs.append(sev_dir)

    return sorted(exp_dirs)


# ------------------------------------------------
# Load outputs helper
# ------------------------------------------------

def load_outputs_21_from_27_csvs(output_files, whitelist_21):

    N = len(output_files)
    C = len(whitelist_21)

    binary_outputs = np.zeros((N, C), dtype=np.bool_)
    scalar_outputs = np.zeros((N, C), dtype=np.float64)

    for i, csv_path in enumerate(output_files):

        lines = Path(csv_path).read_text().splitlines()

        classes27 = lines[1].split(",")
        binary27 = np.array(lines[2].split(","), dtype=int)
        probs27 = np.array(lines[3].split(","), dtype=float)

        idx_map = {c: j for j, c in enumerate(classes27)}

        for k, cls in enumerate(whitelist_21):

            j = idx_map[cls]

            binary_outputs[i, k] = binary27[j]
            scalar_outputs[i, k] = probs27[j]

    return binary_outputs, scalar_outputs


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    print("\nEvaluating model:", MODEL_NAME)

    weights_classes21 = load_weights_classes(WEIGHTS_CSV)

    thr27 = np.full((len(weights_classes21),), 0.5, dtype=np.float32)

    print("\nUsing Emory thresholds:")
    print(thr27)

    tasks150 = load_tasks(TASKS_TXT)

    mapping = build_groups_from_tasks(tasks150, weights_classes21)

    evaluator, old_cwd = import_official_evaluator(EVAL2021_DIR)

    rows = []

    exp_dirs = find_experiment_dirs(MODEL_ROOT)

    # ------------------------------------------------
    # OPTIONAL: evaluate only one exp
    # ------------------------------------------------

    ONLY_THIS = None   # set None to evaluate everything

    if ONLY_THIS is not None:
        exp_dirs = [d for d in exp_dirs if d.name == ONLY_THIS]

    print("Experiments to evaluate:", len(exp_dirs))

    for exp_dir in exp_dirs:

        artifact = exp_dir.parent.name
        severity = exp_dir.name

        print(f"\nProcessing: {artifact} | {severity}")

        probs_path = exp_dir / "probs21.npy"
        rids_path = exp_dir / "record_ids.npy"

        record_ids = np.load(rids_path, allow_pickle=True).tolist()

        if (exp_dir / "probs150.npy").exists():

            probs150 = np.load(exp_dir / "probs150.npy").astype(np.float32)

            probs21 = map_probs150_to_probs21(
                probs150,
                weights_classes21,
                mapping.groups
            )

        elif (exp_dir / "probs21.npy").exists():

            probs21 = np.load(exp_dir / "probs21.npy").astype(np.float32)

        else:
            raise RuntimeError(f"No probs file found in {exp_dir}")
        

        print("\n=== DEBUG: PROB DISTRIBUTION ===")
        print("global mean:", probs21.mean())
        print("global std :", probs21.std())

        print("\nmean per class:")
        print(probs21.mean(axis=0))

        print("\nstd per class:")
        print(probs21.std(axis=0))
        print("================================\n")



        outputs_dir = exp_dir / "outputs_2021"

        write_outputs_physionet2021(
            outputs_dir,
            record_ids,
            probs21,
            thr27,
            weights_classes21
        )

        metrics = run_official_eval(
            evaluator,
            LABELS_DIR,
            outputs_dir,
            #record_ids ###debug subset check
        )

        row = {
            "model": MODEL_NAME,
            "artifact": artifact,
            "severity": severity,
        }

        row.update(metrics)

        rows.append(row)

        print("Challenge score:", metrics["challenge_metric"])

    os.chdir(old_cwd)

    df = pd.DataFrame(rows)

    if SUMMARY_FILE.exists():

        old = pd.read_csv(SUMMARY_FILE)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(SUMMARY_FILE, index=False)

    print("\nSaved summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()