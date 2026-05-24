import os
import math
from xml.parsers.expat import model
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn as nn

# ==========================================================
# CONFIG
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_ROOT = PROJECT_ROOT / "inference"
OUT_CSV = PROJECT_ROOT / "evaluation" / "feature_collapse_metrics.csv"

BATCH_SIZE = 64
EPS = 1e-8
NUM_CLASSES = 21

# If True, recalculates rows even if they already exist in the CSV
FORCE_RECOMPUTE = False

# Known checkpoints for loading classifier heads
CHECKPOINTS = {
    "jepa": PROJECT_ROOT / "training" / "out_jepa_emory21" / "ckpt_best.pt",
    "st-mem": PROJECT_ROOT / "training" / "st_mem" / "stmem_best.pt",
    "xecg": PROJECT_ROOT / "training" / "out_xecg_emory21" / "ckpt_best.pt"
}


# ==========================================================
# HEAD LOADING FOR CENTROID INEQUITY
# ==========================================================

def load_linear_head(model_name: str, feature_dim: int, device: torch.device):
    """
    Loads only the final classifier head.

    Needed for feature-centroid inequity:
        feature centroid -> classifier head -> sigmoid probabilities

    For unknown models, this returns None.
    Then centroid inequity is saved as NaN.
    """

    model_name_lower = model_name.lower()
    ckpt_path = CHECKPOINTS.get(model_name_lower)

    if ckpt_path is None or not ckpt_path.exists():
        print(f"[HEAD] No checkpoint found for model '{model_name}'. Centroid inequity will be NaN.")
        return None

    ckpt = torch.load(ckpt_path, map_location=device)

    if model_name_lower == "jepa":
        head_state = ckpt["head_state"]
        weight = head_state["weight"].float()
        bias = head_state["bias"].float()

    elif model_name_lower in ["st-mem", "st_mem", "stmem"]:
        model_state = ckpt["model_state"]

        weight_key = None
        bias_key = None

        for k in model_state.keys():
            if k.endswith("head.weight"):
                weight_key = k
            elif k.endswith("head.bias"):
                bias_key = k

        if weight_key is None or bias_key is None:
            print(f"[HEAD] Could not find head.weight/head.bias for '{model_name}'.")
            return None

        weight = model_state[weight_key].float()
        bias = model_state[bias_key].float()

    elif model_name_lower == "xecg":
        model_state = ckpt["model_state"]

        weight_key = None
        bias_key = None

        # Find classifier weight by shape, not by name
        for k, v in model_state.items():
            if (
                k.endswith(".weight")
                and v.ndim == 2
                and v.shape[0] == NUM_CLASSES
                and v.shape[1] == feature_dim
            ):
                candidate_bias = k.replace(".weight", ".bias")

                if candidate_bias in model_state:
                    weight_key = k
                    bias_key = candidate_bias
                    break

        if weight_key is None or bias_key is None:
            print(f"[HEAD] Could not find xECG classifier head for feature_dim={feature_dim}.")
            print("[HEAD] Candidate 2D weights:")
            for k, v in model_state.items():
                if k.endswith(".weight") and v.ndim == 2:
                    print("   ", k, tuple(v.shape))
            return None

        print(f"[HEAD] xECG using keys: {weight_key}, {bias_key}")

        weight = model_state[weight_key].float()
        bias = model_state[bias_key].float()

    else:
        print(f"[HEAD] Model '{model_name}' not configured. Centroid inequity will be NaN.")
        return None

    if weight.shape[1] != feature_dim:
        print(
            f"[HEAD] Feature dimension mismatch for {model_name}: "
            f"features={feature_dim}, head expects={weight.shape[1]}. "
            f"Centroid inequity will be NaN."
        )
        return None

    head = nn.Linear(weight.shape[1], weight.shape[0])
    head.weight.data.copy_(weight)
    head.bias.data.copy_(bias)
    head.to(device)
    head.eval()

    print(f"[HEAD] Loaded {model_name} head: in_dim={weight.shape[1]}, out_dim={weight.shape[0]}")
    return head


# ==========================================================
# FEATURE REDUNDANCY
# ==========================================================

def feature_redundancy(features: np.ndarray, eps: float = EPS) -> float:
    """
    Feature redundancy from Z with shape [B, D].

    Higher value = feature dimensions are more correlated.
    """

    Z = np.asarray(features, dtype=np.float64)

    if Z.ndim != 2:
        raise ValueError(f"Expected features with shape [B, D], got {Z.shape}")

    B, D = Z.shape

    if B < 2:
        return np.nan

    # Center each feature dimension over records
    Zc = Z - Z.mean(axis=0, keepdims=True)

    # Normalize each dimension
    std = Zc.std(axis=0, keepdims=True) + eps
    Zn = Zc / std

    # Correlation matrix between feature dimensions
    C = (Zn.T @ Zn) / B

    # Sum squared off-diagonal correlations
    diag_sum = np.sum(np.diag(C) ** 2)
    off_diag_sum = np.sum(C ** 2) - diag_sum

    R = off_diag_sum / max(D - 1, 1)
    return float(R)


def batchwise_feature_redundancy(features: np.ndarray, batch_size: int = BATCH_SIZE):
    values = []

    for start in range(0, len(features), batch_size):
        batch = features[start:start + batch_size]
        if len(batch) < 2:
            continue
        values.append(feature_redundancy(batch))

    values = np.asarray(values, dtype=np.float64)

    return {
        "feature_redundancy_mean": float(np.nanmean(values)) if len(values) else np.nan,
        "feature_redundancy_std": float(np.nanstd(values)) if len(values) else np.nan,
        "feature_redundancy_global": feature_redundancy(features),
        "feature_redundancy_batches": int(len(values)),
    }


# ==========================================================
# MULTILABEL FEATURE-CENTROID INEQUITY
# ==========================================================

def binary_entropy(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def centroid_multilabel_inequity(features: np.ndarray, head, device: torch.device) -> float:
    """
    Multilabel adaptation of feature inequity.

    1. Compute centroid of features.
    2. Pass centroid through classifier head.
    3. Apply sigmoid.
    4. Compute normalized binary entropy.

    Value:
        close to 0 = centroid output is uncertain around 0.5
        high       = centroid output is confident / biased
    """

    if head is None:
        return np.nan

    Z = np.asarray(features, dtype=np.float32)

    if Z.ndim != 2:
        raise ValueError(f"Expected features with shape [B, D], got {Z.shape}")

    mu = Z.mean(axis=0)

    with torch.no_grad():
        mu_t = torch.from_numpy(mu).float().to(device).unsqueeze(0)
        logits = head(mu_t)
        p = torch.sigmoid(logits).cpu().numpy().squeeze(0)

    H = binary_entropy(p)
    inequity = 1.0 - (H.mean() / math.log(2.0))

    return float(inequity)


def batchwise_centroid_inequity(features: np.ndarray, head, device: torch.device, batch_size: int = BATCH_SIZE):
    values = []

    for start in range(0, len(features), batch_size):
        batch = features[start:start + batch_size]
        if len(batch) < 2:
            continue
        values.append(centroid_multilabel_inequity(batch, head, device))

    values = np.asarray(values, dtype=np.float64)

    return {
        "centroid_inequity_mean": float(np.nanmean(values)) if len(values) else np.nan,
        "centroid_inequity_std": float(np.nanstd(values)) if len(values) else np.nan,
        "centroid_inequity_global": centroid_multilabel_inequity(features, head, device),
        "centroid_inequity_batches": int(len(values)),
    }


# ==========================================================
# PREDICTION-SHARE INEQUITY
# ==========================================================

def prediction_share_inequity(probs: np.ndarray, eps: float = EPS):
    """
    Global output-level prediction-share inequity from probs21.npy.

    This computes the average predicted probability per class over all records.
    The class-wise average probabilities are normalized into a class-share
    distribution q. Entropy of q measures how evenly probability mass is
    distributed across classes.

    Higher inequity = probability mass is concentrated on fewer classes.
    Lower inequity  = probability mass is more evenly distributed across classes.

    Note:
        This is a global dataset-level metric, not a batchwise metric.
    """

    P = np.asarray(probs, dtype=np.float64)

    if P.ndim != 2:
        raise ValueError(f"Expected probabilities with shape [N, C], got {P.shape}")

    C = P.shape[1]

    # Average predicted probability per class over all records
    mean_prob_per_class = P.mean(axis=0)

    # Average total probability mass per record
    mean_probability_mass_global = mean_prob_per_class.sum()

    if mean_probability_mass_global <= eps:
        return {
            "prediction_inequity_norm": np.nan,
            "prediction_inequity_raw": np.nan,
            "prediction_entropy": np.nan,
            "mean_probability_mass": float(mean_probability_mass_global),

            # clearer duplicate names
            "prediction_inequity_global_norm": np.nan,
            "prediction_inequity_global_raw": np.nan,
            "prediction_share_entropy_global": np.nan,
            "mean_probability_mass_global": float(mean_probability_mass_global),
        }

    # Normalize average probabilities into class-share distribution
    q = mean_prob_per_class / (mean_probability_mass_global + eps)
    q = np.clip(q, eps, 1.0)

    # Entropy over class-share distribution
    share_entropy = -np.sum(q * np.log(q))

    # Raw and normalized inequity
    inequity_raw = math.log(C) - share_entropy
    inequity_norm = 1.0 - share_entropy / math.log(C)


    binary_H = binary_entropy(P)
    prediction_binary_entropy_global = binary_H.mean()
    prediction_binary_entropy_norm_global = prediction_binary_entropy_global / math.log(2.0)

    return {
        # old names, keep for compatibility
        "prediction_inequity_norm": float(inequity_norm),
        "prediction_inequity_raw": float(inequity_raw),
        "prediction_entropy": float(share_entropy),
        "mean_probability_mass": float(mean_probability_mass_global),

        # clearer names
        "prediction_inequity_global_norm": float(inequity_norm),
        "prediction_inequity_global_raw": float(inequity_raw),
        "prediction_share_entropy_global": float(share_entropy),
        "mean_probability_mass_global": float(mean_probability_mass_global),

        "prediction_binary_entropy_global": float(prediction_binary_entropy_global),
        "prediction_binary_entropy_norm_global": float(prediction_binary_entropy_norm_global),
    }


# ==========================================================
# PATH HANDLING
# ==========================================================

def parse_result_path(features_path: Path):
    """
    Expected:
        bm/inference/<model>/<artifact>/<severity>/features.npy

    Example:
        bm/inference/jepa/physionet_em/physionet_em_Sev1/features.npy
        bm/inference/st-mem/physionet_clean/physionet_clean/features.npy
    """

    rel = features_path.relative_to(INFERENCE_ROOT)
    parts = rel.parts

    if len(parts) < 4:
        raise ValueError(f"Unexpected result path: {features_path}")

    model = parts[0]
    artifact = parts[1]
    severity = parts[2]

    return model, artifact, severity


def find_all_feature_files():
    return sorted(INFERENCE_ROOT.rglob("features.npy"))


def load_existing_csv():
    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
    else:
        df = pd.DataFrame()

    return df


def existing_keys(df: pd.DataFrame):
    if df.empty:
        return set()

    required = {"model", "artifact", "severity"}

    if not required.issubset(set(df.columns)):
        return set()

    return set(zip(df["model"].astype(str), df["artifact"].astype(str), df["severity"].astype(str)))


def save_rows(rows):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate_one(features_path: Path, head_cache: dict, device: torch.device):
    model, artifact, severity = parse_result_path(features_path)

    folder = features_path.parent
    probs_path = folder / "probs21.npy"

    print(f"\nEvaluating: model={model} | artifact={artifact} | severity={severity}")
    print(f"Folder: {folder}")

    features = np.load(features_path)

    if features.ndim != 2:
        raise ValueError(f"features.npy must have shape [N, D], got {features.shape} at {features_path}")

    n_records, feature_dim = features.shape

    # Load classifier head once per model + feature dimension
    head_key = (model, feature_dim)

    if head_key not in head_cache:
        head_cache[head_key] = load_linear_head(model, feature_dim, device)

    head = head_cache[head_key]
    print(f"[DEBUG] model={model}, feature_dim={feature_dim}, head is None? {head is None}")

    # 1) Feature redundancy
    redundancy = batchwise_feature_redundancy(features, batch_size=BATCH_SIZE)

    # 2) Feature-centroid inequity
    centroid_ineq = batchwise_centroid_inequity(
        features,
        head=head,
        device=device,
        batch_size=BATCH_SIZE,
    )

    # 3) Prediction-share inequity
    if probs_path.exists():
        probs = np.load(probs_path)

        if len(probs) != len(features):
            print(
                f"[WARNING] Length mismatch: features={len(features)}, probs={len(probs)}. "
                f"Prediction inequity will be NaN."
            )
            pred_ineq = {
                "prediction_inequity_norm": np.nan,
                "prediction_inequity_raw": np.nan,
                "prediction_entropy": np.nan,
                "mean_probability_mass": np.nan,

                "prediction_inequity_global_norm": np.nan,
                "prediction_inequity_global_raw": np.nan,
                "prediction_share_entropy_global": np.nan,
                "mean_probability_mass_global": np.nan,

                "prediction_binary_entropy_global": np.nan,
                "prediction_binary_entropy_norm_global": np.nan,
            }
        else:
            pred_ineq = prediction_share_inequity(probs)
    else:
        print("[WARNING] No probs21.npy found. Prediction inequity will be NaN.")
        pred_ineq = {
            "prediction_inequity_norm": np.nan,
            "prediction_inequity_raw": np.nan,
            "prediction_entropy": np.nan,
            "mean_probability_mass": np.nan,

            "prediction_inequity_global_norm": np.nan,
            "prediction_inequity_global_raw": np.nan,
            "prediction_share_entropy_global": np.nan,
            "mean_probability_mass_global": np.nan,

            "prediction_binary_entropy_global": np.nan,
            "prediction_binary_entropy_norm_global": np.nan,
        }

    row = {
        "model": model,
        "artifact": artifact,
        "severity": severity,
        "n_records": int(n_records),
        "feature_dim": int(feature_dim),
        "features_path": str(features_path),
        **redundancy,
        **centroid_ineq,
        **pred_ineq,
    }

    return row


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("INFERENCE_ROOT:", INFERENCE_ROOT)
    print("OUT_CSV:", OUT_CSV)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    existing_df = load_existing_csv()
    keys_done = existing_keys(existing_df)

    rows = existing_df.to_dict("records") if not existing_df.empty else []

    feature_files = find_all_feature_files()

    print(f"\nFound {len(feature_files)} features.npy files.")

    head_cache = {}

    for features_path in feature_files:
        model, artifact, severity = parse_result_path(features_path)
        key = (model, artifact, severity)

        if key in keys_done and not FORCE_RECOMPUTE:
            print(f"Skipping existing row: {key}")
            continue

        row = evaluate_one(features_path, head_cache, device)
        rows.append(row)

        # Save after every folder, so progress is not lost
        save_rows(rows)
        keys_done.add(key)

        print(f"Saved progress to: {OUT_CSV}")

    print("\nDone.")
    print(f"Final CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()