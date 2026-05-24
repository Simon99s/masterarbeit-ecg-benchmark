import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--severity-name", required=True)
    parser.add_argument("--severity-value", required=True, type=float)
    return parser.parse_args()

# ==========================================================
# CONFIG
# ==========================================================

args = parse_args()

ARTIFACT = args.artifact.lower()
SEVERITY_NAME = args.severity_name
SEVERITY_VALUE = args.severity_value

VALID_ARTIFACTS = ["in"]

if ARTIFACT not in VALID_ARTIFACTS:
    raise ValueError(f"Unknown artifact: {ARTIFACT}")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHYSIONET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "physionet.org"
    / "files"
    / "challenge-2021"
    / "1.0.3"
    / "training"
)

BENCH_ROOT = PROJECT_ROOT / "data" / "Benchmark" / f"physionet_{ARTIFACT}"

LEAD_TO_PLOT = 1
PLOT_FIRST_RECORD = False
VERIFY_ALL_RECORDS = True

# ==========================================================
# UTIL
# ==========================================================

def check_limb_consistency(sig):
    I = sig[0]
    II = sig[1]

    III_calc = II - I
    aVR_calc = -0.5 * (I + II)
    aVL_calc = I - 0.5 * II
    aVF_calc = II - 0.5 * I

    errors = {
        "III": np.mean(np.abs(III_calc - sig[2])),
        "aVR": np.mean(np.abs(aVR_calc - sig[3])),
        "aVL": np.mean(np.abs(aVL_calc - sig[4])),
        "aVF": np.mean(np.abs(aVF_calc - sig[5])),
    }
    return errors

# ==========================================================
# LOAD CLEAN RECORDS
# ==========================================================

all_clean_records = glob(os.path.join(PHYSIONET_ROOT, "**", "*.hea"), recursive=True)
all_clean_records = sorted(all_clean_records)

records_by_source = defaultdict(list)

for path in all_clean_records:
    relative = os.path.relpath(path, PHYSIONET_ROOT)
    source = relative.split(os.sep)[0]
    records_by_source[source].append(path)

# ==========================================================
# VERIFICATION
# ==========================================================

density_stats = []
amp_stats = []

for source, paths in records_by_source.items():

    print(f"\nChecking source: {source}")

    for rec_path_hea in sorted(paths)[:10]:

        record_path = rec_path_hea.replace(".hea", "")
        record_id = os.path.basename(record_path)

        clean, fields = wfdb.rdsamp(record_path)
        clean = np.nan_to_num(clean.T.astype(np.float32), nan=0.0)

        # ------------------------------------------------
        # RESAMPLE CLEAN TO 500 Hz 
        # ------------------------------------------------
        fs_in = int(fields["fs"])
        if fs_in != 500:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(fs_in, 500)
            up = 500 // g
            down = fs_in // g
            clean = resample_poly(clean, up, down, axis=1).astype(np.float32)




        corrupt_path = os.path.join(
            BENCH_ROOT,
            f"physionet_{ARTIFACT}_{SEVERITY_NAME}",
            os.path.relpath(record_path, PHYSIONET_ROOT),
        )

        try:
            corrupt, _ = wfdb.rdsamp(corrupt_path)
        except FileNotFoundError:
            print(f"Missing corrupted record: {record_id}")
            continue
        corrupt = np.nan_to_num(corrupt.T.astype(np.float32), nan=0.0)

        # ------------------------------------------------
        # Align lengths if needed (PTB fix)
        # ------------------------------------------------
        clean_aligned = clean
        corrupt_aligned = corrupt

        if clean.shape[1] != corrupt.shape[1]:
            min_len = min(clean.shape[1], corrupt.shape[1])
            clean_aligned = clean[:, :min_len]
            corrupt_aligned = corrupt[:, :min_len]


        # ------------------------------------------------
        # Independent leads only (I, II, V1–V6)
        # ------------------------------------------------
        independent_idx = [0, 1, 6, 7, 8, 9, 10, 11]

        for idx in independent_idx:

            clean_lead = clean_aligned[idx]
            corrupt_lead = corrupt_aligned[idx]

            diff = corrupt_lead - clean_lead

            rms = np.sqrt(np.mean(clean_lead**2) + 1e-12)
            threshold = 0.5 * rms

            mask = np.abs(diff) > threshold

            p_measured = np.sum(mask) / len(mask)

            density_stats.append(p_measured)

        # ------------------------------------------------
        # Amplitude ratio
        # ------------------------------------------------
        ratios = []

        for idx in independent_idx:

            clean_lead = clean_aligned[idx]
            corrupt_lead = corrupt_aligned[idx]
            diff_lead = corrupt_lead - clean_lead

            rms = np.sqrt(np.mean(clean_lead**2) + 1e-12)
            threshold = 0.3 * rms

            mask_lead = np.abs(diff_lead) > threshold

            if np.sum(mask_lead) == 0:
                continue

            mean_amp = np.mean(np.abs(diff_lead[mask_lead]))

            ratios.append(mean_amp / rms)

        amp_ratio_measured = np.mean(ratios) if len(ratios) > 0 else 0
        amp_stats.append(amp_ratio_measured)


        # ------------------------------------------------
        # Limb consistency
        # ------------------------------------------------
        limb_errors = check_limb_consistency(corrupt)

        print(
            f"{record_id} | {SEVERITY_NAME} | "
            f"p_target={SEVERITY_VALUE:.3f} | "
            f"p_measured={p_measured:.3f} | "
            f"amp_ratio≈{amp_ratio_measured:.2f}"
        )

        for lead_name, err in limb_errors.items():
            if err > 1e-3:
                print(f"⚠ Limb inconsistency {lead_name}: {err}")

        # ------------------------------------------------
        # Plot
        # ------------------------------------------------
        if PLOT_FIRST_RECORD:
            plt.figure(figsize=(12,4))
            plt.plot(clean[LEAD_TO_PLOT], label="clean")
            plt.plot(corrupt[LEAD_TO_PLOT], label="corrupted")
            plt.legend()
            plt.title(f"{record_id} - {SEVERITY_NAME} - Lead II")
            plt.show()
            PLOT_FIRST_RECORD = False

        if not VERIFY_ALL_RECORDS:
            break

# ==========================================================
# SUMMARY
# ==========================================================

print("\n==============================")
print("IMPULSE DENSITY SUMMARY")
print("==============================")

density_values = np.array(density_stats)

if len(density_values) > 0:
    print(
        f"{ARTIFACT} {SEVERITY_NAME} | "
        f"Target p: {SEVERITY_VALUE:.4f} | "
        f"Mean p: {density_values.mean():.4f} | "
        f"Std: {density_values.std():.4f}"
    )
else:
    print("No impulse density values computed.")

print("\n==============================")
print("AMPLITUDE RATIO SUMMARY")
print("==============================")

amp_values = np.array(amp_stats)

if len(amp_values) > 0:
    print(
        f"{ARTIFACT} {SEVERITY_NAME} | "
        f"Mean ratio: {amp_values.mean():.2f} | "
        f"Std: {amp_values.std():.2f}"
    )
else:
    print("No amplitude ratio values computed.")
