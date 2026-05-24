import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from scipy import signal
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

VALID_ARTIFACTS = ["ma", "em", "gn", "dn"]

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
# UTILITY
# ==========================================================



def butter_highpass_np(x, fs, cutoff=0.5, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype="high")
    return signal.filtfilt(b, a, x)

def compute_snr_db(clean, corrupted, fs=500):
    noise = corrupted - clean

    clean_ref = butter_highpass_np(clean, fs, 0.5)
    noise_ref = butter_highpass_np(noise, fs, 0.5)

    Px = np.mean(clean_ref**2)
    Pn = np.mean(noise_ref**2)

    return 10 * np.log10(Px / Pn)

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
# FIND RECORDS
# ==========================================================

all_clean_records = glob(os.path.join(PHYSIONET_ROOT, "**", "*.hea"), recursive=True)
all_clean_records = sorted(all_clean_records)

print(f"Found {len(all_clean_records)} clean records.")

# Group by source
records_by_source = defaultdict(list)

for path in all_clean_records:
    relative = os.path.relpath(path, PHYSIONET_ROOT)
    source = relative.split(os.sep)[0]
    records_by_source[source].append(path)

print("Sources:", records_by_source.keys())

# ==========================================================
# VERIFICATION LOOP
# ==========================================================

snr_stats_ind = []

for source, paths in records_by_source.items():

    print(f"\nChecking source: {source}")

    for rec_path_hea in sorted(paths)[:10]:  

        record_path = rec_path_hea.replace(".hea", "")
        record_id = os.path.basename(record_path)

        clean, fields = wfdb.rdsamp(record_path)
        clean = np.nan_to_num(clean.T.astype(np.float32), nan=0.0)

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

        corrupt, _ = wfdb.rdsamp(corrupt_path)
        corrupt = np.nan_to_num(corrupt.T, nan=0.0)


        # -----------------------------
        # NaN check
        # -----------------------------
        nan_clean = np.isnan(clean).any()
        nan_corrupt = np.isnan(corrupt).any()

        # -----------------------------
        # SNR (independent vs all)
        # -----------------------------
        independent_leads = [0, 1, 6, 7, 8, 9, 10, 11]

        for lead in independent_leads:
            snr = compute_snr_db(clean[lead], corrupt[lead])
            snr_stats_ind.append(snr)

        # -----------------------------
        # Limb consistency
        # -----------------------------
        limb_errors = check_limb_consistency(corrupt)

        # -----------------------------
        # Amplitude check
        # -----------------------------
        max_clean = np.max(np.abs(clean))
        max_corrupt = np.max(np.abs(corrupt))

        if max_corrupt > 1e6:
            print("⚠ Extreme amplitude detected!")

        for lead_name, err in limb_errors.items():
            if err > 1e-3:
                print(f"⚠ Limb inconsistency {lead_name}: {err}")

        # -----------------------------
        # Plot first record 
        # -----------------------------
        if PLOT_FIRST_RECORD:
            plt.figure(figsize=(12,4))
            plt.plot(clean[LEAD_TO_PLOT], label="clean")
            plt.plot(corrupt[LEAD_TO_PLOT], label="corrupted")
            plt.legend()
            plt.title(f"{record_id} - {SEVERITY_NAME} - Lead II")
            plt.show()
            PLOT_FIRST_RECORD = False  # only once

        if not VERIFY_ALL_RECORDS:
            break

# ==========================================================
# SUMMARY
# ==========================================================

print("\n==============================")
print("SNR SUMMARY (independent leads)")
print("==============================")

values = np.array(snr_stats_ind)

if len(values) > 0:
    print(
        f"{ARTIFACT} {SEVERITY_NAME} | "
        f"Target value: {SEVERITY_VALUE} | "
        f"Mean: {values.mean():.2f} dB | "
        f"Std: {values.std():.2f}"
    )
else:
    print("No SNR values computed.")
