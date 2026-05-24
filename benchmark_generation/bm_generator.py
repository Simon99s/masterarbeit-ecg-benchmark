import os
import wfdb
import numpy as np
import hashlib
from pathlib import Path
from tqdm import tqdm
from scipy.signal import resample_poly
from math import gcd
from scipy import signal
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--severity-name", required=True)
    parser.add_argument("--severity-value", required=True, type=float)
    return parser.parse_args()

def butter_highpass_np(x, fs, cutoff=0.5, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype="high")
    return signal.filtfilt(b, a, x)

# ===========================
# CONFIG
# ===========================
args = parse_args()

ARTIFACT = args.artifact.lower()
SEVERITY_NAME = args.severity_name
SEVERITY_VALUE = args.severity_value

if ARTIFACT not in ["em", "ma"]:
    raise ValueError(f"bm_generator.py only supports em and ma, got: {ARTIFACT}")

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

NSTDB_ROOT = PROJECT_ROOT / "data" / "mit-bih-noise-stress-test-database-1.0.0"

BENCH_ROOT = PROJECT_ROOT / "data" / "Benchmark" / f"physionet_{ARTIFACT}"

FS = 500

# ===========================
# LOAD NSTDB EM
# ===========================

noise_sig, fields = wfdb.rdsamp(str(NSTDB_ROOT / ARTIFACT))

# shape: (T, C) → transpose to (C, T)
noise = noise_sig.astype(np.float32).T  

fs_in = int(fields["fs"])

# Resample along time axis (axis=1!)
if fs_in != FS:
    g = gcd(fs_in, FS)
    up = FS // g
    down = fs_in // g
    noise = resample_poly(noise, up, down, axis=1).astype(np.float32)

# Remove mean per channel
noise = noise - noise.mean(axis=1, keepdims=True)

T_noise = noise.shape[1]

print(f"NSTDB {ARTIFACT.upper()} loaded (resampled):", noise.shape)


# ===========================
# inject nstdb 
# ===========================

def inject_nstdb(ecg, artifact, fs, snr_db, artifact_type):

    if snr_db is None:
        return ecg

    clean = ecg.copy()
    noise = artifact.copy()

    idx_I = 0
    idx_II = 1
    idx_V = list(range(6, 12))

    out = np.zeros_like(clean)

    # 1️⃣ corrupt I and II
    for idx in [idx_I, idx_II]:

        clean_lead = clean[idx]
        noise_lead = noise[idx] - np.mean(noise[idx])

        clean_ref = butter_highpass_np(clean_lead, fs, 0.5)
        noise_ref = butter_highpass_np(noise_lead, fs, 0.5)

        Px = np.mean(clean_ref ** 2)
        Pn = np.mean(noise_ref ** 2)

        alpha = np.sqrt(Px / (Pn * 10 ** (snr_db / 10)))

        out[idx] = clean_lead + alpha * noise_lead


    # 2️⃣ recompute limb leads
    I_prime = out[idx_I]
    II_prime = out[idx_II]

    out[2] = II_prime - I_prime
    out[3] = -0.5 * (I_prime + II_prime)
    out[4] = I_prime - 0.5 * II_prime
    out[5] = II_prime - 0.5 * I_prime


    # 3️⃣ corrupt V1–V6
    for idx in idx_V:

        clean_lead = clean[idx]
        noise_lead = noise[idx] - np.mean(noise[idx])

        clean_ref = butter_highpass_np(clean_lead, fs, 0.5)
        noise_ref = butter_highpass_np(noise_lead, fs, 0.5)

        Px = np.mean(clean_ref ** 2)
        Pn = np.mean(noise_ref ** 2)

        alpha = np.sqrt(Px / (Pn * 10 ** (snr_db / 10)))

        out[idx] = clean_lead + alpha * noise_lead

    return out

# ===========================
# MAIN LOOP
# ===========================

from glob import glob

# recursively find all .hea files
all_records = glob(os.path.join(PHYSIONET_ROOT, "**", "*.hea"), recursive=True)

# optional: limit to first N per source if needed
all_records = sorted(all_records)

print(f"Found {len(all_records)} records.")

from collections import defaultdict

# group records by source (first folder under training/)
records_by_source = defaultdict(list)

for path in all_records:
    relative = os.path.relpath(path, PHYSIONET_ROOT)
    source = relative.split(os.sep)[0]  # first folder (chapman_shaoxing, ptb, etc.)
    records_by_source[source].append(path)

print("Sources found:", records_by_source.keys())

# Get list of source names
sources = list(records_by_source.keys())

# Move georgia to front
if "georgia" in sources:
    sources.remove("georgia")
    sources.insert(0, "georgia")

# Iterate in new order
for source in sources:
    paths = records_by_source[source]

    print(f"\nProcessing source: {source}")

    for rec_path_hea in tqdm(sorted(paths)):

        record_path = rec_path_hea.replace(".hea", "")
        record_id = os.path.basename(record_path)

        try:
            record = wfdb.rdrecord(record_path)
        except FileNotFoundError:
            print(f"Missing signal file for {record_id}, skipping.")
            continue
        except Exception as e:
            print(f"Error loading {record_id}: {e}")
            continue

        clean = np.nan_to_num(record.p_signal.T.astype(np.float32), nan=0.0)

        sig_names = record.sig_name

        target_leads = [
            'I','II','III','aVR','aVL','aVF',
            'V1','V2','V3','V4','V5','V6'
        ]

        idx = [sig_names.index(l) for l in target_leads]

        clean = clean[idx]

        fs_in = int(record.fs)

        if fs_in != FS:
            g = gcd(fs_in, FS)
            up = FS // g
            down = fs_in // g
            clean = resample_poly(clean, up, down, axis=1).astype(np.float32)

        T = clean.shape[1]

        seed = int(hashlib.sha1(record_id.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)

        start = rng.integers(0, T_noise)
        end = start + T

        # ---- wrap NSTDB if needed ----
        if end <= T_noise:
            noise_slice = noise[:, start:end]
        else:
            remaining = T
            pos = start
            parts = []

            while remaining > 0:
                take = min(T_noise - pos, remaining)
                parts.append(noise[:, pos:pos + take])
                remaining -= take
                pos = 0

            noise_slice = np.concatenate(parts, axis=1)

        # ---- map 2 NSTDB channels → 12 ECG leads ----
        noise12 = np.zeros((12, T), dtype=np.float32)

        for lead in range(12):
            ch = 0 if lead < 6 else 1
            noise12[lead] = noise_slice[ch]

        relative_path = os.path.relpath(rec_path_hea, PHYSIONET_ROOT)
        relative_dir = os.path.dirname(relative_path)

        out_dir = BENCH_ROOT / f"physionet_{ARTIFACT}_{SEVERITY_NAME}" / relative_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        corrupted = inject_nstdb(
            ecg=clean,
            artifact=noise12,
            fs=FS,
            snr_db=SEVERITY_VALUE,
            artifact_type=ARTIFACT,
        )

        wfdb.wrsamp(
            record_name=record_id,
            fs=FS,
            units=["mV"] * 12,
            sig_name=target_leads,
            p_signal=corrupted.T.astype(np.float64),
            fmt=["16"] * 12,
            write_dir=str(out_dir),
        )




print(f"\nBenchmark generation finished for {ARTIFACT} {SEVERITY_NAME}.")
