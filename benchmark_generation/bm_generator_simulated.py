import os
import wfdb
import numpy as np
import hashlib
from pathlib import Path
from tqdm import tqdm
from glob import glob
from scipy.signal import resample_poly
from math import gcd
from collections import defaultdict
import torch
from scipy import signal
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

ARTIFACT_SHORT = args.artifact.lower()
SEVERITY_NAME = args.severity_name
SEVERITY_VALUE = args.severity_value

ARTIFACT_MAP = {
    "gn": "gaussian_noise",
    "in": "impulse_noise",
    "dn": "discretization",
}

if ARTIFACT_SHORT not in ARTIFACT_MAP:
    raise ValueError(
        f"bm_generator_simulated.py only supports gn, in, dn. Got: {ARTIFACT_SHORT}"
    )

ARTIFACT = ARTIFACT_MAP[ARTIFACT_SHORT]

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

BENCH_ROOT = PROJECT_ROOT / "data" / "Benchmark" / f"physionet_{ARTIFACT_SHORT}"

FS = 500
IMPULSE_AMP_RATIO = 10


# ==========================================================
# Helper
# ==========================================================

def deterministic_rng(record_id):
    seed = int(hashlib.sha1(record_id.encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed)

# ----------------------------------------------------------
# Gaussian Noise
# ----------------------------------------------------------
USE_HIGHPASS_FOR_SNR = True
HP_CUTOFF = 0.5

def butter_highpass_np(x, fs, cutoff=0.5, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype="high")
    return signal.filtfilt(b, a, x, axis=-1)

def inject_gaussian_noise(ecg, snr_db, rng, fs):

    if snr_db is None:
        return ecg

    clean = ecg.copy()
    out = np.zeros_like(clean)

    idx_I = 0
    idx_II = 1
    idx_V = list(range(6, 12))

    # --------------------------------------------------
    # Corrupt Leads I and II
    # --------------------------------------------------

    for idx in [idx_I, idx_II]:

        clean_lead = clean[idx]

        noise_lead = rng.normal(0, 1, clean_lead.shape).astype(np.float32)

        # remove mean (same as eval)
        noise_lead = noise_lead - np.mean(noise_lead)

        if USE_HIGHPASS_FOR_SNR:
            clean_ref = butter_highpass_np(clean_lead[None, :], fs, HP_CUTOFF)[0]
            noise_ref = butter_highpass_np(noise_lead[None, :], fs, HP_CUTOFF)[0]
        else:
            clean_ref = clean_lead
            noise_ref = noise_lead

        Px = np.mean(clean_ref ** 2)
        Pn = np.mean(noise_ref ** 2) + 1e-12

        alpha = np.sqrt(Px / (Pn * 10 ** (snr_db / 10)))

        out[idx] = clean_lead + alpha * noise_lead

    I_prime = out[idx_I]
    II_prime = out[idx_II]

    # --------------------------------------------------
    # Recompute dependent limb leads
    # --------------------------------------------------

    out[2] = II_prime - I_prime
    out[3] = -0.5 * (I_prime + II_prime)
    out[4] = I_prime - 0.5 * II_prime
    out[5] = II_prime - 0.5 * I_prime

    # --------------------------------------------------
    # Corrupt V1–V6
    # --------------------------------------------------

    for idx in idx_V:

        clean_lead = clean[idx]

        noise_lead = rng.normal(0, 1, clean_lead.shape).astype(np.float32)
        noise_lead = noise_lead - np.mean(noise_lead)

        if USE_HIGHPASS_FOR_SNR:
            clean_ref = butter_highpass_np(clean_lead[None, :], fs, HP_CUTOFF)[0]
            noise_ref = butter_highpass_np(noise_lead[None, :], fs, HP_CUTOFF)[0]
        else:
            clean_ref = clean_lead
            noise_ref = noise_lead

        Px = np.mean(clean_ref ** 2)
        Pn = np.mean(noise_ref ** 2) + 1e-12

        alpha = np.sqrt(Px / (Pn * 10 ** (snr_db / 10)))

        out[idx] = clean_lead + alpha * noise_lead

    return out

# ----------------------------------------------------------
# Impulse Noise
# ----------------------------------------------------------

def inject_impulse_noise(ecg, p, amp_ratio, rng):

    clean = ecg.copy()
    out = np.zeros_like(clean)

    idx_I = 0
    idx_II = 1
    idx_V = list(range(6, 12))

    def corrupt_lead(clean_lead):
        rms = np.sqrt(np.mean(clean_lead**2) + 1e-12)
        amp = amp_ratio * rms
        mask = rng.random(clean_lead.shape) < p
        sign = rng.choice([-1.0, 1.0], size=clean_lead.shape)
        return clean_lead + mask * sign * amp

    out[idx_I] = corrupt_lead(clean[idx_I])
    out[idx_II] = corrupt_lead(clean[idx_II])

    I = out[idx_I]
    II = out[idx_II]

    out[2] = II - I
    out[3] = -0.5 * (I + II)
    out[4] = I - 0.5 * II
    out[5] = II - 0.5 * I

    for idx in idx_V:
        out[idx] = corrupt_lead(clean[idx])

    return out

# ----------------------------------------------------------
# Discretization
# ----------------------------------------------------------

def inject_discretization(ecg, snr_db):

    x = torch.from_numpy(ecg).float()

    sigma = torch.std(x, dim=1, keepdim=True, unbiased=False)
    sigma = torch.clamp(sigma, min=1e-8)

    snr_linear = 10 ** (snr_db / 10.0)
    delta = sigma * torch.sqrt(
        torch.tensor(12.0 / snr_linear, dtype=x.dtype, device=x.device)
    )


    x_q = delta * torch.round(x / delta)

    return x_q.numpy()


def inject_discretization_consistent(ecg, snr_db):

    clean = ecg.copy()
    out = np.zeros_like(clean)

    idx_I = 0
    idx_II = 1
    idx_V = list(range(6, 12))

    # Discretize I, II
    for idx in [idx_I, idx_II]:
        out[idx] = inject_discretization(clean[idx:idx+1], snr_db)[0]

    I = out[idx_I]
    II = out[idx_II]

    # Recompute dependent leads
    out[2] = II - I
    out[3] = -0.5 * (I + II)
    out[4] = I - 0.5 * II
    out[5] = II - 0.5 * I

    # Discretize V1–V6
    for idx in idx_V:
        out[idx] = inject_discretization(clean[idx:idx+1], snr_db)[0]

    return out

# ==========================================================
# MAIN
# ==========================================================

print("Searching PhysioNet records...")
all_records = glob(os.path.join(PHYSIONET_ROOT, "**", "*.hea"), recursive=True)
print(f"Found {len(all_records)} records.")

records_by_source = defaultdict(list)

for path in all_records:
    relative = os.path.relpath(path, PHYSIONET_ROOT)
    source = relative.split(os.sep)[0]
    records_by_source[source].append(path)


# ----------------------------------------------------------
# Do st-petersburg_incart first (30 min records)
# ----------------------------------------------------------

source_order = ["st_petersburg_incart"]

# add remaining sources automatically
source_order += sorted([s for s in records_by_source.keys() if s not in source_order])

for source in source_order:

    paths = records_by_source[source]

    print(f"\nProcessing source: {source}")

    for rec_path_hea in tqdm(sorted(paths)):

        record_path = rec_path_hea.replace(".hea", "")
        record_id = os.path.basename(record_path)

        try:
            sig, fields = wfdb.rdsamp(record_path)
        except FileNotFoundError:
            print(f"Missing signal file for {record_id}, skipping.")
            continue
        except Exception as e:
            print(f"Error loading {record_id}: {e}")
            continue

        clean = np.nan_to_num(sig.T.astype(np.float32), nan=0.0)

        sig_names = fields["sig_name"]

        target_leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

        idx = [sig_names.index(l) for l in target_leads]

        clean = clean[idx]

        fs_in = int(fields["fs"])

        # resample if needed
        if fs_in != FS:
            g = gcd(fs_in, FS)
            up = FS // g
            down = fs_in // g
            clean = resample_poly(clean, up, down, axis=1).astype(np.float32)

        rng = deterministic_rng(record_id)

        relative_path = os.path.relpath(rec_path_hea, PHYSIONET_ROOT)
        relative_dir = os.path.dirname(relative_path)

        out_dir = (
            BENCH_ROOT
            / f"physionet_{ARTIFACT_SHORT}_{SEVERITY_NAME}"
            / relative_dir
        )

        out_dir.mkdir(parents=True, exist_ok=True)

        if ARTIFACT == "gaussian_noise":
            corrupted = inject_gaussian_noise(clean, SEVERITY_VALUE, rng, FS)

        elif ARTIFACT == "impulse_noise":
            corrupted = inject_impulse_noise(
                clean,
                p=SEVERITY_VALUE,
                amp_ratio=IMPULSE_AMP_RATIO,
                rng=rng,
            )

        elif ARTIFACT == "discretization":
            corrupted = inject_discretization_consistent(clean, SEVERITY_VALUE)

        else:
            raise ValueError(f"Unknown ARTIFACT: {ARTIFACT}")

        wfdb.wrsamp(
            record_name=record_id,
            fs=FS,
            units=["mV"] * 12,
            sig_name=target_leads,
            p_signal=corrupted.T.astype(np.float32),
            fmt=["16"] * 12,
            write_dir=str(out_dir),
        )

print("✅ Simulated benchmark generation finished.")
