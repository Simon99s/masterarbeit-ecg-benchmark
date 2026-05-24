import os
import scipy.signal as signal
from config import NSTDB_ROOT
import numpy as np
from math import gcd
import wfdb
from scipy.signal import resample_poly

_NSTDB_CACHE = {}


def butter_highpass_np(x, fs, cutoff=0.5, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype="high")
    return signal.filtfilt(b, a, x, axis=-1)



def apply_artifact(ecg, artifact_type, severity, fs, rng):

    if artifact_type is None:
        return ecg

    if artifact_type in ["em", "ma"]:
        noise = load_nstdb_full(
            artifact_type,
            seg_len=ecg.shape[1],
            fs_out=fs,
            rng=rng,
        )
        return inject_nstdb(ecg, noise, fs, severity)

    if artifact_type == "gn":
        return add_gaussian_noise(ecg, severity, fs, rng)

    if artifact_type == "dn":
        return discretize_signal(ecg, severity)

    if artifact_type == "in":
        return add_impulse_noise(
            ecg,
            p=severity["p"],
            amp_ratio=severity["amp"],
            rng=rng
        )
    

def load_nstdb_full(artifact_type, seg_len, fs_out, rng):
    """
    Load continuous NSTDB noise of length seg_len.
    Returns (12, seg_len)
    """

    if artifact_type not in _NSTDB_CACHE:
        rec = {"bw": "bw", "em": "em", "ma": "ma"}[artifact_type]
        rec_path = os.path.join(NSTDB_ROOT, rec)

        sig, fields = wfdb.rdsamp(rec_path)
        noise = sig[:, 0].astype(np.float32)
        noise -= noise.mean()

        fs_in = int(fields["fs"])

        if fs_in != fs_out:
            g = gcd(fs_in, fs_out)
            up = fs_out // g
            down = fs_in // g
            noise = resample_poly(noise, up, down)
            noise -= noise.mean()

        _NSTDB_CACHE[artifact_type] = noise

    noise = _NSTDB_CACHE[artifact_type]

    if len(noise) < seg_len:
        raise RuntimeError("NSTDB signal too short")

    max_start = len(noise) - seg_len
    noise12 = np.zeros((12, seg_len), dtype=np.float32)

    for lead in range(12):
        start = rng.integers(0, max_start + 1)
        noise12[lead] = noise[start:start + seg_len]

    return noise12



def inject_nstdb(ecg, noise, fs, severity):
    """
    Inject NSTDB noise with controlled SNR (in dB).

    Parameters
    ----------
    ecg : np.ndarray (12, T)
        Clean ECG segment
    noise : np.ndarray (12, T)
        NSTDB noise segment (already sliced + resampled)
    fs : int
        Sampling rate (not needed here but kept for API consistency)
    severity : float
        Target SNR in dB (e.g. 4, -1, -4)

    Returns
    -------
    np.ndarray (12, T)
        Corrupted ECG with preserved limb equations
    """

    if severity is None:
        return ecg

    snr_db = severity

    clean = ecg.copy()
    noise = noise.copy()

    out = np.zeros_like(clean)

    # -------------------------
    # Limb leads I & II
    # -------------------------
    idx_I = 0
    idx_II = 1

    for idx in [idx_I, idx_II]:

        clean_lead = clean[idx]
        noise_lead = noise[idx] - np.mean(noise[idx])

        Px = np.mean(clean_lead ** 2)
        Pn = np.mean(noise_lead ** 2)

        Pn = max(Pn, 1e-8)  # numerical stability

        alpha = np.sqrt(Px / (Pn * 10 ** (snr_db / 10)))

        out[idx] = clean_lead + alpha * noise_lead

    # -------------------------
    # Recompute derived limb leads
    # -------------------------
    I_prime = out[idx_I]
    II_prime = out[idx_II]

    out[2] = II_prime - I_prime                 # III
    out[3] = -0.5 * (I_prime + II_prime)       # aVR
    out[4] = I_prime - 0.5 * II_prime          # aVL
    out[5] = II_prime - 0.5 * I_prime          # aVF

    # -------------------------
    # Precordial leads V1–V6
    # -------------------------
    for idx in range(6, 12):

        clean_lead = clean[idx]
        noise_lead = noise[idx] - np.mean(noise[idx])

        Px = np.mean(clean_lead ** 2)
        Pn = np.mean(noise_lead ** 2)

        Pn = max(Pn, 1e-8)

        alpha = np.sqrt(Px / (Pn * 10 ** (snr_db / 10)))

        out[idx] = clean_lead + alpha * noise_lead

    return out

def add_gaussian_noise(ecg, snr_db, fs, rng):
    """
    Per-lead SNR controlled Gaussian noise.
    Preserves limb lead equations.
    ecg: (12, T)
    """

    if snr_db is None:
        return ecg

    clean = ecg.copy()
    out = np.zeros_like(clean)

    idx_I = 0
    idx_II = 1
    idx_V = list(range(6, 12))

    # Generate white noise
    noise = rng.normal(0, 1, clean.shape).astype(np.float32)

    # ---- I & II ----
    for idx in [idx_I, idx_II]:

        clean_lead = clean[idx]
        noise_lead = noise[idx] - np.mean(noise[idx])

        clean_ref = butter_highpass_np(clean_lead[None, :], fs, 0.5)[0]
        noise_ref = butter_highpass_np(noise_lead[None, :], fs, 0.5)[0]

        Px = np.mean(clean_ref ** 2)
        Pn = np.mean(noise_ref ** 2) + 1e-12

        alpha = np.sqrt(Px / (Pn * 10 ** (snr_db / 10)))

        out[idx] = clean_lead + alpha * noise_lead

    # ---- Recompute derived limb leads ----
    I_prime = out[idx_I]
    II_prime = out[idx_II]

    out[2] = II_prime - I_prime
    out[3] = -0.5 * (I_prime + II_prime)
    out[4] = I_prime - 0.5 * II_prime
    out[5] = II_prime - 0.5 * I_prime

    # ---- Precordial leads ----
    for idx in idx_V:

        clean_lead = clean[idx]
        noise_lead = noise[idx] - np.mean(noise[idx])

        clean_ref = butter_highpass_np(clean_lead[None, :], fs, 0.5)[0]
        noise_ref = butter_highpass_np(noise_lead[None, :], fs, 0.5)[0]

        Px = np.mean(clean_ref ** 2)
        Pn = np.mean(noise_ref ** 2) + 1e-12

        alpha = np.sqrt(Px / (Pn * 10 ** (snr_db / 10)))

        out[idx] = clean_lead + alpha * noise_lead

    return out


def add_impulse_noise(ecg, p, amp_ratio, rng):
    """
    Impulse noise on independent leads only.
    Limb equations preserved.
    """

    clean = ecg.copy()
    out = np.zeros_like(clean)

    idx_I = 0
    idx_II = 1
    idx_V = list(range(6, 12))

    # ---- I & II ----
    for idx in [idx_I, idx_II]:

        clean_lead = clean[idx]

        rms = np.sqrt(np.mean(clean_lead**2) + 1e-12)
        amp = amp_ratio * rms

        mask = rng.random(clean_lead.shape) < p
        sign = rng.choice([-1.0, 1.0], size=clean_lead.shape)

        impulses = mask * sign * amp

        out[idx] = clean_lead + impulses

    # ---- Recompute derived limb leads ----
    I_prime = out[idx_I]
    II_prime = out[idx_II]

    out[2] = II_prime - I_prime
    out[3] = -0.5 * (I_prime + II_prime)
    out[4] = I_prime - 0.5 * II_prime
    out[5] = II_prime - 0.5 * I_prime

    # ---- Precordial leads ----
    for idx in idx_V:

        clean_lead = clean[idx]

        rms = np.sqrt(np.mean(clean_lead**2) + 1e-12)
        amp = amp_ratio * rms

        mask = rng.random(clean_lead.shape) < p
        sign = rng.choice([-1.0, 1.0], size=clean_lead.shape)

        impulses = mask * sign * amp

        out[idx] = clean_lead + impulses

    return out

def discretize_signal(ecg, target_snr_db):
    """
    Per-lead SNR-defined quantization.
    ecg: (12, T)
    """

    if target_snr_db is None:
        return ecg

    clean = ecg.copy()
    out = np.zeros_like(clean)

    snr_linear = 10 ** (target_snr_db / 10.0)

    for idx in range(12):

        x = clean[idx]

        sigma_x = np.std(x)
        sigma_x = max(sigma_x, 1e-8)

        delta = sigma_x * np.sqrt(12.0 / snr_linear)

        out[idx] = delta * np.round(x / delta)

    return out