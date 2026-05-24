from pathlib import Path
import h5py
import numpy as np
from scipy.signal import resample_poly

# ==========================================================
# CONFIG
# ==========================================================

INPUT_PATH = r"C:\Users\simon\Desktop\heedb_i0006_100.h5"

OUTPUT_DIR = Path(r"C:\Users\simon\Desktop\bm\training\jepa")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "heedb_i0006_250hz_jepa.h5"

EXPECTED_INPUT_LEN = 1000
EXPECTED_OUTPUT_LEN = 2500
EXPECTED_LEADS = 12

BATCH_SIZE = 64
STRICT_MODE = False


from scipy.signal import butter, filtfilt

def get_filter(fs=250, low=0.67, high=40.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return b, a

b, a = get_filter()

def bandpass_filter(x):
    return filtfilt(b, a, x, axis=-1, method="pad")

# ==========================================================
# MAIN
# ==========================================================

with h5py.File(INPUT_PATH, "r") as fin, h5py.File(OUTPUT_PATH, "w") as fout:

    X = fin["data"]
    Y = fin["label"]

    N = X.shape[0]

    print(f"Input dataset shape: {X.shape}")
    print(f"Saving to: {OUTPUT_PATH}")

    fout.create_dataset(
        "data",
        shape=(N, EXPECTED_LEADS, EXPECTED_OUTPUT_LEN),
        dtype="float32",
        chunks=(BATCH_SIZE, EXPECTED_LEADS, EXPECTED_OUTPUT_LEN)
    )

    fout.create_dataset("label", data=Y[:])

    if "subject_id" in fin:
        fout.create_dataset("subject_id", data=fin["subject_id"][:])
    if "csv_id" in fin:
        fout.create_dataset("csv_id", data=fin["csv_id"][:])

    skipped = 0

    for start in range(0, N, BATCH_SIZE):

        end = min(start + BATCH_SIZE, N)

        if start % 1000 == 0:
            print(f"{start}/{N} | skipped={skipped}")

        batch = X[start:end]
        buffer = []

        for i in range(batch.shape[0]):

            try:
                x = batch[i]

                # checks
                if x.shape[0] != EXPECTED_LEADS:
                    raise ValueError(f"Wrong leads: {x.shape}")

                if x.shape[1] != EXPECTED_INPUT_LEN:
                    raise ValueError(f"Not 10s: {x.shape[1]} samples")

                if not np.isfinite(x).all():
                    raise ValueError("NaN/Inf in raw signal")

                # 🔥 resample first (100Hz → 250Hz)
                x = resample_poly(x, up=5, down=2, axis=1)

                if x.shape[1] != EXPECTED_OUTPUT_LEN:
                    raise ValueError(f"Wrong length after resample: {x.shape}")

                # 🔥 bandpass (float64 for stability)
                x = x.astype(np.float64)
                x = bandpass_filter(x)

                if not np.isfinite(x).all():
                    raise ValueError("NaN/Inf after filtering")

                buffer.append(x.astype(np.float32))

            except Exception as e:
                skipped += 1
                print(f"[WARNING] Skipping record {start+i}: {e}")

                buffer.append(
                    np.zeros((EXPECTED_LEADS, EXPECTED_OUTPUT_LEN), dtype=np.float32)
                )

        fout["data"][start:end] = np.stack(buffer)

    print(f"\n✅ Done. Skipped {skipped} / {N} samples.")
    print(f"Saved file: {OUTPUT_PATH}")