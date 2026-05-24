import os
import wfdb
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd
from ecg_jepa import ecg_jepa

import torch.nn as nn

# ------------------------------------------------
# PATHS
# ------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = INFERENCE_ROOT.parent

BENCHMARK_ROOT = PROJECT_ROOT / "data" / "Benchmark"
OUT_ROOT = INFERENCE_ROOT / "jepa"

CHECKPOINT = PROJECT_ROOT / "training" / "out_jepa_emory21" / "ckpt_best.pt"
ENCODER_WEIGHTS = PROJECT_ROOT / "training" / "jepa" / "encoder.pth"  # pretrained encoder

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

TARGET_FS = 250
SEG_LEN = 2500  # 🔥 JEPA = full 10s

############################################
# RUN CONFIG
############################################

RUN_FIRST_ONLY = False  # 🔥 run only selected folders and exit
MAX_RECORDS = None      # e.g. 200 for debugging

FIRST_RUNS = [
]

SKIP_ARTIFACTS = [
    # "physionet_dn",
]

# ------------------------------------------------
# FILTER
# ------------------------------------------------

def get_filter(fs=250, low=0.67, high=40.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return b, a

B, A = get_filter()

def bandpass_filter(x):
    return filtfilt(B, A, x, axis=-1, method="pad")



def load_encoder(ckpt_dir=None, leads=None):

    if leads is None:
        leads = [0,1,2,3,4,5,6,7]

    params = {
        'encoder_embed_dim': 768,
        'encoder_depth': 12,
        'encoder_num_heads': 16,
        'predictor_embed_dim': 384,
        'predictor_depth': 6,
        'predictor_num_heads': 12,
        'c': 8,
        'pos_type': 'sincos',
        'mask_scale': (0, 0),
        'leads': leads
    }

    encoder = ecg_jepa(**params).encoder
    embed_dim = 768

    # 🔥 only load weights if path is given
    if ckpt_dir is not None:
        ckpt = torch.load(ckpt_dir, map_location="cpu")

        if "encoder" in ckpt:
            # pretrained JEPA format
            encoder.load_state_dict(ckpt["encoder"])

        elif "encoder_state" in ckpt:
            # your fine-tuned format
            encoder.load_state_dict(ckpt["encoder_state"])

        else:
            raise ValueError("Unknown checkpoint format")

    return encoder, embed_dim

# ------------------------------------------------
# DATASET
# ------------------------------------------------

class BenchmarkDataset(Dataset):

    def __init__(self, folder):
        self.records = []

        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".hea"):
                    self.records.append(os.path.join(root, f.replace(".hea", "")))

        self.records = sorted(self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        record = self.records[idx]

        # -------------------------
        # LOAD
        # -------------------------
        sig, meta = wfdb.rdsamp(record)
        fs_in = int(meta["fs"])

        # -------------------------
        # RESAMPLE → 250 Hz
        # -------------------------
        if fs_in != TARGET_FS:
            g = gcd(fs_in, TARGET_FS)
            up = TARGET_FS // g
            down = fs_in // g
            sig = resample_poly(sig, up, down, axis=0).astype(np.float32)

        data = np.nan_to_num(sig.T.astype(np.float32))

        # -------------------------
        # LEAD ORDER
        # -------------------------
        target_leads = ['I','II','III','aVR','aVL','aVF',
                        'V1','V2','V3','V4','V5','V6']
        idxs = [meta["sig_name"].index(l) for l in target_leads]
        data = data[idxs]

        # -------------------------
        # BANDPASS
        # -------------------------
        data = bandpass_filter(data)

        # -------------------------
        # SPLIT INTO WINDOWS
        # -------------------------
        segments = self.split_segments(data)
            

        # -------------------------
        # SELECT 8 LEADS
        # -------------------------
        segments = segments[:, [0, 1, 6, 7, 8, 9, 10, 11], :]  # [N, 8, 2500]

        x = torch.from_numpy(segments).float()

        record_id = os.path.basename(record)

        return x, record_id
    
    def split_segments(self, data):
        L = data.shape[1]
        seg_len = SEG_LEN

        # Pad to make length divisible by SEG_LEN
        pad_len = (seg_len - (L % seg_len)) % seg_len
        if pad_len > 0:
            data = np.pad(data, ((0, 0), (0, pad_len)))

        # [12, T] -> [num_segments, 12, SEG_LEN]
        segments = data.reshape(12, -1, seg_len)
        segments = np.transpose(segments, (1, 0, 2))

        return segments

# ------------------------------------------------
# MODEL
# ------------------------------------------------

def load_model(device):

    # build encoder ONLY (no weights yet)
    encoder, embed_dim = load_encoder(None)

    # build head
    head = nn.Linear(embed_dim, 21)

    # load fine-tuned checkpoint
    ckpt = torch.load(CHECKPOINT, map_location=device)

    encoder.load_state_dict(ckpt["encoder_state"])
    head.load_state_dict(ckpt["head_state"])

    encoder = encoder.to(device).eval()
    head = head.to(device).eval()

    return encoder, head

# ------------------------------------------------
# INFERENCE
# ------------------------------------------------

def run_inference(folder, max_records=None):

    dataset = BenchmarkDataset(folder)

    if max_records is not None:
        dataset.records = dataset.records[:max_records]

    print(f"Records in dataset: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, head = load_model(device)

    record_probs = []
    record_features = []
    record_ids = []

    use_amp = device.type == "cuda"

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):

        for segments, rid in tqdm(loader, total=len(loader), desc=str(folder)):

            # DataLoader adds batch dimension:
            # [1, num_segments, 8, 2500] -> [num_segments, 8, 2500]
            segments = segments.squeeze(0)
            segments = segments.to(device, non_blocking=True)

            features = encoder.representation(segments)   # [num_segments, D]
            logits = head(features)                       # [num_segments, 21]

            avg_logits = logits.mean(dim=0)               # [21]
            probs = torch.sigmoid(avg_logits)             # [21]

            avg_features = features.float().mean(dim=0)   # [D]

            record_probs.append(probs.cpu().numpy())
            record_features.append(avg_features.cpu().numpy())
            record_ids.append(rid[0])

    record_probs = np.stack(record_probs)
    record_features = np.stack(record_features)

    print("probs shape:", record_probs.shape)
    print("features shape:", record_features.shape)

    return record_probs, record_features, record_ids

# ------------------------------------------------
# RUN
# ------------------------------------------------

def main():

    ############################################
    # RUN FIRST 
    ############################################

    for folder in FIRST_RUNS:

        print("\nRunning FIRST:", folder)

        probs, features, rids = run_inference(folder, max_records=MAX_RECORDS)

        out_dir = OUT_ROOT / folder.parent.name / folder.name
        os.makedirs(out_dir, exist_ok=True)

        np.save(out_dir / "probs21.npy", probs)
        np.save(out_dir / "features.npy", features)
        np.save(out_dir / "record_ids.npy", np.array(rids))

    if RUN_FIRST_ONLY:
        print("\nExiting after FIRST_RUNS")
        return

    ############################################
    #RUN ALL
    ############################################

    for artifact in BENCHMARK_ROOT.iterdir():

        if not artifact.is_dir():
            continue

        if artifact.name in SKIP_ARTIFACTS:
            print(f"Skipping artifact: {artifact.name}")
            continue

        for severity in artifact.iterdir():

            if not severity.is_dir():
                continue

            print(f"\nArtifact: {artifact.name} | Severity: {severity.name}")

            probs, features, rids = run_inference(severity, max_records=MAX_RECORDS)

            out_dir = OUT_ROOT / artifact.name / severity.name
            os.makedirs(out_dir, exist_ok=True)

            np.save(out_dir / "probs21.npy", probs)
            np.save(out_dir / "features.npy", features)
            np.save(out_dir / "record_ids.npy", np.array(rids))


if __name__ == "__main__":
    main()