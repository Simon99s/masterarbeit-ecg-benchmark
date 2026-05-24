import os
import wfdb
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd

# ------------------------------------------------
# ROOT PATHS
# ------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = INFERENCE_ROOT.parent

BENCHMARK_ROOT = PROJECT_ROOT / "data" / "Benchmark"
OUT_ROOT = INFERENCE_ROOT / "st-mem"

CHECKPOINT = PROJECT_ROOT / "training" / "st_mem" / "out_stmem_emory21" / "stmem_best.pt"

############################################
# CONFIG
############################################

RUN_FIRST_ONLY = False
MAX_RECORDS = None

FIRST_RUNS = [
]

SKIP_ARTIFACTS = []

# ------------------------------------------------

TARGET_FS = 250
SEG_LEN = 2250

# ------------------------------------------------
# FILTER
# ------------------------------------------------

def create_bandpass(fs=250, low=0.67, high=40.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


B, A = create_bandpass(TARGET_FS)


def bandpass_filter(x):
    return filtfilt(B, A, x, axis=-1)


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

    def split_segments(self, data):
        L = data.shape[1]
        n = L // SEG_LEN

        if n == 0:
            pad = SEG_LEN - L
            data = np.pad(data, ((0, 0), (0, pad)))
            return data[None, :, :]

        data = data[:, :n * SEG_LEN]

        segments = data.reshape(12, n, SEG_LEN)
        segments = np.transpose(segments, (1, 0, 2))  # [N, 12, 2250]

        return segments

    def __getitem__(self, idx):
        record = self.records[idx]

        sig, meta = wfdb.rdsamp(record)
        fs_in = int(meta["fs"])

        # -------------------------
        # RESAMPLE to 250 Hz
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
        target_leads = [
            "I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6"
        ]
        idxs = [meta["sig_name"].index(l) for l in target_leads]
        data = data[idxs]

        # -------------------------
        # BANDPASS
        # -------------------------
        data = bandpass_filter(data)

        # -------------------------
        # SEGMENTATION
        # -------------------------
        segments = self.split_segments(data)

        # -------------------------
        # Z-SCORE NORMALIZATION
        # -------------------------
        mean = segments.mean(axis=(1, 2), keepdims=True)
        std = segments.std(axis=(1, 2), keepdims=True) + 1e-8
        segments = (segments - mean) / std

        segments = torch.from_numpy(segments).float()

        record_id = os.path.basename(record)

        return segments, record_id


# ------------------------------------------------
# MODEL
# ------------------------------------------------

def load_model(device):
    import models.encoder as encoder

    model = encoder.st_mem_vit_base(
        seq_len=2250,
        patch_size=75,
        num_classes=21,
        num_leads=12,
        qkv_bias=True,
    )

    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    model.to(device)
    model.eval()

    print("\n=== HEAD PARAM DEBUG ===")
    for name, param in model.named_parameters():
        if "head" in name:
            print(name, param.mean().item(), param.std().item())
    print("========================\n")

    return model


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
        prefetch_factor=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    record_probs = []
    record_features = []
    record_ids = []

    captured = {}

    def save_head_input(module, inputs, output):
        # Feature tensor before the classification head
        captured["features"] = inputs[0].detach()

    hook_handle = model.head.register_forward_hook(save_head_input)

    use_amp = device.type == "cuda"

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):

        for segments, rid in tqdm(loader, total=len(dataset), desc=str(folder)):

            segments = segments.squeeze(0)
            segments = segments.to(device, non_blocking=True)

            captured.clear()

            logits = model(segments)

            # Captured features before classification head
            features = captured["features"]  # [num_segments, D]

            avg_logits = logits.mean(dim=0)  # [21]
            avg_features = features.float().mean(dim=0)  # [D]

            probs = torch.sigmoid(avg_logits).cpu().numpy()

            record_probs.append(probs)
            record_features.append(avg_features.cpu().numpy())
            record_ids.append(rid[0])

    hook_handle.remove()

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
    # RUN ALL
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