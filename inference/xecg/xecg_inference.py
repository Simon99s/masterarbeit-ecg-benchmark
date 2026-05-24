import os
import wfdb
import json
import torch
import numpy as np
import pandas as pd
import time

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from net1d import Net1D
from util import filter_bandpass


from pathlib import Path


from downstream_models import xECGClassification
from safetensors.torch import load_file

PROFILE = False
# ------------------------------------------------
# ROOT PATHS
# ------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent          # .../inference/ECGFounder
INFERENCE_ROOT = SCRIPT_DIR.parent                    #.../inference
PROJECT_ROOT = INFERENCE_ROOT.parent                  # bm

# benchmark ECGs
BENCHMARK_ROOT = PROJECT_ROOT / "data" / "Benchmark"

# inference outputs
OUT_ROOT = INFERENCE_ROOT / "xECG"

# model checkpoint
CHECKPOINT = PROJECT_ROOT / "training" / "xecg" / "out_xecg_emory21" / "ckpt_best.pt"



############################################
# CONFIG 
############################################

MODEL_TYPE = "xecg"   

USE_FILTER = False    # ECGFounder=True, xECG=False
USE_ZSCORE = False    # ECGFounder=True, xECG=False

SEGMENT_LENGTH = 1000   # MUST match training!


def load_xecg_safetensors_with_fix(model, weights_path):
    sd = load_file(weights_path)
    fixed = {}
    for k, v in sd.items():
        if k.startswith("head."):
            continue
        if "slstm_cell._recurrent_kernel_" in k and v.ndim == 3:
            fixed[k] = v.permute(0, 2, 1).contiguous()
        else:
            fixed[k] = v
    model.load_state_dict(fixed, strict=False)


def collate_fn(batch):
    segments_list = [item[0] for item in batch]
    rids = [item[1] for item in batch]
    return segments_list, rids


############################################
# Dataset
############################################

class BenchmarkDataset(Dataset):

    def __init__(self, benchmark_folder, max_records=None):

        self.records = []

        for root, _, files in os.walk(benchmark_folder):
            for file in files:
                if file.endswith(".hea"):
                    rec = os.path.join(root, file.replace(".hea", ""))
                    self.records.append(rec)

        self.records = sorted(self.records)

        if max_records is not None:
            self.records = self.records[:max_records]

        self.fs = 100
        self.seg_len = SEGMENT_LENGTH


    def z_score(self, x):

        return (x - np.mean(x)) / (np.std(x) + 1e-8)


    def split_segments(self, data):

        L = data.shape[1]
        seg_len = self.seg_len

        # pad once
        pad_len = (seg_len - (L % seg_len)) % seg_len
        if pad_len > 0:
            data = np.pad(data, ((0,0),(0,pad_len)))

        # reshape directly (NO LOOP)
        segments = data.reshape(12, -1, seg_len)   # [12, N, 1000]
        segments = np.transpose(segments, (1,0,2)) # [N, 12, 1000]

        return segments


    def __len__(self):

        return len(self.records)


    def __getitem__(self, idx):

        t0 = time.time()

        record_name = self.records[idx]

        # -------------------------
        # WFDB LOAD
        # -------------------------
        t = time.time()
        sig, meta = wfdb.rdsamp(record_name)
        fs_in = int(meta["fs"])
        if fs_in != 100:
            from math import gcd
            from scipy.signal import resample_poly

            g = gcd(fs_in, 100)
            up = 100 // g
            down = fs_in // g

            sig = resample_poly(sig, up, down, axis=0).astype(np.float32)
            sig = sig.astype(np.float32)
        t_wfdb = time.time() - t
        # -------------------------
        t = time.time()
        data = np.nan_to_num(sig.T.astype(np.float32), nan=0.0)
        t_numpy = time.time() - t

        # -------------------------
        # LEAD SELECTION
        # -------------------------
        t = time.time()
        sig_names = meta["sig_name"]
        target_leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
        lead_idx = [sig_names.index(l) for l in target_leads]
        data = data[lead_idx]
        t_leads = time.time() - t

        # -------------------------
        # SEGMENTATION
        # -------------------------
        t = time.time()
        segments = self.split_segments(data)
        t_segment = time.time() - t

        # -------------------------
        # Z-SCORE
        # -------------------------
        t = time.time()
        if USE_ZSCORE:
            segments = (segments - segments.mean(axis=(1,2), keepdims=True)) / \
                    (segments.std(axis=(1,2), keepdims=True) + 1e-8)
        t_z = time.time() - t

        # -------------------------
        # TORCH CONVERSION
        # -------------------------
        t = time.time()
        segments_proc = torch.from_numpy(segments).float()
        t_torch = time.time() - t

        record_id = os.path.basename(record_name)

        return segments_proc, record_id


############################################
# MAIN
############################################

def run_inference(folder, max_records=None):

    dataset = BenchmarkDataset(folder, max_records=max_records)
    print(f"Records in dataset: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ############################################
    # Load model
    ############################################

    with open(SCRIPT_DIR / "config.json", "r") as f:
        xecg_config = json.load(f)

    model = xECGClassification(
        config=xecg_config,
        num_classes=21,
        linear_probing=False,
        cls_type=xecg_config.get("cls_type", "avg"),
    )

    weights_path = SCRIPT_DIR / "checkpoint" / "xecg_pretrained.safetensors"

    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    model.to(device)
    model.eval()

    print("\n=== XECG MODEL MODULES ===")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(name, module)
    print("==========================\n")

    # ------------------------------------------------
    # Find final classifier layer automatically
    # ------------------------------------------------
    classifier_name = None
    classifier_layer = None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.out_features == 21:
            classifier_name = name
            classifier_layer = module

    if classifier_layer is None:
        raise RuntimeError("Could not find final Linear classifier with out_features=21.")

    print(f"[HOOK] Using classifier layer: {classifier_name} -> {classifier_layer}")

    captured = {}

    def save_classifier_input(module, inputs, output):
        captured["features"] = inputs[0].detach()

    hook_handle = classifier_layer.register_forward_hook(save_classifier_input)


    ############################################
    # Inference
    ############################################

    record_probs = []
    record_features = []
    record_ids = []

    with torch.no_grad(), torch.amp.autocast("cuda"):

        t_prev = time.time()

        for i, (segments, rid) in enumerate(tqdm(loader, total=len(dataset), desc=str(folder))):

            t_now = time.time()
            t_dataloader = t_now - t_prev

            # -------------------------
            # PREP
            # -------------------------
            t = time.time()
            segments = segments.squeeze(0)
            segments = segments.permute(0, 2, 1)
            segments = segments.to(device, non_blocking=True)
            t_prep = time.time() - t

            # -------------------------
            # MODEL
            # -------------------------
            captured.clear()

            t = time.time()
            logits = model(segments)
            t_model = time.time() - t

            if "features" not in captured:
                raise RuntimeError(
                    "Feature hook did not capture anything. "
                    "The selected classifier layer may not be used in model.forward()."
                )

            features = captured["features"]  # probably [num_segments, D]

            t = time.time()
            probs = torch.sigmoid(logits).mean(dim=0)
            t_post = time.time() - t
            avg_features = features.float().mean(dim=0)

            probs = probs.cpu().numpy()
            avg_features = avg_features.cpu().numpy()

            record_probs.append(probs)
            record_features.append(avg_features)
            record_ids.append(rid[0])

            if PROFILE and i < 50:
                print(f"[LOOP] dataloader={t_dataloader:.4f}s | prep={t_prep:.4f}s | "
                    f"model={t_model:.4f}s | post={t_post:.4f}s")

            t_prev = time.time()


    hook_handle.remove()

    record_probs = np.stack(record_probs)
    record_features = np.stack(record_features)

    print("probs shape:", record_probs.shape)
    print("features shape:", record_features.shape)

    return record_probs, record_features, record_ids


############################################
# RUN
############################################

def main():

    first_runs = [
        #BENCHMARK_ROOT / "physionet_dn" / "physionet_dn_Sev3",
        #BENCHMARK_ROOT / "physionet_em" / "physionet_em_Sev3",
        #BENCHMARK_ROOT / "physionet_gn" / "physionet_gn_Sev3",
        #BENCHMARK_ROOT / "physionet_in" / "physionet_in_Sev3",
        #BENCHMARK_ROOT / "physionet_ma" / "physionet_ma_Sev3",
    ]

    for folder in first_runs:

        print("\nRunning FIRST:", folder)

        probs, features, rids = run_inference(folder)

        artifact_name = folder.parent.name
        severity_name = folder.name

        out_dir = OUT_ROOT / artifact_name / severity_name
        os.makedirs(out_dir, exist_ok=True)

        np.save(out_dir / "probs21.npy", probs)
        np.save(out_dir / "features.npy", features)
        np.save(out_dir / "record_ids.npy", np.array(rids))

    ############################################
    # RUN ALL 
    ############################################

    for artifact in BENCHMARK_ROOT.iterdir():

        if not artifact.is_dir():
            continue

        for severity in artifact.iterdir():

            if not severity.is_dir():
                continue

            print(f"\nArtifact: {artifact.name} | Severity: {severity.name}")

            probs, features, rids = run_inference(severity)

            out_dir = OUT_ROOT / artifact.name / severity.name
            os.makedirs(out_dir, exist_ok=True)

            np.save(out_dir / "probs21.npy", probs)
            np.save(out_dir / "features.npy", features)
            np.save(out_dir / "record_ids.npy", np.array(rids))


if __name__ == "__main__":
    main()