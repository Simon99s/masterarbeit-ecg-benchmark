import os
import wfdb
import json
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from net1d import Net1D
from util import filter_bandpass


from pathlib import Path

# ------------------------------------------------
# ROOT PATHS
# ------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent          # .../inference/ECGFounder
INFERENCE_ROOT = SCRIPT_DIR.parent                    # .../inference
PROJECT_ROOT = INFERENCE_ROOT.parent                  

# benchmark ECGs
BENCHMARK_ROOT = PROJECT_ROOT / "data" / "Benchmark"

# inference outputs
OUT_ROOT = INFERENCE_ROOT / "ECGFounder"

# checkpoint
CHECKPOINT = SCRIPT_DIR / "checkpoint" / "12_lead_ECGFounder.pth"


############################################
# Dataset
############################################

class BenchmarkDataset(Dataset):

    def __init__(self, benchmark_folder):

        self.records = []

        for root, _, files in os.walk(benchmark_folder):
            for file in files:
                if file.endswith(".hea"):
                    rec = os.path.join(root, file.replace(".hea", ""))
                    self.records.append(rec)

        self.records = sorted(self.records)

        self.fs = 500
        self.seg_len = 10 * self.fs


    def z_score(self, x):

        return (x - np.mean(x)) / (np.std(x) + 1e-8)


    def split_segments(self, data):

        segments = []

        for start in range(0, data.shape[1], self.seg_len):

            seg = data[:, start:start+self.seg_len]

            if seg.shape[1] < self.seg_len:

                pad = np.zeros((12, self.seg_len - seg.shape[1]))
                seg = np.concatenate([seg, pad], axis=1)

            segments.append(seg)

        return np.stack(segments)


    def __len__(self):

        return len(self.records)


    def __getitem__(self, idx):

        record_name = self.records[idx]

        sig, meta = wfdb.rdsamp(record_name)

        data = np.nan_to_num(sig.T)

        sig_names = meta["sig_name"]

        target_leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

        lead_idx = [sig_names.index(l) for l in target_leads]
        data = data[lead_idx]

        fs = int(meta["fs"])

        if fs != self.fs:
            raise RuntimeError("Benchmark must already be 500Hz")

        data = filter_bandpass(data, self.fs)

        segments = self.split_segments(data)

        segments_proc = []

        for seg in segments:

            seg = self.z_score(seg)

            segments_proc.append(seg)

        segments_proc = torch.FloatTensor(np.stack(segments_proc))

        record_id = os.path.basename(record_name)

        return segments_proc, record_id


############################################
# MAIN
############################################

def run_inference(folder):

    dataset = BenchmarkDataset(folder)
    print(f"Records in dataset: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=1,   # GPU batch across records
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ############################################
    # Load model
    ############################################

    model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64,160,160,400,400,1024,1024],
        m_blocks_list=[2,2,2,3,3,4,4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=150
    )

    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.to(device)
    model.eval()


    ############################################
    # Inference
    ############################################

    record_probs = []
    record_ids = []

    with torch.no_grad():

        for segments, rid in tqdm(loader, total=len(dataset), desc=str(folder)):

            segments = segments.squeeze(0).to(device)

            logits = model(segments)

            probs = torch.sigmoid(logits).cpu().numpy()

            probs = np.mean(probs, axis=0)

            record_probs.append(probs)
            record_ids.append(rid[0])


    record_probs = np.stack(record_probs)

    return record_probs, record_ids


############################################
# RUN
############################################

def main():

    ############################################
    #RUN physionet_dn Sev1–Sev3 FIRST
    ############################################

    first_runs = [
        # BENCHMARK_ROOT / "physionet_in" / "physionet_in_Sev017",
        # BENCHMARK_ROOT / "physionet_in" / "physionet_in_Sev039",
        # BENCHMARK_ROOT / "physionet_in" / "physionet_in_Sev007",
    ]

    for folder in first_runs:

        print("\nRunning FIRST:", folder)

        probs, rids = run_inference(folder)

        out_dir = OUT_ROOT / "physionet_in" / folder.name
        os.makedirs(out_dir, exist_ok=True)

        np.save(out_dir / "probs150.npy", probs)
        np.save(out_dir / "record_ids.npy", np.array(rids))

    #exit()
    ############################################
    #RUN ALL OTHER ARTIFACTS / SEVERITIES
    ############################################

    for artifact in BENCHMARK_ROOT.iterdir():

        if not artifact.is_dir():
            continue

        for severity in artifact.iterdir():

            if not severity.is_dir():
                continue

            # skip physionet_dn because already done
            if artifact.name == "physionet_dn":
                continue

            print(f"\nArtifact: {artifact.name} | Severity: {severity.name}")

            probs, rids = run_inference(severity)

            out_dir = OUT_ROOT / artifact.name / severity.name
            os.makedirs(out_dir, exist_ok=True)

            np.save(out_dir / "probs150.npy", probs)
            np.save(out_dir / "record_ids.npy", np.array(rids))


if __name__ == "__main__":
    main()