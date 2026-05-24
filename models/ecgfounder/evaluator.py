import numpy as np
import torch


def split_into_10s(signal, fs=500):
    seg_len = 10 * fs
    T = signal.shape[1]

    segments = []
    for start in range(0, T, seg_len):
        seg = signal[:, start:start+seg_len]

        if seg.shape[1] < seg_len:
            pad = np.zeros((12, seg_len - seg.shape[1]))
            seg = np.concatenate([seg, pad], axis=1)

        segments.append(seg)

    return np.stack(segments)  # (K, 12, 5000)


def zscore(x):
    # x shape: (K, 12, 5000)
    mean = np.mean(x, axis=2, keepdims=True)
    std = np.std(x, axis=2, keepdims=True) + 1e-8
    return (x - mean) / std


def evaluate_ecgfounder(model, dataset, device):

    record_probs = []
    record_gt = []
    record_ids = []

    with torch.no_grad():

        for corrupted_full, label, rid in dataset:

            # 1️⃣ split full record into 10s windows
            windows = split_into_10s(corrupted_full)

            # 2️⃣ normalization (same as original ECGFounder)
            windows = zscore(windows)

            windows = torch.FloatTensor(windows).to(device)

            # 3️⃣ forward pass
            logits = model(windows)
            probs = torch.sigmoid(logits).cpu().numpy()

            # 4️⃣ pool windows → record level
            record_prob = np.mean(probs, axis=0)

            record_probs.append(record_prob)
            record_gt.append(label)   # 🔥 THIS IS THE IMPORTANT LINE
            record_ids.append(rid)

    return (
        np.stack(record_probs),
        np.stack(record_gt),    # 🔥 added
        record_ids,
    )