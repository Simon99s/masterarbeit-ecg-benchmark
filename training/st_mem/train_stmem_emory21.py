import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler
import time
import yaml

from mapping_rules import build_groups_from_tasks
from scipy.signal import butter, filtfilt, resample_poly


DEBUG_MAX_BATCHES = None   # set None to disable
DEBUG_VALIDATE = False

EXPECTED_INPUT_LEN = 1000
EXPECTED_OUTPUT_LEN = 2500
EXPECTED_LEADS = 12
CROP_LEN = 2250

def get_filter(fs=250, low=0.67, high=40.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a

B_BANDPASS, A_BANDPASS = get_filter()




# ==========================================================
# CONFIG
# ==========================================================

WEIGHTS_CLASSES_21 = [
    "164889003","164890007","733534002|164909002","713427006|59118001",
    "270492004","713426002","39732003","445118002","47665007",
    "251146004","111975006","698252002","426783006","284470004|63593006",
    "10370003","427172004|17338001","427393009","426177001",
    "427084000","164934002","59931005",
]


# ==========================================================
# DATASET
# ==========================================================

class EmoryH5_21(Dataset):
    def __init__(self, h5_path, groups, class_order21, mode="train"):
        self.h5_path = h5_path
        self.groups = groups
        self.class_order21 = list(class_order21)

        self.map_matrix = np.zeros((21, 150), dtype=np.float32)
        for j, cls in enumerate(self.class_order21):
            for idx in self.groups.get(cls, []):
                self.map_matrix[j, idx] = 1.0

        self._f = None
        self.X = None
        self.Y = None

        with h5py.File(self.h5_path, "r") as f:
            self.n = f["data"].shape[0]

        self.mask21 = torch.tensor(
            [1.0 if len(self.groups.get(c, [])) > 0 else 0.0 for c in self.class_order21],
            dtype=torch.float32,
        )
        self.mode = mode

    def _init(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r", swmr=True)
            self.X = self._f["data"]
            self.Y = self._f["label"]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._init()

        x = np.asarray(self.X[idx], dtype=np.float32)  # expected: (12, 1000)

        if x.shape[0] != EXPECTED_LEADS:
            raise ValueError(f"Wrong number of leads for index {idx}: {x.shape}")

        if x.shape[1] != EXPECTED_INPUT_LEN:
            raise ValueError(f"Expected raw HEEDB length {EXPECTED_INPUT_LEN}, got {x.shape[1]} for index {idx}")

        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 100 Hz -> 250 Hz: 1000 samples -> 2500 samples
        x = resample_poly(x, up=5, down=2, axis=1)

        # ST-MEM preprocessing: bandpass 0.67-40 Hz
        x = filtfilt(B_BANDPASS, A_BANDPASS, x, axis=-1)

        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = x[:, :CROP_LEN]

        x = torch.from_numpy(x.astype(np.float32))

        mean = x.mean()
        std = x.std().clamp_min(1e-8)
        x = (x - mean) / std

        # return as (2250, 12) because of training loop
        x = x.transpose(0, 1)

        y150 = np.array(self.Y[idx], dtype=np.int32)
        y21 = (self.map_matrix @ y150 > 0).astype(np.float32)

        return x, torch.from_numpy(y21), self.mask21
    



# ==========================================================
# UTILS
# ==========================================================

def split_indices(n, val_frac, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_frac))
    return idx[n_val:], idx[:n_val]


def masked_bce_with_logits(logits, targets, mask, pos_weight):
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    loss = loss * mask.unsqueeze(0)
    return loss.sum() / (targets.size(0) * mask.sum().clamp_min(1.0))


def estimate_pos_weight(dataset, indices):
    ys = []
    for i in indices[:50000]:  # sample subset
        _, y, _ = dataset[int(i)]
        ys.append(y.numpy())

    Y = np.stack(ys)
    pos = Y.sum(0)
    neg = Y.shape[0] - pos
    pw = neg / np.clip(pos, 1.0, None)

    return torch.tensor(pw, dtype=torch.float32)

def estimate_pos_weight_fast(h5_path, map_matrix):
    import h5py
    import numpy as np
    import torch

    print("Loading all labels into memory...")

    with h5py.File(h5_path, "r") as f:
        Y150 = f["label"][:]   # [N,150]

    print("Mapping to 21 classes...")

    Y21 = (map_matrix @ Y150.T > 0).T.astype(np.float32)

    print("Computing pos_weight...")

    pos = Y21.sum(axis=0)
    neg = Y21.shape[0] - pos

    pw = neg / np.clip(pos, 1.0, None)

    return torch.tensor(pw, dtype=torch.float32)


# ==========================================================
# MODEL
# ==========================================================

def build_stmem_model(encoder_path, config_model, device):
    import models.encoder as encoder

    model = encoder.__dict__[config_model["model_name"]](**config_model["model"])

    ckpt = torch.load(encoder_path, map_location="cpu")
    ckpt_model = ckpt["model"]

    state_dict = model.state_dict()

    # remove head mismatch
    for k in ["head.weight", "head.bias"]:
        if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
            del ckpt_model[k]

    msg = model.load_state_dict(ckpt_model, strict=False)
    print(msg)

    return model.to(device)


# ==========================================================
# TRAIN 
# ==========================================================

@torch.no_grad()
def eval_val(model, loader, device, pos_weight, patch_size):
    model.eval()
    total, n = 0.0, 0

    num_batches = len(loader)
    print(f"\n[Val] Starting validation ({num_batches} batches)")

    start_time = time.time()
    batch_timer = time.time()

    for i, (x, y, mask) in enumerate(loader):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x = x.permute(0, 2, 1)

        logits = model(x)
        loss = masked_bce_with_logits(logits, y, mask, pos_weight)

        bs = x.size(0)
        total += loss.item() * bs
        n += bs

        # PRINT EVERY 100 BATCHES
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - batch_timer
            time_per_batch = elapsed / 100

            print(f"[Val] {i}/{num_batches} | "
                  f"loss {loss.item():.4f} | {time_per_batch:.4f}s/batch")

            batch_timer = time.time()

    total_time = time.time() - start_time
    print(f"[Val] Done in {total_time:.1f}s")

    return total / max(n, 1)


# ==========================================================
# MAIN
# ==========================================================

def main(cfg):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # mapping
    tasks = Path(cfg["paths"]["tasks_txt"]).read_text().splitlines()
    groups = build_groups_from_tasks(tasks, WEIGHTS_CLASSES_21).groups

    ds = EmoryH5_21(cfg["paths"]["h5_path"], groups, WEIGHTS_CLASSES_21)
    train_idx, val_idx = split_indices(len(ds), cfg["data"]["val_frac"], 0)

    train_ds = Subset(EmoryH5_21(cfg["paths"]["h5_path"], groups, WEIGHTS_CLASSES_21, mode="train"), train_idx.tolist())
    val_ds   = Subset(EmoryH5_21(cfg["paths"]["h5_path"], groups, WEIGHTS_CLASSES_21, mode="eval"), val_idx.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=12,              
        pin_memory=True,            
        persistent_workers=True,    
        prefetch_factor=4           
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=12,
        num_workers=0,
        pin_memory=True,
        #persistent_workers=True
    )

    print("Dataset:", len(ds), "Train:", len(train_ds), "Val:", len(val_ds))

    # pos weight
    pw_path = out_dir / "pos_weight.pt"

    if pw_path.exists():
        print("Loading pos_weight from disk...")
        pos_weight = torch.load(pw_path).float().to(device)
    else:
        print("Computing pos_weight...")
        pos_weight = estimate_pos_weight_fast(
            cfg["paths"]["h5_path"],
            ds.map_matrix
        ).float().to(device)
        torch.save(pos_weight.cpu(), pw_path)
        print("Saved pos_weight to disk.")

    # model
    model = build_stmem_model(
        cfg["paths"]["stmem_checkpoint"],
        cfg["stmem"],
        device
    )

    head_params = []
    enc_params = []

    for n, p in model.named_parameters():
        if "head" in n:
            head_params.append(p)
        else:
            enc_params.append(p)

    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": 2e-5},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-2)


    ckpt_path = out_dir / "stmem_last.pt"

    start_epoch = 0

    best_val = float("inf")

    early_stopping = bool(cfg["train"].get("early_stopping", True))
    patience = int(cfg["train"].get("patience", 3))
    min_delta = float(cfg["train"].get("min_delta", 0.0))
    bad_epochs = 0

    if ckpt_path.exists():
        print("Loading checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])

        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        bad_epochs = ckpt.get("bad_epochs", 0)

        print(f"Resuming from epoch {start_epoch}")

    train_losses_path = out_dir / "train_losses.npy"
    val_losses_path = out_dir / "val_losses.npy"

    if train_losses_path.exists() and val_losses_path.exists():
        train_losses = list(np.load(train_losses_path))
        val_losses = list(np.load(val_losses_path))
    else:
        train_losses = []
        val_losses = []

    scaler = GradScaler("cuda")

    for ep in range(start_epoch, cfg["train"]["epochs"]):
        num_train_batches = len(train_loader)
        print(f"\nEpoch {ep} → {num_train_batches} training batches")
        model.train()
        running = 0.0
        seen = 0

        batch_timer = time.time()

        for b, (x, y, mask) in enumerate(train_loader):

            if DEBUG_MAX_BATCHES is not None and b >= DEBUG_MAX_BATCHES:
                print(f"Stopping early after {DEBUG_MAX_BATCHES} batches (debug mode)")
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                x = x.permute(0,2,1)  # [B,12,L]

                logits = model(x)
                if ep == 0 and b == 0:
                    print("logits mean:", logits.mean().item())
                    print("logits std :", logits.std().item())
                loss = masked_bce_with_logits(logits, y, mask, pos_weight)
                if b % 50 == 0 and b > 0:
                    elapsed = time.time() - batch_timer
                    time_per_batch = elapsed / 50

                    print(f"[Train] {b}/{num_train_batches} | "
                        f"loss {loss.item():.4f} | {time_per_batch:.4f}s/batch")

                    batch_timer = time.time()

            scaler.scale(loss).backward()
            if ep == 0 and b == 0:
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        print("grad check:", n, p.grad.abs().mean().item())
                        break
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            running += loss.item() * x.size(0)
            seen += x.size(0)

        if DEBUG_VALIDATE:
            print("Running quick validation...")
            val_loss = eval_val(
                model,
                val_loader,
                device,
                pos_weight,
                cfg["stmem"]["model"]["patch_size"]
            )

            print(f"[DEBUG] Epoch {ep} (partial) | train {running/seen:.4f} | val {val_loss:.4f}")
            break  

        train_loss = running / seen
        val_loss = eval_val(model, val_loader, device, pos_weight,
                    cfg["stmem"]["model"]["patch_size"])

        print(f"Epoch {ep} | train {train_loss:.4f} | val {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = (best_val - val_loss) > min_delta

        if improved:
            best_val = val_loss
            bad_epochs = 0

            torch.save({
                "model_state": model.state_dict(),
                "best_val": best_val,
            }, out_dir / "stmem_best.pt")

            print("  ✓ saved stmem_best.pt")
        else:
            bad_epochs += 1
            print(f"  no improvement (bad_epochs={bad_epochs}/{patience})")

        # always save last checkpoint
        torch.save({
            "epoch": ep,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "best_val": best_val,
            "bad_epochs": bad_epochs,
        }, out_dir / "stmem_last.pt")

        if early_stopping and bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {ep} with best_val={best_val:.6f}.")
            break
    
    np.save(train_losses_path, np.array(train_losses))
    np.save(val_losses_path, np.array(val_losses))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    PROJECT_ROOT = config_path.parents[2]

    def resolve_project_path(path_str):
        path = Path(path_str)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    for key in [
        "h5_path",
        "tasks_txt",
        "stmem_checkpoint",
        "out_dir",
    ]:
        cfg["paths"][key] = str(resolve_project_path(cfg["paths"][key]))

    main(cfg)