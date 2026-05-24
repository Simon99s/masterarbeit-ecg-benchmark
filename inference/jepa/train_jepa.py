import os
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, get_worker_info
from torch.amp import autocast, GradScaler
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from models import load_encoder

import warnings
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")


# ---- YAML loader (no extra dependency required) ----
try:
    import yaml
except ImportError:
    raise ImportError("Please `pip install pyyaml` in your ECGFounder env.")

from mapping_rules import build_groups_from_tasks  # your mapping_rules.py


# -----------------------------
# PhysioNet 21 class order (weights.csv header order)
# -----------------------------
WEIGHTS_CLASSES_21 = [
    "164889003",
    "164890007",
    "733534002|164909002",
    "713427006|59118001",
    "270492004",
    "713426002",
    "39732003",
    "445118002",
    "47665007",
    "251146004",
    "111975006",
    "698252002",
    "426783006",
    "284470004|63593006",
    "10370003",
    "427172004|17338001",
    "427393009",
    "426177001",
    "427084000",
    "164934002",
    "59931005",
]

def log_time(name, start):
    dt = time.perf_counter() - start
    print(f"[TIMER] {name}: {dt:.2f}s")

def load_tasks150(tasks_path: str) -> List[str]:
    tasks = [ln.strip() for ln in Path(tasks_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(tasks) != 150:
        raise ValueError(f"Expected 150 lines in tasks.txt, got {len(tasks)}")
    return tasks

def build_jepa_model(encoder_path, num_classes, device):
    encoder, embed_dim = load_encoder(encoder_path)

    head = nn.Linear(embed_dim, num_classes)

    encoder = encoder.to(device)
    head = head.to(device)

    return encoder, head





from scipy.signal import resample_poly

class EmoryH5_JEPA(Dataset):
    def __init__(self, h5_path, groups, class_order21):
        self.h5_path = h5_path
        self.groups = groups
        self.class_order21 = list(class_order21)

        self.map_matrix = np.zeros((21, 150), dtype=np.float32)
        for j, cls in enumerate(self.class_order21):
            for idx in self.groups.get(cls, []):
                self.map_matrix[j, idx] = 1.0

        with h5py.File(self.h5_path, "r") as f:
            self.n = f["data"].shape[0]


        self._f = None

        self.mask21 = torch.tensor(
            [1.0 if len(self.groups.get(c, [])) > 0 else 0.0 for c in self.class_order21],
            dtype=torch.float32,
        )

    def _init(self):
        if self._f is None:
            worker = get_worker_info()
            if worker is None:
                self._f = h5py.File(self.h5_path, "r")
            else:
                self._f = h5py.File(self.h5_path, "r", libver="latest")
            self.X = self._f["data"]
            self.Y = self._f["label"]

    def __getitem__(self, idx):
        self._init()

        x = self.X[idx]  # (12,2500) ✅ already preprocessed

        # 🔥 select 8 leads (JEPA expects 8)
        x = x[[0,1,6,7,8,9,10,11]]  # (8,2500)

        x = torch.from_numpy(x).float()

        y150 = np.array(self.Y[idx], dtype=np.int32)
        y21 = (self.map_matrix @ y150 > 0).astype(np.float32)

        return x, torch.from_numpy(y21), self.mask21


    def __len__(self):
        return self.n


def split_indices(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def estimate_pos_weight_21(dataset: Dataset, indices: np.ndarray, sample_n: int, seed: int) -> torch.Tensor:
    """
    Estimate pos_weight on a subset of training indices for stability.
    pos_weight[j] = neg/pos
    """
    rng = np.random.default_rng(seed)
    if sample_n <= 0:
        sample_n = len(indices)
    take = rng.choice(indices, size=min(sample_n, len(indices)), replace=False)

    ys = []
    mask = None
    for i in take:
        _, y, m = dataset[int(i)]
        ys.append(y.numpy())
        mask = m  # same for all
    Y = np.stack(ys, axis=0)  # [S,21]
    pos = Y.sum(axis=0)
    neg = Y.shape[0] - pos
    pw = neg / np.clip(pos, 1.0, None)

    # for unmapped classes, set pos_weight=1 (won't be used if we mask loss correctly)
    if mask is not None:
        pw = np.where(mask.numpy() > 0.5, pw, 1.0)

    return torch.tensor(pw, dtype=torch.float32)


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor | None):
    """
    logits/targets: [B,21]
    mask: [21]
    """
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )  # [B,21]
    loss = loss * mask.unsqueeze(0)  # [B,21]
    denom = mask.sum().clamp_min(1.0)  # active classes
    return loss.sum() / (targets.size(0) * denom)


@torch.no_grad()
def eval_val_loss(encoder, head, loader, device, pos_weight):
    encoder.eval()
    head.eval()

    total = 0
    n = 0

    for i, (x, y, mask) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)
        mask = mask[0].to(device)

        t0 = time.perf_counter()

        with autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            features = encoder.representation(x)
            logits = head(features)
            loss = masked_bce_with_logits(logits, y, mask, pos_weight)

        step_time = time.perf_counter() - t0

        if (i + 1) % 20 == 0:
            print(
                f"[VAL] batch {i+1}/{len(loader)} | "
                f"loss={loss.item():.4f} | time={step_time:.3f}s"
            )

        total += loss.item() * x.size(0)
        n += x.size(0)

    return total / n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config_train_xecg_emory21.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # Repro
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    h5_path = cfg["paths"]["h5_path"]
    tasks_path = cfg["paths"]["tasks_txt"]
    out_dir = Path(cfg["paths"]["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build mapping groups
    tasks150 = load_tasks150(tasks_path)
    mapping_res = build_groups_from_tasks(tasks150, WEIGHTS_CLASSES_21)
    groups = mapping_res.groups

    # Dataset + split
    full_ds = EmoryH5_JEPA(h5_path, groups, WEIGHTS_CLASSES_21)
    train_idx, val_idx = split_indices(len(full_ds), float(cfg["data"]["val_frac"]), seed=int(cfg["data"]["split_seed"]))

    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds = Subset(full_ds, val_idx.tolist())

    print("Total samples:", len(full_ds))
    print("Train/Val:", len(train_ds), len(val_ds))
    print("Mapped classes (mask sum):", int(full_ds.mask21.sum().item()), "/ 21")

    # pos_weight (optional but recommended)
    t0 = time.perf_counter()
    print(">> starting pos_weight estimation ...")
    pos_weight = None
    pos_weight_path = out_dir / "pos_weight21.npy"

    if cfg["train"].get("use_pos_weight", True):

        if pos_weight_path.exists():
            print("Loading existing pos_weight21.npy")
            pw = np.load(pos_weight_path)
            pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)

        else:
            print("Computing pos_weight21 (first run only)")
            pw = estimate_pos_weight_21(
                full_ds,
                train_idx,
                sample_n=int(cfg["train"].get("pos_weight_sample_n", 100000)),
                seed=seed,
            )

            np.save(pos_weight_path, pw.cpu().numpy())
            pos_weight = pw.to(device)

            print("Saved pos_weight21.npy")
    log_time("pos_weight estimation", t0)


    t0 = time.perf_counter()
    weights_path = cfg["paths"]["jepa_checkpoint"]

    encoder, head = build_jepa_model(
        weights_path,
        num_classes=21,
        device=device
    )
    log_time("loading pretrained weights", t0)

    print("Loaded pretrained JEPA encoder.")

    # Dataloaders
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"].get("num_workers", 12))
    t0 = time.perf_counter()

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    log_time("dataloader creation", t0)

    # Optimizer
    opt = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": cfg["train"]["lr_encoder"]},
        {"params": head.parameters(), "lr": cfg["train"]["lr_head"]},
    ], weight_decay=cfg["train"]["weight_decay"])

    resume_last = out_dir / "ckpt_last.pt"
    resume_best = out_dir / "ckpt_best.pt"

    start_epoch = 1
    best_val = float("inf")

    if resume_last.exists():

        print("Resuming training from ckpt_last...")

        ckpt = torch.load(resume_last, map_location=device)

        encoder.load_state_dict(ckpt["encoder_state"])
        head.load_state_dict(ckpt["head_state"])
        opt.load_state_dict(ckpt["optimizer_state"])

        torch.cuda.empty_cache()
        
        start_epoch = ckpt["epoch"] + 1

        # default fallback
        best_val = ckpt.get("best_val", float("inf"))

    # overwrite best_val with real best checkpoint if available
    if resume_best.exists():

        best_ckpt = torch.load(resume_best, map_location=device)

        best_val = best_ckpt["best_val"]

    print("Starting epoch:", start_epoch)
    print("Best validation so far:", best_val)

    if torch.cuda.is_available():
        print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
        print("Reserved :", torch.cuda.memory_reserved() / 1e9, "GB")

    epochs = int(cfg["train"]["epochs"])

    early_stopping = bool(cfg["train"].get("early_stopping", True))
    patience = int(cfg["train"].get("patience", 2))
    min_delta = float(cfg["train"].get("min_delta", 0.0))
    bad_epochs = 0

    scaler = GradScaler("cuda")

    for ep in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        encoder.train()
        head.train()
        running = 0.0
        seen = 0

        data_start = time.perf_counter()

        grad_accum = 1

        opt.zero_grad(set_to_none=True)

        validate_after = int(cfg.get("debug", {}).get("validate_after_batches", -1))
        print_every = int(cfg.get("debug", {}).get("print_every", 20))

        val_triggered = False


        for b, (x, y, mask) in enumerate(train_loader):

            data_time = time.perf_counter() - data_start
            compute_start = time.perf_counter()

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask[0].to(device)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):

                features = encoder.representation(x)
                logits = head(features)
                loss = masked_bce_with_logits(logits, y, mask, pos_weight)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (b + 1) % grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            compute_time = time.perf_counter() - compute_start
            data_start = time.perf_counter()

            # 🔥 PROGRESS PRINT
            if (b + 1) % print_every == 0:
                print(
                    f"[TRAIN] batch {b+1}/{len(train_loader)} | "
                    f"loss={(loss.item()*grad_accum):.4f} | "
                    f"data={data_time:.3f}s | compute={compute_time:.3f}s"
                )

            # 🔥 EARLY VALIDATION TRIGGER
            if (validate_after > 0) and (b + 1 == validate_after) and not val_triggered:
                print(f"\n🚀 Running EARLY validation after {validate_after} batches...\n")

                encoder.eval()
                head.eval()

                t_val = time.perf_counter()
                val_loss = eval_val_loss(encoder, head, val_loader, device, pos_weight)
                val_time = time.perf_counter() - t_val

                print(f"[EARLY VAL] loss={val_loss:.6f} | time={val_time:.2f}s\n")

                encoder.train()
                head.train()

                val_triggered = True

            bs_ = x.size(0)
            running += float(loss.item()) * grad_accum * bs_
            seen += bs_

        # ✅ correct position (AFTER loop)
        if (b + 1) % grad_accum != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        train_loss = running / max(seen, 1)
        t0 = time.perf_counter()
        #torch.cuda.empty_cache()
        val_loss = eval_val_loss(encoder, head, val_loader, device, pos_weight)
        log_time("validation", t0)

        print(f"Epoch {ep:02d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        improved = (best_val - val_loss) > min_delta

        if improved:
            best_val = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        ckpt_last = {
            "epoch": ep,
            "encoder_state": encoder.state_dict(),
            "head_state": head.state_dict(),
            "optimizer_state": opt.state_dict(),
            "best_val": best_val,
        }

        torch.save(ckpt_last, out_dir / "ckpt_last.pt")

        if improved:
            torch.save(ckpt_last, out_dir / "ckpt_best.pt")
            print("  ✓ saved ckpt_best.pt")
        else:
            print(f"  no improvement (bad_epochs={bad_epochs}/{patience})")

        if early_stopping and bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {ep} (best_val={best_val:.6f}).")
            break

        log_time(f"epoch {ep}", epoch_start)

    print("Done. Best val loss:", best_val)
    print("Output dir:", str(out_dir))


if __name__ == "__main__":
    main()
