import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, get_worker_info
from safetensors.torch import load_file
from torch.amp import autocast, GradScaler
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
from torch.utils.checkpoint import checkpoint_sequential


# ---- YAML loader ----
try:
    import yaml
except ImportError:
    raise ImportError("Please `pip install pyyaml` in your ECGFounder env.")

from mapping_rules import build_groups_from_tasks  


# -----------------------------
# PhysioNet 21 class order 
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


class EmoryH5_21(Dataset):
    """
    Returns:
      x: FloatTensor [12,1000]
      y21: FloatTensor [21]
      mask21: FloatTensor [21]  (1 if class has mapping indices else 0)
    """
    def __init__(self, h5_path: str, groups: Dict[str, List[int]], class_order21: Sequence[str]):
        self.h5_path = h5_path
        self.groups = groups
        self.class_order21 = list(class_order21)

        # build fast 150->21 mapping matrix
        self.map_matrix = np.zeros((len(self.class_order21), 150), dtype=np.float32)

        for j, cls in enumerate(self.class_order21):
            for idx in self.groups.get(cls, []):
                self.map_matrix[j, idx] = 1.0

        self._f = None
        self.X = None
        self.Y = None

        # Only read metadata here
        with h5py.File(self.h5_path, "r") as f:
            self.n = f["data"].shape[0]

        self.mask21 = torch.tensor(
            [1.0 if len(self.groups.get(c, [])) > 0 else 0.0 for c in self.class_order21],
            dtype=torch.float32,
        )

    def _init(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r", swmr=True, libver="latest")
            self.X = self._f["data"]
            self.Y = self._f["label"]

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        self._init()
        x = torch.from_numpy(self.X[idx]).float().transpose(0,1)
        y150 = np.array(self.Y[idx], dtype=np.int32)                   # [150]

        y21 = (self.map_matrix @ y150 > 0).astype(np.float32)

        return x, torch.from_numpy(y21), self.mask21
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_f"] = None
        state["X"] = None
        state["Y"] = None
        return state


def split_indices(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def estimate_pos_weight_21(dataset: Dataset, indices: np.ndarray, sample_n: int, seed: int) -> torch.Tensor:

    rng = np.random.default_rng(seed)
    if sample_n <= 0:
        sample_n = len(indices)
    take = rng.choice(indices, size=min(sample_n, len(indices)), replace=False)

    ys = []
    mask = None
    for i in take:
        _, y, m = dataset[int(i)]
        ys.append(y.numpy())
        mask = m  
    Y = np.stack(ys, axis=0)  
    pos = Y.sum(axis=0)
    neg = Y.shape[0] - pos
    pw = neg / np.clip(pos, 1.0, None)

    # for unmapped classes, set pos_weight=1
    if mask is not None:
        pw = np.where(mask.numpy() > 0.5, pw, 1.0)

    return torch.tensor(pw, dtype=torch.float32)


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor | None):

    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )  
    loss = loss * mask.unsqueeze(0)  
    denom = mask.sum().clamp_min(1.0)  
    return loss.sum() / (targets.size(0) * denom)


def build_xecg_model(xecg_config: dict, num_classes: int):
    """
    Adjust these imports to match your repo.
    This assumes you have a downstream model wrapper that adds a head.
    """
    try:
        # common in xECG repos
        from xecg.downstream_models import xECGClassification
    except Exception:
        try:
            from downstream_models import xECGClassification  
        except Exception as e:
            raise ImportError(
                "Could not import xECGClassification. "
                "Adjust the import in build_xecg_model() to your xECG repo structure."
            ) from e

    model = xECGClassification(
        config=xecg_config,
        num_classes=num_classes,
        linear_probing=False,   # finetune everything 
        cls_type=xecg_config.get("cls_type", "avg"),
    )
    return model

def load_xecg_safetensors_with_fix(model: nn.Module, weights_path: str):
    sd = load_file(weights_path)
    fixed = {}
    for k, v in sd.items():
        if k.startswith("head."):
            continue  
        if "slstm_cell._recurrent_kernel_" in k and v.ndim == 3:
            fixed[k] = v.permute(0, 2, 1).contiguous()
        else:
            fixed[k] = v
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    return missing, unexpected


def make_optimizer(model: nn.Module, lr_encoder: float, lr_head: float, wd: float):
    head_params = []
    enc_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in n:
            head_params.append(p)
        else:
            enc_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": enc_params, "lr": lr_encoder},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=wd,
    )


@torch.no_grad()
def eval_val_loss(model, loader, device, pos_weight):
    model.eval()

    total = 0.0
    n = 0

    start = time.perf_counter()

    for i, (x, y, mask) in enumerate(loader):

        if i % 50 == 0:
            batch_start = time.perf_counter()
            elapsed = time.perf_counter() - start
            print(f"val batch {i}/{len(loader)} | elapsed={elapsed:.1f}s")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask1 = mask[0].to(device)

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits = model(x)
            loss = masked_bce_with_logits(logits, y, mask1, pos_weight=pos_weight)

        if i % 50 == 0:
            batch_time = time.perf_counter() - batch_start
            print(f"   batch_time={batch_time:.3f}s")

        bs = x.size(0)
        total += float(loss.item()) * bs
        n += bs

    total_time = time.perf_counter() - start

    print(f"Validation finished in {total_time:.1f}s "
          f"({total_time/len(loader):.3f}s per batch)")

    return total / max(n, 1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config_train_xecg_emory21.yaml")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    PROJECT_ROOT = config_path.parents[2]

    # Repro
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    def resolve_project_path(path_str):
        path = Path(path_str)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path


    h5_path = resolve_project_path(cfg["paths"]["h5_path"])
    tasks_path = resolve_project_path(cfg["paths"]["tasks_txt"])
    out_dir = resolve_project_path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load xECG config yaml/dict
    xecg_cfg_path = resolve_project_path(cfg["paths"]["xecg_config_json"])
    xecg_config = json.loads(xecg_cfg_path.read_text(encoding="utf-8"))

    # Build mapping groups
    tasks150 = load_tasks150(tasks_path)
    mapping_res = build_groups_from_tasks(tasks150, WEIGHTS_CLASSES_21)
    groups = mapping_res.groups

    # Dataset + split
    full_ds = EmoryH5_21(h5_path, groups, WEIGHTS_CLASSES_21)
    train_idx, val_idx = split_indices(len(full_ds), float(cfg["data"]["val_frac"]), seed=int(cfg["data"]["split_seed"]))

    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds = Subset(full_ds, val_idx.tolist())

    print("Total samples:", len(full_ds))
    print("Train/Val:", len(train_ds), len(val_ds))
    print("Mapped classes (mask sum):", int(full_ds.mask21.sum().item()), "/ 21")

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
    model = build_xecg_model(xecg_config, num_classes=21)
    model = model.to(device)

    torch.set_float32_matmul_precision("high")


    weights_path = resolve_project_path(cfg["paths"]["xecg_safetensors"])
    t0 = time.perf_counter()
    missing, unexpected = load_xecg_safetensors_with_fix(model, weights_path)
    log_time("loading pretrained weights", t0)

    # We expect missing head weights because we changed num_classes
    print("Loaded pretrained safetensors.")
    print("Missing keys (expected head.*):", missing[:10], "...", len(missing))
    print("Unexpected keys:", unexpected[:10], "...", len(unexpected))

    # Dataloaders
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"].get("num_workers", 4))
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
    opt = make_optimizer(
        model,
        lr_encoder=float(cfg["train"]["lr_encoder"]),
        lr_head=float(cfg["train"]["lr_head"]),
        wd=float(cfg["train"]["weight_decay"]),
    )

    resume_last = out_dir / "ckpt_last.pt"
    resume_best = out_dir / "ckpt_best.pt"

    start_epoch = 1
    best_val = float("inf")

    if resume_last.exists():

        print("Resuming training from ckpt_last...")

        ckpt = torch.load(resume_last, map_location=device)

        model.load_state_dict(ckpt["model_state"])
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

    epochs = int(cfg["train"]["epochs"])

    early_stopping = bool(cfg["train"].get("early_stopping", True))
    patience = int(cfg["train"].get("patience", 2))
    min_delta = float(cfg["train"].get("min_delta", 0.0))
    bad_epochs = 0

    scaler = GradScaler("cuda")

    for ep in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running = 0.0
        seen = 0

        data_start = time.perf_counter()

        grad_accum = 2

        opt.zero_grad(set_to_none=True)


        for b, (x, y, mask) in enumerate(train_loader):

            data_time = time.perf_counter() - data_start
            compute_start = time.perf_counter()

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask[0].to(device)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                if b == 0:
                    print("Input shape to model:", x.shape)

                logits = model(x)
                loss = masked_bce_with_logits(logits, y, mask, pos_weight=pos_weight)

                # normalize loss for accumulation
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (b + 1) % grad_accum == 0:

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            compute_time = time.perf_counter() - compute_start
            data_start = time.perf_counter()

            if (b + 1) % 20 == 0:
                print(
                    f"batch {b+1}/{len(train_loader)} | "
                    f"loss={(loss.item()*grad_accum):.4f} | "
                    f"data={data_time:.3f}s | "
                    f"compute={compute_time:.3f}s"
                )

            bs_ = x.size(0)
            running += float(loss.item()) * grad_accum * bs_
            seen += bs_

        train_loss = running / max(seen, 1)
        t0 = time.perf_counter()
        val_loss = eval_val_loss(model, val_loader, device, pos_weight)
        #torch.cuda.empty_cache()
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
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "best_val": best_val,
            "xecg_config": xecg_config,
            "weights_classes_21": WEIGHTS_CLASSES_21,
            "mask21": full_ds.mask21.cpu().numpy(),
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