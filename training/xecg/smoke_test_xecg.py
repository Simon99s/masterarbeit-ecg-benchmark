import json
import torch
from safetensors.torch import load_file

from downstream_models import xECGClassification

CFG_PATH = "config.json"
WEIGHTS_PATH = "model.safetensors"

cfg = json.load(open(CFG_PATH, "r"))

m = xECGClassification(
    config=cfg,
    num_classes=21,
    linear_probing=False,
    cls_type=cfg.get("cls_type", "avg")
)

sd = load_file(WEIGHTS_PATH)

# ---- FIX: permute recurrent kernel tensors to match current xlstm expected shape ----
fixed = {}
for k, v in sd.items():
    if "slstm_cell._recurrent_kernel_" in k and v.ndim == 3:
        # checkpoint: [heads, 256, 1024] -> model expects [heads, 1024, 256]
        fixed[k] = v.permute(0, 2, 1).contiguous()
    else:
        fixed[k] = v

missing, unexpected = m.load_state_dict(fixed, strict=False)
print("missing:", missing[:10], "…", len(missing))
print("unexpected:", unexpected[:10], "…", len(unexpected))

x = torch.randn(2, 1000, 12)  # [B,T,12]
y = m(x)
print("out:", y.shape)