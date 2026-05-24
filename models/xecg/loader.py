

def load_xecg(device):

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from .xECG import xECG

    # -----------------------------
    # download weights + config
    # -----------------------------
    repo = "riccardolunelli/xECG_base_model_v1"

    model_file = hf_hub_download(repo, "model.safetensors")
    config_file = hf_hub_download(repo, "config.json")

    # -----------------------------
    # load config
    # -----------------------------
    import json
    with open(config_file, "r") as f:
        cfg = json.load(f)

    cls_type = cfg["cls_type"]
    config = dict(cfg)        # copy dictionary
    config.pop("cls_type")    # remove it from model config

    # -----------------------------
    # build model
    # -----------------------------
    model = xECG(cls_type, config)

    # -----------------------------
    # load weights manually
    # -----------------------------
    state_dict = load_file(model_file, device="cpu")

    fixed_state = {}

    for k, v in state_dict.items():

        # rename keys the same way HF does
        if k.startswith("model."):
            k = k[6:]

        k = k.replace("xlstm.model", "core.model")

        # ---- CRITICAL FIX ----
        if "slstm_cell._recurrent_kernel_" in k:
            v = v.permute(0, 2, 1)

        if "fc" not in k:
            fixed_state[k] = v

    model.load_state_dict(fixed_state, strict=False)

    model.to(device)
    model.eval()

    return model