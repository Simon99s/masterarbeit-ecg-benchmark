import numpy as np
import torch


def evaluate_xecg(model, dataset, device):

    probs_all = []
    gt_all = []
    record_ids = []

    probe = None

    print("EVALUATOR STARTED")   # ✅ confirms function entered

    with torch.no_grad():

        for i, (ecg, label, rid) in enumerate(dataset):

            if i == 0:
                print("FIRST SAMPLE LOADED FROM DATASET")

            if i % 50 == 0:
                print(f"processing record {i}")

            # ----------------------------
            # build input tensor
            # ----------------------------
            if i == 0:
                print("tensor conversion start")

            x = torch.tensor(ecg, dtype=torch.float32)

            if i == 0:
                print("tensor conversion done")

            # (12,T) → (B,12,T)
            x = x.unsqueeze(0).to(device)

            # xECG expects (B,T,12)
            x = x.permute(0, 2, 1)

            # ----------------------------
            # ensure patch compatibility
            # ----------------------------
            patch = model.patch_size
            T = x.shape[1]
            x = x[:, : (T // patch) * patch, :]

            # ----------------------------
            # MODEL FORWARD (biggest suspect)
            # ----------------------------
            if i == 0:
                print("MODEL FORWARD START")

            cls, _ = model(x)

            if i == 0:
                print("MODEL FORWARD DONE")

            # ----------------------------
            # create probe
            # ----------------------------
            if probe is None:

                print("creating probe head")

                num_classes = len(label)
                emb_dim = cls.shape[-1]

                probe = torch.nn.Linear(
                    emb_dim,
                    num_classes
                ).to(device)

                probe.eval()

            logits = probe(cls)
            probs = torch.sigmoid(logits)

            probs_all.append(probs.squeeze(0).cpu().numpy())
            gt_all.append(label)
            record_ids.append(rid)

    print("EVALUATION COMPLETE")

    return (
        np.stack(probs_all),
        np.stack(gt_all),
        record_ids
    )