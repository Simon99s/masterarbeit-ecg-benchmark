import h5py, numpy as np, torch
from torch.utils.data import Dataset

class EmoryH5_21(Dataset):
    def __init__(self, h5_path, groups, class_order):
        self.h5_path = h5_path
        self.groups = groups
        self.class_order = class_order
        self._f = None
        with h5py.File(h5_path, "r") as f:
            self.n = f["data"].shape[0]

        # loss mask: only classes that have at least one mapped task index
        self.loss_mask = torch.tensor(
            [1.0 if len(groups.get(c, [])) > 0 else 0.0 for c in class_order],
            dtype=torch.float32
        )

    def _init(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")
            self.X = self._f["data"]
            self.Y = self._f["label"]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self._init()

        # H5: [12, 1000] -> xECG expects per-sample [T, 12]
        x = np.array(self.X[i], dtype=np.float32)          # (12, 1000)
        x = torch.from_numpy(x).transpose(0, 1)            # (1000, 12)

        y150 = np.array(self.Y[i], dtype=np.int32)

        y21 = np.zeros(len(self.class_order), dtype=np.float32)
        for j, cls in enumerate(self.class_order):
            idxs = self.groups.get(cls, [])
            if idxs:
                y21[j] = 1.0 if np.any(y150[idxs] != 0) else 0.0

        return x, torch.from_numpy(y21), self.loss_mask