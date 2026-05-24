import numpy as np
import pandas as pd


def load_tasks(tasks_txt):
    lines = open(tasks_txt, encoding="utf-8").read().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def load_weights_classes(weights_csv):
    dfw = pd.read_csv(weights_csv, header=0, index_col=0)
    return list(dfw.columns)


def max_pool_probs(probs, idxs):
    if len(idxs) == 0:
        return np.zeros((probs.shape[0],), dtype=np.float32)
    return probs[:, idxs].max(axis=1).astype(np.float32)


def map_probs150_to_probs21(probs150, weights_classes21, groups):
    N = probs150.shape[0]
    probs21 = np.zeros((N, len(weights_classes21)), dtype=np.float32)

    for j, cls in enumerate(weights_classes21):
        idxs = groups.get(cls, [])
        probs21[:, j] = max_pool_probs(probs150, idxs)

    return probs21