import numpy as np

def compute_top1_error(probs, gt):
    """
    probs: (N, C)
    gt:    (N, C) binary multi-label
    """

    pred_top1 = np.argmax(probs, axis=1)

    correct = 0
    for i in range(len(pred_top1)):
        if gt[i, pred_top1[i]] == 1:
            correct += 1

    acc = correct / len(pred_top1)

    return 1.0 - acc