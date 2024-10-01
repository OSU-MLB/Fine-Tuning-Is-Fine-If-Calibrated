import torch
import numpy as np

from collections import OrderedDict
from . import constant as C


def topk_accuracy(similarity, y, topk=C.EVALUATE_TOP_K):
    maxk = max(topk)
    pred = similarity.argsort(axis=1, descending=True)[:, :maxk]
    correct = pred == y[:, None]
    metric = {}
    for k in topk:
        correct_k = correct[:, :k]
        correct_k = correct_k.sum(axis=1, dtype=torch.float32)
        accuracy = correct_k.mean() * 100
        accuracy = accuracy.item()
        metric[f'Top-{k:2d} Accuracy'] = accuracy
    metric = OrderedDict(sorted(metric.items()))
    return metric
