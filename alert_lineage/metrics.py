"""Evaluation metrics: edge recovery and co-occurrence baseline."""

from __future__ import annotations

import numpy as np

from .simulate import AlertStream


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann Whitney U formulation with average ranks for ties."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(scores)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def offdiag(m: np.ndarray) -> np.ndarray:
    return m[~np.eye(m.shape[0], dtype=bool)]


def cooccurrence_lift(stream: AlertStream, dim: int, tau: float = 10.0) -> np.ndarray:
    """Lift of 'u fires, then v fires within tau minutes' over independence."""
    t, m = stream.times, stream.marks
    counts = np.zeros((dim, dim))
    lo = 0
    for i in range(t.shape[0]):
        while t[i] - t[lo] > tau:
            lo += 1
        for j in range(lo, i):
            counts[m[j], m[i]] += 1
    n = np.bincount(m, minlength=dim).astype(float)
    expected = np.outer(n, n) * tau / stream.horizon
    return counts / np.maximum(expected, 1e-12)


def direction_accuracy(score: np.ndarray, adj: np.ndarray) -> float:
    """Share of true edges u->v where score[u, v] > score[v, u]."""
    us, vs = np.nonzero(adj)
    return float(np.mean(score[us, vs] > score[vs, us]))
