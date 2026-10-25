"""Turning a fitted Hawkes model into triage answers.

Two questions are answered for an alert storm window:

1. Per alert: which earlier alert (or none) triggered it?  This is the MAP
   parent under the E step posterior.
2. Per window: which service started the cascade?  For every alert we compute
   the full distribution over its *ultimate* ancestor (the immigrant at the top
   of its branching tree) by the recursion

       R_i = p_i0 * e_i + sum_j p_ij * R_j

   processed in time order, then credit each service with the expected number
   of window alerts whose ultimate ancestor it emitted. This is a soft, exact
   (given the posterior) version of "follow the arrows back to the source",
   and it does not trust whichever alert happened to fire first.
"""

from __future__ import annotations

import numpy as np

from . import topology
from .hawkes import parent_posterior
from .simulate import AlertStream, HawkesParams


def map_parents(times: np.ndarray, marks: np.ndarray, params: HawkesParams) -> np.ndarray:
    """MAP parent per alert, -1 meaning 'background immigrant'."""
    p_imm, pairs, p_pair = parent_posterior(times, marks, params)
    n = times.shape[0]
    best_p = p_imm.copy()
    best_j = np.full(n, -1)
    order = np.argsort(p_pair, kind="stable")  # ascending so the max is written last
    for k in order:
        i = pairs.child[k]
        if p_pair[k] > best_p[i]:
            best_p[i] = p_pair[k]
            best_j[i] = pairs.parent[k]
    return best_j


def window_slice(stream: AlertStream, start: float, end: float, context: float = 60.0):
    lo = int(np.searchsorted(stream.times, start - context, side="left"))
    a = int(np.searchsorted(stream.times, start, side="left"))
    b = int(np.searchsorted(stream.times, end, side="right"))
    return lo, a, b


def root_service_scores(stream: AlertStream, params: HawkesParams, start: float, end: float,
                        context: float = 60.0) -> np.ndarray:
    """Expected number of window alerts descending from each service's immigrants."""
    lo, a, b = window_slice(stream, start, end, context)
    times = stream.times[lo:b]
    marks = stream.marks[lo:b]
    n = times.shape[0]
    d = params.dim
    if n == 0:
        return np.zeros(d)
    p_imm, pairs, p_pair = parent_posterior(times, marks, params)
    # R[i, s]: probability that alert i's ultimate ancestor was emitted by service s
    R = np.zeros((n, d))
    by_child: list[list[int]] = [[] for _ in range(n)]
    for k, i in enumerate(pairs.child):
        by_child[i].append(k)
    for i in range(n):
        R[i, marks[i]] += p_imm[i]
        for k in by_child[i]:
            R[i] += p_pair[k] * R[pairs.parent[k]]
    return R[a - lo:].sum(axis=0)


def hawkes_root(stream: AlertStream, params: HawkesParams, start: float, end: float) -> str:
    scores = root_service_scores(stream, params, start, end)
    return topology.SERVICES[int(np.argmax(scores))]


# ---------------------------------------------------------------- baselines

def first_to_fire_root(stream: AlertStream, start: float, end: float) -> str:
    _, a, b = window_slice(stream, start, end)
    if b <= a:
        return topology.SERVICES[0]
    return topology.SERVICES[int(stream.marks[a])]


def _downstream_closure(service: str) -> set[str]:
    out, stack = set(), [service]
    while stack:
        u = stack.pop()
        for x, y in topology.EDGES:
            if x == u and y not in out:
                out.add(y)
                stack.append(y)
    return out


def topology_root(stream: AlertStream, start: float, end: float) -> str:
    """Strong SRE heuristic: among alerting services with no alerting upstream,
    pick the one whose downstream closure covers the most alerting services;
    break ties by earliest first alert."""
    _, a, b = window_slice(stream, start, end)
    if b <= a:
        return topology.SERVICES[0]
    marks = stream.marks[a:b]
    times = stream.times[a:b]
    alerting = {topology.SERVICES[m] for m in marks}
    first = {}
    for t, m in zip(times, marks):
        first.setdefault(topology.SERVICES[m], t)
    sources = [s for s in alerting if not (set(topology.upstream_of(s)) & alerting)]
    return max(sources, key=lambda s: (len(_downstream_closure(s) & alerting), -first[s]))


def recent_parent(times: np.ndarray, marks: np.ndarray, tau: float = 10.0,
                  topology_aware: bool = False) -> np.ndarray:
    """Heuristic parent: the most recent earlier alert within ``tau`` minutes
    (optionally restricted to the same service or a direct upstream)."""
    adj = topology.adjacency()
    n = times.shape[0]
    out = np.full(n, -1)
    for i in range(n):
        j = i - 1
        while j >= 0 and times[i] - times[j] <= tau:
            if times[j] < times[i] and (not topology_aware or marks[j] == marks[i]
                                        or adj[marks[j], marks[i]]):
                out[i] = j
                break
            j -= 1
    return out
