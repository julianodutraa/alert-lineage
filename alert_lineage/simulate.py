"""Synthetic alert streams from a multivariate Hawkes process.

Simulation uses the branching (cluster) representation of the Hawkes process:
background "immigrant" alerts arrive as independent Poisson processes, and
every alert of service ``u`` spawns ``Poisson(alpha[u, v])`` children on each
service ``v`` after an ``Exponential(beta)`` delay. Because the branching tree
is generated explicitly, every simulated alert carries its true parent, which
is what lets us score per-alert attribution and not just aggregate fit.

Incident storms are injected as immigrants whose first generation is boosted
by ``storm_boost`` (an outage fans out harder than a routine blip). The fitted
model never sees that boost; it has to cope with the misspecification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import topology


@dataclass
class HawkesParams:
    mu: np.ndarray  # (D,) background rate per minute
    alpha: np.ndarray  # (D, D) branching ratios, alpha[u, v]: u excites v
    beta: float  # exponential kernel decay, 1/minutes

    @property
    def dim(self) -> int:
        return int(self.mu.shape[0])


@dataclass
class AlertStream:
    times: np.ndarray  # (N,) minutes, sorted ascending
    marks: np.ndarray  # (N,) service index
    parents: np.ndarray  # (N,) index of true parent, -1 for immigrants
    storm_id: np.ndarray  # (N,) storm index, -1 for background
    horizon: float
    storms: list[dict] = field(default_factory=list)  # {"t0", "root"}

    def __len__(self) -> int:
        return int(self.times.shape[0])


def ground_truth_params(seed: int) -> HawkesParams:
    """Draw a ground-truth parameter set consistent with the reference DAG."""
    rng = np.random.default_rng(seed)
    d = len(topology.SERVICES)
    adj = topology.adjacency()
    alpha = np.zeros((d, d))
    alpha[adj] = rng.uniform(0.35, 0.70, size=int(adj.sum()))
    np.fill_diagonal(alpha, 0.10)
    noisy = topology.INDEX["metrics-exporter"]
    alpha[noisy, noisy] = 0.45
    mu = rng.uniform(0.001, 0.004, size=d)
    mu[noisy] = 0.020
    return HawkesParams(mu=mu, alpha=alpha, beta=1.0 / 3.0)


def simulate(
    params: HawkesParams,
    horizon: float,
    seed: int,
    storms_per_day: float = 1.0,
    storm_boost: float = 6.0,
    storm_roots: list[str] | None = None,
    min_gap: float = 180.0,
) -> AlertStream:
    rng = np.random.default_rng(seed)
    d = params.dim
    roots = storm_roots or topology.storm_capable_roots()

    t_list: list[float] = []
    m_list: list[int] = []
    p_list: list[int] = []
    s_list: list[int] = []
    boost_list: list[float] = []

    for k in range(d):
        n = rng.poisson(params.mu[k] * horizon)
        for t in rng.uniform(0.0, horizon, size=n):
            t_list.append(float(t)); m_list.append(k); p_list.append(-1)
            s_list.append(-1); boost_list.append(1.0)

    storms: list[dict] = []
    n_storms = rng.poisson(storms_per_day * horizon / 1440.0)
    # keep storms away from the horizon edges and from each other (min_gap
    # minutes) so every evaluation window contains exactly one injected storm
    starts: list[float] = []
    for t0 in rng.uniform(30.0, horizon - 90.0, size=n_storms):
        if all(abs(t0 - x) >= min_gap for x in starts):
            starts.append(float(t0))
    for sid, t0 in enumerate(sorted(starts)):
        root = roots[rng.integers(len(roots))]
        storms.append({"t0": float(t0), "root": root})
        t_list.append(float(t0)); m_list.append(topology.INDEX[root]); p_list.append(-1)
        s_list.append(sid); boost_list.append(storm_boost)

    queue = list(range(len(t_list)))
    while queue:
        i = queue.pop()
        u, t, boost = m_list[i], t_list[i], boost_list[i]
        for v in range(d):
            rate = params.alpha[u, v] * boost
            if rate <= 0:
                continue
            for _ in range(rng.poisson(rate)):
                tc = t + rng.exponential(1.0 / params.beta)
                if tc >= horizon:
                    continue
                t_list.append(tc); m_list.append(v); p_list.append(i)
                s_list.append(s_list[i]); boost_list.append(1.0)
                queue.append(len(t_list) - 1)

    order = np.argsort(np.asarray(t_list), kind="stable")
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    parents = np.asarray(p_list)[order]
    parents = np.where(parents >= 0, remap[np.maximum(parents, 0)], -1)
    return AlertStream(
        times=np.asarray(t_list)[order],
        marks=np.asarray(m_list)[order],
        parents=parents,
        storm_id=np.asarray(s_list)[order],
        horizon=float(horizon),
        storms=storms,
    )
