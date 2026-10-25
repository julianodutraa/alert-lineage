"""Multivariate Hawkes process with exponential kernel, fitted by EM.

Intensity of service ``v``:

    lambda_v(t) = mu_v + sum_{t_j < t} alpha[u_j, v] * beta * exp(-beta (t - t_j))

The kernel ``beta * exp(-beta s)`` integrates to one, so ``alpha[u, v]`` is
directly the expected number of ``v`` alerts directly triggered by one ``u``
alert (the branching ratio). EM treats the latent parent of every alert as the
missing data (Lewis and Mohler style): the E step computes the posterior that
alert ``i`` is an immigrant or a child of each earlier alert ``j``; the M step
has closed form updates. ``beta`` is chosen by profile likelihood on a grid.

Pairs are truncated at ``window_mult / beta``; with the default of 12 the
truncated kernel mass is ``exp(-12)``, about 6e-6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulate import AlertStream, HawkesParams


@dataclass
class Pairs:
    child: np.ndarray  # index i (later event)
    parent: np.ndarray  # index j (earlier event)
    dt: np.ndarray  # t_i - t_j > 0


def build_pairs(times: np.ndarray, window: float) -> Pairs:
    """All (i, j) with j < i and t_i - t_j <= window, fully vectorised."""
    n = times.shape[0]
    lo = np.searchsorted(times, times - window, side="left")
    counts = np.arange(n) - lo
    child = np.repeat(np.arange(n), counts)
    offsets = np.arange(child.size) - np.repeat(np.cumsum(counts) - counts, counts)
    parent = np.repeat(lo, counts) + offsets
    dt = times[child] - times[parent]
    keep = dt > 0
    return Pairs(child[keep], parent[keep], dt[keep])


def log_likelihood(stream: AlertStream, params: HawkesParams, pairs: Pairs | None = None,
                   window_mult: float = 12.0) -> float:
    beta = params.beta
    if pairs is None:
        pairs = build_pairs(stream.times, window_mult / beta)
    m = stream.marks
    lam = params.mu[m].copy()
    contrib = params.alpha[m[pairs.parent], m[pairs.child]] * beta * np.exp(-beta * pairs.dt)
    np.add.at(lam, pairs.child, contrib)
    comp = params.mu.sum() * stream.horizon
    tail = 1.0 - np.exp(-beta * (stream.horizon - stream.times))
    comp += float((params.alpha[m].sum(axis=1) * tail).sum())
    return float(np.log(np.maximum(lam, 1e-300)).sum() - comp)


def fit_em(stream: AlertStream, dim: int, beta: float, n_iter: int = 200, tol: float = 1e-6,
           window_mult: float = 12.0, mask: np.ndarray | None = None) -> tuple[HawkesParams, list[float]]:
    """EM for (mu, alpha) at fixed beta. ``mask`` optionally zeroes forbidden edges."""
    times, m, T = stream.times, stream.marks, stream.horizon
    pairs = build_pairs(times, window_mult / beta)
    n = times.shape[0]
    counts = np.bincount(m, minlength=dim).astype(float)
    mu = np.maximum(counts / T * 0.5, 1e-6)
    alpha = np.full((dim, dim), 0.1)
    if mask is not None:
        alpha = alpha * mask
    kern = beta * np.exp(-beta * pairs.dt)
    tail = 1.0 - np.exp(-beta * (T - times))
    exposure = np.bincount(m, weights=tail, minlength=dim)  # sum_j G(T - t_j) per source
    pu, pv = m[pairs.parent], m[pairs.child]
    history: list[float] = []
    prev = -np.inf
    for _ in range(n_iter):
        w_pair = alpha[pu, pv] * kern
        w_imm = mu[m]
        denom = w_imm.copy()
        np.add.at(denom, pairs.child, w_pair)
        p_imm = w_imm / denom
        p_pair = w_pair / denom[pairs.child]
        mu = np.maximum(np.bincount(m, weights=p_imm, minlength=dim) / T, 1e-9)
        num = np.zeros((dim, dim))
        np.add.at(num, (pu, pv), p_pair)
        alpha = num / np.maximum(exposure[:, None], 1e-12)
        if mask is not None:
            alpha = alpha * mask
        params = HawkesParams(mu=mu, alpha=alpha, beta=beta)
        ll = log_likelihood(stream, params, pairs)
        history.append(ll)
        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            break
        prev = ll
    return HawkesParams(mu=mu, alpha=alpha, beta=beta), history


def fit(stream: AlertStream, dim: int, betas: tuple[float, ...] = (0.1, 0.2, 1 / 3, 0.5, 1.0),
        **kw) -> tuple[HawkesParams, dict[float, float]]:
    """Profile likelihood over a beta grid; returns the best fit and the profile."""
    best, best_ll, profile = None, -np.inf, {}
    for b in betas:
        params, hist = fit_em(stream, dim, b, **kw)
        profile[float(b)] = hist[-1]
        if hist[-1] > best_ll:
            best, best_ll = params, hist[-1]
    assert best is not None
    return best, profile


def parent_posterior(times: np.ndarray, marks: np.ndarray, params: HawkesParams,
                     window_mult: float = 12.0) -> tuple[np.ndarray, Pairs, np.ndarray]:
    """Posterior over each alert's parent under fitted params.

    Returns ``(p_immigrant, pairs, p_pair)`` where ``p_pair[k]`` is the
    probability that ``pairs.child[k]`` was triggered by ``pairs.parent[k]``.
    """
    beta = params.beta
    pairs = build_pairs(times, window_mult / beta)
    w_pair = params.alpha[marks[pairs.parent], marks[pairs.child]] * beta * np.exp(-beta * pairs.dt)
    w_imm = params.mu[marks].copy()
    denom = w_imm.copy()
    np.add.at(denom, pairs.child, w_pair)
    return w_imm / denom, pairs, w_pair / denom[pairs.child]
