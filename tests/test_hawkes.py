import numpy as np
import pytest

from alert_lineage import hawkes as H
from alert_lineage import metrics as M
from alert_lineage import simulate as S
from alert_lineage import topology as T

D = len(T.SERVICES)


@pytest.fixture(scope="module")
def world():
    gt = S.ground_truth_params(7)
    train = S.simulate(gt, 28 * 1440, seed=4242, storms_per_day=1.0)
    return gt, train


def test_build_pairs_matches_brute_force():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 100, size=200))
    p = H.build_pairs(t, window=5.0)
    got = set(zip(p.child.tolist(), p.parent.tolist()))
    want = {(i, j) for i in range(len(t)) for j in range(i) if 0 < t[i] - t[j] <= 5.0}
    assert got == want
    assert np.allclose(p.dt, t[p.child] - t[p.parent])


def test_em_log_likelihood_is_monotone(world):
    _, train = world
    _, hist = H.fit_em(train, D, beta=1 / 3, n_iter=60, tol=0.0)
    diffs = np.diff(hist)
    assert (diffs >= -1e-6 * np.abs(hist[1:])).all()


def test_profile_likelihood_picks_true_beta(world):
    gt, train = world
    params, profile = H.fit(train, D)
    assert params.beta == pytest.approx(gt.beta)
    assert max(profile, key=profile.get) == pytest.approx(gt.beta)


def test_recovers_edges_and_branching_ratios(world):
    gt, train = world
    params, _ = H.fit_em(train, D, beta=gt.beta)
    adj = T.adjacency()
    assert M.roc_auc(M.offdiag(params.alpha), M.offdiag(adj)) > 0.97
    assert M.direction_accuracy(params.alpha, adj) == 1.0
    assert np.abs(params.alpha[adj] - gt.alpha[adj]).mean() < 0.15


def test_mask_zeroes_forbidden_edges(world):
    gt, train = world
    mask = (T.adjacency() | np.eye(D, dtype=bool)).astype(float)
    params, _ = H.fit_em(train, D, beta=gt.beta, mask=mask)
    assert (params.alpha[mask == 0] == 0).all()


def test_loglik_higher_at_truth_than_at_null(world):
    gt, train = world
    null = S.HawkesParams(mu=np.bincount(train.marks, minlength=D) / train.horizon,
                          alpha=np.zeros((D, D)), beta=gt.beta)
    assert H.log_likelihood(train, gt) > H.log_likelihood(train, null)
