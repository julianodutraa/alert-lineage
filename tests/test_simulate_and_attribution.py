import numpy as np
import pytest

from alert_lineage import attribution as A
from alert_lineage import simulate as S
from alert_lineage import topology as T


@pytest.fixture(scope="module")
def stream():
    gt = S.ground_truth_params(11)
    return gt, S.simulate(gt, 10 * 1440, seed=5, storms_per_day=1.0)


def test_simulation_parents_precede_children(stream):
    _, s = stream
    has_parent = s.parents >= 0
    assert (s.times[s.parents[has_parent]] < s.times[has_parent]).all()
    assert np.all(np.diff(s.times) >= 0)


def test_storms_are_spaced_and_rooted(stream):
    _, s = stream
    t0s = [x["t0"] for x in s.storms]
    assert all(b - a >= 180.0 for a, b in zip(t0s, t0s[1:]))
    for sid, storm in enumerate(s.storms):
        members = np.nonzero(s.storm_id == sid)[0]
        first = members[0]
        assert s.parents[first] == -1
        assert T.SERVICES[s.marks[first]] == storm["root"]


def test_root_scores_conserve_mass(stream):
    gt, s = stream
    storm = s.storms[0]
    a, b = storm["t0"] - 10, storm["t0"] + 20
    scores = A.root_service_scores(s, gt, a, b)
    _, lo, hi = A.window_slice(s, a, b)
    assert scores.sum() == pytest.approx(hi - lo, rel=1e-9)


def _toy(events):
    times = np.array([t for t, _ in events], dtype=float)
    marks = np.array([T.INDEX[m] for _, m in events])
    n = len(events)
    return S.AlertStream(times=times, marks=marks, parents=np.full(n, -1),
                         storm_id=np.full(n, -1), horizon=200.0)


def test_baselines_on_handcrafted_storm():
    # noisy exporter fires first, then a postgres-rooted cascade
    s = _toy([(100.0, "metrics-exporter"), (103.0, "postgres-primary"), (103.5, "dag-scheduler"),
              (104.0, "api-serving"), (105.0, "spark-batch"), (106.0, "dashboard-svc")])
    assert A.first_to_fire_root(s, 95, 120) == "metrics-exporter"
    assert A.topology_root(s, 95, 120) == "postgres-primary"
    gt = S.ground_truth_params(7)
    assert A.hawkes_root(s, gt, 95, 120) == "postgres-primary"


def test_map_parent_prefers_upstream_over_noise():
    s = _toy([(100.0, "postgres-primary"), (100.5, "metrics-exporter"), (101.0, "api-serving")])
    gt = S.ground_truth_params(7)
    parents = A.map_parents(s.times, s.marks, gt)
    assert parents[2] == 0  # api-serving blamed on postgres, not on the closer exporter alert
    assert A.recent_parent(s.times, s.marks)[2] == 1  # the naive heuristic gets it wrong
    assert A.recent_parent(s.times, s.marks, topology_aware=True)[2] == 0
