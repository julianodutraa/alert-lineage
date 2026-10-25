import numpy as np
import pytest

from alert_lineage import llm as L
from alert_lineage import metrics as M
from alert_lineage import simulate as S
from alert_lineage import topology as T


def test_roc_auc_known_values():
    assert M.roc_auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)
    assert M.roc_auc(np.array([1.0, 1.0]), np.array([0, 1])) == pytest.approx(0.5)
    assert M.roc_auc(np.array([3.0, 1.0, 2.0]), np.array([1, 0, 0])) == pytest.approx(1.0)


def test_direction_accuracy():
    adj = np.array([[False, True], [False, False]])
    assert M.direction_accuracy(np.array([[0, 2.0], [1.0, 0]]), adj) == 1.0
    assert M.direction_accuracy(np.array([[0, 1.0], [2.0, 0]]), adj) == 0.0


def test_parse_and_score_answers():
    text = 'noise {"answers": [{"storm": "S00", "root_service": "kafka-broker"},' \
           ' {"storm": "S01", "root_service": "not-a-service"}]} trailing'
    ans = L.parse_answers(text)
    assert ans == {"S00": "kafka-broker", "S01": "INVALID"}
    r = L.score(ans, {"S00": "kafka-broker", "S01": "object-store", "S02": "postgres-primary"})
    assert (r["hits"], r["n"], r["invalid"]) == (1, 3, 2)


def test_render_storm_lists_window_alerts():
    gt = S.ground_truth_params(7)
    s = S.simulate(gt, 2 * 1440, seed=3, storms_per_day=1.0)
    st = s.storms[0]
    text = L.render_storm(s, st["t0"] - 7.3, st["t0"] + 22.7, "S00")
    assert text.startswith("STORM S00")
    assert T.ALERT_NAMES[st["root"]] in text
    prompt = L.build_batch_prompt([text])
    assert "metrics-exporter has no dependencies" in prompt and '"answers"' in prompt
