"""End to end demo: fit on a week of alerts, then explain one storm.

    python scripts/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alert_lineage import attribution as A  # noqa: E402
from alert_lineage import hawkes as H  # noqa: E402
from alert_lineage import simulate as S  # noqa: E402
from alert_lineage import topology as T  # noqa: E402


def main(storm_index: int = 5) -> None:
    gt = S.ground_truth_params(7)
    train = S.simulate(gt, 7 * 1440, seed=7107, storms_per_day=1.0)
    test = S.simulate(gt, 60 * 1440, seed=7001, storms_per_day=1.0)
    params, profile = H.fit(train, len(T.SERVICES))
    print(f"trained on {len(train)} alerts over 7 days, beta_hat = {params.beta:.3f}/min "
          f"(true {gt.beta:.3f})")

    storm = test.storms[storm_index]
    start, end = storm["t0"] - 10.0, storm["t0"] + 20.0
    lo, a, b = A.window_slice(test, start, end)
    times, marks = test.times[lo:b], test.marks[lo:b]
    parents = A.map_parents(times, marks, params)
    print(f"\nstorm window: {b - a} alerts")
    for i in range(a - lo, b - lo):
        p = parents[i]
        cause = "background" if p < 0 else f"<- {T.SERVICES[marks[p]]} @ t+{times[p] - start:4.1f}"
        print(f"  t+{times[i] - start:5.1f}  {T.SERVICES[marks[i]]:<17} {cause}")

    scores = A.root_service_scores(test, params, start, end)
    order = np.argsort(-scores)[:3]
    print("\nroot ranking (expected window alerts descending from each service):")
    for i in order:
        print(f"  {T.SERVICES[i]:<17} {scores[i]:5.2f}")
    print(f"\nfirst to fire : {A.first_to_fire_root(test, start, end)}")
    print(f"topology rule : {A.topology_root(test, start, end)}")
    print(f"hawkes        : {T.SERVICES[order[0]]}")
    print(f"ground truth  : {storm['root']}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
