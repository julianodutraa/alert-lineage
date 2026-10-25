"""Full evaluation sweep. Writes eval/results.json and prints a summary table.

For each of several ground-truth worlds (seed), we simulate an independent
training history of 1, 3, 7 or 28 days, fit the Hawkes model (with and without
a topology mask), and evaluate on a separate 60 day test stream with injected
storms. Baselines and an oracle (ground-truth parameters) use the same test
stream, so every method answers the exact same questions.

Usage:  python -m eval.run_eval
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from alert_lineage import attribution as A
from alert_lineage import hawkes as H
from alert_lineage import metrics as M
from alert_lineage import simulate as S
from alert_lineage import topology as T

WORLD_SEEDS = (7, 11, 23, 42, 99)
HORIZONS_DAYS = (1, 3, 7, 28)
TEST_DAYS = 60
LOOKBACK, LOOKAHEAD = 10.0, 20.0  # storm window, minutes around storm start


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 4), round(c + h, 4))


def storm_hits(test: S.AlertStream, root_fn) -> int:
    hits = 0
    for s in test.storms:
        if root_fn(s["t0"] - LOOKBACK, s["t0"] + LOOKAHEAD) == s["root"]:
            hits += 1
    return hits


def main() -> dict:
    d = len(T.SERVICES)
    adj = T.adjacency()
    mask = (adj | np.eye(d, dtype=bool)).astype(float)
    rows = []
    t_start = time.time()
    for ws in WORLD_SEEDS:
        gt = S.ground_truth_params(ws)
        test = S.simulate(gt, TEST_DAYS * 1440, seed=ws * 1000 + 1, storms_per_day=1.0)
        n_storms, n_events = len(test.storms), len(test)
        base = {
            "world": ws, "n_storms": n_storms, "n_test_alerts": n_events,
            "first_to_fire_hits": storm_hits(test, lambda a, b: A.first_to_fire_root(test, a, b)),
            "topology_hits": storm_hits(test, lambda a, b: A.topology_root(test, a, b)),
            "oracle_hits": storm_hits(test, lambda a, b: A.hawkes_root(test, gt, a, b)),
            "recent_parent_correct": int((A.recent_parent(test.times, test.marks) == test.parents).sum()),
            "recent_topo_parent_correct": int((A.recent_parent(test.times, test.marks, topology_aware=True)
                                               == test.parents).sum()),
            "oracle_parent_correct": int((A.map_parents(test.times, test.marks, gt) == test.parents).sum()),
        }
        for days in HORIZONS_DAYS:
            train = S.simulate(gt, days * 1440, seed=ws * 1000 + 100 + days, storms_per_day=1.0)
            co = M.cooccurrence_lift(train, d)
            for variant, vmask in (("hawkes", None), ("hawkes_masked", mask)):
                params, profile = H.fit(train, d, mask=vmask)
                row = dict(base)
                row.update({
                    "variant": variant, "train_days": days, "n_train_alerts": len(train),
                    "beta_hat": params.beta, "beta_profile": profile,
                    "storm_hits": storm_hits(test, lambda a, b: A.hawkes_root(test, params, a, b)),
                    "parent_correct": int((A.map_parents(test.times, test.marks, params)
                                           == test.parents).sum()),
                    "edge_auc": M.roc_auc(M.offdiag(params.alpha), M.offdiag(adj)),
                    "edge_auc_cooccurrence": M.roc_auc(M.offdiag(co), M.offdiag(adj)),
                    "direction_acc": M.direction_accuracy(params.alpha, adj),
                    "direction_acc_cooccurrence": M.direction_accuracy(co, adj),
                    "alpha_mae_on_edges": float(np.abs(params.alpha[adj] - gt.alpha[adj]).mean()),
                })
                rows.append(row)
    summary = summarise(rows)
    out = {"config": {"world_seeds": WORLD_SEEDS, "horizons_days": HORIZONS_DAYS,
                      "test_days": TEST_DAYS, "window": [-LOOKBACK, LOOKAHEAD]},
           "summary": summary, "rows": rows, "runtime_s": round(time.time() - t_start, 1)}
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=2))
    print_table(summary)
    print(f"runtime {out['runtime_s']}s")
    return out


def summarise(rows: list[dict]) -> dict:
    worlds = {r["world"]: r for r in rows}
    n_storms = sum(r["n_storms"] for r in worlds.values())
    n_alerts = sum(r["n_test_alerts"] for r in worlds.values())

    def pooled(key_hits, src):
        k = sum(r[key_hits] for r in src)
        return {"acc": round(k / n_storms, 4), "hits": k, "n": n_storms, "ci95": wilson(k, n_storms)}

    def pooled_parent(key, src):
        k = sum(r[key] for r in src)
        return {"acc": round(k / n_alerts, 4), "correct": k, "n": n_alerts}

    w = list(worlds.values())
    s = {"baselines": {
        "first_to_fire": pooled("first_to_fire_hits", w),
        "topology_heuristic": pooled("topology_hits", w),
        "oracle_true_params": pooled("oracle_hits", w),
        "parent_recent_alert": pooled_parent("recent_parent_correct", w),
        "parent_recent_topology_aware": pooled_parent("recent_topo_parent_correct", w),
        "parent_oracle": pooled_parent("oracle_parent_correct", w),
    }, "hawkes": {}}
    for variant in ("hawkes", "hawkes_masked"):
        for days in HORIZONS_DAYS:
            src = [r for r in rows if r["variant"] == variant and r["train_days"] == days]
            s["hawkes"][f"{variant}@{days}d"] = {
                "root": pooled("storm_hits", src),
                "parent": pooled_parent("parent_correct", src),
                "edge_auc_mean": round(float(np.mean([r["edge_auc"] for r in src])), 4),
                "edge_auc_cooccurrence_mean": round(float(np.mean([r["edge_auc_cooccurrence"] for r in src])), 4),
                "direction_acc_mean": round(float(np.mean([r["direction_acc"] for r in src])), 4),
                "alpha_mae_mean": round(float(np.mean([r["alpha_mae_on_edges"] for r in src])), 4),
                "train_alerts_mean": round(float(np.mean([r["n_train_alerts"] for r in src])), 1),
            }
    return s


def print_table(s: dict) -> None:
    b = s["baselines"]
    print(f"storms evaluated: {b['first_to_fire']['n']}")
    for k in ("first_to_fire", "topology_heuristic", "oracle_true_params"):
        print(f"  {k:<28} root acc {b[k]['acc']:.3f}  ci95 {b[k]['ci95']}")
    for k in ("parent_recent_alert", "parent_recent_topology_aware", "parent_oracle"):
        print(f"  {k:<28} parent acc {b[k]['acc']:.3f}")
    for name, r in s["hawkes"].items():
        print(f"  {name:<20} root {r['root']['acc']:.3f} {r['root']['ci95']}  parent {r['parent']['acc']:.3f}"
              f"  edgeAUC {r['edge_auc_mean']:.3f} (cooc {r['edge_auc_cooccurrence_mean']:.3f})"
              f"  alphaMAE {r['alpha_mae_mean']:.3f}")


if __name__ == "__main__":
    main()
