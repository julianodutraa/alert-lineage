"""LLM versus Hawkes on the same storms.

    python -m eval.llm_experiment export   # writes prompts and the hidden answer key
    python -m eval.llm_experiment score    # scores cached model answers against the key
    python -m eval.llm_experiment live     # optional: query the API (needs ANTHROPIC_API_KEY)

The storms come from world seed 7 of the main sweep; the Hawkes hint uses the
unmasked model fitted on 7 days of independent history.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from alert_lineage import attribution as A
from alert_lineage import hawkes as H
from alert_lineage import llm as L
from alert_lineage import simulate as S
from alert_lineage import topology as T

HERE = Path(__file__).parent
WORLD, N_STORMS, TRAIN_DAYS = 7, 24, 7
LOOKBACK, LOOKAHEAD = 10.0, 20.0


def setup():
    gt = S.ground_truth_params(WORLD)
    test = S.simulate(gt, 60 * 1440, seed=WORLD * 1000 + 1, storms_per_day=1.0)
    train = S.simulate(gt, TRAIN_DAYS * 1440, seed=WORLD * 1000 + 100 + TRAIN_DAYS, storms_per_day=1.0)
    params, _ = H.fit(train, len(T.SERVICES))
    storms = test.storms[:N_STORMS]
    return test, params, storms


def export() -> None:
    test, params, storms = setup()
    raw, assisted, truth, methods = [], [], {}, {"hawkes": {}, "topology": {}, "first_to_fire": {}}
    # The window offset is randomised per storm. With a fixed lookback the true
    # root would always sit at exactly t+10.0, a give-away a language model can
    # pick up on even though no algorithm here uses it.
    rng = np.random.default_rng(WORLD)
    for k, s in enumerate(storms):
        key = f"S{k:02d}"
        lookback = float(rng.uniform(4.0, 16.0))
        a, b = s["t0"] - lookback, s["t0"] + (LOOKBACK + LOOKAHEAD - lookback)
        text = L.render_storm(test, a, b, key)
        raw.append(text)
        assisted.append(text + "\n" + L.render_hawkes_hint(test, params, a, b))
        truth[key] = s["root"]
        methods["hawkes"][key] = A.hawkes_root(test, params, a, b)
        methods["topology"][key] = A.topology_root(test, a, b)
        methods["first_to_fire"][key] = A.first_to_fire_root(test, a, b)
    (HERE / "llm_prompt_raw.txt").write_text(L.build_batch_prompt(raw))
    (HERE / "llm_prompt_assisted.txt").write_text(L.build_batch_prompt(assisted))
    (HERE / "llm_answer_key.json").write_text(json.dumps({"truth": truth, "methods": methods}, indent=2))
    print(f"exported {len(storms)} storms")


def score() -> dict:
    key = json.loads((HERE / "llm_answer_key.json").read_text())
    truth = key["truth"]
    out = {name: L.score(ans, truth) for name, ans in key["methods"].items()}
    for cond in ("raw", "assisted"):
        path = HERE / f"llm_responses_{cond}.json"
        if path.exists():
            answers = L.parse_answers(path.read_text())
            out[f"llm_{cond}"] = L.score(answers, truth)
            out[f"llm_{cond}"]["disagrees_with_hawkes"] = sum(
                answers.get(k) != key["methods"]["hawkes"][k] for k in truth)
    (HERE / "llm_results.json").write_text(json.dumps(out, indent=2))
    for name, r in out.items():
        print(f"  {name:<16} {r['hits']}/{r['n']}  acc {r['acc']:.3f}")
    return out


def live() -> None:
    client = L.AnthropicClient()
    for cond in ("raw", "assisted"):
        prompt = (HERE / f"llm_prompt_{cond}.txt").read_text()
        (HERE / f"llm_responses_{cond}.json").write_text(client.complete(prompt))
    score()


if __name__ == "__main__":
    {"export": export, "score": score, "live": live}[sys.argv[1] if len(sys.argv) > 1 else "score"]()
