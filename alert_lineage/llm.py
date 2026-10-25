"""LLM triage on the same storms: prompt rendering, response parsing, scoring.

The question asked of the model is the one an on-call engineer faces: given
the dependency graph and the raw alert feed around a page, which service
started it?  Two conditions are supported:

* ``raw``:      topology plus alert feed only.
* ``assisted``: the same, plus the Hawkes root posterior (top 3 services with
  their expected descendant counts) as an extra, clearly labelled signal.

No API key is needed to reproduce the published numbers: the model answers
used in the README are cached in ``eval/llm_responses_*.json`` and the scorer
reads them. ``AnthropicClient`` is provided for anyone who wants to re-run the
experiment live against their own key.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

import numpy as np

from . import topology
from .attribution import root_service_scores, window_slice
from .simulate import AlertStream, HawkesParams

SYSTEM = (
    "You are a senior SRE triaging an alert storm on a data platform. "
    "Answer with the single service most likely to have started the cascade."
)


def render_storm(stream: AlertStream, start: float, end: float, storm_key: str) -> str:
    _, a, b = window_slice(stream, start, end)
    lines = [f"STORM {storm_key}: alerts in window (minutes relative to window start)"]
    for k, i in enumerate(range(a, b)):
        svc = topology.SERVICES[int(stream.marks[i])]
        lines.append(f"  a{k:02d}  t+{stream.times[i] - start:5.1f}  {svc:<17} {topology.ALERT_NAMES[svc]}")
    return "\n".join(lines)


def render_hawkes_hint(stream: AlertStream, params: HawkesParams, start: float, end: float) -> str:
    scores = root_service_scores(stream, params, start, end)
    top = np.argsort(-scores)[:3]
    parts = [f"{topology.SERVICES[i]} ({scores[i]:.1f})" for i in top]
    return ("  Hawkes model hint, expected number of window alerts descending from each "
            "service (top 3): " + ", ".join(parts))


def topology_block() -> str:
    edges = "\n".join(f"  {u} -> {v}" for u, v in topology.EDGES)
    return ("Dependency graph (u -> v means v depends on u, so a fault in u can make v alert):\n"
            + edges + "\n  metrics-exporter has no dependencies and no dependents.")


def build_batch_prompt(storm_texts: list[str]) -> str:
    return (
        topology_block()
        + "\n\nFor each storm below, name the root service. Background alerts unrelated to the "
        "storm can appear in any window, including before the real root fires.\n\n"
        + "\n\n".join(storm_texts)
        + '\n\nReply with JSON only: {"answers": [{"storm": "<key>", "root_service": "<service>"}, ...]}'
    )


def parse_answers(text: str) -> dict[str, str]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("no JSON object in model output")
    obj = json.loads(match.group(0))
    out = {}
    for item in obj["answers"]:
        svc = str(item["root_service"]).strip()
        out[str(item["storm"])] = svc if svc in topology.INDEX else "INVALID"
    return out


def score(answers: dict[str, str], truth: dict[str, str]) -> dict:
    hits = sum(answers.get(k) == v for k, v in truth.items())
    invalid = sum(answers.get(k) in (None, "INVALID") for k in truth)
    return {"hits": hits, "n": len(truth), "acc": round(hits / len(truth), 4), "invalid": invalid}


class AnthropicClient:
    """Minimal Messages API client (stdlib only). Optional, not used in tests."""

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]

    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        body = json.dumps({"model": self.model, "max_tokens": max_tokens, "system": SYSTEM,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body, method="POST",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return "".join(part.get("text", "") for part in data["content"])
