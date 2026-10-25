# alert-lineage

Infer which alerts triggered which during an incident storm, using a multivariate Hawkes process fitted by EM, and measure honestly how that compares with the heuristics on-call engineers actually use and with an LLM reading the same alert feed.

## Executive summary

When a shared dependency fails on a data platform, the pager does not ring once. A degraded database or object store produces a storm of alerts across every service that depends on it, and the first ten minutes of the incident are spent answering one question: which of these alerts is the cause and which are symptoms? Getting it wrong is expensive in a very specific way. Teams page the wrong owner, restart healthy services, and the time to mitigation grows while the real fault keeps spreading.

The common answer today is a rule of thumb: look at whichever alert fired first, or walk the dependency graph upstream by hand. Both break in predictable ways. Noisy alert sources fire constantly, so "first to fire" is often just background noise that happened to land a minute before the real fault. Manual graph walks are slow and depend on whoever is on call knowing the topology.

alert-lineage treats the alert history as what it statistically is, a self-exciting point process in which every alert raises the short term probability of alerts on dependent services. It learns from past alerts how strongly each service excites each other service, and then, for a live storm, computes for every alert the probability that it was caused by each earlier alert, and for every service the expected number of storm alerts that trace back to it.

On 227 synthetic incident storms across five simulated platforms, with noise deliberately included:

| Method | Correct root service | 95% CI |
|---|---|---|
| First alert to fire | 52.4% | 45.9 to 58.8 |
| Topology rule (most downstream reach, no alerting upstream) | 74.5% | 68.4 to 79.7 |
| alert-lineage, 1 day of history | 67.0% | 60.6 to 72.8 |
| alert-lineage, 1 day of history plus known topology | 78.4% | 72.6 to 83.3 |
| alert-lineage, 28 days of history | 85.0% | 79.8 to 89.1 |
| Same model with the true parameters (oracle) | 84.1% | 78.8 to 88.3 |

The business reading is straightforward. Statistical attribution beats the first-alert habit by about 30 points and beats a solid topology rule by 7 to 10 points once it has a week or more of history. It is not free: with a single day of history it is worse than the topology rule, and the fix for that cold start is to feed it the dependency graph you already have. A small blind experiment with an LLM on 24 of the same storms (below) shows the model and the statistics make different mistakes, and that the combination was the best single configuration measured, while the sample is too small to call that difference significant.

Everything runs offline on synthetic data in about ten seconds, with no GPU and no API key.

## How it works

### The model

Each service `v` has an alert intensity

```
lambda_v(t) = mu_v + sum over past alerts j of  alpha[u_j, v] * beta * exp(-beta * (t - t_j))
```

`mu_v` is the background rate (alerts that would happen anyway), `alpha[u, v]` is the expected number of `v` alerts directly triggered by one `u` alert, and `beta` is how fast that influence decays. Because the kernel `beta * exp(-beta s)` integrates to one, `alpha` is directly interpretable as a branching ratio, and the matrix of alphas is a learned, directed, weighted dependency graph.

### Fitting

The log likelihood of a Hawkes process is awkward to maximise directly, but it has a clean latent variable structure: every alert was either a background immigrant or the child of exactly one earlier alert. EM uses that structure.

* E step: for each alert `i` and each earlier alert `j` inside the kernel window, the posterior that `j` caused `i` is proportional to `alpha[u_j, u_i] * beta * exp(-beta (t_i - t_j))`, and the posterior that `i` is background is proportional to `mu_{u_i}`.
* M step: `mu_v` is the expected number of background `v` alerts divided by the horizon; `alpha[u, v]` is the expected number of `u -> v` parent links divided by the exposure of `u` alerts, `sum_j (1 - exp(-beta (T - t_j)))`.

Each iteration is vectorised over the list of candidate parent pairs, which is built once with `searchsorted`. Pairs are truncated at `12 / beta`, which drops a kernel mass of about `6e-6`. `beta` is chosen by profile likelihood over a small grid; it recovered the true value in 39 of 40 fits, the exception being one unmasked fit on a single day of history. An optional mask restricts `alpha` to known dependency edges plus self excitation, which is how you inject an existing service catalogue.

### From a fitted model to a triage answer

For a storm window, two answers are produced.

1. Per alert, the MAP parent: the earlier alert (or background) with the highest posterior.
2. Per service, a root score. For every alert we compute the full distribution over its ultimate ancestor with the recursion `R_i = p_i0 e_i + sum_j p_ij R_j`, processed in time order, and credit each service with the expected number of window alerts whose ultimate ancestor it emitted. The scores sum exactly to the number of alerts in the window (a test checks this). This is a soft version of "follow the arrows back to the source" that does not trust whichever alert happened to arrive first.

## Evaluation protocol

* A fixed 12 service reference topology of a generic data platform (object store, Kafka, Postgres, scheduler, Spark, feature store, warehouse loader, serving API, dashboards, and an isolated, flapping metrics exporter that exists to make noise).
* Five ground truth worlds (seeds 7, 11, 23, 42, 99). Each draws branching ratios in [0.35, 0.70] on the real edges, self excitation of 0.10 (0.45 for the flapping exporter), background rates of 0.001 to 0.004 alerts per minute (0.02 for the exporter), and a mean propagation delay of three minutes.
* Simulation uses the branching representation, so every simulated alert carries its true parent. Incident storms are injected about once a day at a random cascade capable service, with the first generation boosted six fold. The fitted model is never told about that boost, so it has to cope with a realistic misspecification.
* For each world, independent training histories of 1, 3, 7 and 28 days, and a separate 60 day test stream. Storms are at least three hours apart so each evaluation window, from 10 minutes before to 20 minutes after the storm start, contains exactly one injected storm plus whatever background noise happens to fall in it.
* Every method answers the exact same questions on the exact same test streams.

## Results

Full output is in [`eval/results.json`](eval/results.json). Pooled over 227 storms and 54,451 test alerts.

### Root service of the storm

| Training history | Hawkes | Hawkes with topology mask |
|---|---|---|
| 1 day | 67.0% | 78.4% |
| 3 days | 81.1% | 81.9% |
| 7 days | 81.5% | 83.7% |
| 28 days | 85.0% | 84.1% |

Baselines: first to fire 52.4%, topology rule 74.5%, oracle with true parameters 84.1%.

### Parent of every individual alert

| Method | Correct parent |
|---|---|
| Most recent earlier alert within 10 minutes | 48.4% |
| Most recent earlier alert from the same service or a direct upstream | 70.6% |
| Hawkes, 1 day | 69.8% |
| Hawkes, 1 day, masked | 73.3% |
| Hawkes, 28 days | 76.9% |
| Oracle | 77.2% |

The oracle number is the real ceiling here: even with perfect parameters, about 23% of parent links are not identifiable from timestamps, because two plausible parents fired within seconds of each other.

### Learned dependency graph

| Training history | Edge recovery AUC, Hawkes | Edge recovery AUC, lagged co-occurrence lift | Mean abs. error of alpha on true edges |
|---|---|---|---|
| 1 day | 0.923 | 0.903 | 0.259 |
| 3 days | 0.988 | 0.963 | 0.141 |
| 7 days | 1.000 | 0.978 | 0.116 |
| 28 days | 1.000 | 0.988 | 0.066 |

### What the numbers say, including the parts that are not flattering

* The cold start is real. With one day of history the unmasked model is worse than a zero data topology rule (67.0% versus 74.5%). Its branching ratios are off by 0.26 on average, which is most of their range. Giving it the dependency graph as a mask fixes most of that (78.4%), so in practice the service catalogue should be treated as a prior, not ignored.
* After a week, history beats the rule. From 3 days onward both variants sit above the topology rule, and the confidence intervals at 28 days no longer overlap with first to fire at all.
* Edge discovery is not where this approach earns its keep. A simple lagged co-occurrence lift already recovers the graph with AUC 0.90 to 0.99 and gets direction right, because "u then v within ten minutes" is already directional. The value of the Hawkes model is in the per alert and per storm attribution, where co-occurrence has no answer at all.
* The fitted model at 28 days scored one point above the oracle. That is two storms out of 227, well inside the noise, so it is not a result. A plausible reason it can happen at all is that the oracle uses the true unboosted parameters, while the fitted alphas absorb some of the storm boost from training data, which is exactly the regime the evaluation tests. Treat it as a reminder that the oracle is the ceiling for this model family, not for the problem.
* A known failure mode, visible in the LLM experiment below: the root score counts all descendants, including self excited repeats. A flapping source that re-fires seven times inside the window can outscore a genuine but small cascade. That was not tuned away after seeing the test set, on purpose.

## LLM experiment: can a model read the storm instead?

The same question was put to a Claude model on 24 storms from world 7, in two blind conditions. The prompt contains the dependency graph and the raw alert feed for each window (service, alert name, minutes from window start). The window offset is randomised per storm; with a fixed offset the true root always sat at exactly `t+10.0`, which a language model could have exploited even though no algorithm here does. In the assisted condition each storm also includes the Hawkes root scores for the top three services, labelled as a model hint. The answer key never left the evaluation harness. Prompts, cached answers and the key are in [`eval/`](eval/).

| Method | Correct root (of 24) |
|---|---|
| First to fire | 12 |
| Topology rule | 16 |
| Hawkes (7 days of history) | 19 |
| LLM, raw feed | 19 |
| LLM, raw feed plus Hawkes hint | 20 |

The interesting part is not the totals, which are within one storm of each other, but where they differ.

* Storm S00: a metrics exporter flaps seven times in the window, while the real cascade starts at the serving API and spreads to dashboards. Hawkes picked the exporter (the failure mode above). The LLM, in both conditions, discounted the flapping exporter as noise from its name and isolation, and was right.
* Storm S20: the LLM alone picked a downstream service; with the hint it switched to the correct upstream root.
* Four storms were missed by Hawkes, the topology rule and the LLM in both conditions. In two of them (S02, S14) a service that genuinely sits upstream of the real root alerted a few minutes before it, so from the feed alone the upstream explanation is the more likely one. In the other two (S16, S17) an unrelated background cascade overlapped a small storm and was the bigger of the two. These look like cases that need extra signal (change events, deploy logs) rather than a better reader of the same feed. First to fire happened to get S17 right.

With n = 24 none of these differences is statistically significant, and a single run of one model is not a benchmark. The qualitative point is the useful one: statistics and language priors fail on different storms, which argues for using them together rather than choosing one.

To re-run against your own key: `ANTHROPIC_API_KEY=... python -m eval.llm_experiment live`.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest                   # 15 tests, under 2 seconds
python scripts/run_demo.py         # fit on a week, explain one storm alert by alert
python -m eval.run_eval            # full sweep, about 10 seconds, writes eval/results.json
python -m eval.llm_experiment score
```

All randomness is seeded. The published `eval/` outputs were regenerated on a second machine with a different NumPy version (2.3.5 and 2.5.3) and matched byte for byte, apart from the recorded runtime.

## Layout

```
alert_lineage/
  topology.py      reference service graph and alert names
  simulate.py      branching simulation with ground truth parents and injected storms
  hawkes.py        pair construction, log likelihood, EM, beta profile likelihood
  attribution.py   MAP parents, root scores, baselines
  metrics.py       ROC AUC, co-occurrence lift, direction accuracy
  llm.py           prompt rendering, answer parsing, scoring, optional API client
eval/              sweep, LLM experiment, cached prompts, answers and results
scripts/run_demo.py
tests/
```

## Limitations

* Synthetic data only. The simulator is a Hawkes process, so the model is well specified apart from the storm boost; real alert streams have alert deduplication, `for:` durations, maintenance windows and paging thresholds that bend the kernel.
* One kernel decay shared by all service pairs. Real propagation delays differ by edge (a cache miss propagates in seconds, a failed batch job in hours).
* Static parameters. A real deployment would refit on a rolling window and should detect when the graph itself changes.
* The LLM comparison is a small, single run study meant to show where the methods disagree, not to rank models.

## Further reading

* A. G. Hawkes, Spectra of some self-exciting and mutually exciting point processes, Biometrika, 1971.
* E. Lewis and G. Mohler, A nonparametric EM algorithm for multiscale Hawkes processes, 2011.
* Y. Ogata, On Lewis' simulation method for point processes, IEEE Transactions on Information Theory, 1981.

## License

MIT, see [LICENSE](LICENSE).
