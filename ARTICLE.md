# The first alert is usually lying to you

It is 03:12 and the pager has fired eleven times in four minutes. The dashboard service is timing out, the warehouse loader has failed twice, Spark is retrying stages, and somewhere near the top of the list a metrics exporter is complaining about flapping scrapes, as it does several times a day. The on-call engineer has to decide which of these alerts is the disease and which are symptoms, and they have to decide fast, because every minute spent restarting the wrong thing is a minute the real fault keeps spreading.

Most teams answer that question with one of two habits. The first is to look at whichever alert fired first. The second is to open the service map and walk upstream by hand until you hit something that is alerting and has nothing alerting above it. I wanted to know how good those habits actually are, and whether a model that learns from alert history can do better. This article walks through the project that came out of it, alert-lineage, including the parts where the answer was "not yet".

## Alerts are a self-exciting process

The key observation is not new, it is just rarely applied to alerting. An alert on a shared dependency raises the short term probability of alerts on everything that depends on it. A database alert is followed, within minutes, by alerts on the scheduler and the serving API, which in turn are followed by alerts on the dashboards. That is the defining property of a Hawkes process: every event temporarily increases the rate of future events, possibly on other dimensions.

Formally, each service has an intensity made of a background rate plus a sum of decaying bumps, one for every past alert, with an amplitude `alpha[u, v]` that says how strongly an alert on `u` excites alerts on `v`. With an exponential kernel normalised to integrate to one, `alpha[u, v]` has a lovely operational meaning: it is the expected number of `v` alerts directly caused by one `u` alert. The whole matrix of alphas is a learned, directed, weighted dependency graph, estimated from nothing but timestamps.

## Fitting it without fighting it

The likelihood of a Hawkes process is unpleasant to optimise directly, but it hides a clean latent structure. Every alert is either background, something that would have happened anyway, or the child of exactly one earlier alert. If you knew the parent of every alert, estimating the parameters would be counting. You do not know the parents, which is precisely the situation EM was invented for.

The E step computes, for every alert, the posterior probability that it is background or that it was triggered by each earlier alert inside the kernel window. The M step turns those soft parent assignments back into rates: expected background alerts over the horizon for `mu`, expected parent links over exposure for `alpha`. Each iteration is a handful of vectorised operations over a list of candidate pairs that is built once. The kernel decay is chosen by profile likelihood on a small grid. On a month of simulated alerts the whole fit takes about half a second.

The part that matters for triage is that the E step is not just a fitting device. Run on a live storm with fitted parameters, it gives you, for every alert, a probability distribution over who caused it. Chaining those distributions backwards gives each alert a distribution over its ultimate ancestor, and summing over the storm gives each service an expected number of storm alerts that trace back to it. That number is the root score.

## Building a world where the answer is known

You cannot evaluate root cause attribution on real incidents without ground truth, and postmortems are not ground truth. So the project simulates it, using the branching representation of the Hawkes process, which generates every alert together with its true parent. The world is a generic twelve service data platform: object store, Kafka, Postgres, a scheduler, Spark, a feature store, a warehouse loader, a serving API, dashboards, and one deliberately isolated, flapping metrics exporter whose only job is to make noise.

Incident storms are injected roughly once a day, and their first generation fans out six times harder than a normal alert. The fitted model is never told about that boost. Five independent worlds, training histories from one to twenty eight days, and a separate sixty day test stream per world give 227 storms, each evaluated in a window from ten minutes before to twenty minutes after the storm starts.

## What happened

The first-alert habit picked the right root 52.4% of the time. That is not a strawman number; it comes straight from the background noise rate. With the flapping exporter and ordinary background alerts, there is roughly an even chance that something unrelated fires in the ten minutes before the real fault.

The topology rule, picking the alerting service with no alerting upstream and the widest downstream reach, scored 74.5%. That is a genuinely good heuristic and it needs no data at all.

The Hawkes model with a single day of history scored 67.0%, worse than the topology rule. Its branching ratios were off by 0.26 on average, most of their range. This was the first honest lesson: a statistical model on a cold start is worse than the knowledge already sitting in the service catalogue. The fix was to give it that knowledge, by masking alpha to the known dependency edges plus self excitation. With one day of history and the mask, it scored 78.4%.

With three days it reached 81.1%, with a week 81.5%, and with twenty eight days 85.0%, against an oracle, the same model with the true parameters, at 84.1%. The fitted model edging past the oracle is two storms out of 227 and I do not read anything into it beyond a reminder: the oracle is the ceiling for this model family, not for the problem, and the fitted parameters absorb some of the storm boost the oracle does not know about.

Two other findings were less flattering. Plain lagged co-occurrence, counting how often `v` follows `u` within ten minutes, already recovers the dependency graph with AUC between 0.90 and 0.99. If all you want is the graph, you do not need a point process. The point process earns its keep in per alert attribution, where it reaches 76.9% correct parents against 48.4% for "the most recent earlier alert", and in root scoring, where co-occurrence has nothing to say. And the root score has a real failure mode: it counts self excited repeats, so a source that flaps seven times inside the window can outscore a small but genuine cascade. I left that visible instead of tuning it away on the test set.

## Then I asked a language model

The obvious question in 2026 is whether you need any of this, or whether an LLM can read the storm. I gave a Claude model the dependency graph and the raw feed for 24 storms, blind, with the window offset randomised so the true root did not always sit at the same timestamp. In a second condition, each storm also carried the Hawkes root scores as a labelled hint.

The totals are close: first-to-fire 12 of 24, topology rule 16, Hawkes 19, the LLM alone 19, the LLM with the hint 20. The disagreements are the interesting part. On one storm a metrics exporter flapped seven times while the real cascade started at the serving API. Hawkes fell for the exporter. The LLM discounted it from its name and its isolation in the graph, and was right. On another storm the LLM alone picked a downstream service and switched to the correct upstream root once it saw the hint. Four storms were missed by all of the stronger methods, and in each the feed alone genuinely pointed the wrong way: an upstream service alerted first, or an unrelated cascade overlapped a small storm.

With 24 storms none of this is significant, and one run of one model is not a benchmark. But the shape is clear enough to act on. Statistics and language priors fail on different storms. A triage assistant should not choose between them; it should let the model that has read your alert history set the ranking, and let the model that knows what an exporter is argue with it.

## What I would do next

Per edge kernel decays, because a cache miss and a failed nightly job do not propagate on the same timescale. Rolling refits with change detection on the alpha matrix, so the graph tracks reality. Change events and deploys as exogenous inputs, which is exactly the signal the four unsolvable storms were missing. And a root score that discounts self excitation, evaluated on fresh data rather than the storms that revealed the problem.

The code, the full sweep, the cached prompts and answers and fifteen tests are at https://github.com/julianodutraa/alert-lineage
