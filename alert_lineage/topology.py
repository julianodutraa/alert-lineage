"""Reference service topology for a generic cloud data platform.

The graph is a dependency DAG: an edge ``(u, v)`` means "v depends on u", so a
fault in ``u`` tends to make ``v`` alert shortly afterwards. ``metrics-exporter``
is deliberately isolated: it is the classic flapping, noisy alert source that
fires on its own and confuses "first alert wins" triage.
"""

from __future__ import annotations

import numpy as np

SERVICES: list[str] = [
    "object-store",
    "kafka-broker",
    "postgres-primary",
    "ingest-gateway",
    "stream-enricher",
    "dag-scheduler",
    "spark-batch",
    "feature-store",
    "warehouse-loader",
    "api-serving",
    "dashboard-svc",
    "metrics-exporter",
]

EDGES: list[tuple[str, str]] = [
    ("object-store", "spark-batch"),
    ("object-store", "warehouse-loader"),
    ("object-store", "feature-store"),
    ("kafka-broker", "stream-enricher"),
    ("kafka-broker", "ingest-gateway"),
    ("stream-enricher", "feature-store"),
    ("stream-enricher", "warehouse-loader"),
    ("postgres-primary", "dag-scheduler"),
    ("postgres-primary", "api-serving"),
    ("postgres-primary", "feature-store"),
    ("dag-scheduler", "spark-batch"),
    ("dag-scheduler", "warehouse-loader"),
    ("spark-batch", "warehouse-loader"),
    ("warehouse-loader", "dashboard-svc"),
    ("feature-store", "api-serving"),
    ("api-serving", "dashboard-svc"),
]

ALERT_NAMES: dict[str, str] = {
    "object-store": "ObjectStoreHighErrorRate",
    "kafka-broker": "KafkaUnderReplicatedPartitions",
    "postgres-primary": "PostgresConnectionSaturation",
    "ingest-gateway": "IngestGateway5xxBudgetBurn",
    "stream-enricher": "ConsumerLagGrowing",
    "dag-scheduler": "SchedulerHeartbeatMissed",
    "spark-batch": "SparkStageRetriesHigh",
    "feature-store": "FeatureFreshnessSLOBreach",
    "warehouse-loader": "LoadJobFailed",
    "api-serving": "ServingP99LatencyHigh",
    "dashboard-svc": "DashboardQueryTimeouts",
    "metrics-exporter": "ExporterScrapeFlapping",
}

INDEX: dict[str, int] = {s: i for i, s in enumerate(SERVICES)}


def adjacency() -> np.ndarray:
    """Boolean D x D matrix, ``A[u, v]`` True iff v depends on u."""
    d = len(SERVICES)
    a = np.zeros((d, d), dtype=bool)
    for u, v in EDGES:
        a[INDEX[u], INDEX[v]] = True
    return a


def upstream_of(service: str) -> list[str]:
    return [u for u, v in EDGES if v == service]


def storm_capable_roots() -> list[str]:
    """Services with at least one dependent, i.e. able to cause a cascade."""
    return sorted({u for u, _ in EDGES}, key=INDEX.get)
