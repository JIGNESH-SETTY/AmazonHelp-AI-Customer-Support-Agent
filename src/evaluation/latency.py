"""
latency.py
----------
STAGE 10: Runtime Latency Profiler

Computes:
  - Mean latency (ms)
  - Median latency (ms)
  - 95th percentile latency (P95 ms)
  - Min / Max latency (ms)
  - Component-level latency breakdown where available
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np


@dataclass
class LatencyProfile:
    """Statistical summary of execution latencies."""
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "sample_count": self.sample_count,
        }


class LatencyProfiler:
    """Collects execution latencies and computes percentile summaries."""

    def __init__(self):
        self.latencies: List[float] = []

    def record(self, elapsed_ms: float) -> None:
        self.latencies.append(elapsed_ms)

    def summarize(self) -> LatencyProfile:
        if not self.latencies:
            return LatencyProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0)

        arr = np.array(self.latencies)
        return LatencyProfile(
            mean_ms=round(float(np.mean(arr)), 3),
            median_ms=round(float(np.median(arr)), 3),
            p95_ms=round(float(np.percentile(arr, 95)), 3),
            min_ms=round(float(np.min(arr)), 3),
            max_ms=round(float(np.max(arr)), 3),
            sample_count=len(self.latencies),
        )
