from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from async_rl_lab.serialization import JsonSerializable


@dataclass(frozen=True, slots=True)
class MetricsSnapshot(JsonSerializable):
    counters: dict[str, float]
    histograms: dict[str, tuple[float, ...]]


class MetricsStore:
    def __init__(self) -> None:
        self.counters: dict[str, float] = defaultdict(float)
        self.histograms: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, value: float = 1.0) -> None:
        self.counters[name] += value

    def observe(self, name: str, value: float) -> None:
        self.histograms[name].append(value)

    def snapshot(self) -> MetricsSnapshot:
        return MetricsSnapshot(
            counters=dict(self.counters),
            histograms={name: tuple(values) for name, values in self.histograms.items()},
        )
