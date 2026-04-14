from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from async_rl_lab.ids import utc_ts
from async_rl_lab.serialization import JsonSerializable, JsonValue


@dataclass(frozen=True, slots=True)
class EventRecord(JsonSerializable):
    event_type: str
    ts: float
    run_id: str
    payload: dict[str, JsonValue] = field(default_factory=dict)
    actor_id: str | None = None
    learner_step: int | None = None
    policy_version: int | None = None
    group_id: str | None = None
    trajectory_id: str | None = None


class JsonlEventLogger:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event_type: str,
        *,
        payload: dict[str, JsonValue] | None = None,
        actor_id: str | None = None,
        learner_step: int | None = None,
        policy_version: int | None = None,
        group_id: str | None = None,
        trajectory_id: str | None = None,
    ) -> EventRecord:
        record = EventRecord(
            event_type=event_type,
            ts=utc_ts(),
            run_id=self.run_id,
            payload=payload or {},
            actor_id=actor_id,
            learner_step=learner_step,
            policy_version=policy_version,
            group_id=group_id,
            trajectory_id=trajectory_id,
        )
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(record.to_json_line())
                handle.write("\n")
        return record
