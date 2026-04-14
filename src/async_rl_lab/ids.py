from __future__ import annotations

import time
import uuid


def utc_ts() -> float:
    return time.time()


def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
