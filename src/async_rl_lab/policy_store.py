from __future__ import annotations

import asyncio
import json
from pathlib import Path

from async_rl_lab.ids import utc_ts
from async_rl_lab.models import PolicyRef
from async_rl_lab.serialization import JsonValue


class LocalPolicyStore:
    def __init__(
        self,
        root_dir: Path,
        *,
        run_id: str,
        model_family: str = "mock-llm",
    ) -> None:
        self.root_dir = root_dir
        self.run_id = run_id
        self.model_family = model_family
        self.lock = asyncio.Lock()
        self.policy_dir = self.root_dir / "policies"
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.current = PolicyRef(
            run_id=run_id,
            policy_version=0,
            policy_tag="bootstrap",
            checkpoint_step=0,
            checkpoint_ts=utc_ts(),
            model_family=model_family,
            policy_path=None,
            metadata={"published_by": "bootstrap"},
        )

    def current_policy(self) -> PolicyRef:
        return self.current

    async def publish_policy(
        self,
        *,
        checkpoint_step: int,
        policy_tag: str,
        checkpoint_path: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> PolicyRef:
        async with self.lock:
            next_policy = PolicyRef(
                run_id=self.run_id,
                policy_version=self.current.policy_version + 1,
                policy_tag=policy_tag,
                checkpoint_step=checkpoint_step,
                checkpoint_ts=utc_ts(),
                model_family=self.model_family,
                policy_path=checkpoint_path,
                parent_policy_version=self.current.policy_version,
                metadata=metadata or {},
            )
            manifest_path = self.policy_dir / f"policy-v{next_policy.policy_version:06d}.json"
            manifest_path.write_text(json.dumps(next_policy.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            self.current = next_policy
            return next_policy

    def load_policy(self, policy_version: int) -> PolicyRef | None:
        manifest_path = self.policy_dir / f"policy-v{policy_version:06d}.json"
        if not manifest_path.exists():
            return None
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return PolicyRef(**payload)
