from __future__ import annotations

import unittest

from async_rl_lab.buffer import InMemoryGroupedRolloutBuffer
from async_rl_lab.ids import make_id, utc_ts
from async_rl_lab.models import Action, Observation, PolicyRef, Trajectory, Transition


def make_policy(version: int) -> PolicyRef:
    return PolicyRef(
        run_id="run-test",
        policy_version=version,
        policy_tag=f"v{version}",
        checkpoint_step=version,
        checkpoint_ts=utc_ts(),
        model_family="mock",
    )


def make_trajectory(group_id: str, sample_index: int, policy_version: int) -> Trajectory:
    policy = make_policy(policy_version)
    action = Action(
        action_id=make_id("action"),
        action_type="finish",
        raw_text='{"type":"finish","answer":"4"}',
        parsed_ts=utc_ts(),
        parser_status="ok",
        final_text="4",
    )
    observation = Observation(
        observation_id=make_id("obs"),
        task_id="task-0",
        episode_id="ep-0",
        turn_index=0,
        role="user",
        text="What is 2 + 2?",
        created_ts=utc_ts(),
    )
    transition = Transition(
        transition_id=make_id("tr"),
        task_id="task-0",
        episode_id="ep-0",
        step_index=0,
        observation=observation,
        action=action,
        created_ts=utc_ts(),
        reward=1.0,
        done=True,
    )
    return Trajectory(
        trajectory_id=make_id("traj"),
        task_id="task-0",
        prompt_id="prompt-0",
        group_id=group_id,
        sample_index_within_group=sample_index,
        behavior_policy_version=policy_version,
        policy_ref=policy,
        policy_checksum_or_ref=policy.policy_tag,
        actor_id="actor-0",
        episode_start_ts=utc_ts(),
        episode_end_ts=utc_ts(),
        env_steps=1,
        prompt_tokens=5,
        completion_tokens=3,
        total_tokens=8,
        raw_text='{"type":"finish","answer":"4"}',
        termination_reason="done",
        transitions=(transition,),
        parsed_action_trace=(action,),
        observations=(observation,),
        tool_calls=(),
        tool_results=(),
        per_step_rewards=(1.0,),
        terminal_reward=1.0,
        behavior_logprobs=(-0.1,),
    )


class BufferTests(unittest.TestCase):
    def test_insert_requires_full_groups(self) -> None:
        buffer = InMemoryGroupedRolloutBuffer(capacity_groups=2, required_group_size=2)
        with self.assertRaises(ValueError):
            buffer.insert_group((make_trajectory("group-a", 0, 0),))

    def test_sample_preserves_group(self) -> None:
        buffer = InMemoryGroupedRolloutBuffer(capacity_groups=2, required_group_size=2)
        buffer.insert_group(
            (
                make_trajectory("group-a", 0, 0),
                make_trajectory("group-a", 1, 0),
            )
        )
        batch = buffer.sample_groups(max_groups=1, learner_policy_version=0)
        self.assertIsNotNone(batch)
        self.assertEqual(batch.group_ids, ("group-a",))
        self.assertEqual(len(batch.trajectories), 2)
        self.assertTrue(all(t.learner_consume_ts is not None for t in batch.trajectories))

    def test_drop_most_stale_prefers_old_behavior_policy(self) -> None:
        buffer = InMemoryGroupedRolloutBuffer(
            capacity_groups=2,
            required_group_size=2,
            drop_policy="drop_most_stale",
        )
        buffer.insert_group((make_trajectory("group-a", 0, 0), make_trajectory("group-a", 1, 0)))
        buffer.insert_group((make_trajectory("group-b", 0, 2), make_trajectory("group-b", 1, 2)))
        result = buffer.insert_group((make_trajectory("group-c", 0, 3), make_trajectory("group-c", 1, 3)))
        self.assertEqual(result.dropped_group_ids, ("group-a",))
        self.assertEqual(tuple(buffer.group_order), ("group-b", "group-c"))
