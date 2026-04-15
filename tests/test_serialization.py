from __future__ import annotations

import json
import unittest

from async_rl_lab.models import (
    Action,
    Observation,
    PolicyRef,
    RewardResult,
    ToolCall,
    ToolResult,
    Trajectory,
    Transition,
)
from tests.test_buffer import make_policy, make_trajectory


def policy_ref_from_dict(payload: dict) -> PolicyRef:
    return PolicyRef(**payload)


def reward_result_from_dict(payload: dict) -> RewardResult:
    return RewardResult(**payload)


def tool_call_from_dict(payload: dict | None) -> ToolCall | None:
    if payload is None:
        return None
    return ToolCall(**payload)


def tool_result_from_dict(payload: dict | None) -> ToolResult | None:
    if payload is None:
        return None
    return ToolResult(**payload)


def action_from_dict(payload: dict) -> Action:
    return Action(
        action_id=payload["action_id"],
        action_type=payload["action_type"],
        raw_text=payload["raw_text"],
        parsed_ts=payload["parsed_ts"],
        parser_status=payload["parser_status"],
        final_text=payload.get("final_text"),
        tool_call=tool_call_from_dict(payload.get("tool_call")),
        metadata=payload.get("metadata", {}),
    )


def observation_from_dict(payload: dict) -> Observation:
    return Observation(
        observation_id=payload["observation_id"],
        task_id=payload["task_id"],
        episode_id=payload["episode_id"],
        turn_index=payload["turn_index"],
        role=payload["role"],
        text=payload["text"],
        created_ts=payload["created_ts"],
        available_tools=tuple(payload.get("available_tools", ())),
        tool_result=tool_result_from_dict(payload.get("tool_result")),
        metadata=payload.get("metadata", {}),
    )


def transition_from_dict(payload: dict) -> Transition:
    return Transition(
        transition_id=payload["transition_id"],
        task_id=payload["task_id"],
        episode_id=payload["episode_id"],
        step_index=payload["step_index"],
        observation=observation_from_dict(payload["observation"]),
        action=action_from_dict(payload["action"]),
        created_ts=payload["created_ts"],
        next_observation=observation_from_dict(payload["next_observation"]) if payload.get("next_observation") else None,
        reward=payload.get("reward"),
        done=payload.get("done", False),
        truncated=payload.get("truncated", False),
        env_latency_ms=payload.get("env_latency_ms"),
        verifier_pending=payload.get("verifier_pending", False),
        metadata=payload.get("metadata", {}),
    )


def trajectory_from_dict(payload: dict) -> Trajectory:
    return Trajectory(
        trajectory_id=payload["trajectory_id"],
        task_id=payload["task_id"],
        prompt_id=payload["prompt_id"],
        group_id=payload["group_id"],
        sample_index_within_group=payload["sample_index_within_group"],
        behavior_policy_version=payload["behavior_policy_version"],
        policy_ref=policy_ref_from_dict(payload["policy_ref"]),
        policy_checksum_or_ref=payload["policy_checksum_or_ref"],
        actor_id=payload["actor_id"],
        episode_start_ts=payload["episode_start_ts"],
        episode_end_ts=payload.get("episode_end_ts"),
        env_steps=payload["env_steps"],
        prompt_tokens=payload["prompt_tokens"],
        completion_tokens=payload["completion_tokens"],
        total_tokens=payload["total_tokens"],
        raw_text=payload["raw_text"],
        termination_reason=payload["termination_reason"],
        transitions=tuple(transition_from_dict(item) for item in payload["transitions"]),
        parsed_action_trace=tuple(action_from_dict(item) for item in payload["parsed_action_trace"]),
        observations=tuple(observation_from_dict(item) for item in payload["observations"]),
        tool_calls=tuple(tool_call_from_dict(item) for item in payload["tool_calls"]),
        tool_results=tuple(tool_result_from_dict(item) for item in payload["tool_results"]),
        per_step_rewards=tuple(payload["per_step_rewards"]),
        metadata=payload.get("metadata", {}),
        verifier_start_ts=payload.get("verifier_start_ts"),
        verifier_end_ts=payload.get("verifier_end_ts"),
        queue_insert_ts=payload.get("queue_insert_ts"),
        learner_consume_ts=payload.get("learner_consume_ts"),
        terminal_reward=payload.get("terminal_reward"),
        is_truncated=payload.get("is_truncated", False),
        behavior_logprobs=tuple(payload["behavior_logprobs"]) if payload.get("behavior_logprobs") else None,
        behavior_token_logprobs=tuple(tuple(row) for row in payload["behavior_token_logprobs"])
        if payload.get("behavior_token_logprobs")
        else None,
        reference_logprobs=tuple(payload["reference_logprobs"]) if payload.get("reference_logprobs") else None,
        reference_token_logprobs=tuple(tuple(row) for row in payload["reference_token_logprobs"])
        if payload.get("reference_token_logprobs")
        else None,
        logprob_stats=payload.get("logprob_stats", {}),
        invalid_action_count=payload.get("invalid_action_count", 0),
        generation_latency_ms=payload.get("generation_latency_ms"),
        verifier_latency_ms=payload.get("verifier_latency_ms"),
        queue_wait_ms=payload.get("queue_wait_ms"),
        reward_result=reward_result_from_dict(payload["reward_result"]) if payload.get("reward_result") else None,
    )


class SerializationTests(unittest.TestCase):
    def test_policy_ref_roundtrip(self) -> None:
        policy = make_policy(3)
        reconstructed = policy_ref_from_dict(json.loads(policy.to_json_line()))
        self.assertEqual(reconstructed.policy_version, policy.policy_version)
        self.assertEqual(reconstructed.policy_tag, policy.policy_tag)

    def test_reward_result_roundtrip(self) -> None:
        reward = RewardResult(
            reward_id="reward-0",
            trajectory_id="traj-0",
            task_id="task-0",
            prompt_id="prompt-0",
            group_id="group-0",
            sample_index_within_group=0,
            reward_value=1.0,
            verifier_name="exact_match",
            verifier_status="ok",
            started_ts=1.0,
            ended_ts=2.0,
            reward_components={"exact_match": 1.0},
        )
        reconstructed = reward_result_from_dict(json.loads(reward.to_json_line()))
        self.assertEqual(reconstructed.reward_value, 1.0)
        self.assertEqual(reconstructed.verifier_name, "exact_match")

    def test_trajectory_roundtrip(self) -> None:
        trajectory = make_trajectory("group-a", 0, 0, reward=1.0, answer_text="4", expected_answer="4")
        reconstructed = trajectory_from_dict(json.loads(trajectory.to_json_line()))
        self.assertEqual(reconstructed.trajectory_id, trajectory.trajectory_id)
        self.assertEqual(reconstructed.transitions[0].action.final_text, "4")
        self.assertEqual(reconstructed.metadata["expected_answer"], "4")
