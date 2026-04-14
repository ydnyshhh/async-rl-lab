from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from async_rl_lab.serialization import JsonSerializable, JsonValue


ParserStatus = Literal["ok", "recovered", "malformed"]
ActionType = Literal["respond", "tool_call", "finish", "invalid"]
ToolResultStatus = Literal["ok", "timeout", "error", "validation_error"]
VerifierStatus = Literal["ok", "timeout", "error", "pending"]
FinishReason = Literal["stop", "length", "tool_call", "error"]


@dataclass(frozen=True, slots=True)
class PolicyRef(JsonSerializable):
    run_id: str
    policy_version: int
    policy_tag: str
    checkpoint_step: int
    checkpoint_ts: float
    model_family: str
    policy_path: str | None = None
    checkpoint_sha256: str | None = None
    tokenizer_name: str | None = None
    device_hint: str | None = None
    parent_policy_version: int | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationRequest(JsonSerializable):
    request_id: str
    task_id: str
    prompt_id: str
    group_id: str
    sample_index_within_group: int
    actor_id: str
    policy: PolicyRef
    observation_text: str
    available_tools: tuple[str, ...]
    max_new_tokens: int
    temperature: float
    top_p: float
    created_ts: float
    stop_sequences: tuple[str, ...] = ()
    seed: int | None = None
    deadline_ts: float | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCall(JsonSerializable):
    tool_call_id: str
    tool_name: str
    arguments: dict[str, JsonValue]
    created_ts: float
    raw_text_span: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult(JsonSerializable):
    tool_call_id: str
    tool_name: str
    status: ToolResultStatus
    started_ts: float
    ended_ts: float
    output_text: str
    output_json: dict[str, JsonValue] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Action(JsonSerializable):
    action_id: str
    action_type: ActionType
    raw_text: str
    parsed_ts: float
    parser_status: ParserStatus
    final_text: str | None = None
    tool_call: ToolCall | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationResult(JsonSerializable):
    result_id: str
    request_id: str
    task_id: str
    prompt_id: str
    group_id: str
    sample_index_within_group: int
    actor_id: str
    policy: PolicyRef
    generated_text: str
    finish_reason: FinishReason
    created_ts: float
    started_ts: float
    ended_ts: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    parser_status: ParserStatus
    queue_wait_ms: float
    latency_ms: float
    action: Action | None = None
    token_logprobs: tuple[float, ...] | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Observation(JsonSerializable):
    observation_id: str
    task_id: str
    episode_id: str
    turn_index: int
    role: Literal["system", "user", "assistant", "tool", "env"]
    text: str
    created_ts: float
    available_tools: tuple[str, ...] = ()
    tool_result: ToolResult | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Transition(JsonSerializable):
    transition_id: str
    task_id: str
    episode_id: str
    step_index: int
    observation: Observation
    action: Action
    created_ts: float
    next_observation: Observation | None = None
    reward: float | None = None
    done: bool = False
    truncated: bool = False
    env_latency_ms: float | None = None
    verifier_pending: bool = False
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Episode(JsonSerializable):
    episode_id: str
    task_id: str
    prompt_id: str
    actor_id: str
    environment_name: str
    behavior_policy_version: int
    policy_ref: PolicyRef
    started_ts: float
    transitions: tuple[Transition, ...]
    ended_ts: float | None = None
    termination_reason: str | None = None
    is_truncated: bool = False
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RewardResult(JsonSerializable):
    reward_id: str
    trajectory_id: str
    task_id: str
    prompt_id: str
    group_id: str
    sample_index_within_group: int
    reward_value: float
    verifier_name: str
    verifier_status: VerifierStatus
    started_ts: float
    ended_ts: float
    reward_components: dict[str, float] = field(default_factory=dict)
    failure_tag: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Trajectory(JsonSerializable):
    trajectory_id: str
    task_id: str
    prompt_id: str
    group_id: str
    sample_index_within_group: int
    behavior_policy_version: int
    policy_ref: PolicyRef
    policy_checksum_or_ref: str
    actor_id: str
    episode_start_ts: float
    episode_end_ts: float | None
    env_steps: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_text: str
    termination_reason: str
    transitions: tuple[Transition, ...]
    parsed_action_trace: tuple[Action, ...]
    observations: tuple[Observation, ...]
    tool_calls: tuple[ToolCall, ...]
    tool_results: tuple[ToolResult, ...]
    per_step_rewards: tuple[float, ...]
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    verifier_start_ts: float | None = None
    verifier_end_ts: float | None = None
    queue_insert_ts: float | None = None
    learner_consume_ts: float | None = None
    terminal_reward: float | None = None
    is_truncated: bool = False
    behavior_logprobs: tuple[float, ...] | None = None
    reference_logprobs: tuple[float, ...] | None = None
    logprob_stats: dict[str, float] = field(default_factory=dict)
    invalid_action_count: int = 0
    generation_latency_ms: float | None = None
    verifier_latency_ms: float | None = None
    queue_wait_ms: float | None = None
    reward_result: RewardResult | None = None


@dataclass(frozen=True, slots=True)
class StalenessStats(JsonSerializable):
    batch_id: str
    learner_policy_version: int
    lag_by_trajectory: tuple[int, ...]
    age_ms_by_trajectory: tuple[float, ...]
    freshness_mask: tuple[bool, ...]
    created_ts: float
    min_policy_version: int | None = None
    max_policy_version: int | None = None
    dropped_for_staleness: int = 0
    mean_lag: float = 0.0
    max_lag: int = 0


@dataclass(frozen=True, slots=True)
class GroupedTrajectoryBatch(JsonSerializable):
    batch_id: str
    trajectories: tuple[Trajectory, ...]
    group_sizes: dict[str, int]
    created_ts: float
    required_group_size: int
    group_ids: tuple[str, ...]
    sampled_ts: float | None = None
    allow_partial_groups: bool = False
    staleness_stats: StalenessStats | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActorHeartbeat(JsonSerializable):
    actor_id: str
    current_policy_version: int
    queue_depth: int
    running_episodes: int
    completed_episodes: int
    failed_episodes: int
    last_seen_ts: float
    mean_generation_latency_ms: float | None = None
    mean_verifier_latency_ms: float | None = None
    last_error: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LearnerStepResult(JsonSerializable):
    learner_step: int
    consumed_batch_id: str
    consumed_groups: int
    consumed_trajectories: int
    learner_policy_version: int
    objective_name: str
    step_start_ts: float
    step_end_ts: float
    mean_reward: float
    metrics: dict[str, float]
    mean_advantage: float | None = None
    loss: float | None = None
    gradient_norm: float | None = None
    clipped_fraction: float | None = None
    stale_fraction: float | None = None
    published_policy: PolicyRef | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)
