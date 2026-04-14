# Async RL Lab Implementation Spec

## 1. Core implementation philosophy

`async-rl-lab` is a single-process, `asyncio`-first research library for studying asynchronous GRPO-family post-training for LLM and agentic policies. The codebase is meant to make the control plane and data plane visible: generation, environment stepping, reward verification, grouping, queueing, policy publication, and learner consumption are all explicit subsystems.

Problem definition:
- Train sequence-generating LLM policies with grouped prompt sampling, non-zero rollout latency, policy lag, and optionally delayed rewards.
- Treat inference as a serving subsystem with request queues, batching, refresh latency, and policy identity.
- Preserve enough metadata to measure where async instability comes from: policy staleness, queueing, verifier delay, malformed actions, and incomplete groups.

Explicit exclusions:
- PPO-centric abstractions, replay-style off-policy RL infrastructure, distributed RPC orchestration, cluster schedulers, and generic MDP APIs that hide sequence semantics.
- Hidden “sample then train” helpers that collapse actor, inference, verifier, and queue into one black box.
- Full optimizer or checkpoint sharding in v1.

Why the center is async GRPO-family LLM training:
- Grouped completions per prompt are the natural unit for relative reward objectives.
- LLMs experience serving lag, tokenizer overhead, and batched generation artifacts that do not look like classical action-at-each-step RL.
- Agentic systems care about tool-use traces, verifier delay, and sequence-level reward much more than PPO-style dense timestep bookkeeping.

Smallest nontrivial implementation:
- One learner coroutine, one or more actor coroutines, one in-memory grouped rollout buffer, one mock or HF inference engine, exact-match verifier, grouped GRPO loss, JSONL events, and explicit policy publication.

Hard invariants:
- Every `Trajectory` carries `behavior_policy_version` and a `PolicyRef`.
- Every grouped sample preserves `prompt_id`, `group_id`, and `sample_index_within_group`.
- Inference and learner are decoupled at the interface level.
- Verifier latency is measured independently from generation latency.
- Queue insertion and learner consumption timestamps are explicit fields, never inferred later.
- Grouped rollout insertion is atomic in v1.
- No hidden state transitions inside helpers: transitions are materialized as `Transition` records.
- Parser failures are surfaced as `Action(action_type="invalid")`, not swallowed.
- Policy refresh is explicit and logged.

## 2. Repo structure and file responsibilities

```text
docs/implementation_spec.md
src/async_rl_lab/__init__.py
src/async_rl_lab/ids.py
src/async_rl_lab/serialization.py
src/async_rl_lab/models.py
src/async_rl_lab/events.py
src/async_rl_lab/metrics.py
src/async_rl_lab/inference.py
src/async_rl_lab/policy_store.py
src/async_rl_lab/buffer.py
src/async_rl_lab/objectives.py
src/async_rl_lab/tools.py
src/async_rl_lab/environments.py
src/async_rl_lab/verifiers.py
src/async_rl_lab/runtime.py
src/async_rl_lab/demo.py
tests/
```

File-by-file responsibilities:
- `ids.py`: timestamp and stable ID generation. Must not contain business logic. Exports `utc_ts`, `make_id`.
- `serialization.py`: dataclass-to-JSONL conversion. Must not mutate records or own storage. Exports `JsonSerializable`, `JsonValue`, `to_jsonable`.
- `models.py`: immutable transport records and learner/actor metadata. Must not contain control flow. Depends only on `serialization.py`.
- `events.py`: structured event record and JSONL logger. Must not compute RL metrics. Depends on `ids.py`, `serialization.py`.
- `metrics.py`: simple counters and histogram accumulation. Must not write files. Exports `MetricsStore`.
- `inference.py`: `InferenceEngine` interface, request batching, mock backend, action parsing, backend stubs. Must not contain learner logic or queue sampling.
- `policy_store.py`: version publication and local manifest persistence. Must not hold model weights in memory for the learner.
- `buffer.py`: grouped rollout retention, freshness filtering, sampling, capacity management. Must not know about tokenization or loss math.
- `objectives.py`: batch preparation and GRPO-family loss math. Must not read from inference queues or publish policies.
- `tools.py`: tool registry, validation, timeout handling, result serialization. Must not call the learner.
- `environments.py`: environment state schemas and environment transition logic. Must not parse model text directly; it consumes `Action`.
- `verifiers.py`: reward pipeline implementations. Must not manage actor scheduling.
- `runtime.py`: actor loop, learner loop, verifier worker loop, policy refresh logic, trajectory construction. Must not contain backend-specific generation code.
- `demo.py`: minimal single-process hot path. Must not accumulate production logic.

## 3. Core record schemas

All records in `models.py` are frozen dataclasses and should be treated as append-only event payloads. Any update should use `dataclasses.replace(...)`. All are JSONL-safe; the trajectory-like records are also parquet-safe with flattening.

### `PolicyRef`
- Fields: `run_id`, `policy_version`, `policy_tag`, `checkpoint_step`, `checkpoint_ts`, `model_family`, optional `policy_path`, `checkpoint_sha256`, `tokenizer_name`, `device_hint`, `parent_policy_version`, `metadata`.
- Required: all except the optional checkpoint details.
- Created: bootstrap at runtime start and after each learner publish.
- Owner: `LocalPolicyStore`.
- Async use: copied into every request and trajectory so policy lag is measurable without asking the live inference engine.

### `GenerationRequest`
- Fields: `request_id`, `task_id`, `prompt_id`, `group_id`, `sample_index_within_group`, `actor_id`, `policy`, `observation_text`, `available_tools`, `max_new_tokens`, `temperature`, `top_p`, `created_ts`, optional `stop_sequences`, `seed`, `deadline_ts`, `metadata`.
- Important metadata: gold answer, expression hints, prompt template tag, tool schema version.
- Created: by the actor on every model turn.
- Owner: actor until submitted, inference engine afterward.
- Async use: batching key is compatible across requests while retaining group identity.

### `GenerationResult`
- Fields: request identity, actor identity, `policy`, raw generation text, `finish_reason`, timestamps, token counts, parser status, `queue_wait_ms`, `latency_ms`, optional `action`, optional `token_logprobs`, `metadata`.
- Created: by inference engine after decoding and parsing.
- Owner: inference engine, then actor.
- Async use: separates model latency from environment and verifier latency.

### `Observation`
- Fields: `observation_id`, `task_id`, `episode_id`, `turn_index`, `role`, `text`, `created_ts`, optional `available_tools`, optional `tool_result`, `metadata`.
- Created: by environment on reset and after every step.
- Owner: environment session.
- Async use: immutable turn snapshots prevent hidden env mutation from leaking into already-emitted transitions.

### `Action`
- Fields: `action_id`, `action_type`, `raw_text`, `parsed_ts`, `parser_status`, optional `final_text`, optional `tool_call`, `metadata`.
- Created: parser immediately after generation.
- Owner: inference engine first, then environment transition log.
- Async use: malformed generations become first-class training data instead of dropped exceptions.

### `ToolCall`
- Fields: `tool_call_id`, `tool_name`, `arguments`, `created_ts`, optional `raw_text_span`, `metadata`.
- Created: parser when action type is tool call.
- Owner: action trace, tool subsystem.
- Async use: survives across verifier and queue boundaries.

### `ToolResult`
- Fields: `tool_call_id`, `tool_name`, `status`, `started_ts`, `ended_ts`, `output_text`, optional `output_json`, optional error fields, `metadata`.
- Created: tool registry after invocation or validation failure.
- Owner: environment session.
- Async use: lets verifier separate model quality from tool backend quality.

### `Transition`
- Fields: `transition_id`, task/episode IDs, `step_index`, `observation`, `action`, `created_ts`, optional `next_observation`, optional `reward`, `done`, `truncated`, optional `env_latency_ms`, `verifier_pending`, `metadata`.
- Created: environment on every step.
- Owner: environment session, then trajectory.
- Async use: pending reward episodes keep structural transitions even before reward arrives.

### `Episode`
- Fields: episode identity, prompt/task IDs, actor/environment identity, behavior policy info, timestamps, full transition tuple, termination metadata.
- Created: when environment declares terminal or truncated.
- Owner: actor.
- Async use: intermediate shell before verifier attaches reward.

### `RewardResult`
- Fields: reward identity, trajectory/prompt/group/sample identity, scalar reward, component dictionary, verifier identity/status, verifier timestamps, optional failure tag, metadata.
- Created: verifier pipeline.
- Owner: verifier worker or actor if inline.
- Async use: carries delayed reward provenance back into the queue path.

### `Trajectory`
- Fields: `trajectory_id`, `task_id`, `prompt_id`, `group_id`, `sample_index_within_group`, `behavior_policy_version`, `policy_ref`, `policy_checksum_or_ref`, `actor_id`, `episode_start_ts`, `episode_end_ts`, `verifier_start_ts`, `verifier_end_ts`, `queue_insert_ts`, `learner_consume_ts`, `env_steps`, `tool_calls`, `tool_results`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `terminal_reward`, `per_step_rewards`, `is_truncated`, `termination_reason`, `behavior_logprobs`, `reference_logprobs`, `logprob_stats`, `raw_text`, `parsed_action_trace`, `observations`, `transitions`, `invalid_action_count`, generation/verifier/queue latencies, `reward_result`, `metadata`.
- Why the important fields exist:
  - `behavior_policy_version`: required for lag-aware objectives.
  - `group_id` and `sample_index_within_group`: required for GRPO normalization and verifier alignment.
  - `queue_insert_ts` and `learner_consume_ts`: required for age-based freshness and observability.
  - `verifier_start_ts` and `verifier_end_ts`: required to split serving latency from reward latency.
  - `tool_calls` and `tool_results`: needed for tool-trace-aware verification and debugging.
  - `behavior_logprobs`: needed for future importance weighting or KL-style diagnostics.
- Created: actor after episode close, then enriched after verifier.
- Owner: rollout buffer once queued.
- Async use: the single source of truth for stale sample handling.

### `GroupedTrajectoryBatch`
- Fields: `batch_id`, `trajectories`, `group_sizes`, `created_ts`, `required_group_size`, `group_ids`, `sampled_ts`, `allow_partial_groups`, optional `staleness_stats`, `metadata`.
- Created: rollout buffer at learner sample time.
- Owner: learner.
- Async use: the buffer, not the learner, is responsible for preserving group completeness and staleness stats.

### `StalenessStats`
- Fields: `batch_id`, `learner_policy_version`, `lag_by_trajectory`, `age_ms_by_trajectory`, `freshness_mask`, `created_ts`, optional min/max behavior version, dropped count, mean lag, max lag.
- Created: when the learner samples a batch.
- Owner: rollout buffer.
- Async use: objective variants can consume it without re-querying runtime state.

### `ActorHeartbeat`
- Fields: actor identity, current policy version, queue depth, running/completed/failed episodes, `last_seen_ts`, mean generation and verifier latency, optional last error, metadata.
- Created: actor loop on a fixed interval.
- Owner: runtime event stream.
- Async use: lets the scheduler detect dead actors and queue starvation.

### `LearnerStepResult`
- Fields: learner step index, consumed batch identity, counts, learner policy version, objective name, start/end timestamps, mean reward, metrics, optional loss, advantage, gradient norm, clipped fraction, stale fraction, published policy, metadata.
- Created: learner after each optimization step.
- Owner: learner loop and event logger.
- Async use: ties model publication to a concrete consumed batch.

## 4. Inference subsystem

Primary interface in `inference.py`:

```python
class InferenceEngine(Protocol):
    async def start(self) -> None: ...
    async def submit(self, request: GenerationRequest) -> GenerationResult: ...
    async def submit_many(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]: ...
    async def refresh_policy(self, policy: PolicyRef) -> None: ...
    def current_policy(self) -> PolicyRef: ...
    async def close(self) -> None: ...
```

Request lifecycle:
1. Actor creates `GenerationRequest` from the latest `Observation`, current `PolicyRef`, and group metadata.
2. Request enters the engine queue with enqueue timestamp preserved in `PendingGeneration`.
3. Engine batcher waits `batch_window_ms`, drains up to `max_batch_size`, then issues one backend decode call.
4. Raw text is parsed immediately into `Action`.
5. `GenerationResult` returns raw text, parser status, token counts, queue wait, decode latency, and the behavior policy used.

Backend design:
- `MockInferenceEngine`: deterministic, low latency, explicit batching, ideal for queue and staleness tests. Failure points: parser mismatch, queue starvation, wrong policy refresh sequencing.
- `HFInferenceEngine`: future local `transformers` backend. Latency profile dominated by tokenize/decode and weight reload. Failure points: tokenizer drift, partial batch OOM, refresh pauses.
- `VLLMInferenceEngine`: future serving backend with stronger batching and KV cache reuse. Failure points: engine-side policy refresh semantics and request cancellation.

Primary action format:
- Strict JSON object per turn.
- Tool call: `{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"2+2"}}`
- Final answer: `{"type":"finish","answer":"4"}`
- Reason: it minimizes ambiguous parsing, serializes cleanly, and maps directly into tool and verifier schemas.

Parser behavior:
- First parse strict JSON.
- Then attempt fenced-block stripping and single-object recovery.
- On failure emit `Action(action_type="invalid", parser_status="malformed")`.
- No LLM repair pass in v1; retry is an environment/runtime concern, not parser magic.

Inference metrics:
- request queue wait
- active queue depth
- batch size
- prompt token count
- completion token count
- decode latency
- tokens per second
- policy refresh latency
- invalid action rate
- backend failure count

## 5. Policy storage and versioning

`PolicyRef` is the public policy identity. The learner never hands live model objects directly to actors. It publishes manifests via `LocalPolicyStore`, and actors/inference engines pull by reading the store’s latest published reference.

Publish path:
1. Learner finishes step `t`.
2. Learner writes `policy-v00000t.json` under `artifacts/<run_id>/policies/`.
3. `LocalPolicyStore.current` advances.
4. Actor heartbeat and `policy_refresh_logic` notice version drift.
5. Inference engine calls `refresh_policy(policy_ref)` and logs refresh latency.

Versioning rules:
- Every trajectory stores behavior version and policy tag.
- Freshness is computed as `learner_policy_version - behavior_policy_version`.
- Push model sharing is avoided in v1 to keep actor/inference decoupled from learner internals.
- Optimizer state is excluded from v1 manifests; store only policy metadata and optional checkpoint path.
- Checkpoint frequency in v1: every learner step or every small fixed interval so lag experiments are visible immediately.

Publish/refresh pseudocode:

```python
published = await policy_store.publish_policy(
    checkpoint_step=learner_step,
    policy_tag=f"learner-step-{learner_step}",
    metadata={"loss": loss},
)
await inference_engine.refresh_policy(published)

latest = policy_store.current_policy()
if inference_engine.current_policy().policy_version != latest.policy_version:
    await inference_engine.refresh_policy(latest)
```

## 6. Async runtime control flow

`runtime.py` owns the hot path:
- `actor_main_loop`: pull task, refresh policy, create one group, run `G` episodes, verify, insert group atomically, emit heartbeat.
- `execute_episode`: loop `Observation -> GenerationRequest -> GenerationResult -> Action -> Environment.step`.
- `learner_main_loop`: sample intact groups, compute staleness stats, prepare objective batch, compute loss, publish new policy.
- `verifier_loop`: optional separate path for delayed verification.
- `policy_refresh_logic`: actor-side pull refresh.

Timestamp capture rules:
- `GenerationRequest.created_ts`: before submission.
- `GenerationResult.started_ts` and `ended_ts`: inside engine batch execution.
- `Transition.created_ts`: at environment step entry.
- `Trajectory.queue_insert_ts`: at buffer insertion.
- `Trajectory.learner_consume_ts`: at learner sample.
- `Trajectory.verifier_start_ts` and `verifier_end_ts`: around verifier call, whether inline or external.

Runtime modes to keep distinct:
- Synchronous grouped GRPO collection: actor blocks until all `G` samples for one prompt are complete.
- Overlapped rollout and learning: actor keeps producing groups while learner consumes older full groups.
- True async stale consumption: learner may train on groups generated by policy versions older than current learner version.

## 7. Queue and rollout buffer

The rollout buffer is a real systems object, not a Python list. `InMemoryGroupedRolloutBuffer` uses:
- `deque[group_id]` for global order.
- `dict[group_id, tuple[Trajectory, ...]]` for atomic grouped storage.
- `Counter[actor_id]` for contribution accounting.

Insertion policy:
- Insert only full groups in v1.
- Stamp `queue_insert_ts` at insert time.
- Enforce capacity in units of groups, not individual trajectories.
- Default overflow policy is `drop_oldest`; `drop_most_stale` is reserved for a later pass.

Sampling policy:
- Drop stale groups first based on `max_policy_lag` and `max_age_ms`.
- Sample FIFO by intact group.
- Stamp `learner_consume_ts` on all emitted trajectories.
- Compute `StalenessStats` once during sample and attach to the batch.

Buffer observability surface:
- `current_size`
- `group_count`
- `average_age_ms`
- `oldest_item_age_ms`
- `staleness_histogram(learner_policy_version)`
- `group_completeness`
- `per_actor_contribution`

Why partial groups are disallowed in v1:
- GRPO normalization is group-relative.
- Partial groups silently change reward statistics.
- Allowing them early makes queue policy and verifier alignment much harder to reason about.

Insert pseudocode:

```python
def insert_group(trajectories):
    assert len(trajectories) == required_group_size
    group_id = trajectories[0].group_id
    stamped = [replace(t, queue_insert_ts=now()) for t in trajectories]
    while len(group_order) >= capacity_groups:
        drop_one_group()
    group_order.append(group_id)
    groups[group_id] = tuple(stamped)
```

Sample pseudocode:

```python
def sample_groups(max_groups, learner_policy_version):
    drop_stale_groups(learner_policy_version)
    selected = []
    while group_order and len(selected_group_ids) < max_groups:
        group_id = group_order.popleft()
        group = groups.pop(group_id)
        selected.extend(replace(t, learner_consume_ts=now()) for t in group)
    return GroupedTrajectoryBatch(..., staleness_stats=compute_staleness(...))
```

## 8. GRPO-family objective implementation

### GRPO baseline

Prompt sampling:
- Sample prompt tasks from a task source.
- For each prompt, actor generates `G` independent completions under one `group_id`.
- Rewards attach after verifier completion.

Tensor and shape model for the eventual torch implementation:
- Grouped sequences: `tokens[B, T]`
- Attention mask: `attn_mask[B, T]`
- Action mask over generated suffix: `action_mask[B, T]`
- Sequence rewards: `reward[B]`
- Group IDs: `group_index[B]`
- Group-relative advantages: `advantage[B]`

Where `B = num_groups * G`.

Relative advantage computation:
- For each group `g`, compute `A_i = r_i - mean(r_group)` in v1.
- Optional later variant: z-score normalize within group if verifier scale drifts.

Loss aggregation:
- Sequence-level surrogate in v1 scaffold: `L = -mean_i(advantage_i * logprob_sum_i)`.
- Real torch version: use token-level logprob ratios masked to generated tokens and average per sequence, then average across sequences.
- Token mask excludes prompt prefix and any padding.

### DAPO-style variants

Required code changes relative to plain GRPO:
- Clip extreme advantages before weighting.
- Weight sequences by reward magnitude or verifier confidence.
- Log `clipped_fraction`, `reward_scale`, and invalid-sample fraction.
- Filter or downweight malformed action trajectories instead of treating them equally.

Recommended v1 DAPO-inspired safeguards:
- `clip_advantage_abs = 5.0`
- reward RMS scaling for per-sequence weights
- `min_weight <= weight <= max_weight`
- optional drop of trajectories with verifier status not equal to `ok`

### Sequence-level lag-aware variant

Use freshness-weighted GRPO:
- Additional metadata required: `behavior_policy_version`, learner policy version at consume, and age in ms.
- Define `lag_i = learner_version - behavior_version_i`.
- Define `fresh_weight_i = exp(-lag_penalty * lag_i)`.
- Final loss weight becomes `advantage_i * fresh_weight_i`.

Why this variant:
- No fragile tokenwise importance correction in v1.
- Directly studies whether stale rollouts should count less.
- Works cleanly with grouped batching and delayed rewards.

### Objective interface

```python
class Objective(Protocol):
    def prepare_batch(self, batch: GroupedTrajectoryBatch) -> PreparedBatch: ...
    def compute_loss(self, prepared: PreparedBatch, model_logprob_summaries: Sequence[float]) -> float: ...
    def compute_metrics(self, prepared: PreparedBatch) -> dict[str, float]: ...
```

Division of labor:
- `prepare_batch`: derive rewards, group-relative advantages, masks, and freshness weights.
- `compute_loss`: consume already-prepared tensors or summaries and return scalar loss.
- `compute_metrics`: emit stable diagnostics independent of optimizer code.

## 9. Grouped rollout construction

Opinionated v1 design:
- One actor owns one prompt-group at a time.
- All `G` completions for that prompt are generated by that actor.
- Full group completion is required before queue insertion.
- Verifier aligns back to samples by `(group_id, sample_index_within_group)`.

Why not distribute one group across actors in v1:
- It complicates atomic insertion, completion tracking, and timeout handling.
- It introduces another source of lag unrelated to the core teaching objective.

Delayed verifier interaction:
- If verifier is inline, actor queues only fully rewarded groups.
- If verifier is separate, hold a pending group map keyed by `group_id` until all samples return reward, then insert atomically.

## 10. Environment implementations

### Single-turn exact-match
- State: prompt text, expected answer, started/ended timestamps.
- Observation: one user prompt.
- Action: `finish` only.
- Reward: `1.0` on exact string match, else `0.0`.
- Termination: always one step.
- Best use: parser, queue, grouping, and policy lag debugging.

### Tool-use multi-turn
- State: prompt, expected answer, tool registry, transition history, tool call/results, step counter.
- Observation after tool use: role `tool`, text equals tool output, tool metadata included.
- Action parsing: JSON tool calls or final answers.
- Max steps: fixed small integer, truncate on overflow.
- Failure cases: unknown tool, bad arguments, tool timeout, repeated invalid actions.
- Metrics: tool success rate, invalid action count, steps to completion, tool latency.

### Delayed-verifier environment
- Wraps another environment.
- Environment transition can finish with `reward=None` and `verifier_pending=True`.
- Pending episodes are tracked externally by `group_id`.
- Late rewards re-enter via `verifier_loop` and a pending-to-completed handoff queue.

## 11. Tool subsystem

Tool interface in `tools.py`:

```python
class Tool(Protocol):
    name: str
    description: str
    async def invoke(self, tool_call: ToolCall, *, timeout_s: float | None = None) -> ToolResult: ...
```

Concrete tools:
- `CalculatorTool`: safe arithmetic AST evaluator.
- `SearchRetrievalMockTool`: in-memory lookup corpus.
- `TextPatchTool`: in-memory document mutation to simulate editor actions.
- `ShellMockTool`: deterministic safe shell stub for agent traces.

Invocation path:
1. Model emits JSON tool call.
2. Parser constructs `ToolCall`.
3. Environment routes it through `ToolRegistry.invoke`.
4. Tool validates args and returns `ToolResult` instead of raising to the actor.
5. Result is attached to `Observation`, `Transition`, and final `Trajectory`.

## 12. Verifier and reward pipeline

Verifier interface:

```python
class Verifier(Protocol):
    name: str
    async def verify(self, trajectory: Trajectory) -> RewardResult: ...
    async def verify_many(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]: ...
```

Implementations:
- `ExactMatchVerifier`: strict deterministic reward.
- `ProgrammaticVerifier`: user-supplied scorer.
- `ToolTraceAwareVerifier`: rewards both answer correctness and required tool usage.
- `DelayedVerifierWrapper`: injects controlled latency.

Tradeoffs:
- Inline verifier keeps v1 simple and preserves group atomicity.
- Separate verifier worker is required once reward latency dominates actor throughput or when rewards depend on expensive external checks.

Required verifier logs:
- start timestamp
- end timestamp
- verifier name
- verifier status
- failure tag
- reward components

## 13. Observability and event logging

Primary event types:
- `ActorEpisodeStarted`
- `ActorEpisodeFinished`
- `ActorEpisodeFailed`
- `ActorHeartbeat`
- `InferenceBatchStarted`
- `InferenceBatchFinished`
- `VerifierStarted`
- `VerifierFinished`
- `TrajectoryQueued`
- `TrajectoryDropped`
- `LearnerBatchBuilt`
- `LearnerStepCompleted`
- `PolicyPublished`
- `PolicyRefreshed`

Event schema requirements:
- always include `run_id`, `ts`, `event_type`
- include `actor_id` when actor-owned
- include `learner_step` when learner-owned
- include `policy_version` whenever versioned state is involved
- include `group_id` and `trajectory_id` when applicable for timeline joins

Metrics and histograms:
- rollout queue depth
- group completeness ratio
- queue age ms
- policy lag histogram
- generation queue wait ms
- generation latency ms
- verifier latency ms
- env step latency ms
- invalid action rate
- tool success/error counts
- reward mean/std
- advantage mean/std
- stale fraction
- dropped group count

## 14. Failure modes and edge cases

- Malformed action output: mark `Action` invalid, log parser failure, continue env policy, count toward invalid-action metrics.
- Partial grouped rollouts: do not insert in v1; log `TrajectoryDropped` with reason `partial_group`.
- Actor dies mid-episode: no queue insert; heartbeat disappears; scheduler can restart actor later.
- Verifier timeout: emit `RewardResult(verifier_status="timeout", reward_value=0.0)` unless experiment explicitly wants retry semantics.
- Policy updated while requests are in flight: request keeps its original `PolicyRef`; result is still attributed to that behavior version.
- Stale samples dominate queue: buffer drops by freshness threshold before learner batch formation and logs drop counts.
- Queue fills with incomplete groups: avoided by atomic insertion rule.
- Tool call fails: return serialized `ToolResult(status="error")`; never crash actor loop because of tool backend behavior.
- Reward arrives after batch cutoff: if using separate verifier, keep sample out of the queue until reward exists or route into a pending-group map.
- Ambiguous parser recovery: record `parser_status="recovered"` and keep raw text for offline audit.

## 15. Minimal v1 code path

Best first implementation choice:
- `asyncio`
- single process
- one learner coroutine
- one actor coroutine at first, then scale to `N`
- in-memory grouped buffer
- `MockInferenceEngine` first, `HFInferenceEngine` second
- exact-match env first
- grouped rollouts generated entirely inside one actor
- inline verifier first
- GRPO baseline plus freshness-weighted variant
- JSONL event log
- no distributed infra, no RPC, no vLLM in v1

Cut from v1:
- distributed group assembly
- importance sampling over tokenwise old/new policy ratios
- optimizer state checkpoints
- cross-process queues
- external databases

## 16. Hot-path pseudocode

Actor episode:

```python
session = await env.start_episode(task, actor_id, policy_ref)
while not session.done:
    req = GenerationRequest(..., policy=policy_ref, group_id=group_id)
    gen = await inference.submit(req)
    action = gen.action or invalid_action(gen.generated_text)
    transition = await env.step(session, action)
episode = env.build_episode(session)
trajectory = build_trajectory(...)
reward = await verifier.verify(trajectory)
completed = replace(trajectory, reward_result=reward, terminal_reward=reward.reward_value)
```

Grouped rollout generation:

```python
group = []
for sample_index in range(G):
    group.append(await execute_episode(..., sample_index=sample_index))
buffer.insert_group(tuple(group))
```

Learner batch assembly:

```python
batch = buffer.sample_groups(max_groups=k, learner_policy_version=current_version)
prepared = objective.prepare_batch(batch)
loss = objective.compute_loss(prepared, model_logprob_summaries)
published = await policy_store.publish_policy(...)
await inference.refresh_policy(published)
```

## 17. Testing strategy

High-value tests:
- schema serialization roundtrip for `PolicyRef`, `Trajectory`, `RewardResult`
- parser robustness for strict JSON, fenced JSON, malformed text
- grouping correctness: group IDs preserved, partial groups rejected
- staleness accounting: lag and age computed correctly on sample
- queue policies: capacity overflow drops expected group
- policy version propagation: published version appears in later requests and trajectories
- delayed verifier handling: reward pending samples do not enter learner batch early
- actor-learner invariants: `learner_consume_ts >= queue_insert_ts`
- objective shapes: advantages are group-centered and preserve batch order
- environment determinism: exact-match env is deterministic for fixed input

Concrete examples:
- Build a group with rewards `[1, 0, 0, 1]` and assert GRPO advantages sum to zero within the group.
- Publish policy version `3`, sample with behavior version `1`, assert staleness lag is `2`.
- Feed parser text `{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"2+2"}}` and assert tool call survives into the trajectory.

## 18. Strong opinionated guidance

Best first implementation choice:
- Keep groups actor-local, verifiers inline, and inference explicit.

Biggest technical traps:
- hiding policy identity in mutable model objects
- allowing partial groups too early
- measuring only total episode latency and losing the split between serving and verification
- coupling actors directly to learner state instead of a published policy reference

Most important abstractions to get right:
- `Trajectory`
- `InferenceEngine`
- `InMemoryGroupedRolloutBuffer`
- `PolicyRef`

Most under-designed subsystem in projects like this:
- verifier and reward latency plumbing. People log scalar reward but not when it was produced, why it was delayed, or which subsystem caused the stall.

What makes this educational instead of merely functional:
- every important async boundary is explicit and inspectable
- every rollout can be traced back to a concrete behavior policy version
- group preservation is enforced instead of assumed
- the code lets you study the systems pathologies, not just run the happy path
