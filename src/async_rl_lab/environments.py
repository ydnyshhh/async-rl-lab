from __future__ import annotations

from dataclasses import dataclass, field

from async_rl_lab.ids import make_id, utc_ts
from async_rl_lab.models import Action, Episode, Observation, PolicyRef, Transition
from async_rl_lab.tools import ToolRegistry


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    prompt_id: str
    prompt_text: str
    expected_answer: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class EnvironmentSession:
    episode_id: str
    task: TaskSpec
    actor_id: str
    environment_name: str
    policy_ref: PolicyRef
    observations: list[Observation]
    transitions: list[Transition]
    tool_calls: list
    tool_results: list
    done: bool = False
    truncated: bool = False
    termination_reason: str | None = None
    started_ts: float = field(default_factory=utc_ts)
    ended_ts: float | None = None
    reward: float | None = None
    reward_pending: bool = False


class SingleTurnExactMatchEnvironment:
    name = "single_turn_exact_match"

    async def start_episode(self, task: TaskSpec, actor_id: str, policy_ref: PolicyRef) -> EnvironmentSession:
        episode_id = make_id("ep")
        first_observation = Observation(
            observation_id=make_id("obs"),
            task_id=task.task_id,
            episode_id=episode_id,
            turn_index=0,
            role="user",
            text=task.prompt_text,
            created_ts=utc_ts(),
        )
        return EnvironmentSession(
            episode_id=episode_id,
            task=task,
            actor_id=actor_id,
            environment_name=self.name,
            policy_ref=policy_ref,
            observations=[first_observation],
            transitions=[],
            tool_calls=[],
            tool_results=[],
        )

    async def step(self, session: EnvironmentSession, action: Action) -> Transition:
        if session.done:
            raise RuntimeError("episode already completed")
        started_ts = utc_ts()
        is_correct = (action.final_text or "").strip() == (session.task.expected_answer or "").strip()
        reward = 1.0 if is_correct else 0.0
        next_observation = Observation(
            observation_id=make_id("obs"),
            task_id=session.task.task_id,
            episode_id=session.episode_id,
            turn_index=1,
            role="env",
            text="accepted",
            created_ts=utc_ts(),
        )
        transition = Transition(
            transition_id=make_id("tr"),
            task_id=session.task.task_id,
            episode_id=session.episode_id,
            step_index=0,
            observation=session.observations[-1],
            action=action,
            created_ts=started_ts,
            next_observation=next_observation,
            reward=reward,
            done=True,
            truncated=False,
            env_latency_ms=(utc_ts() - started_ts) * 1000.0,
        )
        session.transitions.append(transition)
        session.observations.append(next_observation)
        session.done = True
        session.reward = reward
        session.termination_reason = "exact_match"
        session.ended_ts = utc_ts()
        return transition

    def build_episode(self, session: EnvironmentSession) -> Episode:
        return Episode(
            episode_id=session.episode_id,
            task_id=session.task.task_id,
            prompt_id=session.task.prompt_id,
            actor_id=session.actor_id,
            environment_name=self.name,
            behavior_policy_version=session.policy_ref.policy_version,
            policy_ref=session.policy_ref,
            started_ts=session.started_ts,
            transitions=tuple(session.transitions),
            ended_ts=session.ended_ts,
            termination_reason=session.termination_reason,
            is_truncated=session.truncated,
            metadata={"expected_answer": session.task.expected_answer or ""},
        )


class ToolUseMultiTurnEnvironment:
    name = "tool_use_multi_turn"

    def __init__(self, tool_registry: ToolRegistry, *, max_steps: int = 4) -> None:
        self.tool_registry = tool_registry
        self.max_steps = max_steps

    async def start_episode(self, task: TaskSpec, actor_id: str, policy_ref: PolicyRef) -> EnvironmentSession:
        episode_id = make_id("ep")
        first_observation = Observation(
            observation_id=make_id("obs"),
            task_id=task.task_id,
            episode_id=episode_id,
            turn_index=0,
            role="user",
            text=task.prompt_text,
            created_ts=utc_ts(),
            available_tools=self.tool_registry.names(),
        )
        return EnvironmentSession(
            episode_id=episode_id,
            task=task,
            actor_id=actor_id,
            environment_name=self.name,
            policy_ref=policy_ref,
            observations=[first_observation],
            transitions=[],
            tool_calls=[],
            tool_results=[],
        )

    async def step(self, session: EnvironmentSession, action: Action) -> Transition:
        if session.done:
            raise RuntimeError("episode already completed")
        started_ts = utc_ts()
        step_index = len(session.transitions)
        reward = 0.0
        done = False
        truncated = False
        if action.action_type == "tool_call" and action.tool_call is not None:
            tool_result = await self.tool_registry.invoke(action.tool_call, timeout_s=1.0)
            session.tool_calls.append(action.tool_call)
            session.tool_results.append(tool_result)
            next_observation = Observation(
                observation_id=make_id("obs"),
                task_id=session.task.task_id,
                episode_id=session.episode_id,
                turn_index=step_index + 1,
                role="tool",
                text=tool_result.output_text,
                created_ts=utc_ts(),
                tool_result=tool_result,
                available_tools=self.tool_registry.names(),
            )
        elif action.action_type == "finish":
            reward = 1.0 if (action.final_text or "").strip() == (session.task.expected_answer or "").strip() else 0.0
            done = True
            next_observation = Observation(
                observation_id=make_id("obs"),
                task_id=session.task.task_id,
                episode_id=session.episode_id,
                turn_index=step_index + 1,
                role="env",
                text="final answer accepted",
                created_ts=utc_ts(),
            )
        else:
            next_observation = Observation(
                observation_id=make_id("obs"),
                task_id=session.task.task_id,
                episode_id=session.episode_id,
                turn_index=step_index + 1,
                role="env",
                text="invalid action",
                created_ts=utc_ts(),
                available_tools=self.tool_registry.names(),
            )

        if step_index + 1 >= self.max_steps and not done:
            truncated = True
            done = True
            session.truncated = True
            session.termination_reason = "max_steps"

        transition = Transition(
            transition_id=make_id("tr"),
            task_id=session.task.task_id,
            episode_id=session.episode_id,
            step_index=step_index,
            observation=session.observations[-1],
            action=action,
            created_ts=started_ts,
            next_observation=next_observation,
            reward=reward,
            done=done,
            truncated=truncated,
            env_latency_ms=(utc_ts() - started_ts) * 1000.0,
        )
        session.transitions.append(transition)
        session.observations.append(next_observation)
        session.done = done
        if done:
            session.reward = reward
            session.termination_reason = session.termination_reason or "finish"
            session.ended_ts = utc_ts()
        return transition

    def build_episode(self, session: EnvironmentSession) -> Episode:
        return Episode(
            episode_id=session.episode_id,
            task_id=session.task.task_id,
            prompt_id=session.task.prompt_id,
            actor_id=session.actor_id,
            environment_name=self.name,
            behavior_policy_version=session.policy_ref.policy_version,
            policy_ref=session.policy_ref,
            started_ts=session.started_ts,
            transitions=tuple(session.transitions),
            ended_ts=session.ended_ts,
            termination_reason=session.termination_reason,
            is_truncated=session.truncated,
            metadata={"expected_answer": session.task.expected_answer or ""},
        )


class DelayedVerifierEnvironment:
    name = "delayed_verifier"

    def __init__(self, inner_environment: SingleTurnExactMatchEnvironment | ToolUseMultiTurnEnvironment) -> None:
        self.inner_environment = inner_environment

    async def start_episode(self, task: TaskSpec, actor_id: str, policy_ref: PolicyRef) -> EnvironmentSession:
        return await self.inner_environment.start_episode(task, actor_id, policy_ref)

    async def step(self, session: EnvironmentSession, action: Action) -> Transition:
        transition = await self.inner_environment.step(session, action)
        if transition.done:
            session.reward = None
            session.reward_pending = True
            session.termination_reason = "pending_verifier"
            session.ended_ts = utc_ts()
            return Transition(
                transition_id=transition.transition_id,
                task_id=transition.task_id,
                episode_id=transition.episode_id,
                step_index=transition.step_index,
                observation=transition.observation,
                action=transition.action,
                created_ts=transition.created_ts,
                next_observation=transition.next_observation,
                reward=None,
                done=True,
                truncated=transition.truncated,
                env_latency_ms=transition.env_latency_ms,
                verifier_pending=True,
                metadata=transition.metadata,
            )
        return transition

    def build_episode(self, session: EnvironmentSession) -> Episode:
        return self.inner_environment.build_episode(session)
