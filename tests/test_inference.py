from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from async_rl_lab.ids import make_id, utc_ts
from async_rl_lab.inference import HFInferenceEngine, MockInferenceEngine
from async_rl_lab.models import GenerationRequest
from async_rl_lab.policy_store import LocalPolicyStore


class InferenceEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_engine_serves_requested_policy_version_and_records_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_store = LocalPolicyStore(root, run_id="run-test")
            initial_policy = policy_store.current_policy()
            updated = policy_store.train_on_sequences(
                policy_version=initial_policy.policy_version,
                sequences=['{"type":"finish","answer":"4"}'],
                advantages=(1.0,),
                sequence_weights=(1.0,),
                learning_rate=0.1,
            )
            policy_v1 = await policy_store.publish_policy(
                checkpoint_step=1,
                policy_tag="v1",
                state=updated.updated_state,
            )
            engine = MockInferenceEngine(policy_v1, policy_store=policy_store, batch_window_ms=1.0, artificial_latency_ms=1.0)
            request = GenerationRequest(
                request_id=make_id("req"),
                task_id="task-0",
                prompt_id="prompt-0",
                group_id="group-0",
                sample_index_within_group=0,
                actor_id="actor-0",
                policy=policy_v1,
                observation_text="What is 2 + 2?",
                available_tools=(),
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.95,
                created_ts=utc_ts(),
                metadata={"gold_answer": "4"},
            )

            result = await engine.submit(request)

            self.assertEqual(result.policy.policy_version, policy_v1.policy_version)
            self.assertEqual(result.metadata["served_policy_version"], policy_v1.policy_version)
            metrics = engine.metrics_snapshot()
            self.assertGreater(metrics.counters.get("inference.requests", 0.0), 0.0)
            self.assertTrue(metrics.histograms.get("inference.batch_size"))

    async def test_engine_partitions_mixed_policy_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_store = LocalPolicyStore(root, run_id="run-test")
            policy_v0 = policy_store.current_policy()
            update_v1 = policy_store.train_on_sequences(
                policy_version=policy_v0.policy_version,
                sequences=['{"type":"finish","answer":"4"}'],
                advantages=(1.0,),
                sequence_weights=(1.0,),
                learning_rate=0.1,
            )
            policy_v1 = await policy_store.publish_policy(
                checkpoint_step=1,
                policy_tag="v1",
                state=update_v1.updated_state,
            )
            update_v2 = policy_store.train_on_sequences(
                policy_version=policy_v1.policy_version,
                sequences=['{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"2+2"}}'],
                advantages=(1.0,),
                sequence_weights=(1.0,),
                learning_rate=0.1,
            )
            policy_v2 = await policy_store.publish_policy(
                checkpoint_step=2,
                policy_tag="v2",
                state=update_v2.updated_state,
            )

            engine = MockInferenceEngine(policy_v2, policy_store=policy_store, batch_window_ms=1.0, artificial_latency_ms=1.0)
            request_v1 = GenerationRequest(
                request_id=make_id("req"),
                task_id="task-0",
                prompt_id="prompt-0",
                group_id="group-0",
                sample_index_within_group=0,
                actor_id="actor-0",
                policy=policy_v1,
                observation_text="What is 2 + 2?",
                available_tools=("calculator",),
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.95,
                created_ts=utc_ts(),
                metadata={"gold_answer": "4", "expression": "2 + 2"},
            )
            request_v2 = GenerationRequest(
                request_id=make_id("req"),
                task_id="task-1",
                prompt_id="prompt-1",
                group_id="group-1",
                sample_index_within_group=0,
                actor_id="actor-0",
                policy=policy_v2,
                observation_text="What is 2 + 2?",
                available_tools=("calculator",),
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.95,
                created_ts=utc_ts(),
                metadata={"gold_answer": "4", "expression": "2 + 2"},
            )

            result_v1, result_v2 = await engine.submit_many((request_v1, request_v2))

            self.assertEqual(result_v1.metadata["served_policy_version"], policy_v1.policy_version)
            self.assertEqual(result_v2.metadata["served_policy_version"], policy_v2.policy_version)

    async def test_hf_engine_queue_and_metrics_without_external_model(self) -> None:
        class FakeHFInferenceEngine(HFInferenceEngine):
            async def ensure_policy_loaded(self, policy):
                self.loaded_policy = policy
                return 0.0

            def generate_batch_sync(self, prompt_texts, batch):
                return [
                    {
                        "generated_text": '{"type":"finish","answer":"4"}',
                        "prompt_tokens": 4,
                        "completion_tokens": 3,
                        "token_logprobs": (),
                    }
                    for _ in batch
                ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_store = LocalPolicyStore(root, run_id="run-test")
            policy = policy_store.current_policy()
            engine = FakeHFInferenceEngine(policy)
            request = GenerationRequest(
                request_id=make_id("req"),
                task_id="task-0",
                prompt_id="prompt-0",
                group_id="group-0",
                sample_index_within_group=0,
                actor_id="actor-0",
                policy=policy,
                observation_text="What is 2 + 2?",
                available_tools=(),
                max_new_tokens=16,
                temperature=0.0,
                top_p=1.0,
                created_ts=utc_ts(),
            )

            result = await engine.submit(request)
            await engine.close()
            restarted_result = await engine.submit(
                GenerationRequest(
                    request_id=make_id("req"),
                    task_id="task-1",
                    prompt_id="prompt-1",
                    group_id="group-1",
                    sample_index_within_group=0,
                    actor_id="actor-0",
                    policy=policy,
                    observation_text="What is 3 + 3?",
                    available_tools=(),
                    max_new_tokens=16,
                    temperature=0.0,
                    top_p=1.0,
                    created_ts=utc_ts(),
                )
            )
            await engine.close()

            self.assertEqual(result.metadata["backend"], "hf")
            self.assertEqual(result.generated_text, '{"type":"finish","answer":"4"}')
            self.assertEqual(restarted_result.metadata["backend"], "hf")
            self.assertGreater(engine.metrics_snapshot().counters.get("inference.requests", 0.0), 0.0)
