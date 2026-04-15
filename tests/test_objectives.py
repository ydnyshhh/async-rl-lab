from __future__ import annotations

from dataclasses import replace
import unittest

from async_rl_lab.buffer import InMemoryGroupedRolloutBuffer
from async_rl_lab.objectives import DAPOObjective, GRPOObjective, RescoredBatch, StalenessWeightedGRPO
from tests.test_buffer import make_trajectory


class ObjectiveTests(unittest.TestCase):
    def test_group_centered_advantages_sum_to_zero(self) -> None:
        buffer = InMemoryGroupedRolloutBuffer(capacity_groups=2, required_group_size=4)
        group = tuple(
            replace(
                make_trajectory("group-a", index, 0),
                terminal_reward=float(index),
                per_step_rewards=(float(index),),
            )
            for index in range(4)
        )
        buffer.insert_group(group)
        batch = buffer.sample_groups(max_groups=1, learner_policy_version=0)
        prepared = GRPOObjective().prepare_batch(batch)
        self.assertAlmostEqual(sum(prepared.advantages), 0.0, places=6)

    def test_staleness_weight_drops_for_older_policy(self) -> None:
        buffer = InMemoryGroupedRolloutBuffer(capacity_groups=2, required_group_size=2)
        buffer.insert_group((make_trajectory("group-a", 0, 0), make_trajectory("group-a", 1, 0)))
        batch = buffer.sample_groups(max_groups=1, learner_policy_version=3)
        prepared = StalenessWeightedGRPO(lag_penalty=0.5).prepare_batch(batch)
        self.assertTrue(all(weight < 1.0 for weight in prepared.sequence_weights))

    def test_dapo_uses_token_level_ratio_clipping(self) -> None:
        buffer = InMemoryGroupedRolloutBuffer(capacity_groups=2, required_group_size=2)
        buffer.insert_group(
            (
                replace(
                    make_trajectory("group-a", 0, 0),
                    terminal_reward=1.0,
                    behavior_token_logprobs=((0.0, 0.0),),
                ),
                replace(
                    make_trajectory("group-a", 1, 0, reward=0.0, answer_text="5"),
                    terminal_reward=0.0,
                    behavior_token_logprobs=((0.0, 0.0),),
                ),
            )
        )
        batch = buffer.sample_groups(max_groups=1, learner_policy_version=0)
        objective = DAPOObjective(clip_ratio=0.2)
        prepared = objective.prepare_batch(batch)
        rescored = RescoredBatch(
            current_policy_version=1,
            reference_policy_version=0,
            current_token_logprobs=((1.0, 1.0), (-0.2, -0.2)),
            current_sequence_logprobs=(2.0, -0.4),
            reference_token_logprobs=((0.0, 0.0), (0.0, 0.0)),
            reference_sequence_logprobs=(0.0, 0.0),
        )

        result = objective.compute_loss(prepared, rescored)

        self.assertGreater(result.metrics["clip_fraction"], 0.0)
        self.assertEqual(len(result.update_advantages), 2)
