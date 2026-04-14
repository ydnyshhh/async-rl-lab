from __future__ import annotations

import unittest

from async_rl_lab.buffer import InMemoryGroupedRolloutBuffer
from async_rl_lab.objectives import GRPOObjective, StalenessWeightedGRPO
from tests.test_buffer import make_trajectory


class ObjectiveTests(unittest.TestCase):
    def test_group_centered_advantages_sum_to_zero(self) -> None:
        buffer = InMemoryGroupedRolloutBuffer(capacity_groups=2, required_group_size=4)
        group = tuple(make_trajectory("group-a", index, 0) for index in range(4))
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
