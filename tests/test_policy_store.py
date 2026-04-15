from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from async_rl_lab.policy_store import LocalPolicyStore, tokenize_text


class PolicyStoreTests(unittest.TestCase):
    def test_train_on_turns_respects_token_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_store = LocalPolicyStore(root, run_id="run-test")
            sequence = '{"type":"finish","answer":"4"}'
            tokens = tokenize_text(sequence)
            mask = tuple(1 if token == "4" else 0 for token in tokens)

            update = policy_store.train_on_turns(
                policy_version=policy_store.current_policy().policy_version,
                turn_texts=((sequence,),),
                turn_training_masks=((mask,),),
                turn_action_types=(("finish",),),
                advantages=(1.0,),
                sequence_weights=(1.0,),
                learning_rate=0.1,
            )

            self.assertGreater(update.token_count, 0)
            self.assertIn("4", update.updated_state.token_logits)
            self.assertNotIn("answer", update.updated_state.token_logits)
            self.assertEqual(update.updated_state.metadata["last_update_mode"], "turn_masked")
