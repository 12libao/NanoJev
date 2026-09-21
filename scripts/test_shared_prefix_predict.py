#!/usr/bin/env python3
"""CLI parity and accounting test for the shared-prefix inference path.

`DecisionPredictor` requires CUDA, so the inference entry point cannot be exercised
on a CPU-only checkout. This test builds the same call sequence the predictor uses
(`prepare_examples` -> `DecisionModel.forward(prefix_sharing=...)` ->
`answer_from_probabilities`) against a tiny CPU backbone, so the wiring that turns
encoder output into published probabilities is covered without a GPU.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_tokenizer_root():
    """Find a local Qwen3-0.6B tokenizer without network access.

    Honours `NANOJEV_TOKENIZER_DIR`, then the in-repo checkpoint directory, then the
    same path in a sibling checkout so that a linked git worktree still finds the
    tokenizer that only the primary checkout has.
    """
    import os
    relative = Path("checkpoints") / "Qwen3-0.6B"
    # Walk up from the checkout so that a linked worktree under `.worktrees/<name>`
    # still finds the tokenizer held by the primary checkout.
    ancestors = [REPO_ROOT, *list(REPO_ROOT.parents)[:3]]
    candidates = [os.environ.get("NANOJEV_TOKENIZER_DIR")]
    candidates += [directory / relative for directory in ancestors]
    for candidate in candidates:
        if candidate and (Path(candidate) / "tokenizer.json").is_file():
            return Path(candidate)
    return None


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PredictorParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import AutoConfig, AutoModel
        except ImportError as error:  # pragma: no cover
            raise unittest.SkipTest(f"torch/transformers unavailable: {error}")
        cls.torch = torch
        cls.predict = _load("shared_prefix_predict", "predict_toy_decisions.py")
        cls.shared = _load("shared_prefix_shared", "shared_prefix.py")
        cls.trainer = _load("shared_prefix_trainer", "train_toy_decisions.py")

        from transformers import AutoTokenizer
        source = resolve_tokenizer_root()
        if source is None:
            raise unittest.SkipTest("Local Qwen3-0.6B tokenizer absent; set NANOJEV_TOKENIZER_DIR")
        cls.tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
        if cls.tokenizer.pad_token_id is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

        # The tokenizer's `vocab_size` excludes added special tokens; the checkpoint's
        # own config carries the padded embedding size that actually covers EOS.
        reference = AutoConfig.from_pretrained(str(source), local_files_only=True)
        config = AutoConfig.for_model(
            "qwen3", hidden_size=64, intermediate_size=128, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=reference.vocab_size, max_position_embeddings=1024,
            tie_word_embeddings=True)
        assert cls.tokenizer.eos_token_id < config.vocab_size
        config._attn_implementation = "eager"
        torch.manual_seed(31)
        cls.model = cls.trainer.DecisionModel(AutoModel.from_config(config).eval(),
                                              "attention").eval()

    def _payload(self, states=2, candidates=6):
        return {"states": [
            {"id": f"s{index}",
             "state": ("Local map: A is north of the exit, B is a wall, C is open. "
                       "The agent stands in a corridor with two untried exits."),
             "questions": {
                 "action": {"type": "choice", "instructions": "Which room should the agent enter?",
                            "criteria": {f"room_{i}": f"enter room {i} whose north side is "
                                                      f"{'open' if i % 2 else 'blocked'}"
                                                      for i in range(candidates)}},
                 "safe": {"type": "boolean", "instructions": "Is the chosen room safe?",
                          "criteria": {"false": "The room contains a wall.",
                                       "true": "The room is inside the maze and open."}}}}
            for index in range(states)]}

    def _run(self, payload, prefix_sharing, batch_questions=0):
        """Replicate `DecisionPredictor.predict` without the CUDA-only constructor."""
        examples = self.predict.prepare_examples(payload, self.tokenizer, 2048)
        batches = [examples] if prefix_sharing else self.predict.complete_question_batches(
            examples, batch_questions)
        sharing = None
        if prefix_sharing:
            sharing = self.shared.SharedPrefixEncoder(self.model.backbone,
                                                      pad_token_id=self.tokenizer.pad_token_id)
        probabilities = {}
        passes = 0
        with self.torch.inference_mode():
            for batch in batches:
                logits, _ = self.model(batch, self.tokenizer.pad_token_id, prefix_sharing=sharing)
                passes += 1
                for example, values in zip(batch, logits):
                    k = len(example["candidate_ids"])
                    scores = values[:k].float()
                    self.assertTrue(self.torch.isfinite(scores).all())
                    answer = self.predict.answer_from_probabilities(
                        example, scores.softmax(-1).cpu().tolist())
                    probabilities[(example["state_id"], example["qid"])] = answer
        return probabilities, passes

    def test_published_probabilities_match_the_reference_path(self):
        payload = self._payload()
        reference, reference_passes = self._run(payload, prefix_sharing=False)
        shared, shared_passes = self._run(payload, prefix_sharing=True)
        self.assertEqual(set(reference), set(shared))
        self.assertEqual(len(reference), 4)
        worst = 0.0
        for key, answer in reference.items():
            other = shared[key]
            self.assertEqual(answer["type"], other["type"])
            self.assertEqual(answer["value"], other["value"])
            self.assertEqual(set(answer["probabilities"]), set(other["probabilities"]))
            for candidate, value in answer["probabilities"].items():
                worst = max(worst, abs(value - other["probabilities"][candidate]))
        self.assertLess(worst, 1e-4, f"published probability drift {worst}")
        # Unbatched, both paths issue a single backbone call; the saving is in the tokens
        # each call evaluates, not in the number of calls.
        self.assertEqual(reference_passes, 1)
        self.assertEqual(shared_passes, 1)

    def test_reference_path_honours_batch_questions_but_shared_path_does_not(self):
        payload = self._payload()
        reference, reference_passes = self._run(payload, prefix_sharing=False,
                                                batch_questions=1)
        shared, shared_passes = self._run(payload, prefix_sharing=True, batch_questions=1)
        self.assertEqual(reference_passes, 4)
        # Every question owns an independent prefix cache, so a question limit no longer
        # partitions the backbone work.
        self.assertEqual(shared_passes, 1)
        self.assertEqual(set(reference), set(shared))
        for key, answer in reference.items():
            self.assertEqual(answer["value"], shared[key]["value"])

    def test_boolean_and_choice_share_one_forward_pass(self):
        payload = self._payload(states=1, candidates=4)
        probabilities, passes = self._run(payload, prefix_sharing=True)
        self.assertEqual(passes, 1)
        kinds = {key[1] for key in probabilities}
        self.assertEqual(kinds, {"action", "safe"})
        boolean = probabilities[("s0", "safe")]
        self.assertEqual(boolean["type"], "boolean")
        self.assertAlmostEqual(sum(boolean["probabilities"].values()), 1.0, places=4)

    def test_shared_path_reports_structural_token_reduction(self):
        payload = self._payload(states=3, candidates=8)
        examples = self.predict.prepare_examples(payload, self.tokenizer, 2048)
        plan = self.shared.SharedPrefixPlan(examples)
        accounting = plan.accounting()
        self.assertTrue(accounting["fully_shared"])
        self.assertGreater(accounting["token_reduction"], 2.0)
        # Repeating one action set across states is an exact reuse, so the plan holds
        # fewer groups than questions and pays for each unique prefix once.
        self.assertEqual(accounting["questions"], len(plan.groups))
        self.assertEqual(accounting["reused_questions"], len(plan.covered_by))
        self.assertEqual(accounting["covered_examples"], len(examples))
        self.assertEqual(accounting["shared_prefix_tokens"], sum(
            group.prefix_length for group in plan.groups))

    def test_batch_questions_does_not_change_shared_results(self):
        payload = self._payload()
        limited, limited_passes = self._run(payload, prefix_sharing=True, batch_questions=1)
        unbatched, unbatched_passes = self._run(payload, prefix_sharing=True, batch_questions=0)
        self.assertEqual(set(limited), set(unbatched))
        self.assertEqual((limited_passes, unbatched_passes), (1, 1))
        for key, answer in limited.items():
            self.assertEqual(answer["value"], unbatched[key]["value"])
            for candidate, value in answer["probabilities"].items():
                self.assertAlmostEqual(value, unbatched[key]["probabilities"][candidate], places=6)


if __name__ == "__main__":
    unittest.main()
