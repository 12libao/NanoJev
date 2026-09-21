#!/usr/bin/env python3
"""Tests for shared-prefix candidate encoding.

Two layers:

* Pure-logic tests that assert the token accounting, the grouping rules and the
  attention-mask invariants without importing PyTorch. These run everywhere.
* Numerical tests that build a tiny Qwen3 backbone on CPU and assert that the
  shared-prefix forward reproduces the reference forward. The equivalence claim
  is a statement about attention inputs, so it is checked in float32 where the
  only expected deviation is softmax reduction order.

Run: `python3 -m pytest scripts/test_shared_prefix.py -q`
"""
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shared_prefix import (  # noqa: E402
    BLOCKED_MASK_SENTINEL, CacheAdapter, SharedPrefixEncoder, SharedPrefixPlan,
    build_suffix_attention_mask, encode_candidate_suffixes, encode_prefix_parts,
    last_hidden_state, shared_prefix_length,
)
from decision_encoding import (  # noqa: E402
    BOOLEAN_IDS, candidate_path_tokens, candidate_suffix_text, question_candidate_texts,
    question_prefix_segments,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_checkpoint_root():
    """Find a local Qwen3-0.6B directory without network access.

    Honours `NANOJEV_CHECKPOINT_DIR`, then walks up from the checkout so a linked git
    worktree under `.worktrees/<name>` still finds the checkpoint that only the primary
    checkout holds. Returns `None` when nothing local is available.
    """
    import os
    relative = Path("checkpoints") / "Qwen3-0.6B"
    candidates = [os.environ.get("NANOJEV_CHECKPOINT_DIR")]
    candidates += [directory / relative for directory in [REPO_ROOT, *list(REPO_ROOT.parents)[:3]]]
    for candidate in candidates:
        if candidate and (Path(candidate) / "tokenizer.json").is_file():
            return Path(candidate)
    return None


class CharacterTokenizer:
    """Deterministic reversible tokenizer: one token per character."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("These tests never request special tokens")
        return [ord(char) for char in text]


def prepared(example_id, state, question):
    """Build one prepared-question dict through the canonical encoder."""
    tokenizer = CharacterTokenizer()
    ids, texts, prefix, paths = candidate_path_tokens(
        tokenizer.encode, state, question, tokenizer.eos_token_id)
    # `candidate_path_tokens` already returns one path for Boolean and one per
    # candidate otherwise, matching `prepare_examples` and `load_examples`.
    return {"id": example_id, "type": question["type"], "candidate_ids": ids,
            "candidate_texts": texts, "prefix_tokens": prefix, "prefix_length": len(prefix),
            "leaf_tokens": paths}


CHOICE = {"type": "choice", "instructions": "Pick one.",
          "criteria": {"a": "alpha", "b": "beta", "c": "gamma"}}


class PrefixPartTests(unittest.TestCase):
    def test_segments_concatenate_in_order(self):
        prefix, lengths = encode_prefix_parts(["ab", "cde"], CharacterTokenizer().encode)
        self.assertEqual(prefix, [ord(c) for c in "abcde"])
        self.assertEqual(lengths, [2, 3])

    def test_segments_reject_empty_input(self):
        for bad in ([], (), [""], ["ok", ""], "not-a-list"):
            with self.assertRaises(ValueError):
                encode_prefix_parts(bad, CharacterTokenizer().encode)

    def test_suffixes_append_eos_and_keep_template(self):
        suffixes = encode_candidate_suffixes(["x", "yy"], CharacterTokenizer().encode, 0)
        for text, suffix in zip(("x", "yy"), suffixes):
            self.assertEqual(suffix[-1], 0)
            self.assertEqual(bytes(suffix[:-1]).decode(), candidate_suffix_text(text))

    def test_suffixes_reject_bad_arguments(self):
        enc = CharacterTokenizer().encode
        with self.assertRaises(ValueError):
            encode_candidate_suffixes([], enc, 0)
        with self.assertRaises(ValueError):
            encode_candidate_suffixes(["ok"], enc, -1)
        with self.assertRaises(ValueError):
            encode_candidate_suffixes(["ok"], enc, "0")

    def test_shared_prefix_length(self):
        self.assertEqual(shared_prefix_length([[1, 2, 3], [1, 2, 9]]), 2)
        self.assertEqual(shared_prefix_length([[1, 2], [1, 2]]), 2)
        self.assertEqual(shared_prefix_length([[1], [2]]), 0)
        with self.assertRaises(ValueError):
            shared_prefix_length([])
        with self.assertRaises(ValueError):
            shared_prefix_length([[1], []])


class EncodingContractTests(unittest.TestCase):
    def test_boolean_has_one_semantic_path(self):
        ids, texts = question_candidate_texts({"type": "boolean", "instructions": "Is it true?",
                                               "criteria": {"false": "no", "true": "yes"}})
        self.assertEqual(ids, list(BOOLEAN_IDS))
        self.assertEqual(texts, ["The proposition is true."])

    def test_boolean_criteria_are_ordered_false_then_true(self):
        question = {"type": "boolean", "instructions": "Is it true?",
                    "criteria": {"true": "yes", "false": "no"}}
        segment = question_prefix_segments("s", question)[1]
        self.assertLess(segment.index("False criterion"), segment.index("True criterion"))

    def test_score_levels_never_receive_an_injected_ordinal(self):
        ids, texts = question_candidate_texts({"type": "score", "instructions": "Rate.",
                                               "criteria": ["low", "mid", "high"]})
        self.assertEqual(ids, ["0", "1", "2"])
        self.assertEqual(texts, ["low", "mid", "high"])
        self.assertNotIn("0", texts[0])

    def test_choice_keeps_caller_order_and_both_fields(self):
        question = {"type": "choice", "instructions": "Pick.",
                    "criteria": {"z": "last", "a": "first"}}
        ids, texts = question_candidate_texts(question)
        self.assertEqual(ids, ["z", "a"])
        self.assertEqual(texts, ["z: last", "a: first"])

    def test_state_object_serialization_matches_reference(self):
        question = {"type": "choice", "instructions": "Pick.", "criteria": {"a": "x"}}
        _, _, _, paths = candidate_path_tokens(CharacterTokenizer().encode,
                                               {"k": [1, 2]}, question, 0)
        expected = f"State:\n{{'k': [1, 2]}}\n"
        self.assertEqual(bytes(paths[0][:len(expected)]).decode(), expected)

    def test_prefix_segments_reject_bad_questions(self):
        for question in ({"type": "nope", "instructions": "x"},
                         {"type": "choice", "instructions": ""},
                         {"type": "choice"},
                         {"type": "choice", "instructions": "x", "criteria": {}},
                         {"type": "choice", "instructions": "x", "criteria": []},
                         {"type": "boolean", "instructions": "x", "criteria": {"maybe": "y"}}):
            with self.assertRaises(ValueError):
                question_prefix_segments("s", question)


class PlanTests(unittest.TestCase):
    def test_every_candidate_path_reconstructs_from_prefix_and_suffix(self):
        """The exact decomposition property the encoder relies on."""
        example = prepared("e1", "state text", CHOICE)
        group = SharedPrefixPlan([example]).groups[0]
        self.assertEqual(len(group.suffix_tokens), len(example["leaf_tokens"]))
        for path, suffix in zip(example["leaf_tokens"], group.suffix_tokens):
            self.assertEqual(path, group.prefix_tokens + suffix)

    def test_planned_prefix_covers_the_textual_prefix(self):
        """The real common run is at least the state+question text, often longer."""
        example = prepared("e1", "state text", CHOICE)
        group = SharedPrefixPlan([example]).groups[0]
        textual = example["prefix_tokens"]
        self.assertGreaterEqual(group.prefix_length, len(textual))
        self.assertEqual(group.prefix_tokens[:len(textual)], textual)

    def test_declared_prefix_never_absorbs_candidate_text(self):
        """The declared state+question prefix bounds the split, so suffix text stays put."""
        example = prepared("e1", "state text", CHOICE)
        group = SharedPrefixPlan([example]).groups[0]
        self.assertEqual(group.prefix_length, example["prefix_length"])
        self.assertFalse(group.split_unverified)
        for suffix in group.suffix_tokens:
            self.assertEqual(bytes(suffix).decode().split("\n")[0], "Candidate:")

    def test_undeclared_prefix_shares_the_whole_common_run_and_says_so(self):
        """Without a declaration the planner shares as much as it can, flagged unverified."""
        declared = prepared("e1", "state text", CHOICE)
        example = dict(declared)
        del example["prefix_length"]
        plan = SharedPrefixPlan([example])
        group = plan.groups[0]
        # The extra shared run is exactly the candidate boilerplate prefix
        # "Candidate:\n", which all three candidates begin with.
        boilerplate = len("Candidate:\n")
        self.assertEqual(group.prefix_length, declared["prefix_length"] + boilerplate)
        self.assertTrue(group.split_unverified)
        self.assertFalse(declared.get("split_unverified", False))
        self.assertEqual(plan.accounting()["unverified_split_questions"], ["e1"])
        self.assertEqual(plan.accounting()["unverified_split_questions"], ["e1"])

    def test_accounting_counts_each_prefix_once(self):
        examples = [prepared(f"e{i}", f"state {i}", CHOICE) for i in range(3)]
        plan = SharedPrefixPlan(examples)
        prefix, suffix = plan.groups[0].prefix_length, plan.groups[0].suffix_width
        # The reference pads every path in the flat batch to one width, so its token
        # count is `width * candidates` and is NOT the raw leaf-token sum (the last
        # candidate of each question is one token shorter here).
        width = plan.reference_width()
        self.assertEqual(width, max(len(path) for ex in examples for path in ex["leaf_tokens"]))
        self.assertEqual(plan.accounting()["reference_leaf_tokens"], width * 9)
        self.assertGreater(plan.accounting()["reference_leaf_tokens"],
                           sum(len(path) for ex in examples for path in ex["leaf_tokens"]))
        self.assertEqual(plan.accounting()["shared_leaf_tokens"],
                         3 * prefix + sum(len(s) for g in plan.groups for s in g.suffix_tokens))
        self.assertEqual(plan.accounting()["candidates"], 9)
        self.assertEqual(plan.accounting()["shared_prefix_tokens"], 3 * prefix)
        self.assertEqual(plan.accounting()["reference_prefix_tokens"], 9 * prefix)
        self.assertEqual(plan.accounting()["reference_padded_width"], width)

    def test_accounting_pads_all_paths_to_one_width(self):
        """The reference measures padded width: a long candidate inflates every path."""
        criteria = {"a": "x", "b": "y" * 20}
        example = prepared("e1", "s", {"type": "choice", "instructions": "Pick.", "criteria": criteria})
        plan = SharedPrefixPlan([example])
        self.assertEqual(plan.accounting()["reference_leaf_tokens"],
                         2 * max(len(path) for path in example["leaf_tokens"]))
        group = plan.groups[0]
        self.assertEqual(plan.accounting()["shared_leaf_tokens"],
                         group.prefix_length + sum(len(s) for s in group.suffix_tokens))

    def test_reduction_grows_with_candidate_count(self):
        previous = 0.0
        for count in (2, 4, 8, 16):
            criteria = {f"c{i}": "same length text" for i in range(count)}
            example = prepared("e", "a reasonably long shared state " * 3,
                               {"type": "choice", "instructions": "Pick.", "criteria": criteria})
            reduction = SharedPrefixPlan([example]).accounting()["token_reduction"]
            self.assertGreater(reduction, previous)
            previous = reduction
        self.assertGreater(previous, 3.0)

    def test_fully_shared_reported_for_normal_questions(self):
        plan = SharedPrefixPlan([prepared("e1", "s", CHOICE)])
        self.assertTrue(plan.accounting()["fully_shared"])
        self.assertEqual(plan.accounting()["prefix_mismatch_examples"], [])

    def test_question_without_a_shared_prefix_is_reported_not_optimized(self):
        example = {"id": "odd", "type": "choice", "candidate_ids": ["a", "b"],
                   "leaf_tokens": [[1, 2, 3], [9, 2, 3]]}
        plan = SharedPrefixPlan([example])
        self.assertFalse(plan.accounting()["fully_shared"])
        self.assertEqual(plan.accounting()["prefix_mismatch_examples"], ["odd"])
        self.assertEqual(plan.groups, [])
        self.assertIsNone(plan.reference_width())
        self.assertEqual(plan.accounting()["token_reduction"], None)

    def test_identical_samples_are_reused_not_re_encoded(self):
        """Repeating one question is an exact cache hit, not an error."""
        example = prepared("e1", "s", CHOICE)
        duplicate = json.loads(json.dumps(example))
        duplicate["id"] = "e2"
        plan = SharedPrefixPlan([example, duplicate])
        self.assertEqual(len(plan.groups), 1)
        self.assertEqual(plan.reused_samples, ["e2"])
        self.assertEqual(plan.covered_by, {1: 0})
        self.assertTrue(plan.fully_shared())
        # The reused question adds no prefix work and no suffix work of its own.
        self.assertEqual(plan.accounting()["shared_prefix_tokens"], len(example["prefix_tokens"]))
        self.assertEqual(plan.accounting()["questions"], 1)
        self.assertEqual(plan.accounting()["reused_questions"], 1)

    def test_same_state_with_different_candidates_is_two_groups(self):
        first = prepared("e1", "s", CHOICE)
        second = prepared("e2", "s", {"type": "choice", "instructions": "Pick.",
                                      "criteria": {"a": "alpha", "b": "beta", "c": "delta"}})
        plan = SharedPrefixPlan([first, second])
        self.assertEqual(len(plan.groups), 2)
        # Both questions share the whole state+question text; the candidate keys are
        # part of the per-candidate suffix, so only the textual prefix is common.
        for group, example in zip(plan.groups, (first, second)):
            textual = example["prefix_length"]
            self.assertEqual(group.prefix_tokens[:textual], example["prefix_tokens"])
            self.assertGreaterEqual(group.prefix_length, textual)
            for path, suffix in zip(example["leaf_tokens"], group.suffix_tokens):
                self.assertEqual(path, group.prefix_tokens + suffix)

    def test_degenerate_question_is_left_ungrouped_not_optimized(self):
        """A candidate that is exactly the shared prefix has no final position to read."""
        example = {"id": "e", "type": "choice", "candidate_ids": ["a", "b"],
                   "leaf_tokens": [[1, 2, 3, 4], [1, 2, 3]], "prefix_length": 3}
        plan = SharedPrefixPlan([example])
        self.assertEqual(plan.groups, [])
        self.assertFalse(plan.accounting()["fully_shared"])
        self.assertIn("no final position to read", plan.ungrouped[0]["reason"])

    def test_declared_prefix_must_be_shared(self):
        example = {"id": "e", "type": "choice", "candidate_ids": ["a", "b"],
                   "leaf_tokens": [[1, 2, 3, 4], [9, 8, 7, 6]], "prefix_length": 2}
        plan = SharedPrefixPlan([example])
        self.assertEqual(plan.groups, [])
        self.assertIn("diverge inside the declared prefix", plan.ungrouped[0]["reason"])

    def test_declared_prefix_longer_than_shortest_path_is_rejected(self):
        example = {"id": "e", "type": "choice", "candidate_ids": ["a", "b"],
                   "leaf_tokens": [[1, 2, 3], [1, 2, 3, 4]], "prefix_length": 9}
        with self.assertRaises(ValueError) as caught:
            SharedPrefixPlan([example])
        self.assertIn("exceeds the shortest candidate path", str(caught.exception))

    def test_malformed_examples_are_rejected(self):
        for bad in ({"id": "e", "candidate_ids": ["a"], "leaf_tokens": []},
                    {"id": "e", "candidate_ids": ["a", "b"], "leaf_tokens": [[1]]},
                    {"id": "e", "candidate_ids": ["a"], "leaf_tokens": [[]]},
                    {"id": "e", "candidate_ids": ["a"], "leaf_tokens": [[-1]]},
                    {"id": "e", "candidate_ids": ["a"], "leaf_tokens": [[1.5]]},
                    {"id": "e", "candidate_ids": [], "leaf_tokens": []}):
            with self.assertRaises(ValueError):
                SharedPrefixPlan([bad])
        with self.assertRaises(ValueError):
            SharedPrefixPlan([])

    def test_group_geometry_is_consistent(self):
        plan = SharedPrefixPlan([prepared("e1", "s", CHOICE)])
        group = plan.groups[0]
        self.assertEqual(group.path_width, group.prefix_length + group.suffix_width)
        self.assertEqual(group.reference_path_tokens(), group.candidate_count * group.path_width)
        self.assertEqual(group.shared_path_tokens(),
                         group.prefix_length + sum(len(s) for s in group.suffix_tokens))
        self.assertLessEqual(group.shared_path_tokens(), group.reference_path_tokens())
        self.assertGreaterEqual(group.prefix_length, 1)
        self.assertGreaterEqual(group.suffix_width, 2)


class MaskTests(unittest.TestCase):
    """The mask is the part most likely to be wrong, so it is asserted exactly."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError as error:  # pragma: no cover - torch is present in CI
            raise unittest.SkipTest(f"torch unavailable: {error}")

    def test_boolean_allow_mask_is_exact(self):
        import torch
        mask = build_suffix_attention_mask([2, 1], prefix_length=2, dtype=None)
        self.assertEqual(tuple(mask.shape), (2, 1, 2, 4))
        self.assertEqual(mask.dtype, torch.bool)
        expected = torch.tensor([
            [[[True, True, True, False], [True, True, True, True]]],
            [[[True, True, True, False], [True, True, True, True]]],
        ])
        self.assertTrue(torch.equal(mask, expected))

    def test_real_queries_never_attend_to_padding_keys(self):
        """The invariant that matters: padding keys cannot influence a real query."""
        import torch
        mask = build_suffix_attention_mask([3, 1], prefix_length=2, dtype=None)
        # Row 1 has one real suffix token at index 0; its padding is at indices 1 and 2.
        self.assertTrue(mask[1, 0, 0, -2].item() is False)
        self.assertTrue(mask[1, 0, 0, -1].item() is False)
        # Row 0 is full width, so every suffix key is real and causal.
        self.assertTrue(torch.equal(mask[0, 0, 2, 2:], torch.tensor([True, True, True])))

    def test_padding_queries_only_see_their_own_causal_run(self):
        import torch
        mask = build_suffix_attention_mask([3, 1], prefix_length=1, dtype=None)
        for query in range(3):
            self.assertTrue(mask[1, 0, query, 1 + query].item() is True)

    def test_blocked_positions_use_the_dtype_minimum(self):
        import torch
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            mask = build_suffix_attention_mask([2], prefix_length=1, dtype=dtype)
            self.assertEqual(mask.dtype, dtype)
            self.assertEqual(float(mask.max()), 0.0)
            self.assertEqual(float(mask.min()), float(torch.finfo(dtype).min))
            self.assertEqual(BLOCKED_MASK_SENTINEL, "dtype_minimum")

    def test_blocked_weight_is_exactly_zero_after_softmax(self):
        import torch
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            mask = build_suffix_attention_mask([2], prefix_length=1, dtype=dtype)
            weights = torch.softmax(mask[0, 0, 0].float(), dim=-1)
            self.assertEqual(float(weights[-1]), 0.0)

    def test_prefix_columns_are_all_visible(self):
        import torch
        mask = build_suffix_attention_mask([4, 4, 4], prefix_length=5, dtype=None)
        self.assertTrue(mask[:, :, :, :5].all().item())

    def test_mask_rejects_bad_arguments(self):
        import torch
        with self.assertRaises(ValueError):
            build_suffix_attention_mask([], 1, None)
        with self.assertRaises(ValueError):
            build_suffix_attention_mask([0], 1, None)
        with self.assertRaises(ValueError):
            build_suffix_attention_mask([1], -1, None)
        with self.assertRaises(ValueError):
            build_suffix_attention_mask([1], 1, torch.int64)


class TinyModelTests(unittest.TestCase):
    """Numerical equivalence on a tiny real Qwen3 backbone."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import AutoConfig, AutoModel
        except ImportError as error:  # pragma: no cover
            raise unittest.SkipTest(f"torch/transformers unavailable: {error}")
        cls.torch = torch
        config = AutoConfig.for_model(
            "qwen3", hidden_size=64, intermediate_size=128, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=256, max_position_embeddings=512, tie_word_embeddings=True)
        config._attn_implementation = "eager"
        torch.manual_seed(1234)
        cls.backbone = AutoModel.from_config(config).eval()
        cls.decision_class = _load_decision_model_class()
        cls.model = cls.decision_class(cls.backbone, "attention").eval()

    def setUp(self):
        self.torch.manual_seed(7)

    def _batch(self, count=2, candidates=4, state_tokens=12):
        examples = []
        for index in range(count):
            criteria = {f"c{i}": "candidate description %d" % i for i in range(candidates)}
            state = " ".join(["tok"] * state_tokens) + f" number {index}"
            examples.append(prepared(f"e{index}", state,
                                     {"type": "choice", "instructions": "Pick one.",
                                      "criteria": criteria}))
        return examples

    def _forward(self, examples, sharing):
        with self.torch.inference_mode():
            return self.model(examples, 0, prefix_sharing=sharing)[0]

    def test_shared_forward_matches_reference_float32(self):
        examples = self._batch()
        reference = self._forward(examples, None)
        encoder = self._encoder()
        shared = self._forward(examples, encoder)
        self.assertEqual(tuple(reference.shape), tuple(shared.shape))
        difference = (reference - shared).abs().max().item()
        self.assertLess(difference, 1e-4, f"shared-prefix drift too large: {difference}")

    def test_shared_forward_keeps_candidate_ranking(self):
        examples = self._batch(count=3, candidates=5)
        reference = self._forward(examples, None).softmax(-1)
        shared = self._forward(examples, self._encoder()).softmax(-1)
        for row in range(reference.shape[0]):
            self.assertEqual(int(reference[row].argmax()), int(shared[row].argmax()))

    def test_mixed_candidate_widths_and_counts(self):
        examples = self._batch(count=2, candidates=3)
        examples.append(prepared("e9", "another shared state",
                                 {"type": "choice", "instructions": "Pick one.",
                                  "criteria": {"only": "a much longer candidate description here"}}))
        reference = self._forward(examples, None)
        shared = self._forward(examples, self._encoder())
        self.assertLess((reference - shared).abs().max().item(), 1e-4)

    def test_chunked_suffix_encoding_matches_unchunked(self):
        examples = self._batch(count=1, candidates=7)
        reference = self._forward(examples, None)
        for chunk in (1, 2, 3, 7, 100):
            shared = self._forward(examples, self._encoder(suffix_chunk=chunk))
            self.assertLess((reference - shared).abs().max().item(), 1e-4,
                            f"suffix_chunk={chunk} diverged")

    def test_boolean_questions_use_only_the_true_path(self):
        example = prepared("b1", "state", {"type": "boolean", "instructions": "Is it true?",
                                           "criteria": {"false": "no", "true": "yes"}})
        self.assertEqual(len(example["candidate_ids"]), 2)
        self.assertEqual(len(example["leaf_tokens"]), 1)
        reference = self._forward([example], None)
        shared = self._forward([example], self._encoder())
        self.assertEqual(tuple(reference.shape), (1, 2))
        self.assertEqual(float(reference[0, 0]), 0.0)
        self.assertLess((reference - shared).abs().max().item(), 1e-4)

    def test_reference_forward_returns_logits_and_valid_mask(self):
        examples = self._batch(candidates=2)
        with self.torch.inference_mode():
            logits, valid = self.model(examples, 0)
        self.assertEqual(tuple(valid.shape), (2, 2))
        self.assertTrue(valid.all().item())
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_encoder_falls_back_when_a_question_has_no_shared_prefix(self):
        examples = self._batch(candidates=2)
        examples.append({"id": "broken", "type": "choice", "candidate_ids": ["a", "b"],
                         "leaf_tokens": [[1, 2, 3], [9, 8, 7]]})
        reference = self._forward(examples, None)
        shared = self._forward(examples, self._encoder())
        self.assertLess((reference - shared).abs().max().item(), 1e-4)

    def test_reference_path_is_unchanged_by_the_refactor(self):
        """`encode_leaves` + `decision_head` must equal the historical `forward`."""
        examples = self._batch(count=2, candidates=3)
        with self.torch.inference_mode():
            combined = self.model(examples, 0)[0]
            leaves, valid = self.model.encode_leaves(examples, 0)
            split = self.model.decision_head(examples, leaves, valid, return_logits=True)[0]
        self.assertTrue(self.torch.equal(combined, split))

    def test_pack_leaves_rejects_wrong_leaf_count(self):
        examples = self._batch(count=1, candidates=2)
        with self.assertRaises(ValueError):
            self.model.pack_leaves(examples, self.torch.zeros(1, 8))

    def test_pack_leaves_needs_one_leaf_per_choice_candidate(self):
        example = self._batch(count=1, candidates=3)[0]
        with self.assertRaises(ValueError) as caught:
            self.model.pack_leaves([example], self.torch.zeros(2, 8))
        self.assertIn("exactly once", str(caught.exception))

    def test_pack_leaves_fills_boolean_slots_from_one_path(self):
        example = prepared("b1", "state", {"type": "boolean", "instructions": "Is it true?",
                                           "criteria": {"false": "no", "true": "yes"}})
        self.assertEqual(len(example["leaf_tokens"]), 1)
        leaves = self.torch.arange(8, dtype=self.torch.float32).unsqueeze(0)
        h, valid = self.model.pack_leaves([example], leaves)
        self.assertEqual(tuple(h.shape), (1, 2, 8))
        self.assertTrue(valid.all().item())
        self.assertTrue(self.torch.equal(h[0, 0], h[0, 1]))

    def test_gradients_flow_through_the_shared_path(self):
        examples = self._batch(count=2, candidates=3)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(examples, 0, prefix_sharing=self._encoder())[0]
        logits[0, 0].backward()
        touched = [name for name, parameter in self.model.named_parameters()
                   if parameter.grad is not None and parameter.grad.abs().sum() > 0]
        self.assertTrue(any(name.startswith("backbone") for name in touched),
                        "backbone received no gradient through the shared prefix path")
        self.assertTrue(any(name.startswith("scalar") for name in touched))

    def _encoder(self, suffix_chunk=None):
        # The encoder drives the raw backbone; the decision head is applied by
        # DecisionModel.decision_head after the leaves are packed.
        return SharedPrefixEncoder(self.model.backbone, pad_token_id=0, suffix_chunk=suffix_chunk)


class CacheAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise unittest.SkipTest(f"torch unavailable: {error}")
        cls.torch = torch

    def _cache(self, batch=1, layers=2, heads=2, length=3, dim=4):
        class Layer:
            pass
        cache = type("Cache", (), {})()
        cache.layers = []
        for _ in range(layers):
            layer = Layer()
            layer.keys = self.torch.zeros(batch, heads, length, dim)
            layer.values = self.torch.zeros(batch, heads, length, dim)
            cache.layers.append(layer)
        return cache

    def test_broadcast_uses_zero_stride(self):
        adapter = CacheAdapter(self._cache())
        expanded = adapter.prefixed(5)
        for layer in expanded.layers:
            self.assertEqual(tuple(layer.keys.shape), (5, 2, 3, 4))
            self.assertEqual(layer.keys.stride(0), 0)
            self.assertEqual(layer.values.stride(0), 0)

    def test_broadcast_does_not_touch_the_source_cache(self):
        cache = self._cache()
        CacheAdapter(cache).prefixed(4)
        self.assertEqual(tuple(cache.layers[0].keys.shape), (1, 2, 3, 4))

    def test_single_row_broadcast_stays_a_view(self):
        """`expand` to one row is a no-op; it must still share storage, not copy."""
        adapter = CacheAdapter(self._cache())
        single = adapter.prefixed(1)
        self.assertEqual(single.layers[0].keys.data_ptr(),
                         adapter.pristine_tensors()[0][1].data_ptr())
        self.assertIsNot(single.layers[0], adapter._cache.layers[0])

    def test_broadcast_view_shares_storage_with_the_pristine_prefix(self):
        adapter = CacheAdapter(self._cache())
        expanded = adapter.prefixed(3)
        self.assertEqual(expanded.layers[0].keys.data_ptr(),
                         adapter.pristine_tensors()[0][1].data_ptr())

    def test_broadcast_never_hands_the_source_layers_to_the_model(self):
        """A forward that appends to its own cache must not reach the shared prefix."""
        cache = self._cache()
        adapter = CacheAdapter(cache)
        first = adapter.prefixed(2)
        self.assertIsNot(first.layers[0], cache.layers[0])
        # Simulate transformers appending the current step to the cache it received.
        first.layers[0].keys = self.torch.cat(
            [first.layers[0].keys, self.torch.ones(2, 2, 1, 4)], dim=-2)
        second = adapter.prefixed(3)
        self.assertEqual(tuple(second.layers[0].keys.shape), (3, 2, 3, 4))
        self.assertEqual(tuple(cache.layers[0].keys.shape), (1, 2, 3, 4))

    def test_repeated_broadcasts_are_identical(self):
        adapter = CacheAdapter(self._cache())
        first, second = adapter.prefixed(2), adapter.prefixed(5)
        self.assertEqual(tuple(first.layers[0].keys.shape), (2, 2, 3, 4))
        self.assertEqual(tuple(second.layers[0].keys.shape), (5, 2, 3, 4))
        self.assertEqual(first.layers[0].keys[0].data_ptr(),
                         second.layers[0].keys[0].data_ptr())

    def test_prefix_length_is_reported(self):
        self.assertEqual(CacheAdapter(self._cache(length=7)).prefix_length, 7)

    def test_multiple_layers_must_agree(self):
        cache = self._cache()
        cache.layers[1].keys = self.torch.zeros(1, 2, 9, 4)
        cache.layers[1].values = self.torch.zeros(1, 2, 9, 4)
        with self.assertRaises(ValueError):
            CacheAdapter(cache).prefixed(2)

    def test_batched_prefix_cache_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            CacheAdapter(self._cache(batch=4)).prefixed(2)
        self.assertIn("broadcast", str(caught.exception))

    def test_mismatched_key_value_shapes_are_rejected(self):
        cache = self._cache()
        cache.layers[0].values = self.torch.zeros(1, 2, 9, 4)
        with self.assertRaises(ValueError):
            CacheAdapter(cache).prefixed(2)

    def test_bad_row_count_is_rejected(self):
        adapter = CacheAdapter(self._cache())
        for bad in (0, -1, 1.5, "2"):
            with self.assertRaises(ValueError):
                adapter.prefixed(bad)

    def test_single_row_chunk_matches_multi_row_chunk(self):
        """The last partial chunk often has one row; it must still be exact."""
        adapter = CacheAdapter(self._cache())
        one, many = adapter.prefixed(1), adapter.prefixed(4)
        self.assertEqual(tuple(one.layers[0].keys.shape), (1, 2, 3, 4))
        self.assertTrue(self.torch.equal(one.layers[0].keys[0], many.layers[0].keys[0]))

    def test_cache_bytes_scales_with_rows(self):
        adapter = CacheAdapter(self._cache(length=3))
        self.assertEqual(adapter.batch_bytes(1) * 4, adapter.batch_bytes(4))

    def test_missing_past_key_values_is_rejected(self):
        with self.assertRaises(ValueError):
            CacheAdapter.from_forward_output(object())


class RealCheckpointTests(unittest.TestCase):
    """Optional end-to-end check against a local Qwen3-0.6B checkpoint."""

    @classmethod
    def setUpClass(cls):
        root = resolve_checkpoint_root()
        if root is None:
            raise unittest.SkipTest("No local Qwen3-0.6B tokenizer; set NANOJEV_CHECKPOINT_DIR")
        try:
            import torch
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as error:  # pragma: no cover
            raise unittest.SkipTest(f"torch/transformers/safetensors unavailable: {error}")
        cls.torch = torch
        cls.root = root
        cls.tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
        if cls.tokenizer.pad_token_id is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        backbone_root = root / "backbone_config" if (root / "backbone_config").is_dir() else root
        config = AutoConfig.from_pretrained(str(backbone_root), local_files_only=True)
        config.use_cache = True
        config._attn_implementation = "sdpa"
        model = AutoModel.from_config(config).float()
        # The full 28-layer Qwen3-0.6B weights anchor the claim; a directory that only
        # ships a tokenizer still exercises the encoder on random weights and is recorded
        # as such in `weights`.
        weights = None
        for name in ("best.safetensors", "model.safetensors"):
            if (root / name).is_file():
                weights = root / name
                break
        cls.weights = weights
        if weights is not None:
            model.load_state_dict(load_file(str(weights)), strict=False)
        cls.decision_class = _load_decision_model_class()
        cls.model = cls.decision_class(model.eval(), "attention").eval()

    def _examples(self, candidates=6, states=2, distinct_states=False):
        examples = []
        for index in range(states):
            criteria = {f"room_{i}": f"enter room {i} whose north side is "
                                     f"{'open' if i % 2 else 'blocked'}" for i in range(candidates)}
            marker = f" Map variant {index}." if distinct_states else ""
            examples.append(prepared(
                f"state{index}",
                "Local map: A is north of the exit, B is a wall, C is open. "
                "The agent stands in a corridor with two untried exits." + marker,
                {"type": "choice", "instructions": "Which room should the agent enter?",
                 "criteria": criteria}))
        return examples

    def test_distinct_states_each_pay_their_own_prefix(self):
        examples = self._examples(candidates=4, states=3, distinct_states=True)
        encoder = SharedPrefixEncoder(self.model.backbone,
                                      pad_token_id=self.tokenizer.pad_token_id)
        reference = self._predict(examples, None)
        shared = self._predict(examples, encoder)
        self.assertEqual(encoder.stats["questions"], 3)
        self.assertEqual(encoder.stats["reused_questions"], 0)
        self.assertEqual(encoder.stats["prefix_forwards"], 3)
        self.assertLess((reference - shared).abs().max().item(),
                        1e-2 if self.weights is None else 1e-3)

    def _predict(self, examples, sharing):
        with self.torch.inference_mode():
            return self.model(examples, self.tokenizer.pad_token_id, prefix_sharing=sharing)[0]

    def test_bf16_autocast_matches_under_matched_attention(self):
        """The serving precision is bf16 autocast; equivalence must hold there too."""
        torch = self.torch
        examples = self._examples(candidates=4, states=1)
        encoder = SharedPrefixEncoder(self.model.backbone,
                                      pad_token_id=self.tokenizer.pad_token_id)

        def run(sharing):
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                return self.model(examples, self.tokenizer.pad_token_id,
                                  prefix_sharing=sharing)[0]

        reference = run(None)
        shared = run(encoder)
        difference = (reference.float() - shared.float()).abs().max().item()
        scale = max(reference.float().abs().max().item(), 1e-6)
        self.assertLess(difference, 1e-2 * scale,
                        f"bf16 autocast drift {difference} relative to scale {scale}")
        # A broken mask moves logits by order 1 in this configuration, far above the
        # tolerance above, so this test fails loudly rather than silently passing.
        self.assertEqual(encoder.used_shared_prefix, True)

    def test_real_checkpoint_encoding_parity(self):
        """The canonical encoder must reproduce `prepare_examples` token for token."""
        from decision_encoding import build_candidate_paths
        from predict_toy_decisions import prepare_examples
        state = ("Local map: A is north of the exit, B is a wall, C is open. "
                 "The agent stands in a corridor with two untried exits.")
        questions = {
            "choice": {"type": "choice", "instructions": "Which room should the agent enter?",
                       "criteria": {f"room_{i}": f"enter room {i} whose north side is "
                                                 f"{'open' if i % 2 else 'blocked'}"
                                                 for i in range(4)}},
            "boolean": {"type": "boolean", "instructions": "Is the chosen room safe?",
                        "criteria": {"false": "The room contains a wall.",
                                     "true": "The room is inside the maze and open."}},
            "score": {"type": "score", "instructions": "How safe is the chosen room?",
                      "criteria": ["blocked", "risky", "clear", "wide open"]},
        }
        payload = {"states": [{"id": "state0", "state": state, "questions": questions}]}
        reference = {ex["qid"]: ex for ex in prepare_examples(payload, self.tokenizer, 2048)}
        self.assertEqual(set(reference), set(questions))
        for qid, question in questions.items():
            ids, texts, prefix, leaves = build_candidate_paths(
                self.tokenizer.encode, state, question, self.tokenizer.eos_token_id)
            self.assertEqual(ids, reference[qid]["candidate_ids"], qid)
            self.assertEqual(texts, reference[qid]["candidate_texts"], qid)
            self.assertEqual(leaves, reference[qid]["leaf_tokens"], qid)
            self.assertEqual(len(prefix), reference[qid]["prefix_length"], qid)

    def test_real_model_equivalence_and_reduction(self):
        examples = self._examples()
        encoder = SharedPrefixEncoder(self.model.backbone,
                                      pad_token_id=self.tokenizer.pad_token_id)
        reference = self._predict(examples, None)
        shared = self._predict(examples, encoder)
        difference = (reference - shared).abs().max().item()
        tolerance = 1e-3 if self.weights is not None else 1e-2
        self.assertLess(difference, tolerance,
                        f"checkpoint drift too large ({self.weights}): {difference}")
        plan = SharedPrefixPlan(examples)
        accounting = plan.accounting()
        self.assertGreater(accounting["token_reduction"], 1.5)
        self.assertTrue(accounting["fully_shared"])
        # Two states with the same candidate set are byte-identical questions, so one
        # prefix encode serves both; only distinct questions pay their own prefix.
        self.assertEqual(encoder.stats["questions"], len(examples) - encoder.stats["reused_questions"])
        self.assertEqual(encoder.stats["prefix_forwards"], encoder.stats["questions"])
        self.assertEqual(encoder.stats["reused_questions"], 1)


def _load_decision_model_class():
    path = Path(__file__).with_name("train_toy_decisions.py")
    spec = importlib.util.spec_from_file_location("shared_prefix_test_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DecisionModel


if __name__ == "__main__":
    unittest.main()
