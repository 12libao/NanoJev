#!/usr/bin/env python3
"""One command that re-derives every shared-prefix claim on this machine.

The tests assert these facts; this script reproduces them and prints the numbers, so a
reviewer does not have to read test code to decide whether to believe them. It performs
no optimization, no training and no network access.

    python3 scripts/verify_shared_prefix.py
    python3 scripts/verify_shared_prefix.py --checkpoint-dir checkpoints/NanoJev-unified

Checks, in increasing order of strength:

1. **Ranking** — shared and reference paths agree on the argmax for every question.
2. **Magnitude** — the maximum absolute logit deviation, against a float32 tolerance.
3. **Oracle** — the refactored `forward` is bit-identical to a verbatim copy of the
   released `618cea6` body, so the default serving path did not move.
4. **Accounting** — the structural token counts and the reduction ratio, which are a
   function of the token layout and therefore hardware independent.

Set `NANOJEV_VERIFY_MUTATE` to inject a known defect and confirm these checks fail:
`leak`, `no_prefix`, `shifted_positions`.
"""
import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RELEASED_WEIGHTS_SHA256 = "f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_tokenizer():
    relative = Path("checkpoints") / "Qwen3-0.6B"
    roots = [os.environ.get("NANOJEV_CHECKPOINT_DIR")]
    roots += [d / relative for d in [HERE.parent, *list(HERE.parent.parents)[:3]]]
    for root in roots:
        if root and (Path(root) / "tokenizer.json").is_file():
            return Path(root)
    return None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tokenizer(AutoTokenizer, directory):
    """Load a tokenizer, normalizing one upstream export detail.

    transformers 4.51 wrote `extra_special_tokens` as a list; 4.57 expects a mapping and
    raises while reading it. The released `tokenizer.json` is byte-identical to the base
    model's, so only that metadata field is rewritten, in a temp copy of the config.
    """
    import json
    import shutil
    import tempfile
    config_path = Path(directory) / "tokenizer_config.json"
    if not isinstance(json.loads(config_path.read_text()).get("extra_special_tokens"), list):
        return AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    staging = Path(tempfile.mkdtemp(prefix="nanojev-verify-tokenizer-"))
    for item in Path(directory).iterdir():
        if item.is_file():
            shutil.copy2(item, staging / item.name)
    config = json.loads((staging / "tokenizer_config.json").read_text())
    config["extra_special_tokens"] = {}
    (staging / "tokenizer_config.json").write_text(json.dumps(config, indent=2))
    return AutoTokenizer.from_pretrained(str(staging), local_files_only=True)


def payload(candidates, states):
    criteria = {f"room_{i}": f"enter room {i} whose north side is "
                             f"{'open' if i % 2 else 'blocked'}" for i in range(candidates)}
    return {"states": [
        {"id": f"s{index}",
         "state": ("Local map: A is north of the exit, B is a wall, C is open. The agent "
                   "stands in a corridor with two untried exits."),
         "questions": {
             "action": {"type": "choice",
                        "instructions": "Which room should the agent enter?",
                        "criteria": criteria},
             "safe": {"type": "boolean", "instructions": "Is the chosen room safe?",
                      "criteria": {"false": "The room contains a wall.",
                                   "true": "The room is inside the maze and open."}}}}
        for index in range(states)]}


def released_forward(model, examples, pad_token, torch):
    """Verbatim body of `DecisionModel.forward` at commit 618cea6, as an oracle."""
    F = torch.nn.functional
    paths = [ids for ex in examples for ids in ex["leaf_tokens"]]
    device = model.scalar.weight.device
    lengths = torch.tensor([len(ids) for ids in paths], device=device)
    width = int(lengths.max())
    tokens = torch.full((len(paths), width), pad_token, dtype=torch.long, device=device)
    for i, ids in enumerate(paths):
        tokens[i, :len(ids)] = torch.tensor(ids, device=device)
    attention = torch.arange(width, device=device)[None, :] < lengths[:, None]
    hidden = model.backbone(input_ids=tokens, attention_mask=attention,
                            use_cache=False).last_hidden_state
    leaves = hidden[torch.arange(len(paths), device=device), lengths - 1]
    kmax = max(len(ex["candidate_ids"]) for ex in examples)
    h = leaves.new_zeros((len(examples), kmax, leaves.shape[-1]))
    valid = torch.zeros((len(examples), kmax), dtype=torch.bool, device=device)
    offset = 0
    for i, ex in enumerate(examples):
        n = len(ex["leaf_tokens"])
        h[i, :n] = leaves[offset:offset + n]
        valid[i, :len(ex["candidate_ids"])] = True
        offset += n
    h = model.norm(h)
    z = model.scalar(h).squeeze(-1).float()
    choice = torch.tensor([i for i, ex in enumerate(examples) if ex["type"] == "choice"],
                          device=device)
    if model.set_head == "attention" and len(choice):
        log_k = valid[choice].sum(-1).float().log()[:, None, None].expand(-1, kmax, 1)
        u = model.set_project(torch.cat([h[choice], log_k.to(h.dtype)], dim=-1))
        mixed, _ = model.set_attention(u, u, u, key_padding_mask=~valid[choice],
                                        need_weights=False)
        delta = model.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
        z = z.index_add(0, choice, delta)
    out = []
    for i, ex in enumerate(examples):
        if ex["type"] == "boolean":
            out.append(F.pad(torch.stack([z[i, 0] * 0, z[i, 0]]), (0, kmax - 2)))
        else:
            out.append(z[i])
    return torch.stack(out).masked_fill(~valid, -1e9), valid


def inject(mutation, shared_prefix):
    """Install one known defect so the checks can be shown to fail on it."""
    original_mask = shared_prefix.build_suffix_attention_mask
    original_stage = shared_prefix.SharedPrefixEncoder._stage_two_inputs
    import torch

    def finalize(allowed, dtype, device):
        if dtype is None:
            return allowed
        return torch.where(allowed, torch.zeros((), dtype=dtype, device=device),
                           torch.full((), float(torch.finfo(dtype).min), dtype=dtype,
                                      device=device))

    if mutation == "leak":
        def patched(lengths, prefix_length, dtype, device=None):
            allowed = original_mask(lengths, prefix_length, None, device)
            allowed[:, :, :, prefix_length:] = True
            return finalize(allowed, dtype, device)
    elif mutation == "no_prefix":
        def patched(lengths, prefix_length, dtype, device=None):
            allowed = original_mask(lengths, prefix_length, None, device)
            if prefix_length:
                allowed[:, :, :, :prefix_length] = False
            return finalize(allowed, dtype, device)
    elif mutation == "shifted_positions":
        def patched(self, group, rows):
            tokens, lengths, _, cache_position = original_stage(self, group, rows)
            width = tokens.shape[1]
            positions = torch.arange(width, device=self.device)[None, :]
            return tokens, lengths, positions.expand(len(rows), width).contiguous(), cache_position
    else:
        raise SystemExit(f"unknown mutation: {mutation}")
    if mutation == "shifted_positions":
        shared_prefix.SharedPrefixEncoder._stage_two_inputs = patched
    else:
        shared_prefix.build_suffix_attention_mask = patched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", help="Released checkpoint; defaults to local Qwen3-0.6B")
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args(argv)

    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as error:
        raise SystemExit(f"torch/transformers required: {error}")

    shared_prefix = load("verify_shared_prefix", "shared_prefix.py")
    predictor = load("verify_shared_prefix_predict", "predict_toy_decisions.py")
    trainer = load("verify_shared_prefix_trainer", "train_toy_decisions.py")

    released = None
    if args.checkpoint_dir:
        released = Path(args.checkpoint_dir)
        tokenizer_dir = released / "tokenizer"
        backbone_dir = released / "backbone_config"
        weights = released / "best.safetensors"
        for required in (tokenizer_dir, backbone_dir, weights):
            if not required.exists():
                raise SystemExit(f"missing {required}")
        digest = sha256(weights)
        if digest != RELEASED_WEIGHTS_SHA256:
            raise SystemExit(f"not the released weights: {digest}")
        print(f"checkpoint: released unified-games-v1 (sha256 verified)")
    else:
        root = find_tokenizer()
        if root is None:
            raise SystemExit("no local tokenizer; pass --checkpoint-dir or set NANOJEV_CHECKPOINT_DIR")
        released = root
        tokenizer_dir = root
        backbone_dir = root
        weights = None
        print(f"checkpoint: base Qwen3-0.6B, decision head at initialization ({root})")

    tokenizer = load_tokenizer(AutoTokenizer, tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(str(backbone_dir), local_files_only=True)
    config.use_cache = True
    config._attn_implementation = "sdpa"
    backbone = AutoModel.from_config(config).float()
    if weights is not None:
        from safetensors.torch import load_file
        model = trainer.DecisionModel(backbone, "attention")
        missing, unexpected = model.load_state_dict(load_file(str(weights)), strict=False)
        if missing or unexpected:
            raise SystemExit(f"strict load failed: missing={missing} unexpected={unexpected}")
        print("strict load: 0 missing, 0 unexpected tensors")
    else:
        torch.manual_seed(1234)
        model = trainer.DecisionModel(backbone, "attention")
    model = model.eval()

    if os.environ.get("NANOJEV_VERIFY_MUTATE"):
        mutation = os.environ["NANOJEV_VERIFY_MUTATE"]
        inject(mutation, shared_prefix)
        print(f"*** injected defect: {mutation} (checks are expected to FAIL) ***")

    examples = predictor.prepare_examples(payload(args.candidates, args.states), tokenizer, 4096)
    accounting = shared_prefix.SharedPrefixPlan(examples).accounting()
    encoder = shared_prefix.SharedPrefixEncoder(model.backbone,
                                                pad_token_id=tokenizer.pad_token_id,
                                                device=torch.device("cpu"))
    with torch.inference_mode():
        reference_logits, reference_valid = model(examples, tokenizer.pad_token_id)
        shared_logits, shared_valid = model(examples, tokenizer.pad_token_id,
                                            prefix_sharing=encoder)
        oracle_logits, oracle_valid = released_forward(model, examples,
                                                       tokenizer.pad_token_id, torch)

    drift = (reference_logits - shared_logits).abs().max().item()
    oracle_drift = (reference_logits - oracle_logits).abs().max().item()
    argmax_equal = torch.equal(reference_logits.argmax(-1), shared_logits.argmax(-1))
    valid_equal = torch.equal(reference_valid, shared_valid)
    oracle_equal = torch.equal(reference_logits, oracle_logits) and torch.equal(reference_valid,
                                                                                oracle_valid)
    worst_probability = 0.0
    for example, left, right in zip(examples, reference_logits, shared_logits):
        k = len(example["candidate_ids"])
        worst_probability = max(worst_probability,
                                (left[:k].float().softmax(-1)
                                 - right[:k].float().softmax(-1)).abs().max().item())

    print()
    print(f"questions                     {len(examples)}")
    print(f"candidates                    {accounting['candidates']} "
          f"({accounting['encoded_paths']} encoded paths)")
    print(f"leaf tokens reference -> shared  {accounting['reference_leaf_tokens']} -> "
          f"{accounting['shared_leaf_tokens']}  ({accounting['token_reduction']}x)")
    print(f"prefix forwards               {encoder.stats['prefix_forwards']} "
          f"(reused questions {encoder.stats['reused_questions']})")
    print(f"1. argmax identical           {argmax_equal}")
    print(f"2. max logit deviation        {drift:.3e}  (tolerance {args.tolerance:g})")
    print(f"   max probability deviation  {worst_probability:.3e}")
    print(f"   valid mask bitwise equal   {valid_equal}")
    print(f"3. oracle bitwise identical   {oracle_equal}  (drift {oracle_drift:.3e})")

    failures = []
    if not argmax_equal:
        failures.append("decisions changed")
    if drift > args.tolerance:
        failures.append(f"logit deviation {drift:.3e} exceeds {args.tolerance:g}")
    if not valid_equal:
        failures.append("valid mask changed")
    if not oracle_equal:
        failures.append("default path differs from the released implementation")
    if failures:
        print("\nFAIL: " + "; ".join(failures))
        return 1
    print("\nPASS: shared-prefix encoding reproduces the reference path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
