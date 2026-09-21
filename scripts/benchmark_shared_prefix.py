#!/usr/bin/env python3
"""Measure what shared-prefix encoding actually saves, on a local checkpoint.

Two independent quantities are reported:

* **Structural token accounting** — computed from the token layout alone, so it is
  hardware independent and is the headline number.
* **Wall clock and peak accelerator memory** — measured, therefore machine specific.
  The two do not track each other one-for-one: the suffix pass still attends over
  the cached prefix, so attention cost grows with prefix length even though the
  prefix is no longer re-encoded.

Usage:

    python3 scripts/benchmark_shared_prefix.py --checkpoint-dir checkpoints/Qwen3-0.6B \
        --output results/shared_prefix_benchmark.json --device cpu

The command loads a local checkpoint only, performs no optimization and makes no
network requests.
"""
import argparse
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path

DEFAULT_CANDIDATES = (4, 16, 64, 255)


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_criteria(count):
    return {f"room_{i}": f"enter room {i} whose north side is "
                         f"{'open' if i % 2 else 'blocked'} and whose east side is "
                         f"{'clear' if i % 3 else 'walled'}" for i in range(count)}


def build_payload(candidates, states):
    criteria = candidate_criteria(candidates)
    return {"states": [
        {"id": f"state{index}",
         "state": ("Local map: A is north of the exit, B is a wall, C is open. The agent "
                   "stands in a corridor with two untried exits and remembers three "
                   "blocked edges."),
         "questions": {"action": {"type": "choice",
                                  "instructions": "Which room should the agent enter next?",
                                  "criteria": criteria}}}
        for index in range(states)]}


def timed(fn, repeats):
    fn()  # warm up
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), min(samples)


def autocast_context(torch, device, precision):
    """bf16 means autocast over float32 parameters, exactly as the service runs it."""
    if precision == "bf16" and device.type in {"cpu", "cuda", "mps"}:
        return torch.autocast(device.type, dtype=torch.bfloat16)
    import contextlib
    return contextlib.nullcontext()


def measure(module, model, tokenizer, device, candidates, states, repeats, max_length,
            precision, suffix_chunk):
    examples = module.prepare_examples(build_payload(candidates, states), tokenizer, max_length)
    plan = module.SharedPrefixPlan(examples)
    accounting = plan.accounting()
    if not accounting["fully_shared"]:
        raise RuntimeError("Benchmark payload must have a shareable prefix for every question")

    import torch

    def reference():
        with torch.inference_mode(), autocast_context(torch, device, precision):
            return model(examples, tokenizer.pad_token_id)[0]

    def shared():
        encoder = module.SharedPrefixEncoder(model.backbone, pad_token_id=tokenizer.pad_token_id,
                                             device=device, suffix_chunk=suffix_chunk)
        with torch.inference_mode(), autocast_context(torch, device, precision):
            return model(examples, tokenizer.pad_token_id, prefix_sharing=encoder)[0]

    reference_median, reference_best = timed(reference, repeats)
    shared_median, shared_best = timed(shared, repeats)
    with __import__("torch").inference_mode():
        drift = (reference() - shared()).abs().max().item()
    return {
        "candidates": candidates,
        "states": states,
        "questions": len(examples),
        "prefix_length": plan.groups[0].prefix_length,
        "suffix_width": plan.groups[0].suffix_width,
        "reference_leaf_tokens": accounting["reference_leaf_tokens"],
        "shared_leaf_tokens": accounting["shared_leaf_tokens"],
        "token_reduction": accounting["token_reduction"],
        "prefix_fraction_before": accounting["prefix_fraction_before"],
        "prefix_fraction_after": accounting["prefix_fraction_after"],
        "reference_seconds_median": round(reference_median, 6),
        "shared_seconds_median": round(shared_median, 6),
        "reference_seconds_best": round(reference_best, 6),
        "shared_seconds_best": round(shared_best, 6),
        "wall_clock_speedup_median": round(reference_median / shared_median, 4),
        "max_abs_logit_drift": round(drift, 8),
        "repeats": repeats,
        "suffix_chunk": suffix_chunk,
    }


def peak_memory_bytes(device):
    import torch
    if device.type == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--candidates", default=",".join(map(str, DEFAULT_CANDIDATES)))
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--suffix-chunk", type=int,
                        help="Cap suffix rows per forward pass; default processes all candidates")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    trainer = load_module("shared_prefix_benchmark_trainer", "train_toy_decisions.py")
    module = load_module("shared_prefix_benchmark", "shared_prefix.py")
    predictor = load_module("shared_prefix_benchmark_predict", "predict_toy_decisions.py")
    module.prepare_examples = predictor.prepare_examples

    root = Path(args.checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone_root = root / "backbone_config" if (root / "backbone_config").is_dir() else root
    config = AutoConfig.from_pretrained(str(backbone_root), local_files_only=True)
    config.use_cache = True
    config._attn_implementation = "sdpa"
    backbone = AutoModel.from_config(config)
    weights = root / "best.safetensors"
    weights_only = weights.is_file()
    if weights_only:
        backbone.load_state_dict(load_file(str(weights)), strict=False)
    else:
        # A bare base-model directory has no decision head; the backbone weights alone
        # are enough to benchmark encoding, and the head stays at its initialization.
        base = root / "model.safetensors"
        if not base.is_file():
            raise SystemExit(f"No weights at {weights} or {base}")
        backbone.load_state_dict(load_file(str(base)), strict=False)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.precision]
    backbone = backbone.to(dtype).eval()
    device = torch.device(args.device)
    model = trainer.DecisionModel(backbone, "attention").to(device=device, dtype=dtype).eval()

    rows = []
    for candidates in [int(value) for value in args.candidates.split(",")]:
        states = 1 if candidates >= 128 else args.states
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        row = measure(module, model, tokenizer, device, candidates, states, args.repeats,
                      args.max_length, args.precision, args.suffix_chunk)
        row["peak_memory_bytes"] = peak_memory_bytes(device)
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = {
        "schema": "nanojev-shared-prefix-benchmark-v1",
        "device": str(device),
        "precision": args.precision,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "attention_implementation": config._attn_implementation,
        "checkpoint_dir": str(root),
        "note": ("Structural token accounting is exact and hardware independent. Timings and "
                 "peak memory are this machine only; the suffix pass still attends over the "
                 "cached prefix, so wall-clock gains are smaller than the token reduction."),
        "rows": rows,
    }
    print(json.dumps({key: report[key] for key in
                      ("schema", "device", "precision", "attention_implementation")}, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
