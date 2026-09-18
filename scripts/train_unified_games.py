#!/usr/bin/env python3
"""Warm-start one parallel DecisionModel on Maze, Snake and shooting decisions.

Policy rows supply complete API Choice distributions. Outcome rows supply actual
Boolean observations under one frozen continuation policy. No environment, API,
language generation, inferred counterfactual label, or policy improvement runs
inside this trainer. All model files are read from a local checkpoint bundle.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

from predict_toy_decisions import prepare_examples, read_json
from train_pipeline_decisions import (
    SPLITS, distribution_metrics, dump, pack_complete_questions,
    read_training_records, target_for, validate_training_row,
)

TASKS = ("maze", "snake", "shooting")
ROLES = ("policy", "outcome")
LOSSES = ("ce", "brier", "paired_brier_pg")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _action_identity(value):
    if value is None or isinstance(value, bool):
        raise ValueError("Action provenance must specify an executed action, not null/boolean")
    if not isinstance(value, (str, int, list, dict)) or value == "" or value == [] or value == {}:
        raise ValueError("Action provenance must be a nonempty action ID or JSON action")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_unified_records(records, manifest):
    """Validate provenance without treating API probabilities as physical outcomes."""
    if not isinstance(manifest, dict):
        raise ValueError("Dataset manifest must be a JSON object")
    policy_id = manifest.get("continuation_policy_id")
    episode_splits, counts, eligible = {}, Counter(), Counter()
    cell_episodes, state_labels = defaultdict(set), defaultdict(set)
    episode_labels, question_labels, missing_episode_ids = {}, Counter(), Counter()
    outcome_count = 0
    for row in records:
        targets = validate_training_row(row)
        meta = row.get("metadata")
        if not isinstance(meta, dict) or meta.get("task") not in TASKS or meta.get("record_role") not in ROLES:
            raise ValueError(f"{row['id']}: metadata.task and metadata.record_role must be declared")
        task, role = meta["task"], meta["record_role"]
        episode = meta.get("episode_id")
        if episode is not None:
            if not _nonempty_string(episode):
                raise ValueError("metadata.episode_id must be a nonempty string")
            key = (task, episode)
            if key in episode_splits and episode_splits[key] != row["split"]:
                raise ValueError("The same task/episode crosses dataset splits")
            episode_splits[key] = row["split"]
            cell_episodes[(row["split"], task, role)].add(episode)
        else:
            missing_episode_ids[(row["split"], task, role)] += 1
        if role == "outcome":
            if not _nonempty_string(policy_id) or meta.get("continuation_policy_id") != policy_id:
                raise ValueError("Every outcome row must match manifest.continuation_policy_id exactly")
            executed = _action_identity(meta.get("executed_action"))
            conditioned = meta.get("conditioned_action")
            # A single conditioned action covers all event questions on that action.
            # For structured physical actions use conditioned_action_by_question.
            by_question = meta.get("conditioned_action_by_question")
            if by_question is not None:
                if not isinstance(by_question, dict) or set(by_question) != set(row["questions"]):
                    raise ValueError("conditioned_action_by_question must cover exactly all outcome questions")
            elif conditioned is None:
                raise ValueError("Outcome rows must identify the conditioned and executed actions")
            for qid in row["questions"]:
                action = by_question[qid] if by_question is not None else conditioned
                if _action_identity(action) != executed:
                    raise ValueError("Observed outcome cannot label an unexecuted conditioned action")
        for qid, question in row["questions"].items():
            target = targets[qid]
            counts[(row["split"], task, role)] += 1
            if role == "policy":
                if question["type"] != "choice":
                    raise ValueError("Policy supervision requires dynamic Choice questions")
                if target["gold_label_kind"] == "observed_outcome":
                    raise ValueError("Policy preference labels cannot be declared observed event outcomes")
                usable = target["teacher_probs"] is not None
            else:
                outcome_count += 1
                if question["type"] != "boolean":
                    raise ValueError("Outcome supervision requires independent Boolean questions")
                if target["gold_label_kind"] != "observed_outcome" or target["gold_index"] is None:
                    raise ValueError("Outcome rows require actual Boolean gold with gold_label_kind=observed_outcome")
                if target["gold_probs"] is not None:
                    raise ValueError("Observed outcome supervision must not contain a substituted soft gold distribution")
                label = target["gold_index"]
                question_labels[(row["split"], task, label)] += 1
                state_labels[(row["split"], task, label)].add(row["state_id"])
                if episode is not None:
                    key = (row["split"], task, episode)
                    if key in episode_labels and episode_labels[key] != label:
                        raise ValueError("Conflicting eventual-success labels within the same task/episode")
                    episode_labels[key] = label
                usable = True
            if usable:
                eligible[(row["split"], task, role)] += 1
    event_cells = sorted({(split, task) for split, task, _ in question_labels})
    episode_label_counts = Counter((split, task, label) for (split, task, _), label in episode_labels.items())
    return {
        "records": len(records), "outcome_questions": outcome_count,
        "continuation_policy_id": policy_id if outcome_count else None,
        "questions_by_split_task_role": {"/".join(k): v for k, v in sorted(counts.items())},
        "eligible_by_split_task_role": {"/".join(k): v for k, v in sorted(eligible.items())},
        "quarantined_policy_questions": sum(counts[k] - eligible[k] for k in counts if k[2] == "policy"),
        "unique_episodes_by_split_task_role": {"/".join(k): len(cell_episodes[k]) for k in sorted(counts)},
        "rows_missing_episode_id_by_split_task_role": {"/".join(k): v for k, v in sorted(missing_episode_ids.items())},
        "outcome_episode_counts_by_split_task": {
            f"{split}/{task}": {"successes": episode_label_counts[(split, task, 1)],
                               "failures": episode_label_counts[(split, task, 0)]}
            for split, task in event_cells},
        "outcome_state_label_counts_by_split_task": {
            f"{split}/{task}": {"true": len(state_labels[(split, task, 1)]),
                               "false": len(state_labels[(split, task, 0)])}
            for split, task in event_cells},
        "outcome_question_label_counts_by_split_task": {
            f"{split}/{task}": {"true": question_labels[(split, task, 1)],
                               "false": question_labels[(split, task, 0)]}
            for split, task in event_cells},
        "episode_audit_note": "Repeated states/questions are not independent episodes. Single-outcome cells are retained without forced balancing. Missing episode IDs are counted explicitly and excluded from episode totals.",
    }


def read_unified_dataset(path):
    root = Path(path)
    if not root.is_dir():
        raise ValueError("--input must be a directory containing manifest.json and all five split JSONL files")
    required = [root / "manifest.json"] + [root / f"{split}.jsonl" for split in SPLITS]
    if any(not filename.is_file() for filename in required):
        raise ValueError("Dataset requires manifest.json plus train/dev/calibration/test/ood.jsonl")
    manifest = read_json(required[0])
    records, files = read_training_records(root)
    # The shared reader validates row splits and state/map leakage; also enforce file placement.
    for filename in files:
        for line in filename.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line)["split"] != filename.stem:
                raise ValueError(f"A row split disagrees with its file: {filename.name}")
    audit = validate_unified_records(records, manifest)
    return records, manifest, required, audit


def prepare_unified_examples(records, tokenizer, max_length):
    examples, audit = [], []
    for row in records:
        targets = validate_training_row(row)
        meta = row["metadata"]
        prepared = prepare_examples({"states": [{k: row[k] for k in ("id", "state", "questions")}]},
                                    tokenizer, max_length)
        for example in prepared:
            target = targets[example["qid"]]
            example.update(target, state_id=row["state_id"], family_id=row["family_id"],
                           split=row["split"], source=row, task=meta["task"], record_role=meta["record_role"],
                           continuation_policy_id=meta.get("continuation_policy_id"))
            objective = objective_for(example)
            examples.append(example)
            audit.append({"id": example["id"], "split": example["split"], "task": example["task"],
                          "record_role": example["record_role"], "type": example["type"],
                          "candidate_count": len(example["candidate_ids"]), "objective": objective,
                          "eligible": target_for(example, objective) is not None,
                          "max_path_tokens": max(map(len, example["leaf_tokens"])),
                          "continuation_policy_id": example["continuation_policy_id"],
                          "teacher_target_error": target["teacher_target_error"],
                          "target_transform": "identity_rounded_proxy" if objective == "teacher" else "observed_boolean_one_hot"})
    return examples, audit


def objective_for(example):
    role = example["record_role"]
    if role == "policy":
        if example["type"] != "choice":
            raise ValueError("Policy examples must be Choice")
        return "teacher"
    if role == "outcome":
        if example["type"] != "boolean":
            raise ValueError("Outcome examples must be Boolean")
        return "observed_outcome"
    raise ValueError("Unknown record_role")


def population_weights(stage, balance="task", retention_fraction=.25):
    if stage not in {"sft", "critic"} or balance not in {"task", "task_role"}:
        raise ValueError("Unknown stage or balancing mode")
    if not math.isfinite(retention_fraction) or not 0 < retention_fraction < 1:
        raise ValueError("retention_fraction must lie strictly between zero and one")
    if stage == "sft":
        return {(task, "policy"): 1 / len(TASKS) for task in TASKS}
    retention = .5 if balance == "task_role" else retention_fraction
    return {(task, role): (retention if role == "policy" else 1 - retention) / len(TASKS)
            for task in TASKS for role in ROLES}


class BalancedQuestionSampler:
    """Stratified sampling with replacement and exact task/role loss weights.

Every effective update contains each required cell. Weighting by population mass
divided by sampled cell count removes integer-rounding and corpus-size biases.
No held-out row enters a sampling pool, even when supplied by the caller.
"""
    def __init__(self, examples, stage, balance="task", retention_fraction=.25, seed=17):
        self.weights = population_weights(stage, balance, retention_fraction)
        self.rng = random.Random(seed)
        self.pools = {cell: [] for cell in self.weights}
        for ex in examples:
            cell = (ex["task"], ex["record_role"])
            if ex["split"] == "train" and cell in self.pools and target_for(ex, objective_for(ex)) is not None:
                self.pools[cell].append(ex)
        missing = ["/".join(cell) for cell, pool in self.pools.items() if not pool]
        if missing:
            raise ValueError("Missing eligible training cells: " + ", ".join(missing))

    def sample(self, batch_questions):
        if type(batch_questions) is not int or batch_questions < len(self.weights):
            raise ValueError(f"Effective batch must include every task/role cell: at least {len(self.weights)} questions")
        counts = {cell: 1 for cell in self.weights}
        while sum(counts.values()) < batch_questions:
            cells = list(self.weights)
            self.rng.shuffle(cells)
            chosen = max(cells, key=lambda cell: batch_questions * self.weights[cell] - counts[cell])
            counts[chosen] += 1
        batch = []
        for cell, count in counts.items():
            for _ in range(count):
                source = self.rng.choice(self.pools[cell])
                batch.append({**source, "loss_weight": self.weights[cell] / count})
        self.rng.shuffle(batch)
        return batch


def loss_spec(example, stage, critic_loss):
    if critic_loss not in LOSSES:
        raise ValueError("Unknown critic loss")
    objective = objective_for(example)
    if example["record_role"] == "policy":
        return objective, "ce"
    if stage != "critic":
        raise ValueError("Outcome rows cannot enter SFT optimization")
    return objective, critic_loss


def mixed_question_loss(logits, examples, stage, critic_loss, reward_samples=32, generator=None):
    """One weighted sum; microbatches must not divide by their own row counts."""
    import torch
    from calibrated_objectives import grouped_calibrated_loss
    if len(logits) != len(examples) or not examples:
        raise ValueError("Logits and nonempty complete question batch must align")
    losses = []
    for values, ex in zip(logits, examples):
        objective, kind = loss_spec(ex, stage, critic_loss)
        weight = ex.get("loss_weight")
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Every sampled question needs a finite positive population weight")
        loss = grouped_calibrated_loss(values.unsqueeze(0), [ex], objective, kind, reward_samples, generator)[0]
        losses.append(loss * weight)
    return torch.stack(losses).sum()


def summarize_predictions(rows, weights, require_all=False):
    """Deterministic metrics; policy matching and observed-event quality stay separate."""
    cells = defaultdict(list)
    excluded = Counter()
    for row in rows:
        cell = (row["task"], row["record_role"])
        target, logits = row.get("training_target"), row["student_logits"]
        if target is None:
            excluded[cell] += 1
            continue
        metrics = distribution_metrics(target, logits)
        top = max(range(len(logits)), key=logits.__getitem__)
        probs = row["student_probs"]
        metrics["brier"] = math.fsum((p - t) ** 2 for p, t in zip(probs, target))
        metrics["target_argmax_agreement"] = float(top == max(range(len(target)), key=target.__getitem__))
        if row["record_role"] == "outcome":
            metrics["observed_accuracy"] = float(top == row["gold_index"])
        cells[cell].append(metrics)
    by_cell = {}
    for cell in sorted(set(cells) | set(excluded)):
        values = cells[cell]
        bucket = {"questions": len(values), "excluded_questions": excluded[cell],
                  "objective": "api_policy_distribution" if cell[1] == "policy" else "observed_outcome"}
        if values:
            for key in values[0]:
                bucket[key] = math.fsum(value[key] for value in values) / len(values)
        by_cell["/".join(cell)] = bucket
    missing = [cell for cell in weights if not cells[cell]]
    if require_all and missing:
        raise ValueError("Dev selection lacks eligible task/role cells: " + ", ".join("/".join(c) for c in missing))
    selection = None if missing else math.fsum(weights[cell] * by_cell["/".join(cell)]["ce"] for cell in weights)
    by_role = {}
    for role in ROLES:
        buckets = [bucket for cell, bucket in by_cell.items() if cell.endswith("/" + role) and bucket["questions"]]
        if buckets:
            keys = [key for key in buckets[0] if key not in {"questions", "excluded_questions", "objective"}]
            by_role[role] = {"tasks_present": len(buckets), "questions": sum(b["questions"] for b in buckets),
                             **{key: math.fsum(b[key] for b in buckets) / len(buckets) for key in keys}}
    return {"questions": len(rows), "selection_ce": selection,
            "selection_weights": {"/".join(k): v for k, v in weights.items()},
            "missing_selection_cells": ["/".join(cell) for cell in missing],
            "by_task_role": by_cell, "by_role_macro_task": by_role,
            "notes": "Policy CE/TV/KL measure reference-distribution matching; observed Boolean CE/Brier measure event prediction. Binary Brier sums both classes."}


def evaluate_unified(model, examples, pad_token, args, weights, path=None, require_all=False):
    import torch
    from train_toy_decisions import prediction_record
    model.eval()
    rows = []
    with torch.inference_mode():
        for group in pack_complete_questions(examples, args.microbatch_questions, args.max_microbatch_tokens):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                logits, _ = model(group, pad_token)
            for ex, values in zip(group, logits):
                if not torch.isfinite(values[:len(ex["candidate_ids"])]).all():
                    raise RuntimeError("Nonfinite evaluation logits")
                row = prediction_record(ex, values)
                row.update(task=ex["task"], record_role=ex["record_role"],
                           continuation_policy_id=ex["continuation_policy_id"],
                           training_target=target_for(ex, objective_for(ex)),
                           target_objective=objective_for(ex))
                rows.append(row)
    if path:
        Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    return summarize_predictions(rows, weights, require_all)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--init-checkpoint", help="Complete local DecisionModel bundle; fresh optimizer")
    parser.add_argument("--output-dir")
    parser.add_argument("--stage", choices=["sft", "critic"], required=True)
    parser.add_argument("--loss", choices=LOSSES, help="SFT defaults to CE; critic defaults to paired_brier_pg; retention always CE")
    parser.add_argument("--balance", choices=["task", "task_role"], default="task")
    parser.add_argument("--retention-fraction", type=float, default=.25, help="Policy population mass in critic/task mode; task_role fixes it to .5")
    parser.add_argument("--reward-samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--head-steps", type=int, default=0)
    parser.add_argument("--batch-questions", type=int, default=12)
    parser.add_argument("--microbatch-questions", type=int, default=4)
    parser.add_argument("--max-microbatch-tokens", type=int, default=16384)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--head-warmup-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-native-triton", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Stdlib schema/provenance audit; no model or GPU")
    args = parser.parse_args(argv)
    args.loss = args.loss or ("ce" if args.stage == "sft" else "paired_brier_pg")
    if args.stage == "sft" and args.loss != "ce":
        parser.error("SFT policy distribution supervision uses CE; event losses belong to --stage critic")
    weights = population_weights(args.stage, args.balance, args.retention_fraction)
    if min(args.steps, args.microbatch_questions, args.max_length, args.eval_every) <= 0 or args.head_steps < 0 or args.max_microbatch_tokens < 0:
        parser.error("Steps, batch and token limits must be valid positive sizes")
    if args.batch_questions < len(weights) or args.reward_samples < 2:
        parser.error(f"--batch-questions must be >= {len(weights)} and --reward-samples >= 2")
    if any(not math.isfinite(v) or v <= 0 for v in (args.backbone_lr, args.head_lr, args.head_warmup_lr)):
        parser.error("Learning rates must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Weight decay must be finite and nonnegative")
    if not args.validate_only and (not args.init_checkpoint or not args.output_dir):
        parser.error("Training requires --init-checkpoint and --output-dir")
    return args


def main(argv=None):
    args = parse_args(argv)
    records, manifest, files, schema_audit = read_unified_dataset(args.input)
    weights = population_weights(args.stage, args.balance, args.retention_fraction)
    for split in ("train", "dev"):
        for task, role in weights:
            if not schema_audit["eligible_by_split_task_role"].get(f"{split}/{task}/{role}", 0):
                raise ValueError(f"Missing eligible {split}/{task}/{role} questions")
    if args.validate_only:
        print(json.dumps({**schema_audit, "stage": args.stage, "loss": args.loss,
                          "population_weights": {"/".join(k): v for k, v in weights.items()}}, ensure_ascii=False))
        return
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory; existing run artifacts are not overwritten")
    import torch
    from safetensors.torch import load_file, save_file
    from predict_toy_decisions import DecisionPredictor
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runtime = DecisionPredictor(args.init_checkpoint, max_length=args.max_length, precision=args.precision,
                                disable_native_triton=args.disable_native_triton)
    model, tokenizer = runtime.model, runtime.tokenizer
    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    examples, target_audit = prepare_unified_examples(records, tokenizer, args.max_length)
    splits = {split: [ex for ex in examples if ex["split"] == split] for split in SPLITS}
    sampler = BalancedQuestionSampler(examples, args.stage, args.balance, args.retention_fraction, args.seed)
    for split_examples in splits.values():
        pack_complete_questions(split_examples, args.microbatch_questions, args.max_microbatch_tokens)
    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "schema_version": "nanojev-unified-games-v1",
              "model": runtime.run_config.get("model"), "set_head": runtime.run_config["set_head"],
              "resolved_model_revision": runtime.run_config.get("resolved_model_revision"),
              "initialization": "local DecisionModel warm start; fresh optimizer",
              "init_weights_sha256": file_sha256(runtime.root / "best.safetensors"),
              "data_sha256": {str(path): file_sha256(path) for path in files},
              "implementation_sha256": file_sha256(__file__),
              "continuation_policy_id": schema_audit["continuation_policy_id"],
              "population_weights": {"/".join(k): v for k, v in weights.items()},
              "sampling": "stratified with replacement; per-cell exact population weights; train rows only",
              "parameter_storage": "float32", "forward_autocast": args.precision,
              "objective": "Choice API full-distribution CE plus observed Boolean event loss",
              "selection": "minimum fixed population-weighted dev CE including initial checkpoint; test only after selection",
              "temperature": 1.0, "temperature_fitted": False,
              "deps": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")},
              "gpu": torch.cuda.get_device_name(0), "schema_counts": schema_audit,
              "predictive_samples_are_physical_actions": False,
              "frozen_continuation_policy_is_updated_by_this_script": False}
    dump(out / "config.json", config)
    dump(out / "target_audit.json", target_audit)
    tokenizer.save_pretrained(out / "tokenizer")
    model.backbone.config.save_pretrained(out / "backbone_config")
    initial = evaluate_unified(model, splits["dev"], tokenizer.pad_token_id, args, weights,
                               out / "initial_dev.jsonl", require_all=True)
    dump(out / "initial_dev_metrics.json", initial)
    best, best_step = initial["selection_ce"], 0
    if not math.isfinite(best):
        raise RuntimeError("Initial dev metric is nonfinite")

    def save_best():
        save_file({k: value.detach().cpu().contiguous().clone() for k, value in model.state_dict().items()},
                  out / "best.safetensors")

    save_best()
    body = list(model.backbone.parameters())
    head = [param for name, param in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW([{"params": body, "lr": args.backbone_lr},
                                  {"params": head, "lr": args.head_lr}], weight_decay=args.weight_decay)
    reward_rng = torch.Generator(device="cuda").manual_seed(args.seed + 104729)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    logs = []
    for step in range(args.head_steps + args.steps):
        warm = step < args.head_steps
        for param in body:
            param.requires_grad_(not warm)
        optimizer.param_groups[1]["lr"] = args.head_warmup_lr if warm else args.head_lr
        batch = sampler.sample(args.batch_questions)
        groups = pack_complete_questions(batch, args.microbatch_questions, args.max_microbatch_tokens)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for group in groups:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                logits, _ = model(group, tokenizer.pad_token_id)
            loss = mixed_question_loss(logits, group, args.stage, args.loss, args.reward_samples, reward_rng)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            loss_sum += float(loss.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        cells = Counter((ex["task"], ex["record_role"]) for ex in batch)
        item = {"step": step + 1, "phase": "head" if warm else "full", "loss": loss_sum,
                "loss_is_score_function_surrogate": args.loss == "paired_brier_pg",
                "gradient_norm_before_clip": float(grad_norm), "microbatches": len(groups),
                "sample_counts": {"/".join(k): v for k, v in sorted(cells.items())},
                "batch_question_ids_sha256": hashlib.sha256("\n".join(ex["id"] for ex in batch).encode()).hexdigest(),
                "elapsed_seconds": time.perf_counter() - started}
        if not warm and ((step + 1 - args.head_steps) % args.eval_every == 0 or step + 1 == args.steps + args.head_steps):
            metrics = evaluate_unified(model, splits["dev"], tokenizer.pad_token_id, args, weights, require_all=True)
            item["dev"] = metrics
            if metrics["selection_ce"] < best:
                best, best_step = metrics["selection_ce"], step + 1
                save_best()
        logs.append(item)
        if step % 12 == 0 or "dev" in item:
            dump(out / "train_log.json", logs)
            print(json.dumps(item, allow_nan=False), flush=True)
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    max_memory = torch.cuda.max_memory_allocated() / 1e9
    # Loading the selected checkpoint does not alter the frozen dataset policy ID.
    model.load_state_dict(load_file(str(out / "best.safetensors"), device="cpu"), strict=True)
    final = {}
    for split in ("dev", "calibration", "test", "ood"):
        if splits[split]:
            final[split] = evaluate_unified(model, splits[split], tokenizer.pad_token_id, args, weights,
                                           out / f"predictions_{split}.jsonl", require_all=split == "dev")
    summary = {"best_step": best_step, "best_dev_selection_ce": best, "selected_on": "dev only",
               "stage": args.stage, "loss": args.loss, "completed_steps": args.steps + args.head_steps,
               "metrics_by_split": final, "training_seconds": training_seconds,
               "max_gpu_allocated_gb": max_memory, "weights_sha256": file_sha256(out / "best.safetensors"),
               "continuation_policy_id": schema_audit["continuation_policy_id"],
               "temperature": 1.0, "temperature_fitted": False}
    dump(out / "train_log.json", logs)
    dump(out / "summary.json", summary)
    print(json.dumps({"done": str(out), **summary}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
