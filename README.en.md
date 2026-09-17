# NanoJev — A nano replica of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

**English** | [简体中文](README.zh-CN.md)

> Private development repository for NanoJev. The [release repository](https://github.com/TianyuCodings/NanoJev) remains separate.

## Current development

- **Larger mazes:** complete 8×8, 16×16, 32×32, and 50×50 boards, four maze topologies, multiple positions per map, and configurable larger sizes.
- **Atomic judgments + code planning:** matched 5×5 local training inputs, parallel directional judgments, and model-guided edge exploration with movement memory.
- **Snake:** reproducible food generation, body growth, collision and tail-movement rules, action choices, and parallel safety questions.
- **Calibrated rewards:** an implemented paired-sample policy-gradient objective, direct CE/Brier controls, exact gradient checks, and real Qwen3-0.6B training runs.
- **End-to-end evaluation:** frozen game cohorts, live [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) comparisons, map-separated datasets, and independently verified trajectory replay.

Start with [atomic planning](docs/ATOMIC_PLANNING.md), the [scaled-game pipeline](docs/SCALED_GAMES.md), [RLCD experiment](docs/RLCD_EXPERIMENT.md), [TypeSafe input contract](docs/TYPESAFE_CONTRACT.md), and [Fable review](docs/FABLE_REVIEW.md).

```bash
git clone https://github.com/TianyuCodings/NanoJev-dev.git
cd NanoJev-dev
```

### Probability-learning pilot

From the released NanoJev checkpoint, each training arm uses the same observed events and 100 updates. Probability error is `sum((p - q)^2)` against the simulator's exact event distribution; lower is better.

| Model / objective | Held-out test, 364 events | 50×50 OOD, 128 events |
|---|---:|---:|
| NanoJev starting checkpoint | 0.27708 | 0.31995 |
| Observed-outcome CE | 0.12423 | 0.07250 |
| Direct Brier | 0.13844 | 0.06719 |
| Paired proper reward | **0.11844** | **0.06202** |

This seed-17 pilot measures one-step events under a specified random actuator. The [experiment report](docs/RLCD_EXPERIMENT.md) includes event coverage, NLL/Brier, the separate three-seed algorithm benchmark, and the complete reward definition. The [game results](docs/DEVELOPMENT_RESULTS.md) use their own fixed closed-loop protocol.

### Local-judgment pilot

The new local model learns four directional safety questions from 300 maze snapshots, preserving map-separated splits. It processes a 5×5 observation at every board size.

| Local geometry questions | Test, 176 questions | 50×50 OOD, 64 questions |
|---|---:|---:|
| Constant true | 56.25% | 56.25% |
| **Locally trained NanoJev** | **77.84%** | **76.56%** |

On a separate identical 68-question route audit, starting NanoJev scores 64.71%, the locally trained model 75.00%, and [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 76.47%. See the [atomic training and planning report](docs/ATOMIC_PLANNING.md) and [model-guided exploration results](docs/MODEL_EDGE_RESULTS.md) for full probabilities, actual collisions, and completion measurements.

## Released baseline

**A 0.6B parallel decision model that turns states and questions into complete probability distributions.**

Give NanoJev multiple states, multiple questions, and dynamic candidate sets. It evaluates them in one batched backbone forward pass and returns structured decisions with **zero output-token decoding**.

[Model on Hugging Face](https://huggingface.co/C-Tianyu/NanoJev) · [Dataset on Hugging Face](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)

## See it in action

Watch **NanoJev**, **[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)**, and **untuned Qwen3-0.6B** navigate the same maps side by side.

The videos highlight four selected successful examples: two 4×4 test maps and two 6×6 OOD maps where NanoJev and [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) reach the goal while original Qwen does not. Both controller videos use the same four examples.

### Probability sampling

Choose each action from its full probability distribution at **T=1**.

[![NanoJev, Jev, and original Qwen: probability sampling](assets/comparison_sample.gif)](assets/comparison_sample.mp4)

[Watch the full MP4](assets/comparison_sample.mp4) · [Still preview](assets/comparison_sample.png)

### Greedy control

Choose the highest-probability action at each step.

[![NanoJev, Jev, and original Qwen: greedy control](assets/comparison_greedy.gif)](assets/comparison_greedy.mp4)

[Watch the full MP4](assets/comparison_greedy.mp4) · [Still preview](assets/comparison_greedy.png)

The animations replay actual model trajectories. Panels advance by environment step, display action probabilities, and hold their final state when an episode ends. Each GIF shows the complete first example, including its outcome. Each MP4 includes all four examples.

## Navigation results

**Controller: T=1 probability sampling.** Results cover the complete 40-map benchmark: 20 test maps and 20 OOD maps.

| System | 4×4 test | 6×6 OOD |
|---|---:|---:|
| **NanoJev** | **19/20 — 95%** | **18/20 — 90%** |
| [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 20/20 — 100% | 19/20 — 95% |
| Untuned Qwen3-0.6B | 7/20 — 35% | 3/20 — 15% |

Original Qwen is pretrained and has no task-specific fine-tuning. Its action probabilities come from its native language-model head, conditioned on the offered A–D answer tokens.

## Features

- **0.6B LLM backbone.** Built on Qwen3-0.6B with decision heads for structured outputs.
- **Multiple states and questions in one forward.** Batch independent decisions together.
- **Dynamic Choice.** Supply **2–255 candidates** per question and receive a probability for every candidate.
- **Boolean decisions.** Receive the probability that a proposition is true.
- **Ordered Score.** Supply **2–10 levels** and receive the full level distribution and expected score.
- **Complete distributions.** Use the same output for ranking, greedy selection, or probability sampling.
- **Zero output decoding.** Read decisions directly from a forward pass, without generating answer tokens.
- **Persistent serving.** Load a checkpoint once and reuse it across requests.

Measured in the running service: **6 states · 18 questions · 44 candidate paths · 1 backbone forward**.

## How it works

Each decision is defined by a **state**, a **question**, and its **candidate set**. Every candidate path carries the relevant input into the backbone. Shared decision heads return a distribution over the candidates supplied for that question.

Choice uses a shared scalar head and set attention. Boolean uses a single-path sigmoid. Score evaluates its ordered level descriptions and returns their probability-weighted expectation.

The implementation covers the full pipeline:

1. **Build queries.** Generate states, questions, candidate descriptions, and target distributions.
2. **Organize data.** Keep related maps, rules, and their variations in the same data split.
3. **Train the model.** Initialize Qwen3-0.6B, warm up the decision heads, and update the model using complete-question distribution losses.
4. **Evaluate decisions.** Measure probability quality and run closed-loop navigation with greedy and sampling controllers.
5. **Serve and visualize.** Expose a persistent model endpoint and replay actual trajectories in the browser.

[Complete pipeline commands](research/pipeline_runbook.md)

## Quick start: interactive replay

The replay viewer runs with Python's built-in HTTP server:

```bash
git clone https://github.com/TianyuCodings/NanoJev.git
cd NanoJev
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

Open **http://127.0.0.1:8080/comparison.html** to pause playback, switch examples, and inspect all three systems at the same environment step.

## Download and run the model

Prepare a CUDA environment with the recorded [Python dependencies](requirements-toy.txt). Sign in with an account that has access to the currently private model and dataset:

```bash
python -m pip install -r requirements-toy.txt
hf auth login
```

Download the final checkpoint and the dataset. The model file selection retrieves the root checkpoint only:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="C-Tianyu/NanoJev", local_dir="checkpoints/NanoJev",
    allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"],
)
snapshot_download(
    repo_id="C-Tianyu/NanoJev-Data", repo_type="dataset", local_dir="data/NanoJev",
)
```

Start the persistent service:

```bash
python scripts/serve_decisions.py \
  --checkpoint-dir checkpoints/NanoJev \
  --web-root web --port 8765
```

Open **http://127.0.0.1:8765**. The service loads the model once and accepts repeated batches through **`POST /api/evaluate`**.

The [pipeline runbook](research/pipeline_runbook.md) includes data generation, training, evaluation, checkpoint creation, and continuing from the downloaded model and data.

## Roadmap

- [x] **Scale up data** — Add larger mazes, Snake, atomic questions, and observed-event datasets.
- [x] **Calibrated reward prototype** — Implement and test paired proper-reward learning with CE/Brier controls.
- [ ] **RLCD expansion** — Add broader semantic tasks, stochastic long-horizon events, and additional model seeds.
- [ ] **Structured input support** — Version the encoder for structured instructions, criteria, and the native Noul interface.
