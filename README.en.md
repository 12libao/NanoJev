# NanoJev

[中文 README](README.md)

**A parallel decision model prototype trained from Qwen3-0.6B: provide a state, questions, and dynamic candidate sets to receive candidate probabilities directly.** Multiple states, questions, and candidate paths are processed in one batched backbone forward pass, with no autoregressive output-token decoding.

The project includes query generation, distribution supervision, training, held-out evaluation, closed-loop games, model serving, and visualizations. It implements a Jev-style decision interface. The official RLCD training recipe remains unavailable to this project; the cross-entropy training implemented here is not presented as RLCD.

## Watch the actual runs

Compare **trained NanoJev, the Jev API, and original Qwen without task-specific fine-tuning** on the same maps. These animations replay recorded model trajectories. Each full video contains four examples fixed before the comparison: the first two 4×4 test maps and the first two 6×6 out-of-distribution (OOD) maps.

### Probability sampling: choose from the full action distribution

[![Trained NanoJev, Jev, and original Qwen: probability sampling](assets/comparison_sample.gif)](assets/comparison_sample.mp4)

[Watch the full MP4](assets/comparison_sample.mp4) · [Still preview](assets/comparison_sample.png)

### Greedy control: choose the most probable action

[![Trained NanoJev, Jev, and original Qwen: greedy control](assets/comparison_greedy.gif)](assets/comparison_greedy.mp4)

[Watch the full MP4](assets/comparison_greedy.mp4) · [Still preview](assets/comparison_greedy.png)

Each GIF shows the first 12 seconds; the MP4 retains all four examples, including loops and failures. Panels are synchronized by **environment step**, and completed panels hold their final state. The videos do not compare inference latency. The probability bars represent NanoJev action probabilities, normalized rounded Jev probabilities, and Qwen probabilities conditioned on the offered answer tokens, respectively.

Run the interactive replay locally without a model, GPU, or API:

```bash
git clone https://github.com/TianyuCodings/NanoJev.git
cd NanoJev
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

Open **http://127.0.0.1:8080/comparison.html** to pause playback or switch examples. The root page also retains historical V2/V3 comparisons, failures, and multi-state inference evidence. Recorded replay and live model serving are separate modes.

## Results on the same 40 maps

Each system runs both greedy control and sampling at T=1 on 20 held-out 4×4 maps and 20 6×6 OOD maps: **240 episodes** in total. There is no online oracle correction, visit penalty, or early termination of loops. Each episode has a maximum of `2 × size²` steps.

| System | Greedy: 4×4 | Greedy: 6×6 | Sampling: 4×4 | Sampling: 6×6 |
|---|---:|---:|---:|---:|
| **Trained NanoJev** | 18/20 (90%) | 10/20 (50%) | 19/20 (95%) | 18/20 (90%) |
| **Jev: measured API runs** | 20/20 (100%) | 16/20 (80%) | 20/20 (100%) | 19/20 (95%) |
| **Original Qwen3-0.6B** | 6/20 (30%) | 1/20 (5%) | 7/20 (35%) | 3/20 (15%) |

The original Qwen baseline is already pretrained but has no fine-tuning for this project. It uses its original language-model head to obtain probabilities conditioned on A–D answer tokens; it has neither random weights nor a randomly initialized decision head. Trained NanoJev uses a new decision head and a different input format. This table therefore compares three complete systems, rather than isolating training as the only changed factor.

The NanoJev checkpoint was fixed as `v3_teacher_coords_multi_seed17` before this comparison and was not selected again from its results. One training seed, one sampling seed, and 20 maps per size support conclusions about this toy benchmark only. Greedy performance on 6×6 maps remains substantially below Jev. The earlier programmatic-supervision branch, ablations, and non-navigation regression results are also retained.

- [Comparison protocol, fixed examples, and probability semantics](research/nanojev_comparison_protocol_zh.md)
- [Full comparison and interpretation](research/nanojev_comparison_zh.md) · [Metrics and public trajectories](research/nanojev_comparison_public.json) · [Independent verification of 240 episodes](research/nanojev_comparison_verification.json)
- [All V3 experiments and failures](research/navigation_v3_report_zh.md) · [Non-navigation regression evaluation](research/navigation_v3_regression_zh.md)
- [Video sources, frames, and file hashes](assets/comparison_media_manifest.json)

The detailed research reports linked here are currently in Chinese; machine-readable results are provided alongside them.

## How the model makes decisions

Every training example explicitly contains `(state, question, candidate_set, target_distribution)`. The question is part of the input. Candidates are supplied separately for each question, rather than drawn from a fixed classification vocabulary.

1. **Initialize from an LLM.** The backbone is `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`. There is no intermediate yes/no reranker training stage.
2. **Encode candidate paths.** Each path contains the state, question, and current candidate description. Paths are batched into one backbone forward pass. A shared scalar head and set attention produce the K logits for a Choice question; softmax produces its K-way distribution without adding a new classifier for each K. Set attention applies only to Choice. Score bypasses set attention, and Boolean uses one path with a sigmoid.
3. **Train complete distributions.** The loss is `L = -Σ qᵢ log pᵢ`, with reference distributions or independent programmatic ground truth as targets. Normalization covers the complete candidate set of each question; candidates are not treated as unrelated binary decisions.
4. **Return decisions in parallel.** Choice returns a dynamic candidate distribution, Boolean returns a probability, and Score returns a level distribution and its expectation. No output tokens are decoded. The environment still advances step by step because its next state depends on the chosen action.

The live service has been checked with **6 states / 18 questions / 44 candidate paths / 1 backbone forward**. The current interface allows 2–255 Choice candidates and 2–10 Score levels. Valid distributions have been checked at K=2/5/20/64/255; this does not establish semantic accuracy at large K. The current implementation repeats prefix encoding and has not demonstrated Jev-level throughput or shared-prefix optimization. [Six-state batch evidence](research/parallel_example_v3.json) · [Persistent service checks](research/live_service_check_v3.json) · [Dynamic candidate checks](research/dynamic_candidates_v2.json)

## Training and benchmark pipeline

**[Execution runbook: complete commands from data to a runnable model](research/pipeline_runbook_zh.md)** · [Algorithm decisions](research/implementation_plan_zh.md) · [Fable discussions and technical review](research/algorithm_fable_decisions_zh.md)

| Stage | Implementation and checks |
|---|---|
| Query distribution | Authored generators for rules, catalog lookup, probability events, tic-tac-toe, and navigation; environments are constructed before questions and legal candidates |
| Data separation | Train / dev / calibration / test / OOD; source maps, rule groups, and their variations retain their assigned split |
| Supervision | Reference distributions and independent programmatic targets remain distinct; original rounded outputs and derived distributions are recorded separately |
| Training | Qwen initialization, decision-head warmup, and full-parameter training; complete-question microbatches and dev-only checkpoint selection |
| Probability evaluation | KL/TV against reference distributions; NLL/Brier/ECE against independent truth; TV against known conditional distributions |
| Closed-loop evaluation | Identical map cohorts and controller rules, reporting completion, path efficiency, revisits, and all failures |
| Serving and visualization | Load a local checkpoint once for repeated requests; preserve actual trajectories for web and video replay |

Reference agreement and calibration against independent truth answer different questions. On V2's known-probability test questions, the programmatically supervised branch achieved TV `.0454 ± .0081`, versus `.3060 ± .0479` for the reference-supervised branch, across three training seeds each. Action-policy probabilities are not probabilities of eventually winning. [Full V2 results](research/pipeline_v2_report_zh.md)

The runbook provides an end-to-end **zero-API programmatic-supervision path** and a separate path using reference distributions. V2 contains 2,312 states / 6,936 questions. V3 adds 500 maps / 3,000 states / 9,000 questions. Each of the five V3 runs used one A100 80GB for 1,200 updates, taking approximately 8.3–10.0 minutes. All five started from the same V2 checkpoint. A programmatic target in the V3 stage does not imply that the checkpoint's entire training history used only programmatic targets.

Live inference requires a compatible checkpoint that you supply:

```bash
python scripts/serve_decisions.py \
  --checkpoint-dir runs/v3_teacher_coords_multi_seed17 \
  --web-root web --port 8765
```

Python/CUDA dependencies are recorded in [requirements-toy.txt](requirements-toy.txt). The optional `--disable-native-triton` flag is a compatibility fallback for the original experiment host.

This repository publishes source code, generators, evaluation evidence, and replays. **Trained weights and historical private training labels are not included in the Git repository.** You can reconstruct a new programmatic-supervision experiment. Reproducing the recorded reference-supervision experiments requires their frozen targets and checkpoints; public summaries do not replace those artifacts. [Release scope and reproduction limits](research/release_scope_zh.md)

## Verify or rerun the comparison

Verify the public trajectories without API calls or model training:

```bash
python3 scripts/verify_nanojev_comparison.py --public-only \
  --output /tmp/nanojev-public-verification.json
```

The verifier checks legal actions, state transitions, complete episode horizons, probability-based choices, sampling RNG, map cohorts, and Jev source-call references. Additional checks against private journals and original datasets were performed before release. The verifier output and protocol distinguish checks supported by the public files from those requiring private artifacts.

For original Qwen model caching, data construction, and GPU evaluation, see the [complete baseline commands](research/navigation_v3_native_qwen_zh.md). That entry point requires the explicitly cached model revision and a CUDA environment; cloning this repository alone does not provide them.

Collecting new Jev responses requires Node.js ≥22, the locked Node dependencies, and your own local API credentials:

```bash
npm ci
cp .env.example .env
# Set AI_GATEWAY_API_KEY in .env. Do not commit this file.

# First reconstruct data/native_rebuild_v3 using the baseline instructions.
python3 scripts/evaluate_live_jev_navigation.py \
  --data-dir data/native_rebuild_v3 \
  --journal-dir data/jev_rebuild_journal \
  --output-dir artifacts/jev_rebuild --budget-usd 2
```

The new Jev comparison made 145 successful requests covering 435 questions and cost **$0.003966648**. Repeated states reused exact-input responses from this comparison's journal; historical training labels were not used as live API results. Across all gateway experiments, recorded expenditure was **$0.138601428**, with **$24.861398572** remaining as of September 17, 2026. Claude CLI usage estimates and GPU resources are recorded separately. [Budget details](research/budget_summary.json)

## Research notes and license

[Official website, use cases, and public sources](research/jev_public_sources_zh.md) · [Public discussion and RLCD](research/rlcd_public_discussion_zh.md) · [RLCD mathematical analysis](research/rlcd_theory_zh.md) · [Existing OpenJev repository review](research/openjev_repo_audit_zh.md) · [jevlike review](research/jevlike_repo_audit_zh.md) · [Early toy experiments](research/toy_experiment_report_zh.md)

Code is available under the [MIT License](LICENSE). Authored programmatically generated data is marked CC0; third-party models, service materials, and their rights are documented separately. NanoJev is an independent research project with no official affiliation with TypeSafe or Jev. Historical OpenJev names and schemas are retained for traceability; the project is now called NanoJev.
