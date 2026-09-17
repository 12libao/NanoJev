# NanoJev viewer

**A nano replica of Jev.** The English interface reads recorded model results from local JSON files. It uses plain HTML, CSS, and JavaScript, with no framework or external scripts.

From the repository root:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

Open `http://127.0.0.1:8080` for the experiment viewer or `http://127.0.0.1:8080/comparison.html` for the selected success cases. Serve only the `web` directory. Opening the HTML as a local file can prevent JSON loading.

The main viewer offers model and episode selection, step controls, playback, action probabilities, recorded execution details, and parallel batches. V2 and V3 have different evaluation cohorts; their summaries remain separate. Sampled controllers display the action that was actually taken, including when it differs from the most probable action. Completed trajectories hold their final state.

## Selected success cases

The showcase presents **NanoJev / Jev / Untuned Qwen** on two TEST maps and two OOD maps. These maps were selected by outcome: NanoJev and Jev reached the goal while Untuned Qwen reached the step limit under both greedy and sampling controllers. The [selection record](../research/nanojev_showcase_selection.json) contains the four episode IDs, rule, actual outcomes, and source hashes. The [showcase notes](../research/nanojev_showcase.md) explain the selection.

Each MP4 includes all four selected cases. Playback advances by environment step. A completed panel stays at its recorded final state while the other trajectories continue. Each GIF includes the **complete first case**, including the unsuccessful baseline's full step horizon and a final outcome hold. The PNG poster shows that case's actual final state.

The six source trajectory files remain unchanged. The reader JSON contains the drawing fields needed for these 24 recorded trajectories. It retains original action probabilities, sampled actions, and source/checkpoint hashes.

## Recreate the media

Rendering requires Python, Node.js, Playwright, Chrome, and FFmpeg with libx264. The static viewer requires none of these tools. The renderer reads existing trajectories and makes no API or GPU calls.

Use the source filenames recorded in the manifest to construct a fresh render. Set the two tool paths below to your local installations:

```bash
python3 - <<'PY'
import json
import subprocess
from pathlib import Path

manifest = json.loads(Path('assets/comparison_media_manifest.json').read_text())
command = ['python3', 'scripts/render_comparison_video.py']
for source in manifest['sources']:
    command += ['--' + source['role'], str(Path('research') / source['artifact'])]
command += [
    '--ffmpeg', '/path/to/ffmpeg',
    '--playwright-module', '/path/to/playwright/index.mjs',
    '--data-output', '/tmp/nanojev-showcase-new.json',
    '--output-dir', '/tmp/nanojev-showcase-new-media',
]
subprocess.run(command, check=True)
PY
```

Outputs must use new paths. The script refuses to replace existing files. `--assemble-only` validates and writes reader data without launching Chrome or encoding video. `--steps-per-second` controls environment-step playback; its default is 4.

The media manifest records source, renderer, reader, and media SHA256 hashes, as well as per-frame episode IDs and environment steps. Capture checks every rendered frame against the recorded state and action and verifies the expected final outcomes for all selected cases. The playback check records real Chrome interaction and exported-video decoding checks.

## Result data

The main viewer reads `demo_results.json`:

```json
{
  "models": [
    {
      "name": "Model display name",
      "checkpoint_sha256": "Recorded checkpoint hash",
      "summary": {},
      "episodes": []
    }
  ],
  "parallel_batches": [],
  "sources": []
}
```

An episode contains `id`, `game`, `split`, `initial_state`, `steps`, and `outcome`, with optional `final_state`. Each step can include `state`, `action`, `next_state`, `probabilities`, `controller`, `distribution_argmax`, `sample_uniform_draw`, `forced`, and `model_forward`. The viewer uses `step.action` for the executed action. Original question and state text may retain the language of the recorded input.

Supported environment states:

```text
grid_navigation: { game, size, walls:[[row,col],...], position:[row,col], goal:[row,col] }
tic_tac_toe:       { game, board:"Nine characters: . / X / O", player:"X or O" }
```

Grid coordinates are zero-based. Unknown environment schemas are shown as JSON. A missing result file produces an empty state.

Parallel batches contain `states` and recorded `execution` fields such as `forward_passes` and `total_paths`. The interface uses those records for its execution counters. Boolean probabilities can be displayed as `false: 1-p_true` and `true: p_true`; the original response remains available.

## Optional inference endpoint

The live form posts its JSON to the same origin at `/api/evaluate`. A model server must implement this endpoint; Python's static HTTP server does not. The usual request shape is `{states:[{id,state,questions}]}` and the response shape is `{states:[{id,answers}],execution}`.

The interface shows the actual response and browser round-trip duration. The backend should load the model once and record its actual forward-pass metadata. API credentials are never embedded in the page.

## Checks

```bash
node --check web/app.js
node --check web/comparison.js
node --check scripts/capture_comparison.mjs
```

The exported media and viewer were checked with isolated, unsigned-in Chrome 153 and Playwright 1.63. Detailed results are in [the playback check](../assets/comparison_playback_check.json) and [the media manifest](../assets/comparison_media_manifest.json). FFmpeg 7.1 encodes the H.264/yuv420p MP4 files; Pillow is used only for optional GIF frame inspection.
