# Ramer desktop app

The production application, kept separate from `scripts/` (the research
CLI tools and experiments this app is built on top of -- see the repo
root `README.md` and `PRODUCT_HYPOTHESES.md` for that history). This
directory is meant to become the actual "download it, it just works"
product; `scripts/ramer_local.py` and `scripts/mini_agent.py` stay as the
validated research prototypes that proved the underlying approach.

## Status: Milestone 1 of 4

**Done and tested**: hardware detection, Ollama integration, model
selection UI.

**Not built yet**: placement engine/cache wired into the app (still only
in `scripts/ramer_local.py`), agent harness, project workspace, runtime
monitoring/drift-watch, CLI. See "Remaining milestones" below.

## Run it

```bash
pip install -r app/requirements.txt
python -m app.gui.main
```

Requires Ollama running locally (`127.0.0.1:11434`) with at least one
model pulled -- the app can pull one for you on the model-selection
screen, but needs Ollama itself already installed and running.

## Test it

```bash
python -m pytest app/tests/ -v
```

These are **real integration tests against this machine's actual
hardware and live Ollama server**, not mocks -- consistent with this
project's existing rule (see `PRODUCT_HYPOTHESES.md`) against faking
measurements. Running them requires an NVIDIA GPU and at least one
Ollama model installed. 12/12 passing as of this milestone.

## What's actually implemented

- `app/core/hardware.py` -- real hardware detection (GPU name/VRAM via
  `nvidia-smi`, reusing `scripts/ramer_local.py`'s existing helpers
  rather than duplicating them; system RAM via `psutil`; CPU name via
  WMI on Windows; Ollama availability and installed-model list via the
  real API). Every field is either a real measurement or explicitly
  listed in `HardwareSnapshot.errors` -- never silently faked.
- `app/core/model_runtime.py` -- the `ModelRuntime` abstraction the rest
  of the app is meant to depend on, plus `OllamaRuntime`, the only
  implementation today. A curated list of 3 models (the same
  `qwen2.5:*-instruct` sizes already benchmarked in this project's own
  research), a simple, explainable VRAM-based recommendation (not a
  learned model), and a real streaming `pull()` that reports Ollama's
  actual download progress.
- `app/gui/` -- a two-screen PySide6 app: hardware detection (Setup) ->
  model selection (Model), wired to the real core above via background
  `QThread`s so hardware detection and model pulls never block the UI.
  Verified by rendering each screen's actual widget tree to a pixmap
  from inside the Qt process (not an OS screenshot -- that approach was
  tried first and rejected because it captures whatever else is on the
  user's screen) and visually confirming the real detected hardware,
  the real installed-model list, and the real recommendation appear
  correctly.

## What was tested vs. what's a smoke test

Real, automated (`pytest`): hardware detection, Ollama integration,
model-list/metadata/recommendation logic -- 12 tests, all against live
hardware and a live Ollama server, no mocks.

Manual, one-time visual check (not automated): both GUI screens render
correctly and the Setup -> Model navigation works, via a throwaway
script that grabbed each screen's pixmap and simulated the real
Continue button click. Worth turning into an automated GUI test later
(pytest-qt) rather than repeating manually every milestone.

## Remaining milestones (per the original spec)

2. Placement engine + placement cache, adapted from `scripts/ramer_local.py`
   into `app/core/`, wired into model selection (context picker, num_gpu
   probing, headroom-aware tiebreak) instead of just picking a model.
3. Agent harness (filesystem/terminal/git tools, permission prompts,
   workspace boundary), adapted from `scripts/mini_agent.py`.
4. Streaming agent UI, context-usage display, runtime monitoring
   (quick-check + drift-watch surfaced in the UI, not just the CLI),
   observability log, optional CLI.

Each will get its own real tests and an honest status report before
moving to the next, per the incremental approach this was scoped with.
