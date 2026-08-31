# fast-h3-live — continuous FL2VA chain from one prompt

A sibling model to `fast-h3/`, left untouched. Same engine, same 148 GB
checkpoint, same solved infrastructure — but instead of a queue of independent
clips separated by a black flush, this channel chains clips into **one
continuous video-and-audio stream from a single held prompt**, with no
re-prompting. This is our validated FL2VA-chaining logic (the one that produced
the good 30 s bear video from one prompt) housed in this repo's runtime
scaffolding.

## What it does

- `set_prompt` holds a prompt; `start` begins an open-ended chain; `stop` ends
  it; `set_prompt` again mid-run steers the stream at the next clip.
- Clip 0 is **T2VA**. Every clip after is **FL2VA**, anchored on the previous
  clip's last frame, so the scene carries forward off the one prompt.
- Consecutive clips are **stitched** at the seam so the result plays as one
  video, not a sequence of cuts:
  - **linearfade** — a linear-light crossfade with complementary weights and a
    ramped local exposure match (kills the sRGB "flash"; validated monotonic).
  - **per-clip color-match** locked to clip 0's last-frame mean (exposure can't
    ratchet across a long chain — the drift that blew the bear video to white).
  - **equal-power audio overlap-add** across the same window.
- **Double-buffered**: a producer builds clips ahead into a bounded queue while
  the consumer streams and stitches, so generation can stay ahead of playback.

## Files

New model logic:
- `live_h3.py` — `ReactorModel`: full control surface + held-prompt indefinite
  producer/consumer chain.
- `live_h3_backend.py` — `fast-h3`'s backend **plus** an FL2VA `anchor_image`,
  the last frame returned for the next anchor, and **both** the T2VA and FL2VA
  compile shapes warmed at load.
- `live_h3_seam.py` — the seam math (pure numpy, GPU-free): color-match lock,
  linear-light blend, equal-power audio.
- `live_h3_types.py` — the stream contract (tracks + messages).
- `live_h3_assets.py` — config parsing / weights validation (their assets with
  the queue sizes replaced by the seam + buffer knobs).
- `reactor.yaml`, `live_h3.yaml` — mirror `fast-h3`'s manifest and recipe;
  only name/import/description and the queue→chain knobs differ.

Copied verbatim from `fast-h3/` (with parity tests that fail if they drift):
- `sitecustomize.py`, `requirements.txt`, `live_h3_clip_plan.py`
  (← `fasth3_clip_plan.py`).

## Timing fixes carried over (this was our bimodal cause)

- **256-token prompt pad** — one compiled shape for every prompt; without it a
  novel prompt length recompiles the transformer (~23 s) mid-stream.
- **Dynamo recompile-limit raise** — in the parent and, via `sitecustomize.py`
  on `PYTHONPATH`, in every spawned engine worker.
- **Warm BOTH T2VA and FL2VA shapes at load** — the FL2VA anchor adds keyframe
  rows to the packed sequence, a distinct compile shape; unwarmed, the first
  continuation clip stalls ~20 s. `_warmup` now builds both paths per shape.

## CPU checks (this branch)

`PYTHONPATH=. python -m pytest tests/ -q` → **50 passed**. `py_compile` clean on
all files. Covered without a GPU:
- seam: linearfade monotonic / no midpoint flash / endpoints approach each clip;
  color-match locks to clip 0's mean, preserves intra-clip variation, and does
  **not** ratchet across a chain; audio overlap equal-power, no int16 wrap.
- chain end-to-end (fake backend): one held prompt drives an indefinite chain;
  clip 0 T2VA, every clip after FL2VA-anchored; seed advances per clip;
  `set_prompt` steers mid-stream; `stop`/restart/`reset`; A/V locked slice for
  slice; the seam removes exactly one overlap per boundary (`C*N-(C-1)*k`).
- schema renders; command set, tracks, messages, moderation, bounds; manifest
  (name=folder, bare semver, runtime 3.2.6, import resolves, no weights);
  shared-file parity with `fast-h3`; sitecustomize arch list ↔ manifest.

## Honest limits — NOT yet GPU-validated

- **Chaining quality from one prompt** — reproduces the bear video; validated
  previously on GPU with this exact seam + color-match + FL2VA-anchor logic.
- **Gap-free live playback** — **NOT** GPU-validated on this branch. The bimodal
  6 s/12 s per-clip timing we hit before should resolve via the 256-pad +
  recompile-limit fixes carried over here, but that is **unconfirmed** until a
  GPU pass. `live_h3.yaml` ships short 5.167 s clips (max double-buffer
  headroom) at 768p; whether an FL2VA clip builds inside the playout budget at
  768p is exactly what a GPU pass must measure.
- **FL2VA is undistilled on this checkpoint.** FL2VA rides the base path, not
  the 4-step DMD2 distillation; long-run quality/drift at 768p over many clips
  is a real risk. Genuine motion continuity across seams needs a distilled
  `transformer_ref` (Ref2VA) partition — which the DataFree preview checkpoint
  does not contain. The seam blend hides the *appearance* discontinuity, not the
  *momentum* reset; post-hoc tricks (longoverlap, flowwarp) were tried and lost
  to linearfade.

## Needs a GPU validation pass

1. FL2VA clip build time at 768p vs the playout budget → is the stream gap-free?
   If not, the levers are shorter clips, a smaller canvas, or a wider crossfade.
2. Long-run FL2VA quality/drift over many minutes of chain.
3. `recording.enabled` can be turned on once (1) confirms continuity.
