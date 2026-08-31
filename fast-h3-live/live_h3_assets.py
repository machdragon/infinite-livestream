"""Config parsing and weights-bundle validation for the live FastH3 channel.

``live_h3.yaml`` is read here and nowhere else: ``load_config`` turns it into one
validated :class:`LiveH3Config`, and ``require_weights`` fails startup loudly
when the bundle on disk is incomplete. Pure file and dict work — no torch, no
fastvideo — so the schema renders and the tests run on any machine.

This mirrors ``fast-h3/fasth3_assets.py`` but drops the two clip-queue sizes
(the live channel has no client-facing queue: one held prompt drives an
open-ended chain) and adds the stream knobs — the seam crossfade width, the
color-match mode, and how many clips the producer buffers ahead of playout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import live_h3_clip_plan as clip_plan

# The HF snapshot directory inside the weights bundle — the same checkpoint the
# hard-cut fast-h3 channel serves.
DEFAULT_CHECKPOINT_DIR = "FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"

# Component directories the T2VA/FL2VA pipeline loads. An incomplete bundle must
# kill startup, not surface as a loader traceback on the first clip.
REQUIRED_COMPONENTS = (
    "transformer",
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
)


@dataclass(frozen=True)
class LiveH3Config:
    """Everything ``live_h3.yaml`` configures, validated once at load.

    The session-level fields are the defaults a fresh session starts from;
    ``inference`` and ``runtime`` are the raw blocks the backend reads its engine
    knobs (attention kernels, compile flags, parallelism, offload policy) from.
    The stream fields shape the seam: ``crossfade_frames`` is the overlap width,
    ``color_match`` is ``per_clip`` or ``off``, and ``buffer_depth`` is how many
    built clips the producer keeps ahead of the emitter.
    """

    aspect: str
    clip_frames: int
    seed: int
    num_inference_steps: int
    crossfade_frames: int
    color_match: str
    buffer_depth: int
    warmup_aspects: tuple[str, ...]
    warmup_frames: tuple[int, ...]
    inference: dict[str, Any]
    runtime: dict[str, Any]


def load_config(config_path: Path | None) -> LiveH3Config:
    """Parse ``live_h3.yaml`` into a validated :class:`LiveH3Config`.

    Args:
        config_path: Path the runtime hands over from ``runtime.config`` in
            ``reactor.yaml``, or ``None`` when the manifest names no config.

    Raises:
        ValueError: If the configured aspect is not one this model offers, the
            crossfade width is negative or not shorter than a clip, the
            color-match mode is unknown, or the buffer depth is not positive.
    """
    document: dict[str, Any] = {}
    if config_path is not None:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    inference: dict[str, Any] = document.get("inference") or {}
    runtime: dict[str, Any] = document.get("runtime") or {}

    aspect = str(inference.get("aspect", "16:9"))
    if aspect not in clip_plan.ASPECT_CHOICES:
        raise ValueError(
            f"inference.aspect must be one of {list(clip_plan.ASPECT_CHOICES)}, got {aspect!r}"
        )

    clip_frames = clip_plan.frames_for_seconds(
        float(inference.get("clip_seconds", clip_plan.MAX_SECONDS))
    )

    crossfade_frames = int(inference.get("crossfade_frames", 12))
    if crossfade_frames < 0:
        raise ValueError(f"inference.crossfade_frames must be >= 0, got {crossfade_frames}")
    if crossfade_frames >= clip_frames:
        raise ValueError(
            f"inference.crossfade_frames ({crossfade_frames}) must be shorter than a clip "
            f"({clip_frames} frames)"
        )

    color_match = str(inference.get("color_match", "per_clip"))
    if color_match not in ("per_clip", "off"):
        raise ValueError(f"inference.color_match must be 'per_clip' or 'off', got {color_match!r}")

    buffer_depth = int(inference.get("buffer_depth", 2))
    if buffer_depth < 1:
        raise ValueError(f"inference.buffer_depth must be positive, got {buffer_depth}")

    return LiveH3Config(
        aspect=aspect,
        clip_frames=clip_frames,
        seed=int(inference.get("seed", 1000)),
        # Sigma-grid POINTS, not transformer forwards: the distilled schedule is
        # five points and exactly four forwards.
        num_inference_steps=int(inference.get("num_inference_steps", 5)),
        crossfade_frames=crossfade_frames,
        color_match=color_match,
        buffer_depth=buffer_depth,
        warmup_aspects=tuple(str(a) for a in (inference.get("warmup_aspects") or [aspect])),
        warmup_frames=_parse_warmup_lengths(inference.get("warmup_lengths"), clip_frames),
        inference=inference,
        runtime=runtime,
    )


def _parse_warmup_lengths(raw: Any, clip_frames: int) -> tuple[int, ...]:
    """Resolve ``inference.warmup_lengths`` to the frame counts load() warms.

    ``"default"`` (or nothing) warms only the session's default length — the
    right setting for the live channel, whose every clip is that one length;
    ``"all"`` warms every legal length; a list of seconds warms those, snapped to
    legal lengths. The default length is always included. Both the T2VA and the
    FL2VA compile shape are warmed at each of these lengths by the backend.
    """
    if raw in (None, "", "default"):
        return (clip_frames,)
    if raw == "all":
        frames = set(clip_plan.legal_frame_counts())
    elif isinstance(raw, (list, tuple)):
        frames = {clip_plan.frames_for_seconds(float(seconds)) for seconds in raw}
    else:
        raise ValueError(
            f'inference.warmup_lengths must be "default", "all", or a list of seconds, got {raw!r}'
        )
    frames.add(clip_frames)
    return tuple(sorted(frames))


def resolve_model_path(config: LiveH3Config, weights_root: Path) -> Path:
    """The checkpoint directory inside the mounted weights bundle.

    ``checkpoint_dir: "."`` means the snapshot's components sit directly under
    the weights root, which is how ``reactor weights upload`` lays a bundle out
    when the snapshot itself is uploaded.
    """
    subdir = str(config.runtime.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR))
    if subdir in ("", "."):
        return weights_root
    return weights_root / subdir


def require_weights(root: Path, model_path: Path) -> None:
    """Fail startup loudly when the weights bundle is incomplete."""
    problems: list[str] = []
    if not model_path.is_dir():
        problems.append(f"checkpoint directory is missing: {model_path}")
    else:
        index = model_path / "modular_model_index.json"
        if not index.is_file():
            problems.append(f"modular_model_index.json is missing: {index}")
        for component in REQUIRED_COMPONENTS:
            if not (model_path / component).is_dir():
                problems.append(f"component directory is missing: {model_path / component}")
    if problems:
        raise FileNotFoundError(
            f"FastH3 weights bundle under {root} is incomplete:\n  " + "\n  ".join(problems)
        )


__all__ = [
    "DEFAULT_CHECKPOINT_DIR",
    "REQUIRED_COMPONENTS",
    "LiveH3Config",
    "load_config",
    "require_weights",
    "resolve_model_path",
]
