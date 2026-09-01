"""Chunk geometry for the persistent causal H3 channel.

The persistent model generates continuously: an initial 5-frame prefix
followed by 17-frame causal chunks (each producing 5 latents).  Unlike
``fast-h3`` whose clips are independent 5-15 second samples with hard
boundaries, this model never resets between chunks -- the KV cache and
clean x0 state carry forward, so the stream is one unbroken session.

Constants are duplicated from FastVideo/ai-toolkit (importing theirs drags
in torch) with a drift test to catch divergence.
"""

from __future__ import annotations

import math

FPS = 24
"""The only frame rate MiniMax-H3 accepts; the pipeline rejects anything else."""

# The causal VAE consumes video in 17-frame chunks that decode to 5 latents,
# so a valid pixel length is always ``17n + 5``.
_FRAMES_PER_CHUNK = 17
_LATENTS_PER_CHUNK = 5

# The initial prefix: the smallest valid H3 frame count (17*0 + 5 = 5).
# It seeds the KV cache and the clean x0 state before the first real chunk.
PREFIX_FRAMES = 5

# Sigma-grid points for the four-step distilled schedule.
# Five points, exactly four forwards; the checkpoint supports nothing else.
SIGMA_LADDER = (999, 749, 500, 250, 0)
NUM_INFERENCE_STEPS = 5  # sigma-grid POINTS, not transformer forwards
NUM_FORWARDS = 4  # the distilled schedule runs exactly four transformer passes

# Released flow shifts (exponential): video 12, audio 3.
VIDEO_SIGMA_SHIFT = 12.0
AUDIO_SIGMA_SHIFT = 3.0

# Guidance-distilled: no negative prompt, no CFG pass.
GUIDANCE_SCALE = 1.0

# Audio: native stereo at 32 kHz, decoded by a separate audio VAE.
AUDIO_SAMPLE_RATE = 32_000
AUDIO_CHANNELS = 2
AUDIO_LATENTS_PER_SECOND = 40

# Canvas rules: the short edge is fixed, total area is capped, both sides
# multiples of 32.  Same as fast-h3 -- the checkpoint is identical.
_SHORT_EDGE = 768
_MAX_PIXELS = 768 * 1344
_CANVAS_MULTIPLE = 32
_MIN_ASPECT = 1 / 4
_MAX_ASPECT = 4

# The 60-second session target: generate 1,450 valid H3 frames, emit 1,440.
# 1,450 = 5 (prefix) + 85 * 17 (chunks) = 17*85 + 5.
# Emitting 1,440 at 24 fps = exactly 60 seconds; the last 10 frames are
# trimmed so the output is a clean 60s cut with no hard boundary.
SESSION_SECONDS = 60
SESSION_EMIT_FRAMES = SESSION_SECONDS * FPS  # 1,440
SESSION_GENERATE_FRAMES = 1_450  # 17*85 + 5, the nearest valid H3 length >= 1,440
SESSION_TRIM_FRAMES = SESSION_GENERATE_FRAMES - SESSION_EMIT_FRAMES  # 10
SESSION_CHUNKS = (SESSION_GENERATE_FRAMES - PREFIX_FRAMES) // _FRAMES_PER_CHUNK  # 85

# How many chunks to generate for a given number of emit frames.
# The prefix provides PREFIX_FRAMES; each chunk adds _FRAMES_PER_CHUNK.


def chunks_for_emit_frames(emit_frames: int) -> int:
    """Number of 17-frame chunks needed to cover *emit_frames* output frames.

    The prefix contributes ``PREFIX_FRAMES`` frames; each chunk adds
    ``_FRAMES_PER_CHUNK``.  The result is the minimum chunk count whose
    total generated frames reach or exceed *emit_frames*.
    """
    if emit_frames <= PREFIX_FRAMES:
        return 0
    remaining = emit_frames - PREFIX_FRAMES
    return math.ceil(remaining / _FRAMES_PER_CHUNK)


def total_generate_frames(emit_frames: int) -> int:
    """Total H3 frames to generate for *emit_frames* output frames."""
    chunks = chunks_for_emit_frames(emit_frames)
    return PREFIX_FRAMES + chunks * _FRAMES_PER_CHUNK


def trim_frames(emit_frames: int) -> int:
    """Frames to trim from the tail so output is exactly *emit_frames*."""
    return total_generate_frames(emit_frames) - emit_frames


def seconds_for_frames(frames: int) -> float:
    """Exact playout length of *frames* frames, in seconds."""
    return frames / FPS


def frames_for_seconds(seconds: float) -> int:
    """Snap a requested duration to the nearest valid ``17n + 5`` frame count."""
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    frames = round(seconds * FPS)
    # Align up to the next 17n + 5.
    while frames % _FRAMES_PER_CHUNK != _LATENTS_PER_CHUNK:
        frames += 1
    return max(PREFIX_FRAMES, frames)


def align_frames(frames: int) -> int:
    """Round up to the next valid ``17n + 5`` pixel length."""
    if frames < 1:
        raise ValueError(f"frames must be positive, got {frames}")
    while frames % _FRAMES_PER_CHUNK != _LATENTS_PER_CHUNK:
        frames += 1
    return frames


def canvas_for_aspect(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """Resolve an aspect ratio to a ``(height, width)`` the checkpoint accepts.

    Mirrors FastVideo's ``resolve_canvas_size``: pin the short edge to 768,
    shrink to the area cap if the result is too wide, then round both sides
    to a multiple of 32.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"aspect must be positive, got {aspect_width}:{aspect_height}")
    ratio = aspect_width / aspect_height
    if ratio >= 1:
        width, height = _SHORT_EDGE * ratio, float(_SHORT_EDGE)
    else:
        width, height = float(_SHORT_EDGE), _SHORT_EDGE / ratio
    area = width * height
    if area > _MAX_PIXELS:
        scale = (_MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale
    m = _CANVAS_MULTIPLE
    return max(m, round(height / m) * m), max(m, round(width / m) * m)


ASPECT_CHOICES: tuple[str, ...] = ("16:9", "1:1", "9:16", "4:3")

_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "16:9": (16, 9),
    "1:1": (1, 1),
    "9:16": (9, 16),
    "4:3": (4, 3),
}


def canvas_for_choice(aspect: str) -> tuple[int, int]:
    """Resolve one of ``ASPECT_CHOICES`` to ``(height, width)``."""
    try:
        ratio = _ASPECT_RATIOS[aspect]
    except KeyError:
        raise ValueError(f"unknown aspect {aspect!r}; choose from {ASPECT_CHOICES}") from None
    return canvas_for_aspect(*ratio)


# VSA-H3 sparse attention parameters.
VSA_TILE_SIZE = 64
VSA_SPARSITY = 0.9

__all__ = [
    "FPS",
    "PREFIX_FRAMES",
    "SIGMA_LADDER",
    "NUM_INFERENCE_STEPS",
    "NUM_FORWARDS",
    "VIDEO_SIGMA_SHIFT",
    "AUDIO_SIGMA_SHIFT",
    "GUIDANCE_SCALE",
    "AUDIO_SAMPLE_RATE",
    "AUDIO_CHANNELS",
    "AUDIO_LATENTS_PER_SECOND",
    "SESSION_SECONDS",
    "SESSION_EMIT_FRAMES",
    "SESSION_GENERATE_FRAMES",
    "SESSION_TRIM_FRAMES",
    "SESSION_CHUNKS",
    "ASPECT_CHOICES",
    "VSA_TILE_SIZE",
    "VSA_SPARSITY",
    "chunks_for_emit_frames",
    "total_generate_frames",
    "trim_frames",
    "seconds_for_frames",
    "frames_for_seconds",
    "align_frames",
    "canvas_for_aspect",
    "canvas_for_choice",
]
