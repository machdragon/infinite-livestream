"""Bounded KV cache configuration and clean x0 state for persistent causal H3.

The persistent model carries two pieces of state across chunks:

1. **KV cache** -- a ``SlidingWindowKVCache`` (from the vendored
   ``minimax_h3.src.causal``) holding per-layer key/value tensors for the
   sink (text + initial conditions) and a bounded sliding window of recent
   chunks.  After each chunk's four read-only denoise forwards
   (``update_cache=False``), a distinct clean x0 forward at t=0
   (``update_cache=True``) cache-fills the denoised output so the next
   chunk attends to it.  Eviction is deterministic: sink tokens are
   always kept; the oldest non-sink tokens are evicted when the window
   bound is exceeded.

2. **Clean x0 state** -- the denoised video and audio latent rows from
   the clean x0 forward of the previous chunk.  Each chunk starts with
   fresh noise; the previous x0 is history attended to through the KV
   cache, not cloned as the new chunk's input.  The x0 state is kept for
   diagnostic and buffer-management purposes (the VAE decode buffer uses
   the latent tensors for left-context).

The ``update_cache`` flag on ``MiniMaxH3Transformer.forward()`` (added in
ai-toolkit PR #17 head, vendored here) controls whether the cache is
written: ``False`` for read-only denoise forwards, ``True`` for the clean
x0 cache-fill forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CacheConfig:
    """Bounds for the KV cache and decode buffers.

    Attributes:
        sink_tokens: Number of tokens always retained (text + initial
            conditions).  These are the prefix rows the model prefills once.
        window_tokens: Maximum number of non-sink tokens retained.  When
            the cache exceeds ``sink + window``, the oldest non-sink tokens
            are evicted.
        vae_left_latents: Number of left-context latents the VisualVAE
            decoder keeps for incremental decode.
        audio_overlap_latents: Number of overlap latents the AudioVAE
            overlap-save keeps between decode blocks.
    """

    sink_tokens: int = 256
    window_tokens: int = 512
    vae_left_latents: int = 5
    audio_overlap_latents: int = 10


class CleanX0State:
    """Carries the clean video and audio x0 latent rows across chunks.

    After each chunk's clean x0 forward (the 5th forward at t=0 with
    update_cache=True), the denoised video_rows and audio_rows are stored
    here.  Each chunk starts with fresh noise; the previous x0 is history
    attended to through the KV cache.  The x0 state is kept for VAE decode
    buffer management and diagnostics.
    """

    def __init__(self) -> None:
        self._video_x0: Any = None
        self._audio_x0: Any = None
        self._chunk_index: int = -1

    @property
    def video_x0(self) -> Any:
        """The current clean video x0 latent rows, or None before the first chunk."""
        return self._video_x0

    @property
    def audio_x0(self) -> Any:
        """The current clean audio x0 latent rows, or None before the first chunk."""
        return self._audio_x0

    @property
    def chunk_index(self) -> int:
        """The chunk index that produced the current x0, or -1."""
        return self._chunk_index

    def update(self, video_x0: Any, audio_x0: Any, chunk_index: int) -> None:
        """Store the clean x0 from a chunk's clean x0 forward."""
        self._video_x0 = video_x0
        self._audio_x0 = audio_x0
        self._chunk_index = chunk_index

    def reset(self) -> None:
        """Clear the state."""
        self._video_x0 = None
        self._audio_x0 = None
        self._chunk_index = -1


class VisualVAEDecodeBuffer:
    """Bounded left-latent buffer for incremental VisualVAE decode.

    The causal video VAE needs left-context latents to decode a chunk
    correctly.  This buffer holds the last ``vae_left_latents`` latents
    from the previous chunk's decode so the next chunk's decode has the
    context it needs.  The buffer holds real latent tensors.

    The buffer also tracks how many pixel frames the left-context latents
    produced when they were decoded, so the next decode can discard exactly
    that many frames from the concatenated output.
    """

    def __init__(self, left_latents: int) -> None:
        self._left_latents = left_latents
        self._buffer: list[Any] = []
        self._prev_frame_count: int = 0

    @property
    def left_latents(self) -> int:
        return self._left_latents

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    @property
    def prev_frame_count(self) -> int:
        """How many pixel frames the buffered latents produced, or 0."""
        return self._prev_frame_count

    def push(self, latents: Any, frame_count: int = 0) -> None:
        """Push a chunk's decoded latent tensor and its pixel frame count."""
        t = latents.shape[2]
        for i in range(t):
            self._buffer.append(latents[:, :, i:i+1])
        if len(self._buffer) > self._left_latents:
            self._buffer = self._buffer[-self._left_latents:]
        self._prev_frame_count = frame_count

    def context(self) -> Any:
        """The left-context latents for the next decode, or None."""
        if not self._buffer:
            return None
        try:
            import numpy as np
            return np.concatenate(self._buffer, axis=2)
        except (ImportError, TypeError):
            import torch
            return torch.cat(self._buffer, dim=2)

    def reset(self) -> None:
        self._buffer.clear()
        self._prev_frame_count = 0

class AudioVAEOverlapSave:
    """Overlap-save decoder for continuous AudioVAE decode.

    Audio is decoded in overlapping blocks: each block produces output
    samples, and the overlap from the previous block is saved and mixed
    into the next block's output.  This eliminates boundary artifacts at
    chunk transitions.
    """

    def __init__(self, overlap_samples: int) -> None:
        self._overlap_samples = overlap_samples
        self._saved: Any = None
        self._chunk_index: int = -1

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples

    @property
    def has_overlap(self) -> bool:
        return self._saved is not None

    def save(self, overlap: Any, chunk_index: int) -> None:
        self._saved = overlap
        self._chunk_index = chunk_index

    def pop(self) -> Any:
        saved = self._saved
        self._saved = None
        return saved

    def reset(self) -> None:
        self._saved = None
        self._chunk_index = -1


__all__ = [
    "CacheConfig",
    "CleanX0State",
    "VisualVAEDecodeBuffer",
    "AudioVAEOverlapSave",
]
