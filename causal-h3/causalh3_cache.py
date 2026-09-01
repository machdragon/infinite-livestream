"""Bounded KV cache and clean x0 state for persistent causal H3 generation.

The persistent model carries two pieces of state across chunks:

1. **KV cache** -- a ``SlidingWindowKVCache`` (from ai-toolkit ``causal.py``)
   holding per-layer key/value tensors for the sink (text + initial
   conditions) and a bounded sliding window of recent chunks.  After each
   chunk's four read-only denoise forwards, the final clean x0 prediction
   is cache-filled (the transformer appends its new KV) so the next chunk
   attends to the denoised output.  Eviction is deterministic: sink tokens
   are always kept; the oldest non-sink tokens are evicted when the window
   bound is exceeded.

2. **Clean x0 state** -- the denoised video and audio latent rows from the
   last forward of the previous chunk, carried as the starting point for
   the next chunk's noisy initialization.  This is what makes the stream
   continuous rather than a sequence of independent clips.

The ``ReadOnlyKVCacheView`` wraps a real ``SlidingWindowKVCache`` and makes
``append`` a no-op, so the first three forwards of each chunk attend to
the cached KV without modifying it.  Only the fourth forward uses the real
cache (with append) to write the clean x0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


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


class ReadOnlyKVCacheView:
    """A read-only view over a ``SlidingWindowKVCache`` that does not append.

    The transformer's ``forward()`` calls ``kv_cache.get(layer_idx)`` to
    retrieve cached KV and ``kv_cache.append(layer_idx, k, v)`` to write
    new KV.  This view delegates ``get`` (and the validity/role masks) to
    the real cache but makes ``append``, ``append_validity``, and
    ``append_token_roles`` no-ops, so the first three forwards of each
    chunk attend to the existing cache without modifying it.

    Only the fourth forward (the clean x0 cache-fill) uses the real cache
    directly, so its new KV is appended and the cache grows.
    """

    def __init__(self, real_cache: Any) -> None:
        self._real = real_cache

    def get(self, layer_idx: int) -> Optional[Tuple[Any, Any]]:
        return self._real.get(layer_idx)

    def append(self, layer_idx: int, k_new: Any, v_new: Any) -> None:
        pass  # read-only: do not write back

    def append_validity(self, validity: Any) -> None:
        pass

    def append_token_roles(self, roles: Any) -> None:
        pass

    def get_validity_mask(self) -> Optional[Any]:
        return self._real.get_validity_mask()

    def get_token_roles(self) -> Optional[Any]:
        return self._real.get_token_roles()

    def __len__(self) -> int:
        return len(self._real)


class CleanX0State:
    """Carries the clean video and audio x0 latent rows across chunks.

    After each chunk's four read-only denoise forwards, the final forward's
    clean x0 prediction (the denoised video_rows and audio_rows) is stored
    here.  The next chunk initializes its noisy latents from this clean x0
    (plus fresh noise at the first sigma point), so the stream is
    continuous rather than a sequence of independent clips.

    The state is separate from the KV cache because the x0 latent feeds
    the *input* of the next chunk, while the KV cache feeds the *attention*
    context.
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
        """Store the clean x0 from a chunk's final forward."""
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
    correctly: each 17-frame chunk decodes from 5 latents, but the decoder
    benefits from adjacent latents for boundary continuity.  This buffer
    holds the last ``vae_left_latents`` latents from the previous chunk's
    decode so the next chunk's decode has the context it needs.

    The buffer holds real latent tensors ``(C, T, H, W)``, not placeholders.
    """

    def __init__(self, left_latents: int) -> None:
        self._left_latents = left_latents
        self._buffer: list[Any] = []

    @property
    def left_latents(self) -> int:
        """The configured left-context size."""
        return self._left_latents

    @property
    def buffered(self) -> int:
        """Number of latents currently in the buffer."""
        return len(self._buffer)

    def push(self, latents: Any) -> None:
        """Push a chunk's decoded latent tensor into the buffer, evicting old ones.

        Args:
            latents: The latent tensor ``(B, C, T, H, W)`` from the most
                recent chunk's decode.  The last ``left_latents`` temporal
                slices are retained.
        """
        # Store the temporal slices as individual latent frames.
        t = latents.shape[2]
        for i in range(t):
            self._buffer.append(latents[:, :, i:i+1])
        # Keep only the last _left_latents.
        if len(self._buffer) > self._left_latents:
            self._buffer = self._buffer[-self._left_latents:]

    def context(self) -> Any:
        """The left-context latents for the next decode, as a concatenated tensor.

        Returns None when no context is available (the first chunk).
        """
        if not self._buffer:
            return None
        # Use numpy if available (works with both numpy arrays and torch
        # tensors via conversion); fall back to torch in production.
        try:
            import numpy as np
            return np.concatenate(self._buffer, axis=2)
        except (ImportError, TypeError):
            import torch
            return torch.cat(self._buffer, dim=2)

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()


class AudioVAEOverlapSave:
    """Overlap-save decoder for continuous AudioVAE decode.

    Audio is decoded in overlapping blocks: each block produces output
    samples, and the overlap from the previous block is saved and mixed
    into the next block's output.  This eliminates boundary artifacts at
    chunk transitions, which is critical for the persistent stream where
    there are no hard cuts.

    The buffer holds real waveform tensors at the native 32 kHz stereo rate.
    Downmixing and resampling to the Reactor wire format (mono 48 kHz)
    happens after decode, in the backend.
    """

    def __init__(self, overlap_samples: int) -> None:
        self._overlap_samples = overlap_samples
        self._saved: Any = None
        self._chunk_index: int = -1

    @property
    def overlap_samples(self) -> int:
        """The configured overlap size in samples."""
        return self._overlap_samples

    @property
    def has_overlap(self) -> bool:
        """Whether saved overlap from a previous block is available."""
        return self._saved is not None

    def save(self, overlap: Any, chunk_index: int) -> None:
        """Save the overlap waveform from a decoded block for the next decode."""
        self._saved = overlap
        self._chunk_index = chunk_index

    def pop(self) -> Any:
        """Return and clear the saved overlap."""
        saved = self._saved
        self._saved = None
        return saved

    def reset(self) -> None:
        """Clear the saved overlap."""
        self._saved = None
        self._chunk_index = -1


__all__ = [
    "CacheConfig",
    "ReadOnlyKVCacheView",
    "CleanX0State",
    "VisualVAEDecodeBuffer",
    "AudioVAEOverlapSave",
]
