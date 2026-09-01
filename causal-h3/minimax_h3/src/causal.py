"""Causal masking and bounded KV state for MiniMax-H3 Gate 2.

Provides:
  * build_causal_mask: per-query boolean attention mask that prevents future
    video/audio chunk access while retaining text, keyframe, and sink access.
  * ChunkState: dataclass holding the KV tensors for sink + sliding window
    tokens plus metadata, with fixed-format serialization.
  * KVCache / SlidingWindowKVCache: per-layer KV storage with deterministic
    eviction and serialization.

The mask shape is (B, 1, S, S) where True = attend.  This is compatible with
``F.scaled_dot_product_attention`` and broadcasts correctly when ANDed with the
existing (B, 1, 1, S) pad mask.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

# Modality tags — mirror packing.py values (kept inline to avoid a circular
# import; these are a checkpoint contract and never change).
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
PAD_TAG = -1


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------


def build_causal_mask(
    token_tags: torch.Tensor,          # (B, S) long
    position_ids: torch.Tensor,        # (B, S, 3) float — not used directly
    chunk_idx: torch.Tensor,           # (S,) long: per-row chunk index, -1 = not in any chunk
    sink_size: int = 0,
) -> torch.Tensor:
    """Build a per-query causal attention mask.

    Returns ``(B, 1, S, S)`` boolean tensor where ``True`` means the query
    (row) may attend to the key (column).

    *chunk_idx* is a per-row tensor (computed by
    :func:`compute_chunk_boundaries`) where each target row carries its
    temporal chunk id and every other row (text, conditioning, pad) is -1.
    Using actual chunk ids (not sequential indices) lets audio and video
    rows that share the same temporal chunk attend to each other even
    though they occupy separate segments in the packed sequence.

    Rules:
      * Text rows (tag == TEXT_TAG) attend to all text rows.
      * Conditioning rows (video or audio rows not inside any chunk:
        keyframe anchors, ref2va references, reference soundtracks) attend
        to text + all conditioning rows.
      * Target rows (inside a chunk) attend to text + conditioning
        + sink tokens + all rows in the same or earlier chunks, but NOT
        rows in later chunks.
      * Sink tokens (the first ``sink_size`` tokens) are visible to every
        row.
      * Pad rows (tag == PAD_TAG) are never attended to as keys; as queries
        they attend to everything non-pad (their output is discarded).
    """
    B, S = token_tags.shape
    device = token_tags.device

    is_text = token_tags == TEXT_TAG        # (B, S)
    is_pad = token_tags == PAD_TAG          # (B, S)
    is_video = token_tags == VIDEO_TAG      # (B, S)
    is_audio = token_tags == AUDIO_TAG      # (B, S)

    # Conditioning = video or audio rows NOT in any chunk (keyframe anchors,
    # ref2va references, reference soundtracks).
    is_conditioning = (is_video | is_audio) & (chunk_idx[None, :] < 0)

    # text + conditioning keys (visible to conditioning and target queries)
    text_cond_keys = is_text | is_conditioning           # (B, S)

    # Unique chunk ids (sorted), used for cumulative visibility.
    unique_chunks = torch.unique(chunk_idx[chunk_idx >= 0])
    num_chunks = unique_chunks.numel()

    # Cumulative chunk-key visibility: attend_to_chunk[c_idx, j] is True if
    # key j is inside any chunk 0..unique_chunks[c_idx] (inclusive).
    if num_chunks > 0:
        key_in_chunk = torch.zeros(num_chunks, S, dtype=torch.bool, device=device)
        for c_idx in range(num_chunks):
            key_in_chunk[c_idx] = (chunk_idx == unique_chunks[c_idx])
        attend_to_chunk = torch.zeros(num_chunks, S, dtype=torch.bool, device=device)
        cumulative = torch.zeros(S, dtype=torch.bool, device=device)
        for c_idx in range(num_chunks):
            cumulative = cumulative | key_in_chunk[c_idx]
            attend_to_chunk[c_idx] = cumulative
    else:
        attend_to_chunk = torch.zeros(0, S, dtype=torch.bool, device=device)

    # Non-pad key mask (B, 1, 1, S)
    non_pad_key = ~is_pad[:, None, None, :]

    # Build (B, 1, S, S) mask
    mask = torch.zeros(B, 1, S, S, dtype=torch.bool, device=device)

    # --- Text queries: attend to text keys ---
    text_q = is_text[:, None, :]                        # (B, 1, S)
    mask |= text_q.unsqueeze(3) & is_text[:, None, None, :]

    # --- Conditioning queries: attend to text + conditioning keys ---
    cond_q = is_conditioning[:, None, :]                # (B, 1, S)
    mask |= cond_q.unsqueeze(3) & text_cond_keys[:, None, None, :]

    # --- Target queries in chunk c: attend to text+conditioning + chunks 0..c ---
    for c_idx in range(num_chunks):
        target_in_c = (chunk_idx[None, :] == unique_chunks[c_idx])  # (B, S)
        keys_visible = text_cond_keys | attend_to_chunk[c_idx][None, :]
        mask |= target_in_c[:, None, :, None] & keys_visible[:, None, None, :]

    # --- Pad queries: attend to everything non-pad (output discarded) ---
    pad_q = is_pad[:, None, :]                          # (B, 1, S)
    mask |= pad_q.unsqueeze(3) & non_pad_key

    # --- Sink tokens visible to all rows ---
    if sink_size > 0:
        mask[:, :, :, :sink_size] = True

    # --- Pad rows are never attended to as keys ---
    mask &= non_pad_key

    return mask


# ---------------------------------------------------------------------------
# ChunkState
# ---------------------------------------------------------------------------


@dataclass
class ChunkState:
    """KV state for sink + sliding-window tokens plus chunk metadata.

    Fixed serialization keys: ``k``, ``v``, ``position_ids``, ``token_tags``,
    ``chunk_index``.
    """

    k: torch.Tensor            # (B, S_cached, H, D) or (L, B, S_cached, H, D)
    v: torch.Tensor            # same shape as k
    position_ids: torch.Tensor  # (B, S_cached, 3) or (S_cached, 3)
    token_tags: torch.Tensor    # (B, S_cached) or (S_cached,)
    chunk_index: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "k": self.k,
            "v": self.v,
            "position_ids": self.position_ids,
            "token_tags": self.token_tags,
            "chunk_index": self.chunk_index,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "ChunkState":
        return cls(
            k=d["k"],
            v=d["v"],
            position_ids=d["position_ids"],
            token_tags=d["token_tags"],
            chunk_index=int(d["chunk_index"]),
        )


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------


class KVCache:
    """Per-layer KV storage with deterministic eviction and serialization.

    Each layer stores ``(k, v)`` tensors of shape ``(B, S, H, D)``.  The
    ``__len__`` returns the number of cached tokens in layer 0 (all layers
    are assumed to have the same token count).

    A per-sample validity mask ``(B, total_S)`` tracks which cached keys
    are real tokens versus padding.  This lets the attention mask extension
    in the transformer prepend the *historical* validity pattern instead
    of all-true columns, so padding keys from shorter captions stay
    invisible when batch size > 1.
    """

    def __init__(self) -> None:
        self._store: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._validity_mask: Optional[torch.Tensor] = None
        self._token_roles: Optional[torch.Tensor] = None

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Concatenate ``k_new`` / ``v_new`` along the sequence dimension."""
        if layer_idx in self._store:
            k_old, v_old = self._store[layer_idx]
            k = torch.cat([k_old, k_new], dim=2)
            v = torch.cat([v_old, v_new], dim=2)
        else:
            k, v = k_new, v_new
        self._store[layer_idx] = (k, v)

    def append_validity(self, validity: torch.Tensor) -> None:
        """Append a per-sample validity mask ``(B, S)`` for the current chunk.

        ``True`` marks a real (non-padding) key.  Masks accumulate across
        chunks in the same order as K/V, so the first ``len(cache)`` entries
        always correspond to the cached keys.
        """
        if self._validity_mask is not None:
            self._validity_mask = torch.cat([self._validity_mask, validity], dim=1)
        else:
            self._validity_mask = validity

    def get_validity_mask(self) -> Optional[torch.Tensor]:
        """Return the accumulated validity mask ``(B, total_S)`` or ``None``."""
        return self._validity_mask

    def append_token_roles(self, roles: torch.Tensor) -> None:
        """Append per-sample token roles ``(B, S)`` for the current chunk.

        Roles: 0=text, 1=conditioning, 2=target, 3=pad.
        Accumulated in the same order as K/V and validity.
        """
        if self._token_roles is not None:
            self._token_roles = torch.cat([self._token_roles, roles], dim=1)
        else:
            self._token_roles = roles

    def get_token_roles(self) -> Optional[torch.Tensor]:
        """Return accumulated token roles ``(B, total_S)`` or ``None``."""
        return self._token_roles

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self._store.get(layer_idx)

    def truncate(self, max_tokens: int) -> None:
        """Keep only the most recent ``max_tokens`` tokens in every layer.

        Eviction is deterministic: the oldest tokens (lowest sequence index)
        are removed first, retaining the tail that a causal cache most needs.
        """
        if max_tokens <= 0:
            return
        for idx, (k, v) in self._store.items():
            if k.shape[2] > max_tokens:
                self._store[idx] = (
                    k[:, :, -max_tokens:],
                    v[:, :, -max_tokens:],
                )
        if self._validity_mask is not None and self._validity_mask.shape[1] > max_tokens:
            self._validity_mask = self._validity_mask[:, -max_tokens:]
        if self._token_roles is not None and self._token_roles.shape[1] > max_tokens:
            self._token_roles = self._token_roles[:, -max_tokens:]

    def serialize(self) -> Dict[str, object]:
        """Return a plain-dict representation (tensors kept as-is)."""
        out: Dict[str, object] = {}
        for idx, (k, v) in sorted(self._store.items()):
            out[str(idx)] = {"k": k, "v": v}
        if self._token_roles is not None:
            out["_token_roles"] = self._token_roles
        if self._validity_mask is not None:
            out["_validity_mask"] = self._validity_mask
        return out

    @classmethod
    def deserialize(cls, d: Dict[str, object]) -> "KVCache":
        cache = cls()
        for key, val in d.items():
            if key == "_validity_mask":
                cache._validity_mask = val
                continue
            if key == "_token_roles":
                cache._token_roles = val
                continue
            idx = int(key)
            cache._store[idx] = (val["k"], val["v"])
        return cache

    def __len__(self) -> int:
        if not self._store:
            return 0
        # all layers have the same token count; report layer 0
        first = sorted(self._store)[0]
        return self._store[first][0].shape[2]


class SlidingWindowKVCache(KVCache):
    """KV cache with explicit sink tokens and a bounded sliding window.

    After each ``append``, if the total token count exceeds
    ``sink_size + window_size``, the oldest non-sink tokens are evicted
    deterministically (sink tokens are always kept).
    """

    def __init__(self, window_size: int, sink_size: int = 0) -> None:
        super().__init__()
        self.window_size = window_size
        self.sink_size = sink_size

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        super().append(layer_idx, k_new, v_new)
        self._evict()

    def _evict(self) -> None:
        """Keep sink tokens + the most recent ``window_size`` tokens."""
        max_tokens = self.sink_size + self.window_size
        for idx, (k, v) in self._store.items():
            total = k.shape[2]
            if total <= max_tokens:
                continue
            # keep first sink_size tokens + last window_size tokens; when
            # window_size is 0, keep only the sink (k[:, :, -0:] would
            # otherwise slice the entire tensor)
            sink_k, sink_v = k[:, :, : self.sink_size], v[:, :, : self.sink_size]
            if self.window_size > 0:
                win_k = k[:, :, -self.window_size:]
                win_v = v[:, :, -self.window_size:]
            else:
                win_k = k[:, :, self.sink_size:self.sink_size]
                win_v = v[:, :, self.sink_size:self.sink_size]
            self._store[idx] = (
                torch.cat([sink_k, win_k], dim=2),
                torch.cat([sink_v, win_v], dim=2),
            )
        # evict the validity mask to stay in sync with K/V
        if self._validity_mask is not None:
            total = self._validity_mask.shape[1]
            if total > max_tokens:
                sink_m = self._validity_mask[:, : self.sink_size]
                if self.window_size > 0:
                    win_m = self._validity_mask[:, -self.window_size:]
                else:
                    win_m = self._validity_mask[:, self.sink_size:self.sink_size]
                self._validity_mask = torch.cat([sink_m, win_m], dim=1)
        if self._token_roles is not None:
            total = self._token_roles.shape[1]
            if total > max_tokens:
                sink_r = self._token_roles[:, : self.sink_size]
                if self.window_size > 0:
                    win_r = self._token_roles[:, -self.window_size:]
                else:
                    win_r = self._token_roles[:, self.sink_size:self.sink_size]
                self._token_roles = torch.cat([sink_r, win_r], dim=1)

    def serialize(self) -> Dict[str, object]:
        out = super().serialize()
        out["_meta"] = {
            "window_size": self.window_size,
            "sink_size": self.sink_size,
        }
        return out

    @classmethod
    def deserialize(cls, d: Dict[str, object]) -> "SlidingWindowKVCache":
        meta = d.get("_meta", {})
        cache = cls(
            window_size=int(meta.get("window_size", 0)),
            sink_size=int(meta.get("sink_size", 0)),
        )
        for key, val in d.items():
            if key == "_meta":
                continue
            if key == "_validity_mask":
                cache._validity_mask = val
                continue
            if key == "_token_roles":
                cache._token_roles = val
                continue
            idx = int(key)
            cache._store[idx] = (val["k"], val["v"])
        return cache
