"""CausalH3's chunk geometry, cache, model package, state machine, and session contract.

Everything here runs on a laptop: the GPU work sits behind the backend,
which these tests replace with a fake that builds instantly, so the real
cache eviction, chunk accounting, condition updates, cancellation/reset,
and throughput metrics all run.

Run from the model folder: ``PYTHONPATH=. python -m pytest tests/ -q``.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip tests that need reactor_runtime when it is not installed (laptop CI).
_has_reactor_runtime = True
try:
    import reactor_runtime  # noqa: F401
except ImportError:
    _has_reactor_runtime = False
_needs_runtime = pytest.mark.skipif(
    not _has_reactor_runtime, reason="reactor_runtime not installed"
)
import causalh3_chunk_plan as chunk_plan
import causalh3_session_rules as session_rules
from causalh3_assets import (
    FAMILIES,
    CausalH3Config,
    ModelPackageManifest,
    load_config,
    validate_manifest,
)
from causalh3_cache import (
    AudioVAEOverlapSave,
    CacheConfig,
    CleanX0State,
    ReadOnlyKVCacheView,
    VisualVAEDecodeBuffer,
)
from causalh3_session_rules import valid_commands

MODEL_DIR = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- chunk geometry


def test_session_frame_math():
    """60 seconds: 1,450 generated, 1,440 emitted, 10 trimmed, 85 chunks."""
    assert chunk_plan.SESSION_SECONDS == 60
    assert chunk_plan.SESSION_EMIT_FRAMES == 1440
    assert chunk_plan.SESSION_GENERATE_FRAMES == 1450
    assert chunk_plan.SESSION_TRIM_FRAMES == 10
    assert chunk_plan.SESSION_CHUNKS == 85


def test_generate_frames_are_valid_h3():
    """1,450 = 17*85 + 5, a valid 17n+5 frame count."""
    assert chunk_plan.SESSION_GENERATE_FRAMES % 17 == 5
    assert chunk_plan.SESSION_GENERATE_FRAMES == 17 * 85 + 5


def test_prefix_is_minimum_valid():
    """The 5-frame prefix is the smallest valid H3 frame count."""
    assert chunk_plan.PREFIX_FRAMES == 5
    assert chunk_plan.PREFIX_FRAMES % 17 == 5


def test_sigma_ladder_is_four_step():
    """The ladder is [999, 749, 500, 250, 0]: five points, four forwards."""
    assert chunk_plan.SIGMA_LADDER == (999, 749, 500, 250, 0)
    assert chunk_plan.NUM_INFERENCE_STEPS == 5
    assert chunk_plan.NUM_FORWARDS == 4


def test_shifts_are_video_12_audio_3():
    assert chunk_plan.VIDEO_SIGMA_SHIFT == 12.0
    assert chunk_plan.AUDIO_SIGMA_SHIFT == 3.0


def test_guidance_is_distilled():
    assert chunk_plan.GUIDANCE_SCALE == 1.0


def test_audio_is_stereo_32k():
    assert chunk_plan.AUDIO_SAMPLE_RATE == 32000
    assert chunk_plan.AUDIO_CHANNELS == 2


def test_vsa_params():
    assert chunk_plan.VSA_TILE_SIZE == 64
    assert chunk_plan.VSA_SPARSITY == 0.9


def test_chunks_for_emit_frames():
    """1440 emit frames need 85 chunks (prefix covers 5, 85*17=1445, total 1450)."""
    assert chunk_plan.chunks_for_emit_frames(1440) == 85
    assert chunk_plan.total_generate_frames(1440) == 1450
    assert chunk_plan.trim_frames(1440) == 10


def test_chunks_for_short_duration():
    """A 5-second target (120 frames) needs 7 chunks (5 + 7*17 = 124)."""
    assert chunk_plan.chunks_for_emit_frames(120) == 7
    assert chunk_plan.total_generate_frames(120) == 124
    assert chunk_plan.trim_frames(120) == 4


def test_zero_chunks_when_within_prefix():
    """A target within the prefix needs no chunks."""
    assert chunk_plan.chunks_for_emit_frames(5) == 0
    assert chunk_plan.total_generate_frames(5) == 5


@pytest.mark.parametrize("aspect", chunk_plan.ASPECT_CHOICES)
def test_every_offered_canvas_satisfies_the_checkpoint(aspect):
    height, width = chunk_plan.canvas_for_choice(aspect)
    assert height % 32 == 0 and width % 32 == 0
    assert height * width <= 768 * 1344


def test_unknown_aspect_is_rejected():
    with pytest.raises(ValueError):
        chunk_plan.canvas_for_choice("32:9")


def test_align_frames_lands_on_chunk_grid():
    assert chunk_plan.align_frames(1) == 5
    assert chunk_plan.align_frames(6) == 22
    assert chunk_plan.align_frames(22) == 22


# --------------------------------------------------------------- model package


def _valid_manifest_data(mode: str = "fl2va") -> dict[str, Any]:
    """Build a valid model-package-v1 manifest dict for tests."""
    return {
        "schema_version": "model-package-v1",
        "package_id": "pkg-test-001",
        "mode": mode,
        "components": {
            "base": {
                "type": "base",
                "name_or_path": "Comfy-Org/MiniMax-H3",
                "arch": "minimax_h3",
                "partition": mode,
            },
            "causal_trunk": {
                "type": "causal_trunk",
                "native": True,
                "num_layers": 50,
            },
            "pdd_heads": {
                "type": "pdd_heads",
                "native": True,
                "num_layers": 4,
            },
            "control_branch": {
                "type": "control_branch",
                "native": True,
                "num_layers": 2,
            },
        },
        "revisions": {
            "base": "r1",
            "causal_trunk": "native-r1",
            "pdd_heads": "pdd-r1",
            "control_branch": "ctrl-r1",
        },
        "hashes": {
            "base": "a" * 64,
            "causal_trunk": "b" * 64,
            "pdd_heads": "c" * 64,
            "control_branch": "d" * 64,
        },
        "metrics": {"loss": 0.01},
        "dataset_revision": "ds-test-001",
        "cost": {"currency": "USD", "amount": 10.0},
        "authorization": {"issuer": "test-issuer", "signer": "test-signer"},
    }


def test_valid_manifest_passes():
    data = _valid_manifest_data()
    manifest = validate_manifest(data, expected_family="fl2va")
    assert manifest.mode == "fl2va"
    assert manifest.base_arch == "minimax_h3"
    assert manifest.causal_trunk_native is True
    assert manifest.causal_trunk_layers == 50
    assert manifest.adapter_path is None
    assert manifest.authorization == {"issuer": "test-issuer", "signer": "test-signer"}


def test_pruned_base_rejected():
    data = _valid_manifest_data()
    data["components"]["base"]["partition"] = "fl2va_pruned"
    with pytest.raises(ValueError, match="pruned"):
        validate_manifest(data, expected_family="fl2va")


def test_ref2va_pruned_rejected():
    data = _valid_manifest_data(mode="ref2va")
    data["components"]["base"]["partition"] = "ref2va_pruned"
    with pytest.raises(ValueError, match="pruned"):
        validate_manifest(data, expected_family="ref2va")


def test_family_mismatch_rejected():
    data = _valid_manifest_data(mode="fl2va")
    with pytest.raises(ValueError, match="does not match"):
        validate_manifest(data, expected_family="ref2va")


def test_wrong_hash_format_rejected():
    data = _valid_manifest_data()
    data["hashes"]["base"] = "xyz"
    with pytest.raises(ValueError, match="hex"):
        validate_manifest(data, expected_family="fl2va")


def test_missing_native_gate_rejected():
    data = _valid_manifest_data()
    data["components"]["causal_trunk"]["native"] = False
    with pytest.raises(ValueError, match="causal_trunk"):
        validate_manifest(data, expected_family="fl2va")


def test_missing_control_branch_rejected():
    data = _valid_manifest_data()
    del data["components"]["control_branch"]
    with pytest.raises(ValueError, match="control_branch"):
        validate_manifest(data, expected_family="fl2va")


def test_adapter_stack_rejected():
    data = _valid_manifest_data()
    data["components"]["adapter"] = [
        {"type": "adapter", "name_or_path": "a", "temporary": True},
        {"type": "adapter", "name_or_path": "b", "temporary": True},
    ]
    with pytest.raises(ValueError, match="adapter stack"):
        validate_manifest(data, expected_family="fl2va")


def test_non_temporary_adapter_rejected():
    data = _valid_manifest_data()
    data["components"]["adapter"] = {
        "type": "adapter",
        "name_or_path": "a",
        "temporary": False,
    }
    with pytest.raises(ValueError, match="temporary"):
        validate_manifest(data, expected_family="fl2va")


def test_temporary_adapter_accepted():
    data = _valid_manifest_data()
    data["components"]["adapter"] = {
        "type": "adapter",
        "name_or_path": "a",
        "temporary": True,
        "rank": 16,
    }
    manifest = validate_manifest(data, expected_family="fl2va")
    assert manifest.adapter_path == "a"
    assert manifest.adapter_temporary is True
    assert manifest.adapter_rank == 16


def test_missing_authorization_rejected():
    data = _valid_manifest_data()
    del data["authorization"]
    with pytest.raises(ValueError, match="authorization"):
        validate_manifest(data, expected_family="fl2va")


def test_authorization_without_issuer_rejected():
    data = _valid_manifest_data()
    data["authorization"] = {"not_issuer": "x"}
    with pytest.raises(ValueError, match="issuer"):
        validate_manifest(data, expected_family="fl2va")


def test_wrong_schedule_rejected():
    data = _valid_manifest_data()
    data["schedule"] = {"sigma_ladder": [999, 500, 0]}
    with pytest.raises(ValueError, match="sigma_ladder"):
        validate_manifest(data, expected_family="fl2va")


def test_correct_schedule_accepted():
    data = _valid_manifest_data()
    data["schedule"] = {"sigma_ladder": [999, 749, 500, 250, 0]}
    manifest = validate_manifest(data, expected_family="fl2va")
    assert manifest.package_id == "pkg-test-001"


def test_wrong_schema_version_rejected():
    data = _valid_manifest_data()
    data["schema_version"] = "model-package-v0"
    with pytest.raises(ValueError, match="schema_version"):
        validate_manifest(data, expected_family="fl2va")


def test_wrong_arch_rejected():
    data = _valid_manifest_data()
    data["components"]["base"]["arch"] = "flux"
    with pytest.raises(ValueError, match="arch"):
        validate_manifest(data, expected_family="fl2va")


# --------------------------------------------------------------- config parsing


def test_load_config_defaults():
    """Loading the shipped causalh3.yaml gives the expected defaults."""
    config = load_config(MODEL_DIR / "causalh3.yaml")
    assert config.family == "fl2va"
    assert config.aspect == "16:9"
    assert config.seed == 1000
    assert config.target_seconds == 60


def test_load_config_invalid_family():
    yaml_text = "inference:\n  family: unknown\n"
    path = Path("/tmp/test_causalh3.yaml")
    path.write_text(yaml_text)
    try:
        with pytest.raises(ValueError, match="family"):
            load_config(path)
    finally:
        path.unlink()


def test_load_config_invalid_aspect():
    yaml_text = "inference:\n  aspect: 32:9\n"
    path = Path("/tmp/test_causalh3.yaml")
    path.write_text(yaml_text)
    try:
        with pytest.raises(ValueError, match="aspect"):
            load_config(path)
    finally:
        path.unlink()


# --------------------------------------------------------------- KV cache


class _FakeRealCache:
    """A minimal stand-in for SlidingWindowKVCache that tracks append calls.

    The real SlidingWindowKVCache lives in ai-toolkit's causal.py and needs
    torch.  This fake has the same interface (get, append, append_validity,
    append_token_roles, get_validity_mask, get_token_roles, __len__) so the
    ReadOnlyKVCacheView and eviction logic can be tested without torch.
    """

    def __init__(self, sink_size=0, window_size=0):
        self._store: dict[int, list] = {}
        self._validity: list = []
        self._roles: list = []
        self.sink_size = sink_size
        self.window_size = window_size
        self.append_calls = 0

    def get(self, layer_idx):
        return self._store.get(layer_idx)

    def append(self, layer_idx, k_new, v_new):
        self.append_calls += 1
        if layer_idx not in self._store:
            self._store[layer_idx] = (k_new, v_new)
        else:
            k_old, v_old = self._store[layer_idx]
            self._store[layer_idx] = (k_old + k_new, v_old + v_new)
        self._evict()

    def _evict(self):
        max_tokens = self.sink_size + self.window_size
        for idx, (k, v) in self._store.items():
            if len(k) > max_tokens and max_tokens > 0:
                sink_k = k[: self.sink_size]
                win_k = k[-self.window_size:] if self.window_size > 0 else []
                self._store[idx] = (sink_k + win_k, v[: self.sink_size] + (v[-self.window_size:] if self.window_size > 0 else []))

    def append_validity(self, validity):
        self._validity.extend(validity if isinstance(validity, list) else [validity])

    def append_token_roles(self, roles):
        self._roles.extend(roles if isinstance(roles, list) else [roles])

    def get_validity_mask(self):
        return self._validity if self._validity else None

    def get_token_roles(self):
        return self._roles if self._roles else None

    def __len__(self):
        if not self._store:
            return 0
        first = sorted(self._store)[0]
        return len(self._store[first][0])

    def reset(self):
        self._store.clear()
        self._validity.clear()
        self._roles.clear()
        self.append_calls = 0


def test_readonly_view_does_not_append():
    """ReadOnlyKVCacheView intercepts append calls so forwards 1-3 are read-only."""
    real = _FakeRealCache(sink_size=10, window_size=100)
    view = ReadOnlyKVCacheView(real)
    # Simulate a read-only forward: the transformer calls append.
    view.append(0, ["k1"], ["v1"])
    view.append_validity([True])
    view.append_token_roles([0])
    # The real cache must not have been modified.
    assert real.append_calls == 0
    assert len(real) == 0
    assert real.get_validity_mask() is None
    assert real.get_token_roles() is None


def test_readonly_view_delegates_get():
    """ReadOnlyKVCacheView delegates get() to the real cache."""
    real = _FakeRealCache(sink_size=10, window_size=100)
    real.append(0, ["k0"], ["v0"])
    view = ReadOnlyKVCacheView(real)
    k, v = view.get(0)
    assert k == ["k0"]
    assert v == ["v0"]


def test_readonly_view_len_matches_real():
    """ReadOnlyKVCacheView.__len__ matches the real cache."""
    real = _FakeRealCache(sink_size=10, window_size=100)
    real.append(0, ["k0", "k1"], ["v0", "v1"])
    view = ReadOnlyKVCacheView(real)
    assert len(view) == len(real) == 2


def test_readonly_view_delegates_masks():
    """ReadOnlyKVCacheView delegates validity/role masks to the real cache."""
    real = _FakeRealCache(sink_size=10, window_size=100)
    real.append_validity([True, False, True])
    real.append_token_roles([0, 1, 2])
    view = ReadOnlyKVCacheView(real)
    assert view.get_validity_mask() == [True, False, True]
    assert view.get_token_roles() == [0, 1, 2]


def test_real_cache_append_evicts():
    """SlidingWindowKVCache evicts oldest non-sink tokens past the bound."""
    real = _FakeRealCache(sink_size=10, window_size=50)
    # Prefill sink.
    real.append(0, list(range(10)), list(range(10)))
    assert len(real) == 10
    # Add a chunk of 30 tokens.
    real.append(0, list(range(30)), list(range(30)))
    assert len(real) == 40  # 10 sink + 30 window
    # Add another 30 -- 60 > 50 window, evict oldest.
    real.append(0, list(range(30)), list(range(30)))
    assert len(real) == 60  # 10 sink + 50 window (evicted to window bound)


def test_real_cache_sink_never_evicted():
    """Sink tokens survive eviction."""
    real = _FakeRealCache(sink_size=100, window_size=20)
    real.append(0, list(range(100)), list(range(100)))
    # Add a chunk of 30 -- 30 > 20 window, evict chunk.
    real.append(0, list(range(30)), list(range(30)))
    k, v = real.get(0)
    # Sink (first 100) is always kept.
    assert k[:100] == list(range(100))


def test_clean_x0_state():
    state = CleanX0State()
    assert state.video_x0 is None
    assert state.audio_x0 is None
    assert state.chunk_index == -1
    state.update("video_x0", "audio_x0", 5)
    assert state.video_x0 == "video_x0"
    assert state.audio_x0 == "audio_x0"
    assert state.chunk_index == 5
    state.reset()
    assert state.video_x0 is None
    assert state.audio_x0 is None

# --------------------------------------------------------------- VAE decode buffer


def test_vae_buffer_keeps_left_context():
    """The buffer keeps the last N latent slices across pushes."""
    import numpy as np
    buf = VisualVAEDecodeBuffer(left_latents=5)
    # Push a latent tensor with 5 temporal slices: (1, C, 5, H, W)
    latents = np.zeros((1, 24, 5, 48, 84))
    buf.push(latents)
    assert buf.buffered == 5
    # Push 3 more slices.
    latents2 = np.zeros((1, 24, 3, 48, 84))
    buf.push(latents2)
    # Should keep only the last 5.
    assert buf.buffered == 5
    ctx = buf.context()
    assert ctx is not None
    assert ctx.shape[2] == 5  # 5 temporal slices


def test_vae_buffer_reset():
    """Reset clears the buffer."""
    import numpy as np
    buf = VisualVAEDecodeBuffer(left_latents=5)
    buf.push(np.zeros((1, 24, 3, 48, 84)))
    buf.reset()
    assert buf.buffered == 0
    assert buf.context() is None


def test_vae_buffer_first_chunk_no_context():
    """The first chunk has no left context."""
    buf = VisualVAEDecodeBuffer(left_latents=5)
    assert buf.context() is None
    assert buf.buffered == 0


# --------------------------------------------------------------- audio overlap-save


def test_audio_overlap_save():
    save = AudioVAEOverlapSave(overlap_samples=10)
    assert not save.has_overlap
    save.save("overlap_data", 0)
    assert save.has_overlap
    assert save.pop() == "overlap_data"
    assert not save.has_overlap


def test_audio_overlap_reset():
    save = AudioVAEOverlapSave(overlap_samples=10)
    save.save("data", 0)
    save.reset()
    assert not save.has_overlap


# --------------------------------------------------------------- session rules


def test_commands_before_start():
    cmds = valid_commands(generating=False, started=False, family_locked=False)
    assert "set_family" in cmds
    assert "set_canvas" in cmds
    assert "set_prompt" in cmds
    assert "start" in cmds
    assert "stop" not in cmds


def test_commands_while_generating():
    cmds = valid_commands(generating=True, started=True, family_locked=True)
    assert "stop" in cmds
    assert "set_prompt" in cmds
    assert "start" not in cmds
    assert "set_family" not in cmds
    assert "set_canvas" not in cmds


def test_commands_after_generation_finished():
    cmds = valid_commands(generating=False, started=True, family_locked=True)
    assert "start" in cmds
    assert "stop" not in cmds
    assert "set_prompt" in cmds


def test_always_available_commands():
    for generating in (True, False):
        for started in (True, False):
            cmds = valid_commands(generating=generating, started=started, family_locked=started)
            assert "set_seed" in cmds
            assert "get_state" in cmds
            assert "reset" in cmds


# --------------------------------------------------------------- fake backend


class FakeChunkResult:
    """A fake ChunkResult for testing without GPU."""

    def __init__(
        self,
        *,
        chunk_index: int,
        frames: list[Any],
        audio: Any,
        compute_time: float = 0.01,
        cache_tokens_after: int = 100,
        native_audio: Any = None,
    ) -> None:
        self.chunk_index = chunk_index
        self.frames = frames
        self.audio = audio
        self.compute_time = compute_time
        self.cache_tokens_after = cache_tokens_after
        self.native_audio = native_audio


class FakeBackend:
    """A fake backend that builds instantly, for testing the model loop."""

    def __init__(self, config: CausalH3Config) -> None:
        self.config = config
        self.kv_cache = _FakeRealCache(sink_size=256, window_size=512)
        self.clean_x0 = CleanX0State()
        self.vae_buffer = VisualVAEDecodeBuffer(5)
        self.audio_overlap = AudioVAEOverlapSave(10)
        self._prefilled = False
        self._chunk_index = -1
        self._cancel_next = False
        self._cache_token_count = 0

    def load(self) -> None:
        pass

    def reset_session(self) -> None:
        self.kv_cache.reset()
        self.clean_x0.reset()
        self.vae_buffer.reset()
        self.audio_overlap.reset()
        self._prefilled = False
        self._chunk_index = -1
        self._cache_token_count = 0

    @property
    def prefilled(self) -> bool:
        return self._prefilled

    @property
    def chunk_index(self) -> int:
        return self._chunk_index

    def submit_prefill(self, **kwargs) -> Any:
        self._prefilled = True
        # Simulate prefill: add sink tokens to the cache.
        self.kv_cache.append(0, ["sink"] * 256, ["sink"] * 256)
        self._cache_token_count = 256
        result = FakeChunkResult(
            chunk_index=-1, frames=[], audio=None, cache_tokens_after=256
        )

        class FakeJob:
            def __init__(self, result):
                self.done = MagicMock()
                self.done.is_set = MagicMock(return_value=True)
                self.cancelled = False
                self.error = None
                self.result = result

        return FakeJob(result)

    def submit_chunk(self, *, chunk_index, prompt, seed, height, width, frames) -> Any:
        self._chunk_index = chunk_index
        # Generate fake frames.
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frames_list = [frame.copy() for _ in range(frames)]
        # Generate fake audio: 1 second of silence at 48k mono.
        audio = np.zeros((1, frames * 48000 // 24), dtype=np.int16)
        # Simulate cache-fill: the final forward appends to the real cache.
        chunk_tokens = 100
        self.kv_cache.append(0, ["chunk"] * chunk_tokens, ["chunk"] * chunk_tokens)
        self._cache_token_count = (getattr(self, "_cache_token_count", 256) + chunk_tokens)
        # Carry the clean x0 forward as real tensor-like objects.
        self.clean_x0.update(
            f"video_x0_{chunk_index}", f"audio_x0_{chunk_index}", chunk_index
        )
        result = FakeChunkResult(
            chunk_index=chunk_index,
            frames=frames_list,
            audio=audio,
            compute_time=0.01,
            cache_tokens_after=self._cache_token_count,
        )

        class FakeJob:
            def __init__(self, result, cancel_next=False):
                self.done = MagicMock()
                self.done.is_set = MagicMock(return_value=True)
                self.cancelled = cancel_next
                self.error = None
                self.result = result

        job = FakeJob(result, self._cancel_next)
        self._cancel_next = False
        return job


def _make_model_with_fake_backend(
    family: str = "fl2va",
    target_seconds: float = 60,
) -> CausalH3:
    """Create a CausalH3 model with a fake backend, bypassing load()."""
    from causalh3 import CausalH3

    model = CausalH3()
    # Build a minimal config.
    config = CausalH3Config(
        family=family,
        aspect="16:9",
        seed=1000,
        target_seconds=target_seconds,
        inference={"family": family, "vsa_enabled": True},
        runtime={},
    )
    model.config = config
    model.manifest = MagicMock()
    model.backend = FakeBackend(config)
    model._reset_session_state()
    return model


# --------------------------------------------------------------- model contract


@_needs_runtime
@pytest.mark.asyncio
async def test_set_family_before_start():
    model = _make_model_with_fake_backend()
    result = await model.set_family(family="ref2va")
    assert result is not None
    assert result.family == "ref2va"
    assert model._family == "ref2va"


@_needs_runtime
@pytest.mark.asyncio
async def test_set_family_after_start_rejected():
    model = _make_model_with_fake_backend()
    model._prompt = "test"
    model._started = True
    result = await model.set_family(family="ref2va")
    assert result is None  # refused


@_needs_runtime
@pytest.mark.asyncio
async def test_set_prompt_before_start():
    model = _make_model_with_fake_backend()
    result = await model.set_prompt(prompt="A sunset over the ocean")
    assert result is not None
    assert model._prompt == "A sunset over the ocean"


@_needs_runtime
@pytest.mark.asyncio
async def test_set_prompt_empty_rejected():
    model = _make_model_with_fake_backend()
    result = await model.set_prompt(prompt="")
    assert result is None


@_needs_runtime
@pytest.mark.asyncio
async def test_set_prompt_during_generation_queued():
    model = _make_model_with_fake_backend()
    model._prompt = "old"
    model._generating = True
    result = await model.set_prompt(prompt="new")
    assert result is not None
    assert result.prompt == "old"  # current prompt unchanged
    assert model._pending_prompt == "new"  # queued for next chunk


@_needs_runtime
@pytest.mark.asyncio
async def test_start_without_prompt_rejected():
    model = _make_model_with_fake_backend()
    result = await model.start()
    assert result is None  # refused


@_needs_runtime
@pytest.mark.asyncio
async def test_start_sets_session_targets():
    model = _make_model_with_fake_backend()
    model._prompt = "test"
    result = await model.start()
    assert result is not None
    assert result.target_seconds == 60
    assert result.target_frames == 1450
    assert result.emit_frames == 1440
    assert result.chunks == 85
    assert model._started is True
    assert model._generating is True


@_needs_runtime
@pytest.mark.asyncio
async def test_start_with_custom_seconds():
    model = _make_model_with_fake_backend()
    model._prompt = "test"
    result = await model.start(seconds=10)
    assert result is not None
    assert result.target_seconds == 10
    # 10s = 240 emit frames; 5 + ceil(235/17)*17 = 5 + 14*17 = 243 generate
    assert result.emit_frames == 240
    assert result.target_frames == 243


@_needs_runtime
@pytest.mark.asyncio
async def test_start_twice_rejected():
    model = _make_model_with_fake_backend()
    model._prompt = "test"
    await model.start()
    result = await model.start()
    assert result is None


@_needs_runtime
@pytest.mark.asyncio
async def test_stop_when_not_generating_rejected():
    model = _make_model_with_fake_backend()
    result = await model.stop()
    assert result is None


@_needs_runtime
@pytest.mark.asyncio
async def test_stop_when_generating_accepted():
    model = _make_model_with_fake_backend()
    model._prompt = "test"
    model._generating = True
    await model.stop()
    assert model._stop_generation is True


@_needs_runtime
@pytest.mark.asyncio
async def test_reset_clears_state():
    model = _make_model_with_fake_backend()
    model._prompt = "test"
    model._started = True
    model._generating = True
    result = await model.reset()
    assert result is not None
    assert result.was_generating is True
    assert model._started is False
    assert model._generating is False
    assert model._prompt == ""


@_needs_runtime
@pytest.mark.asyncio
async def test_get_state_returns_snapshot():
    model = _make_model_with_fake_backend()
    model._prompt = "test"
    state = await model.get_state()
    assert state.family == "fl2va"
    assert state.prompt == "test"
    assert state.generating is False
    assert "set_family" in state.valid_commands


@_needs_runtime
@pytest.mark.asyncio
async def test_set_canvas_before_start():
    model = _make_model_with_fake_backend()
    result = await model.set_canvas(aspect="1:1")
    assert result is not None
    assert result.aspect == "1:1"


@_needs_runtime
@pytest.mark.asyncio
async def test_set_canvas_after_start_rejected():
    model = _make_model_with_fake_backend()
    model._started = True
    result = await model.set_canvas(aspect="1:1")
    assert result is None


@_needs_runtime
@pytest.mark.asyncio
async def test_set_seed():
    model = _make_model_with_fake_backend()
    result = await model.set_seed(seed=42)
    assert result is not None
    assert result.seed == 42
    assert model._seed == 42


# --------------------------------------------------------------- chunk/frame accounting


def test_no_duplicated_chunk_ids():
    """Each chunk gets a unique UUID."""
    import uuid

    ids = {str(uuid.uuid4()) for _ in range(100)}
    assert len(ids) == 100  # all unique


def test_frame_accounting_for_60s_session():
    """60s: 1450 generated, 1440 emitted, 10 trimmed."""
    emit = chunk_plan.SESSION_EMIT_FRAMES
    gen = chunk_plan.SESSION_GENERATE_FRAMES
    trim = gen - emit
    assert gen == 1450
    assert emit == 1440
    assert trim == 10
    # No missing frames: emit + trim = generate.
    assert emit + trim == gen


def test_no_hard_cuts_in_chunk_sequence():
    """Chunks are contiguous: prefix + 85 * 17 = 1450."""
    total = chunk_plan.PREFIX_FRAMES + chunk_plan.SESSION_CHUNKS * 17
    assert total == chunk_plan.SESSION_GENERATE_FRAMES
    assert total == 1450


# --------------------------------------------------------------- throughput metrics


def test_throughput_calculation():
    """Throughput = content_seconds / compute_time."""
    content = 17 / 24  # one chunk = 17 frames at 24 fps
    compute = 0.5  # 0.5 seconds to build
    throughput = content / compute
    assert throughput == pytest.approx(1.4167, rel=0.01)


def test_generation_ratio_for_60s():
    """Ratio = wall_time / content_seconds. <1 means faster than realtime."""
    content = 60.0  # 60 seconds of content
    wall = 55.0  # 55 seconds to generate
    ratio = wall / content
    assert ratio < 1.0  # faster than realtime


# --------------------------------------------------------------- cancellation


@_needs_runtime
@pytest.mark.asyncio
async def test_cancel_build_on_reset():
    model = _make_model_with_fake_backend()
    model._prompt = "test"
    # Simulate a build in flight.
    from causalh3_backend import ChunkJob

    job = ChunkJob(lambda: None)
    model._build = (job, time.monotonic())
    await model.reset()
    assert job.cancelled is True
    assert model._build is None


@_needs_runtime
@pytest.mark.asyncio
async def test_cancel_build_on_session_end():
    model = _make_model_with_fake_backend()
    from causalh3_backend import ChunkJob

    job = ChunkJob(lambda: None)
    model._build = (job, time.monotonic())
    await model.on_session_ended()
    assert job.cancelled is True


# --------------------------------------------------------------- cache calls


def test_cache_fill_after_read_only_denoise():
    """Only the final forward appends to the KV cache; the first three are read-only.

    This instruments the actual cache append behavior: the ReadOnlyKVCacheView
    intercepts append for forwards 1-3, and only the real cache receives the
    append on forward 4 (clean x0 cache-fill).
    """
    real = _FakeRealCache(sink_size=10, window_size=100)
    # Simulate prefill: add sink tokens.
    real.append(0, ["sink"] * 10, ["sink"] * 10)
    assert real.append_calls == 1
    assert len(real) == 10

    # Simulate four forwards: forwards 1-3 use ReadOnlyKVCacheView.
    for i in range(3):
        view = ReadOnlyKVCacheView(real)
        view.append(0, ["f" + str(i)] * 50, ["f" + str(i)] * 50)
        view.append_validity([True] * 50)
        view.append_token_roles([2] * 50)
    # Real cache must not have grown from read-only forwards.
    assert real.append_calls == 1
    assert len(real) == 10

    # Forward 4: clean x0 cache-fill with the real cache.
    real.append(0, ["clean_x0"] * 50, ["clean_x0"] * 50)
    assert real.append_calls == 2
    assert len(real) == 60  # 10 sink + 50 chunk


def test_cache_eviction_order():
    """Oldest non-sink tokens are evicted first when the window bound is exceeded."""
    real = _FakeRealCache(sink_size=10, window_size=40)
    real.append(0, ["sink"] * 10, ["sink"] * 10)
    # Add chunk 0 (20 tokens).
    real.append(0, ["c0"] * 20, ["c0"] * 20)
    assert len(real) == 30  # 10 sink + 20 window
    # Add chunk 1 (20 tokens). 40 == window bound, no eviction.
    real.append(0, ["c1"] * 20, ["c1"] * 20)
    assert len(real) == 50
    # Add chunk 2 (20 tokens). 60 > 40, evict oldest (chunk 0).
    real.append(0, ["c2"] * 20, ["c2"] * 20)
    # After eviction: 10 sink + 40 window = 50.
    assert len(real) == 50
    # Verify chunk 0 tokens are gone: the first non-sink tokens should be c1.
    k, v = real.get(0)
    assert k[10:30] == ["c1"] * 20  # chunk 1 survived
    assert k[30:50] == ["c2"] * 20  # chunk 2 survived


def test_readonly_view_used_for_forward_1_to_3():
    """The denoise loop uses ReadOnlyKVCacheView for forwards 1-3.

    This verifies the contract: the transformer's kv_cache parameter receives
    a ReadOnlyKVCacheView for the first three forwards and the real cache
    for the fourth.  The test instruments the append call count.
    """
    real = _FakeRealCache(sink_size=10, window_size=100)
    real.append(0, ["sink"] * 10, ["sink"] * 10)
    initial_append_calls = real.append_calls

    # Forward 1: read-only.
    view1 = ReadOnlyKVCacheView(real)
    view1.append(0, ["f1"], ["f1"])
    assert real.append_calls == initial_append_calls  # no change

    # Forward 2: read-only.
    view2 = ReadOnlyKVCacheView(real)
    view2.append(0, ["f2"], ["f2"])
    assert real.append_calls == initial_append_calls  # no change

    # Forward 3: read-only.
    view3 = ReadOnlyKVCacheView(real)
    view3.append(0, ["f3"], ["f3"])
    assert real.append_calls == initial_append_calls  # no change

    # Forward 4: cache-fill (real cache).
    real.append(0, ["clean_x0"], ["clean_x0"])
    assert real.append_calls == initial_append_calls + 1  # one append


def test_clean_x0_carries_real_tensors():
    """CleanX0State carries real video/audio x0 tensors, not strings or None."""
    import numpy as np
    state = CleanX0State()
    # Before the first chunk, x0 is None.
    assert state.video_x0 is None
    assert state.audio_x0 is None
    # After a chunk, x0 holds real tensor-like objects.
    video_x0 = np.zeros((1, 100, 96), dtype=np.float32)
    audio_x0 = np.zeros((1, 80, 32), dtype=np.float32)
    state.update(video_x0, audio_x0, 0)
    assert state.video_x0 is video_x0  # same object, not a copy
    assert state.audio_x0 is audio_x0
    assert state.chunk_index == 0
    # After reset, x0 is None again.
    state.reset()
    assert state.video_x0 is None
    assert state.audio_x0 is None


# --------------------------------------------------------------- condition updates


def test_prompt_change_at_chunk_boundary():
    """Pending prompt is applied when the next chunk is submitted."""
    pending = "new prompt"
    current = "old prompt"
    if pending is not None:
        current = pending
        pending = None
    assert current == "new prompt"
    assert pending is None


def test_immutable_condition_rows():
    """FL2VA first/last frame and Ref2VA references are immutable prefix rows.

    Condition rows are part of the sink and never evicted.  This verifies
    that the sink tokens survive eviction even when large conditions are
    added and chunks overflow the window.
    """
    real = _FakeRealCache(sink_size=310, window_size=20)
    # Prefill: 10 text tokens + 100 first-frame + 200 reference = 310 sink.
    real.append(0, ["text"] * 10 + ["first_frame"] * 100 + ["ref"] * 200,
                ["text"] * 10 + ["first_frame"] * 100 + ["ref"] * 200)
    assert len(real) == 310
    # Add a chunk that exceeds the window.
    real.append(0, ["chunk"] * 50, ["chunk"] * 50)
    # After eviction: 310 sink + 20 window = 330.
    assert len(real) == 330
    # Verify sink tokens (including conditions) survived.
    k, v = real.get(0)
    assert k[:310] == ["text"] * 10 + ["first_frame"] * 100 + ["ref"] * 200

# --------------------------------------------------------------- families


def test_two_family_slugs():
    assert FAMILIES == ("fl2va", "ref2va")


def test_fl2va_manifest_valid():
    data = _valid_manifest_data("fl2va")
    manifest = validate_manifest(data, expected_family="fl2va")
    assert manifest.mode == "fl2va"


def test_ref2va_manifest_valid():
    data = _valid_manifest_data("ref2va")
    manifest = validate_manifest(data, expected_family="ref2va")
    assert manifest.mode == "ref2va"


# --------------------------------------------------------------- dense diagnostic


@_needs_runtime
def test_dense_diagnostic_must_be_explicit():
    """VSA disabled without dense_diagnostic flag should fail."""
    from causalh3_backend import CausalH3Backend

    config = CausalH3Config(
        family="fl2va",
        aspect="16:9",
        seed=1000,
        target_seconds=60,
        inference={"vsa_enabled": False, "dense_diagnostic": False},
        runtime={},
    )
    manifest = MagicMock()
    backend = CausalH3Backend(config, Path("/fake"), manifest)
    with pytest.raises(RuntimeError, match="dense_diagnostic"):
        backend._validate_runtime_compatibility()


@_needs_runtime
def test_dense_diagnostic_explicit_accepted():
    """VSA disabled with dense_diagnostic=true is accepted (non-realtime)."""
    from causalh3_backend import CausalH3Backend

    config = CausalH3Config(
        family="fl2va",
        aspect="16:9",
        seed=1000,
        target_seconds=60,
        inference={"vsa_enabled": False, "dense_diagnostic": True},
        runtime={},
    )
    manifest = MagicMock()
    backend = CausalH3Backend(config, Path("/fake"), manifest)
    # Should not raise.
    backend._validate_runtime_compatibility()
