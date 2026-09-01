"""CausalH3 as a Reactor model: a persistent causal video-and-audio stream.

CausalH3 is MiniMax-H3 served as a *persistent* stream: one continuous
generation session whose KV cache and clean x0 state carry forward across
chunks, with no hard cuts at clip boundaries.  This is the fundamental
difference from ``fast-h3``, which generates independent 5-15 second clips.

The model prefills text and conditions once (the KV cache sink), then
generates 5-latent media chunks continuously.  Each chunk executes four
read-only denoise forwards; only the final clean x0 is cache-filled.  The
sink/window eviction keeps the cache bounded.  Prompt changes take effect
at the next chunk boundary; conditions (first/last frame, references) are
immutable prefix rows.

Two family slugs share one model class:

- ``fl2va``: text-to-video, image-to-video, and first-last-frame conditioning.
- ``ref2va``: reference-to-video with image/video/audio reference blocks.

The 60-second session command generates 1,450 valid H3 frames (85 chunks of
17 frames plus a 5-frame prefix), emits the first 1,440 at 24 fps (exactly
60 seconds), and reports honest generation wall time and ratio.

Layout:
  * ``causalh3_types.py``         -- everything a client sees (tracks, messages).
  * ``causalh3_chunk_plan.py``    -- chunk geometry (frame counts, sigma ladder).
  * ``causalh3_assets.py``        -- model-package validation and config parsing.
  * ``causalh3_cache.py``         -- bounded KV cache, clean x0, decode buffers.
  * ``causalh3_backend.py``       -- the FastVideo engine and worker thread.
  * ``causalh3_session_rules.py`` -- which commands each state accepts.
  * ``causalh3.yaml``             -- the generation recipe and cache bounds.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

from reactor_runtime import (
    ClientInfo,
    InputField,
    ReactorModel,
    connected,
    event,
    get_weights_path,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger

import causalh3_chunk_plan as chunk_plan
import causalh3_session_rules as session_rules
from causalh3_assets import (
    FAMILIES,
    CausalH3Config,
    load_config,
    load_manifest,
    resolve_model_path,
    require_weights,
    validate_manifest,
)
from causalh3_backend import OUTPUT_SAMPLE_RATE, ChunkJob, CausalH3Backend
from causalh3_cache import CacheConfig
from causalh3_types import (
    MAX_METADATA_CHARS,
    MAX_PROMPT_CHARS,
    CanvasAccepted,
    ChunkFinished,
    ChunkGenerated,
    ChunkStarted,
    CausalH3Output,
    CommandError,
    FamilyAccepted,
    PromptAccepted,
    SeedAccepted,
    SessionFinished,
    SessionReset,
    SessionStarted,
    StateUpdate,
)

logger = get_logger(__name__)

FRAME_RATE = chunk_plan.FPS

# Frames per emitted slice.  Same as fast-h3: the runtime recorder's feed
# queue cannot absorb one-second bursts.
EMIT_FRAMES = 3

# How often the idle loop re-checks for a finished build.
POLL_SECONDS = 0.05


class CausalH3(ReactorModel):
    """Generate a persistent causal video-and-audio stream from one prompt."""

    # Pinned: the emitter is a strict 24 fps metronome.
    fps = FRAME_RATE
    # Two seconds of transport-side tolerance at 24 fps.
    buffer_size = 48

    def __init__(self) -> None:
        """Create the model shell; everything session-scoped arrives in load()."""
        super().__init__()
        self._build: tuple[ChunkJob, float] | None = None
        self._session_start_time: float | None = None
        self._generation_wall_time: float = 0.0

    # ------------------------------------------------------------------ load

    def load(self, config_path: Path | None) -> None:
        """Parse the config, validate the model package, and build the warm engine.

        Runs once at startup, before any session.  The runtime marks the
        pod ready only when this returns, so the backend's warm-up means a
        deployed pod never builds a cold chunk.

        Args:
            config_path: Path to ``causalh3.yaml``; its ``inference`` block
                is the generation recipe and cache bounds, and its
                ``runtime`` block holds the weight layout and engine shape.
        """
        self.config = load_config(config_path)
        weights = get_weights_path()
        model_path = resolve_model_path(self.config, weights)

        # Load and validate the model-package-v1 manifest.
        manifest_data = load_manifest(weights)
        self.manifest = validate_manifest(
            manifest_data,
            expected_family=self.config.family,
        )
        require_weights(weights, model_path, self.manifest)

        self.backend = CausalH3Backend(self.config, model_path, self.manifest)
        self._reset_session_state()
        self.backend.load()
        logger.info(
            "causal-h3 loaded",
            family=self.config.family,
            package_id=self.manifest.package_id,
        )

    # -------------------------------------------------------- session state

    def _reset_session_state(self) -> None:
        """Return every session-scoped field to its default.

        Called once at ``load()`` and at every ``@session_started``.
        """
        if self._build is not None:
            job, _submitted = self._build
            job.cancelled = True
            self._build = None

        self._family: str = self.config.family
        self._prompt: str = ""
        self._seed: int = self.config.seed
        self._aspect: str = self.config.aspect
        self._target_seconds: float = self.config.target_seconds

        # Session lifecycle: idle -> started -> generating -> finished.
        self._started: bool = False
        self._generating: bool = False
        self._finished: bool = False

        # Chunk tracking.
        self._current_chunk_index: int = -1
        self._target_emit_frames: int = 0
        self._target_generate_frames: int = 0
        self._total_chunks: int = 0

        # Progress counters.
        self._frames_generated: int = 0
        self._frames_emitted: int = 0
        self._seconds_emitted: float = 0.0
        self._last_compute_time: float = 0.0
        self._last_throughput: float = 0.0

        # Pending prompt change (takes effect at next chunk boundary).
        self._pending_prompt: str | None = None

        # Emission state.
        self._emit_queue: list[tuple[list[Any], Any, int]] = []  # (frames, audio, chunk_index)
        self._stop_generation: bool = False

        # Reset backend cache state.
        if hasattr(self, "backend"):
            self.backend.reset_session()

        self._session_start_time = None
        self._generation_wall_time = 0.0

    def _canvas(self) -> tuple[int, int]:
        """The ``(height, width)`` this session generates at."""
        return chunk_plan.canvas_for_choice(self._aspect)

    def _snapshot(self) -> StateUpdate:
        """Everything a client can observe, in one message."""
        height, width = self._canvas()
        return StateUpdate(
            family=self._family,
            prompt=self._prompt,
            seed=self._seed,
            aspect=self._aspect,
            width=width,
            height=height,
            generating=self._generating,
            current_chunk_index=self._current_chunk_index,
            frames_generated=self._frames_generated,
            frames_emitted=self._frames_emitted,
            seconds_emitted=round(self._seconds_emitted, 2),
            compute_time=round(self._last_compute_time, 3),
            throughput_x=round(self._last_throughput, 2),
            cache_tokens=len(self.backend.kv_cache) if hasattr(self, "backend") and self.backend.kv_cache is not None else 0,
            valid_commands=session_rules.valid_commands(
                generating=self._generating,
                started=self._started,
                family_locked=self._started,
            ),
        )

    async def _send_state_update(self) -> None:
        """Broadcast the snapshot to every connected client."""
        await self.send(self._snapshot())

    async def _refuse(self, command: str, reason: str) -> None:
        """Reject a command: tell every client, and leave its reply bodyless."""
        logger.info("command refused", command=command, reason=reason)
        await self.send(CommandError(command=command, reason=reason))

    # ------------------------------------------------------------ lifecycle

    @session_started
    async def on_session_started(self) -> None:
        """Clear the session state so a new session inherits nothing."""
        self._reset_session_state()

    @session_ended
    async def on_session_ended(self) -> None:
        """Drop the session's work."""
        self._stop_generation = True
        if self._build is not None:
            job, _submitted = self._build
            job.cancelled = True
        if hasattr(self, "backend"):
            self.backend.reset_session()

    @connected
    async def on_connect(self, client: ClientInfo) -> None:
        """Greet the joining client with the full state."""
        await client.send(self._snapshot())

    # ------------------------------------------------------------- commands

    @event(
        name="set_family",
        description=(
            "Choose the model family: `fl2va` for text/image/first-last-frame "
            "conditioning, or `ref2va` for reference-block conditioning. Only "
            "valid before `start`; the family is locked once generation begins. "
            "Emits `family_accepted` and `state_update`, or `command_error` "
            "when the session has already started or the slug is unknown."
        ),
    )
    async def set_family(
        self,
        family: str = InputField(
            default="fl2va",
            choices=list(FAMILIES),
            description=(
                "The family slug: `fl2va` or `ref2va`. `fl2va` handles "
                "text-to-video, image-to-video, and first-last-frame; `ref2va` "
                "handles image/video/audio references as immutable prefix rows."
            ),
        ),
    ) -> FamilyAccepted:
        """Set the model family; refused after `start`."""
        if self._started:
            await self._refuse(
                "set_family",
                "The family is locked once the session has started; "
                "`reset` to change it.",
            )
            return None
        if family not in FAMILIES:
            await self._refuse(
                "set_family",
                f"Unknown family {family!r}; choose from {list(FAMILIES)}.",
            )
            return None
        self._family = family
        await self._send_state_update()
        return FamilyAccepted(family=family)

    @event(
        name="set_prompt",
        description=(
            "Set the prompt driving generation. Before `start`, this is the "
            "initial prompt; after `start`, the new prompt takes effect at "
            "the next chunk boundary (the current chunk finishes with the "
            "old prompt). Emits `prompt_accepted` and `state_update`, or "
            "`command_error` when the prompt is empty."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=MAX_PROMPT_CHARS,
            moderate=True,
            description=(
                "What the stream should show, up to 800 characters. Before "
                "`start` this sets the initial prompt; after `start` the new "
                "prompt applies from the next chunk."
            ),
        ),
    ) -> PromptAccepted:
        """Set the prompt, active immediately or at the next chunk."""
        prompt = prompt.strip()
        if not prompt:
            await self._refuse("set_prompt", "The prompt is empty.")
            return None
        if self._generating:
            # Queue the change for the next chunk boundary.
            self._pending_prompt = prompt
        else:
            self._prompt = prompt
        await self._send_state_update()
        return PromptAccepted(prompt=prompt if not self._generating else self._prompt)

    @event(
        name="set_seed",
        description=(
            "Set the session's seed. Emits `seed_accepted` and `state_update`."
        ),
    )
    async def set_seed(
        self,
        seed: int = InputField(
            default=1000,
            ge=0,
            description="The session's seed.",
        ),
    ) -> SeedAccepted:
        """Set the session seed."""
        self._seed = int(seed)
        await self._send_state_update()
        return SeedAccepted(seed=self._seed)

    @event(
        name="set_canvas",
        description=(
            "Choose the aspect ratio of `main_video`. Only valid before "
            "`start`; the canvas is locked once generation begins. Emits "
            "`canvas_accepted`, carrying the exact pixel size, and "
            "`state_update`, or `command_error` when the session has started."
        ),
    )
    async def set_canvas(
        self,
        aspect: str = InputField(
            default="16:9",
            choices=list(chunk_plan.ASPECT_CHOICES),
            description="Aspect ratio of `main_video`.",
        ),
    ) -> CanvasAccepted:
        """Set the session's canvas; refused after `start`."""
        if self._started:
            await self._refuse(
                "set_canvas",
                "The canvas is locked once the session has started; "
                "`reset` to change it.",
            )
            return None
        try:
            height, width = chunk_plan.canvas_for_choice(aspect)
        except ValueError as error:
            await self._refuse("set_canvas", str(error))
            return None
        self._aspect = aspect
        await self._send_state_update()
        return CanvasAccepted(aspect=aspect, width=width, height=height)

    @event(
        name="start",
        description=(
            "Begin the persistent generation session. Prefills the text and "
            "conditions once, then generates media chunks continuously until "
            "the target duration is reached or `stop` is called. The target "
            "duration defaults to 60 seconds (1,440 emitted frames at 24 fps; "
            "1,450 H3 frames generated). Emits `session_started` and begins "
            "streaming `chunk_generated`, `chunk_started`, and "
            "`chunk_finished` messages, or `command_error` when the prompt "
            "is empty or the session has already started."
        ),
    )
    async def start(
        self,
        seconds: float | None = InputField(
            default=None,
            ge=1.0,
            description=(
                "Target duration in seconds. Omitted, the session default "
                "(60 s) applies. The session generates enough H3 frames to "
                "cover the target and emits exactly `seconds * 24` frames."
            ),
        ),
        metadata: str = InputField(
            default="",
            max_length=MAX_METADATA_CHARS,
            moderate=True,
            description=(
                "Free-form string stored with the session and echoed on "
                "messages. The model never reads it."
            ),
        ),
    ) -> SessionStarted:
        """Start the persistent generation session."""
        if self._started:
            await self._refuse("start", "The session has already started; `reset` to restart.")
            return None
        if not self._prompt:
            await self._refuse("start", "The prompt is empty; `set_prompt` first.")
            return None

        target = float(seconds) if isinstance(seconds, (int, float)) else self._target_seconds
        self._target_seconds = target
        self._target_emit_frames = round(target * FRAME_RATE)
        self._target_generate_frames = chunk_plan.total_generate_frames(self._target_emit_frames)
        self._total_chunks = chunk_plan.chunks_for_emit_frames(self._target_emit_frames)

        self._started = True
        self._generating = True
        self._finished = False
        self._stop_generation = False
        self._session_start_time = time.monotonic()

        height, width = self._canvas()
        await self.send(
            SessionStarted(
                family=self._family,
                prompt=self._prompt,
                seed=self._seed,
                aspect=self._aspect,
                width=width,
                height=height,
                target_seconds=target,
                target_frames=self._target_generate_frames,
                emit_frames=self._target_emit_frames,
                chunks=self._total_chunks,
            )
        )
        await self._send_state_update()
        return SessionStarted(
            family=self._family,
            prompt=self._prompt,
            seed=self._seed,
            aspect=self._aspect,
            width=width,
            height=height,
            target_seconds=target,
            target_frames=self._target_generate_frames,
            emit_frames=self._target_emit_frames,
            chunks=self._total_chunks,
        )

    @event(
        name="stop",
        description=(
            "Stop the persistent generation. The current chunk finishes "
            "emitting, then the stream holds on black. The session remains "
            "started; `start` resumes generation, `reset` clears everything. "
            "Emits `state_update`, or `command_error` when not generating."
        ),
    )
    async def stop(self) -> None:
        """Ask the generation loop to stop after the current chunk."""
        if not self._generating:
            await self._refuse("stop", "The session is not generating.")
            return
        self._stop_generation = True

    @event(
        name="get_state",
        description=(
            "Return a snapshot of everything the session exposes: the "
            "conditions in force, what is generating, progress counters, "
            "and the commands valid right now. Valid at any time."
        ),
    )
    async def get_state(self) -> StateUpdate:
        """Answer with the same snapshot `state_update` broadcasts."""
        return self._snapshot()

    @event(
        name="reset",
        description=(
            "Return every condition to its default, stop generation, clear "
            "the KV cache and all session state, and flush the output tracks. "
            "Valid at any time. Replies `session_reset` and emits "
            "`state_update`."
        ),
    )
    async def reset(self) -> SessionReset:
        """Clear the session back to its defaults."""
        was_generating = self._generating
        self._stop_generation = True
        if self._build is not None:
            job, _submitted = self._build
            job.cancelled = True
        self.output.flush()
        self._reset_session_state()
        await self._send_state_update()
        return SessionReset(was_generating=was_generating)

    # ------------------------------------------------------------- run loop

    async def run(self) -> None:
        """The model's control loop: park without an audience, serve with one."""
        while True:
            await self.connected.wait()
            await self._serve()

    async def _serve(self) -> None:
        """Pump builds and emit chunks while an audience is connected."""
        while self.connected.is_set():
            try:
                await self._pump_builds()
                # Emit any queued chunks.
                while self._emit_queue:
                    frames_list, audio, chunk_index = self._emit_queue.pop(0)
                    await self._emit_chunk(frames_list, audio, chunk_index)
                    if self._stop_generation:
                        break
                if self._stop_generation and self._build is None:
                    self._generating = False
                    self._stop_generation = False
                    await self._send_state_update()
                if self._generating and self._build is None and not self._emit_queue:
                    # Submit the next chunk or finish.
                    if self._current_chunk_index + 1 < self._total_chunks:
                        await self._submit_next_chunk()
                    elif self._frames_generated >= self._target_generate_frames:
                        await self._finish_session()
                    else:
                        await asyncio.sleep(POLL_SECONDS)
                elif not self._generating:
                    await asyncio.sleep(POLL_SECONDS)
            except Exception:  # noqa: BLE001
                logger.exception("error in the causal-h3 serve loop")
                await asyncio.sleep(POLL_SECONDS)

    async def _submit_next_chunk(self) -> None:
        """Submit the next chunk for generation, or the prefill if needed."""
        if not self.backend.prefilled:
            # Prefill text and conditions first.
            height, width = self._canvas()
            self._build = (
                self.backend.submit_prefill(
                    prompt=self._prompt,
                    seed=self._seed,
                    height=height,
                    width=width,
                ),
                time.monotonic(),
            )
            return

        # Apply pending prompt change at the chunk boundary.
        if self._pending_prompt is not None:
            self._prompt = self._pending_prompt
            self._pending_prompt = None

        chunk_index = self._current_chunk_index + 1
        if chunk_index == 0:
            # First chunk: the 5-frame prefix.
            frames = chunk_plan.PREFIX_FRAMES
        else:
            frames = chunk_plan._FRAMES_PER_CHUNK

        height, width = self._canvas()
        logger.info(
            f"chunk submitted: index={chunk_index}, frames={frames}, "
            f"prompt={self._prompt[:50]!r}"
        )
        self._build = (
            self.backend.submit_chunk(
                chunk_index=chunk_index,
                prompt=self._prompt,
                seed=self._seed,
                height=height,
                width=width,
                frames=frames,
            ),
            time.monotonic(),
        )

    async def _pump_builds(self) -> None:
        """Apply a finished build and keep the worker fed, without blocking."""
        if self._build is None:
            return
        job, submitted = self._build
        if not job.done.is_set():
            return
        self._build = None

        if job.cancelled:
            return
        if job.error is not None:
            logger.error("chunk failed", error=str(job.error))
            await self.send(
                CommandError(command="start", reason=f"Chunk generation failed: {job.error}")
            )
            self._generating = False
            await self._send_state_update()
            return

        result = job.result
        if result.chunk_index == -1:
            # Prefill result; nothing to emit.
            logger.info("prefill complete", cache_tokens=result.cache_tokens_after)
            return

        # Record the chunk.
        self._current_chunk_index = result.chunk_index
        self._frames_generated += len(result.frames)
        self._last_compute_time = result.compute_time
        content = len(result.frames) / FRAME_RATE
        self._last_throughput = content / result.compute_time if result.compute_time > 0 else 0.0

        chunk_id = str(uuid.uuid4())
        chunk_info = {
            "chunk_id": chunk_id,
            "chunk_index": result.chunk_index,
            "prompt": self._prompt,
            "metadata": "",
            "frames": len(result.frames),
            "seed": self._seed,
            "ready": True,
        }

        await self.send(
            ChunkGenerated(
                chunk=chunk_info,
                compute_time=round(result.compute_time, 3),
                throughput_x=round(self._last_throughput, 2),
                cache_tokens=result.cache_tokens_after,
            )
        )
        await self._send_state_update()

        # Queue the chunk for emission.
        self._emit_queue.append((result.frames, result.audio, result.chunk_index))

    async def _emit_chunk(
        self, frames_list: list[Any], audio: Any, chunk_index: int
    ) -> None:
        """Emit one chunk as paced slices on a 24 fps metronome."""
        chunk_id = str(uuid.uuid4())
        chunk_info = {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "prompt": self._prompt,
            "metadata": "",
            "frames": len(frames_list),
            "seed": self._seed,
            "ready": True,
        }
        await self.send(ChunkStarted(chunk=chunk_info))
        await self._send_state_update()

        import numpy as np

        samples_per_frame = OUTPUT_SAMPLE_RATE / FRAME_RATE
        total = len(frames_list)
        clock_start: float | None = None
        frames_paced = 0
        for lo in range(0, total, EMIT_FRAMES):
            await self._pump_builds()
            if self._stop_generation:
                break
            if not self.connected.is_set():
                return
            hi = min(lo + EMIT_FRAMES, total)
            alo = round(lo * samples_per_frame)
            ahi = round(hi * samples_per_frame)

            now = asyncio.get_running_loop().time()
            if clock_start is None:
                clock_start = now
            content_pos = frames_paced / FRAME_RATE
            clock_start = max(clock_start, now - content_pos)
            delay = clock_start + content_pos - now
            if delay > 0:
                await asyncio.sleep(delay)

            frames_paced += hi - lo
            self._frames_emitted += hi - lo
            self._seconds_emitted = self._frames_emitted / FRAME_RATE

            # Trim to the target emit frames.
            if self._frames_emitted > self._target_emit_frames:
                excess = self._frames_emitted - self._target_emit_frames
                hi -= excess
                self._frames_emitted -= excess
                ahi = round(hi * samples_per_frame)
                self._seconds_emitted = self._frames_emitted / FRAME_RATE
                video = np.ascontiguousarray(np.stack(frames_list[lo:hi]))
                await self.emit(
                    CausalH3Output(
                        main_video=video,
                        main_audio=audio[:, alo:ahi],
                    )
                )
                break

            video = np.ascontiguousarray(np.stack(frames_list[lo:hi]))
            await self.emit(
                CausalH3Output(
                    main_video=video,
                    main_audio=audio[:, alo:ahi],
                )
            )

        await self.send(
            ChunkFinished(
                chunk=chunk_info,
                frames_emitted=self._frames_emitted,
                seconds_emitted=round(self._seconds_emitted, 2),
            )
        )
        await self._send_state_update()

    async def _finish_session(self) -> None:
        """Report the session's final metrics and stop generating."""
        self._generating = False
        self._finished = True
        if self._session_start_time is not None:
            self._generation_wall_time = time.monotonic() - self._session_start_time
        content_seconds = self._frames_emitted / FRAME_RATE
        ratio = self._generation_wall_time / content_seconds if content_seconds > 0 else 0.0
        await self.send(
            SessionFinished(
                frames_generated=self._frames_generated,
                frames_emitted=self._frames_emitted,
                seconds_emitted=round(self._seconds_emitted, 2),
                generation_wall_time=round(self._generation_wall_time, 2),
                generation_ratio=round(ratio, 3),
            )
        )
        await self._send_state_update()
        logger.info(
            f"session finished: {self._frames_generated}f generated, "
            f"{self._frames_emitted}f emitted ({self._seconds_emitted:.2f}s), "
            f"wall={self._generation_wall_time:.2f}s, ratio={ratio:.3f}"
        )
