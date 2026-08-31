"""Live FastH3: one prompt, an endless continuously-stitched video-and-audio stream.

FastH3 is MiniMax-H3 distilled to four transformer forwards. The hard-cut
``fast-h3`` channel plays independent clips with a black flush between them; this
channel instead **chains** them into one continuous stream from a single prompt:
`set_prompt` holds a prompt, `start` begins, and from then on the model generates
clip after clip with no further input — the first is text-to-video-and-audio, and
every clip after is FL2VA-anchored on the previous clip's last frame, so the scene
carries forward. Consecutive clips are stitched at their seam (a linear-light
crossfade with per-clip color-match, and an equal-power audio overlap), so the
result plays as one video rather than a sequence of cuts. `stop` ends it; a
mid-run `set_prompt` swaps the held prompt in for the next clip, steering the
stream without interrupting it.

The unit of work is a whole clip, not a frame, which is why this subclasses
``ReactorModel`` and owns its own ``run()`` loop rather than using
``ReactorPipeline``. Generation runs ahead of playout in a bounded queue
(double-buffering), so while one clip streams the next is already building — the
only way clip generation can keep ahead of real-time playback.

Layout:
  * ``live_h3_types.py``      — everything a client sees (tracks, messages).
  * ``live_h3_backend.py``    — the FastVideo engine, FL2VA anchor, worker thread.
  * ``live_h3_seam.py``       — the seam stitch (color-match, linear-light blend).
  * ``live_h3_assets.py``     — config parsing and weights validation.
  * ``live_h3_clip_plan.py``  — clip geometry (lengths, frame counts, canvases).
  * ``live_h3.yaml``          — the generation recipe, seam knobs, weight layout.

``sitecustomize.py`` next to this file is the shared interpreter-wide fix (dynamo
recompile limits, VSA arch gate) the manifest puts on ``PYTHONPATH``.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import numpy as np
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

import live_h3_clip_plan as clip_plan
import live_h3_seam as seam
from live_h3_assets import load_config, require_weights, resolve_model_path
from live_h3_backend import OUTPUT_SAMPLE_RATE, LiveH3Backend
from live_h3_types import (
    MAX_PROMPT_CHARS,
    CanvasAccepted,
    ClipComplete,
    ClipLengthAccepted,
    CommandError,
    LiveH3Output,
    PromptAccepted,
    SeedAccepted,
    SessionReset,
    StateUpdate,
    StreamStarted,
    StreamStopped,
)

logger = get_logger(__name__)

FRAME_RATE = clip_plan.FPS

# Exact because 48000 / 24 divides evenly: one video frame is 2000 audio samples,
# so seam slicing lands on sample boundaries with no rounding drift.
SAMPLES_PER_FRAME = OUTPUT_SAMPLE_RATE // FRAME_RATE

# The clip-length range, rendered once so the command text and the schema's own
# bounds can never disagree.
_CLIP_RANGE = f"{clip_plan.MIN_SECONDS_PUBLISHED:g} and {clip_plan.MAX_SECONDS_PUBLISHED:g}"

# Frames per emitted slice. The runtime recorder's feed queue cannot absorb
# one-second bursts, and the emitter is a metronome either way, so smaller
# slices cost nothing.
EMIT_FRAMES = 3

# Idle re-check and job-poll granularity. Runs on the event loop, so this is a
# scheduling granularity, not a busy-wait.
POLL_SECONDS = 0.05


class LiveH3(ReactorModel):
    """Drive an endless continuously-stitched clip chain from one held prompt."""

    # Pinned: the emitter is a strict 24 fps metronome and every emit omits
    # `compute_time`, which is exactly the "unmeasured" path this rate tags.
    # Measuring instead re-estimates the rate from observed timing, whose wobble
    # both drops frames while converging and drifts video against the
    # sample-clocked audio.
    fps = FRAME_RATE
    # Two seconds of transport-side tolerance at 24 fps, so a hiccup dents the
    # buffer instead of dropping frames.
    buffer_size = 48

    # ------------------------------------------------------------------ load

    def load(self, config_path: Path | None) -> None:
        """Parse the config, validate the weights, and build the warm engine.

        Runs once at startup, before any session. The runtime marks the pod
        ready only when this returns, so the backend's warm-up (both the T2VA
        and the FL2VA compile shape) means a deployed pod never builds a cold
        clip — the first continuation would otherwise stall ~20 s.

        Args:
            config_path: Path to ``live_h3.yaml``; its ``inference`` block is the
                generation recipe and the seam knobs, and its ``runtime`` block
                holds the weight layout and the engine shape.
        """
        self.config = load_config(config_path)
        weights = get_weights_path()
        model_path = resolve_model_path(self.config, weights)
        require_weights(weights, model_path)

        self.backend = LiveH3Backend(self.config, model_path)
        # Session-scoped state exists before the first session, so a command
        # racing ahead of `@session_started` reads defaults, never garbage.
        self._reset_session_state()
        self.backend.load()
        logger.info(
            "live fast-h3 loaded",
            clip_frames=self.config.clip_frames,
            crossfade_frames=self.config.crossfade_frames,
            color_match=self.config.color_match,
            buffer_depth=self.config.buffer_depth,
        )

    # -------------------------------------------------------- session state

    def _reset_session_state(self) -> None:
        """Return every session-scoped field to its default.

        Called once at ``load()`` and at every ``@session_started``, which is
        what keeps one session from ever observing another's prompt, conditions,
        or a chain it did not start.
        """
        # Conditions the next chain snapshots at `start`.
        self._prompt: str = ""
        self._seed: int = self.config.seed
        self._clip_frames: int = self.config.clip_frames
        self._aspect: str = self.config.aspect

        # Chain lifecycle. ``_streaming`` is owned by the run loop; the two
        # request flags are how a command asks a running chain to unwind —
        # ``_stop_requested`` cuts it (a `stream_stopped` follows), and
        # ``_reset_requested`` additionally suppresses that message because
        # `reset` reports its own `session_reset`.
        self._streaming: bool = False
        self._stop_requested: bool = False
        self._reset_requested: bool = False

        # Progress, mirrored so a `state_update` is a complete snapshot. The
        # pacing clock lives here too, spanning a whole chain so blocks are
        # emitted gap-free rather than each clip re-starting the metronome.
        self._clips_emitted: int = 0
        self._frames_sent: int = 0
        self._seconds_sent: float = 0.0
        self._clock_start: float | None = None
        self._frames_paced: int = 0

    def _canvas(self) -> tuple[int, int]:
        """The ``(height, width)`` this session generates at."""
        return clip_plan.canvas_for_choice(self._aspect)

    def _valid_commands(self) -> list[str]:
        """The commands the session would accept right now.

        A chain locks the canvas, the seed and the clip length (they are snapped
        at `start` and a mid-chain change would recompile), and offers `stop`; an
        idle session offers `set_canvas` and — once a prompt is held — `start`.
        """
        commands = {"get_state", "set_prompt", "reset"}
        if self._streaming:
            commands.add("stop")
        else:
            commands |= {"set_seed", "set_clip_seconds", "set_canvas"}
            if self._prompt.strip():
                commands.add("start")
        return sorted(commands)

    def _snapshot(self) -> StateUpdate:
        """Everything a client can observe, in one message.

        The single source of the snapshot: `state_update` broadcasts it, a
        joining client is greeted with it, and `get_state` answers with it.
        Built once here so those three can never disagree.
        """
        height, width = self._canvas()
        return StateUpdate(
            prompt=self._prompt,
            streaming=self._streaming,
            clip_seconds=round(clip_plan.seconds_for_frames(self._clip_frames), 3),
            clip_seconds_min=clip_plan.MIN_SECONDS_PUBLISHED,
            clip_seconds_max=clip_plan.MAX_SECONDS_PUBLISHED,
            seed=self._seed,
            aspect=self._aspect,
            width=width,
            height=height,
            clips_emitted=self._clips_emitted,
            seconds_sent=round(self._seconds_sent, 2),
            valid_commands=self._valid_commands(),
        )

    async def _send_state_update(self) -> None:
        """Broadcast the snapshot to every connected client."""
        await self.send(self._snapshot())

    async def _refuse(self, command: str, reason: str) -> None:
        """Reject a command: tell every client, and leave its reply bodyless.

        A handler returns only the message its annotation names, and reports a
        failure by broadcasting `command_error` and returning without a value.
        The runtime answers that with a correlated bodyless acknowledgement, so
        an awaiting client resolves rather than hanging — and unlike a raised
        runtime ``CommandError``, whose failure frame is withheld from v0
        clients, the broadcast reaches every SDK generation.
        """
        logger.info("command refused", command=command, reason=reason)
        await self.send(CommandError(command=command, reason=reason))

    # ------------------------------------------------------------ lifecycle

    @session_started
    async def on_session_started(self) -> None:
        """Clear the prompt and every condition so a new session inherits nothing."""
        self._stop_requested = True  # unwind any chain the run loop still holds
        self._reset_session_state()

    @session_ended
    async def on_session_ended(self) -> None:
        """Cut the chain; the only hook guaranteed to fire on every path."""
        self._stop_requested = True

    @connected
    async def on_connect(self, client: ClientInfo) -> None:
        """Greet the joining client with the full state.

        Addressed rather than broadcast: the clients already watching have this,
        and a late joiner needs it without replaying every command.
        """
        await client.send(self._snapshot())

    # ------------------------------------------------------------- commands

    @event(
        name="set_prompt",
        description=(
            "Hold a prompt for the chain to generate from. Set before `start` it "
            "is the chain's prompt; set during a run it swaps in for the next "
            "clip generated, steering the stream without interrupting it — the "
            "single held prompt then keeps driving generation with no further "
            "input. Replies `prompt_accepted` and emits `state_update`, or "
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
                "What the stream should show, up to 800 characters. One prompt "
                "drives the whole chain; there is no re-prompting per clip unless "
                "you choose to steer with another `set_prompt`."
            ),
        ),
    ) -> PromptAccepted:
        """Hold a prompt; the running chain, if any, picks it up on its next clip."""
        prompt = prompt.strip()
        if not prompt:
            await self._refuse("set_prompt", "The prompt is empty; the chain needs one.")
            return None
        self._prompt = prompt
        applied_live = self._streaming
        await self._send_state_update()
        return PromptAccepted(prompt=self._prompt, applied_live=applied_live)

    @event(
        name="start",
        description=(
            "Begin the continuous chain from the held prompt. The first clip is "
            "text-to-video-and-audio; every clip after is anchored on the "
            "previous clip's last frame and stitched onto it, so the stream runs "
            "indefinitely off the one prompt until `stop`. Emits `stream_started` "
            "and `state_update`, or `command_error` when a chain is already "
            "running or no prompt is held."
        ),
    )
    async def start(self) -> None:
        """Arm the chain; the run loop begins generating and streaming it."""
        if self._streaming:
            await self._refuse("start", "A chain is already running; send `stop` first.")
            return
        if not self._prompt.strip():
            await self._refuse("start", "Set a prompt with `set_prompt` before starting.")
            return
        self._stop_requested = False
        self._reset_requested = False
        self._clips_emitted = 0
        self._frames_sent = 0
        self._seconds_sent = 0.0
        self._clock_start = None
        self._frames_paced = 0
        self._streaming = True
        height, width = self._canvas()
        await self.send(
            StreamStarted(
                prompt=self._prompt,
                aspect=self._aspect,
                width=width,
                height=height,
                clip_seconds=round(clip_plan.seconds_for_frames(self._clip_frames), 3),
                seed=self._seed,
            )
        )
        await self._send_state_update()

    @event(
        name="stop",
        description=(
            "End the running chain. Generation stops, whatever is queued on the "
            "output tracks is dropped, and the picture goes to black within a "
            "fraction of a second. A later `start` begins a fresh chain from the "
            "held prompt. Emits `stream_stopped` and `state_update`, or "
            "`command_error` when no chain is running."
        ),
    )
    async def stop(self) -> None:
        """Ask the run loop to cut the chain."""
        if not self._streaming:
            await self._refuse("stop", "No chain is running.")
            return
        self._stop_requested = True

    @event(
        name="set_seed",
        description=(
            "Set the seed the next chain's first clip generates from; each "
            "continuation advances it by one, so re-running the same prompt "
            "reproduces the same stream. Takes effect on the next `start` — a "
            "running chain keeps its seed. Emits `seed_accepted` and "
            "`state_update`."
        ),
    )
    async def set_seed(
        self,
        seed: int = InputField(
            default=1000,
            ge=0,
            description=(
                "Seed for the next chain. Reproduction is close rather than "
                "exact: the deployment runs fused kernels that can reorder "
                "arithmetic."
            ),
        ),
    ) -> SeedAccepted:
        """Set the seed the next chain starts from."""
        self._seed = int(seed)
        await self._send_state_update()
        return SeedAccepted(seed=self._seed)

    @event(
        name="set_clip_seconds",
        description=(
            "Set the length of each clip the chain generates. Shorter clips seam "
            "more often but keep generation further ahead of playback; longer "
            "ones seam less. The value is snapped to the nearest length the model "
            "can produce, so read the effective one back from "
            "`clip_length_accepted`. Takes effect on the next `start` — a running "
            "chain keeps its length so the compiled shape never changes mid-"
            "stream. Emits `clip_length_accepted` and `state_update`."
        ),
    )
    async def set_clip_seconds(
        self,
        seconds: float = InputField(
            default=clip_plan.MAX_SECONDS_PUBLISHED,
            ge=clip_plan.MIN_SECONDS_PUBLISHED,
            le=clip_plan.MAX_SECONDS_PUBLISHED,
            description=(
                f"Clip length in seconds, between {_CLIP_RANGE}. Snapped to the "
                "nearest length the model can produce, so the value that takes "
                "effect can differ slightly; `state_update.clip_seconds` always "
                "carries the one in force."
            ),
        ),
    ) -> ClipLengthAccepted:
        """Set the per-clip length the next chain generates at."""
        self._clip_frames = clip_plan.frames_for_seconds(float(seconds))
        await self._send_state_update()
        return ClipLengthAccepted(
            clip_seconds=round(clip_plan.seconds_for_frames(self._clip_frames), 3),
            frames=self._clip_frames,
        )

    @event(
        name="set_canvas",
        description=(
            "Choose the aspect ratio of `main_video`. The video track keeps one "
            "size for a whole chain, so this is only valid while no chain is "
            "running. Emits `canvas_accepted`, carrying the exact pixel size, and "
            "`state_update`, or `command_error` while a chain is running or when "
            "the ratio is not one this model offers."
        ),
    )
    async def set_canvas(
        self,
        aspect: str = InputField(
            default="16:9",
            choices=list(clip_plan.ASPECT_CHOICES),
            description=(
                "Aspect ratio of `main_video`. `canvas_accepted` and "
                "`state_update` report the width and height in pixels it resolves "
                "to."
            ),
        ),
    ) -> CanvasAccepted:
        """Set the session's canvas; refused while a chain is running."""
        if self._streaming:
            await self._refuse(
                "set_canvas",
                "The canvas is fixed while a chain is running; `stop` first.",
            )
            return None
        try:
            height, width = clip_plan.canvas_for_choice(aspect)
        except ValueError as error:
            await self._refuse("set_canvas", str(error))
            return None
        self._aspect = aspect
        await self._send_state_update()
        return CanvasAccepted(aspect=aspect, width=width, height=height)

    @event(
        name="reset",
        description=(
            "Return every condition to its default, clear the held prompt, cut "
            "any running chain, and clear the output tracks. A running chain is "
            "cut with a `stream_stopped` to mark it. Valid at any time. Replies "
            "`session_reset` and emits `state_update`."
        ),
    )
    async def reset(self) -> SessionReset:
        """Clear the session back to its defaults, cutting any running chain."""
        was_streaming = self._streaming
        if self._streaming:
            self._reset_requested = True
            self._stop_requested = True
        self._prompt = ""
        self._seed = self.config.seed
        self._clip_frames = self.config.clip_frames
        self._aspect = self.config.aspect
        self.output.flush()
        await self._send_state_update()
        return SessionReset(was_streaming=was_streaming)

    @event(
        name="get_state",
        description=(
            "Return a snapshot of everything the session exposes: the held "
            "prompt, the conditions in force, whether a chain is running, "
            "progress counters, and the commands that are valid right now. The "
            "same payload the model broadcasts as `state_update`. Valid at any "
            "time."
        ),
    )
    async def get_state(self) -> StateUpdate:
        """Answer with the same snapshot `state_update` broadcasts."""
        return self._snapshot()

    # ------------------------------------------------------------- run loop

    async def run(self) -> None:
        """The model's control loop: park without an audience, serve with one.

        Nothing here may raise: an exception out of ``run()`` is an
        unrecoverable crash of the whole model loop, not the end of one session,
        so ``_serve`` owns its own failure reporting.
        """
        while True:
            await self.connected.wait()
            await self._serve()

    async def _serve(self) -> None:
        """Run an armed chain while an audience is connected, else idle.

        Generation is gated on having an audience: with nobody connected no chain
        runs, and the loop parks back in ``run()``.
        """
        while self.connected.is_set():
            try:
                if self._streaming:
                    await self._run_chain()
                else:
                    await asyncio.sleep(POLL_SECONDS)
            except Exception:  # noqa: BLE001 — the model loop must survive anything
                logger.exception("error in the live fast-h3 serve loop")
                self._streaming = False
                self.output.flush()
                await asyncio.sleep(POLL_SECONDS)

    async def _run_chain(self) -> None:
        """Generate and stream one continuous chain until it is stopped.

        A producer task builds clips ahead into a bounded queue (double-buffering
        — while one clip streams the next is already on the worker), and the
        consumer here stitches each onto the last and paces it out. The chain
        ends on `stop`, `reset`, or a lost audience; it then flushes to black and
        reports how it ended, unless `reset` (which reports its own).
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.buffer_depth)
        self._producer_error: BaseException | None = None
        height, width = self._canvas()
        producer = asyncio.create_task(self._producer(queue, height, width, base_seed=self._seed))
        try:
            await self._consumer(queue)
        finally:
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer
            self.output.flush()
            self._streaming = False

        if self._producer_error is not None:
            logger.error("chain ended on a generation failure", error=str(self._producer_error))
        if not self._reset_requested:
            await self.send(
                StreamStopped(
                    clips_emitted=self._clips_emitted,
                    seconds_sent=round(self._seconds_sent, 2),
                )
            )
        await self._send_state_update()

    async def _producer(
        self, queue: asyncio.Queue, height: int, width: int, *, base_seed: int
    ) -> None:
        """Build clips ahead of playout, color-matched and anchored, into ``queue``.

        The first clip is T2VA; every clip after passes the previous clip's
        (color-matched) last frame as the FL2VA anchor, which is what carries the
        scene across the seam. The color-match reference is locked once to clip
        0's last frame and held for the whole chain, so exposure cannot ratchet.
        A ``None`` sentinel is always the last thing put, so the consumer unwinds
        even on stop or failure.
        """
        from PIL import Image

        loop = asyncio.get_running_loop()
        index = 0
        target_rgb: np.ndarray | None = None
        anchor = None
        try:
            while not self._stop_requested and self.connected.is_set():
                prompt = self._prompt  # held; a mid-chain set_prompt steers here
                job = self.backend.submit(
                    frames=self._clip_frames,
                    prompt=prompt,
                    seed=base_seed + index,
                    height=height,
                    width=width,
                    anchor_image=anchor,
                )
                while not job.done.is_set():
                    if self._stop_requested or not self.connected.is_set():
                        job.cancelled = True
                        return
                    await asyncio.sleep(POLL_SECONDS)
                if job.error is not None:
                    self._producer_error = job.error
                    return
                frames_list, samples = job.result
                # Off-loop numpy work: stacking a clip's frames is not trivial at
                # 768p, and blocking the event loop would stutter the emitter.
                frames = await loop.run_in_executor(
                    None, self._post_process, index, frames_list, target_rgb
                )
                if target_rgb is None:
                    target_rgb = seam.reference_rgb(frames[-1])
                anchor = Image.fromarray(frames[-1]).convert("RGB")
                await queue.put((index, prompt, frames, samples))
                index += 1
        except Exception as error:  # noqa: BLE001 — handed to the consumer's caller
            self._producer_error = error
            logger.exception("live fast-h3 producer raised")
        finally:
            await queue.put(None)

    def _post_process(
        self, index: int, frames_list: list, target_rgb: np.ndarray | None
    ) -> np.ndarray:
        """Stack a clip's frames and color-match continuation clips to clip 0.

        Runs in a thread executor. Clip 0 is left as generated (it sets the
        reference); every clip after is shifted by one per-channel offset onto
        that reference.
        """
        frames = np.ascontiguousarray(np.stack(frames_list))
        if index > 0 and self.config.color_match == "per_clip" and target_rgb is not None:
            frames = seam.color_match_to_reference(frames, target_rgb)
        return frames

    async def _consumer(self, queue: asyncio.Queue) -> None:
        """Stitch each built clip onto the last and pace the result out.

        Each clip holds back its last ``crossfade_frames`` as the seam tail; when
        the next clip arrives, that tail is crossfaded (linear light, video;
        equal-power, audio) with the next clip's head, and the middle of the clip
        streams between seams. The held tail carries across the loop, so the
        stitch spans the clip boundary rather than sitting inside one clip.
        """
        xf = self.config.crossfade_frames
        pending_v: np.ndarray | None = None
        pending_a: np.ndarray | None = None
        while self.connected.is_set() and not self._stop_requested:
            item = await queue.get()
            if item is None:
                break
            index, prompt, video, audio = item
            n = int(video.shape[0])
            if pending_v is None:
                # First clip: stream all but the seam tail, hold the tail.
                body_end = max(0, n - xf)
                await self._emit(video[:body_end], audio[:, : body_end * SAMPLES_PER_FRAME])
                pending_v = video[body_end:].copy()
                pending_a = audio[:, body_end * SAMPLES_PER_FRAME :].copy()
            else:
                # Seam: crossfade the held tail with this clip's head.
                k = min(xf, n, int(pending_v.shape[0]))
                if k > 0:
                    blended_v = seam.blend_video_linear(pending_v[:k], video[:k])
                    la = min(int(pending_a.shape[-1]), k * SAMPLES_PER_FRAME, int(audio.shape[-1]))
                    if la > 0:
                        blended_a = seam.blend_audio_equal_power(
                            pending_a[:, :la], audio[:, :la]
                        )
                    else:
                        blended_a = np.zeros((1, 0), dtype=np.int16)
                    await self._emit(blended_v, blended_a)
                # Middle: this clip minus the blended head and the new held tail.
                body_end = max(k, n - xf)
                await self._emit(
                    video[k:body_end],
                    audio[:, k * SAMPLES_PER_FRAME : body_end * SAMPLES_PER_FRAME],
                )
                pending_v = video[body_end:].copy()
                pending_a = audio[:, body_end * SAMPLES_PER_FRAME :].copy()

            self._clips_emitted += 1
            await self.send(
                ClipComplete(
                    index=index,
                    frames=n,
                    prompt=prompt,
                    seconds_sent=round(self._seconds_sent, 2),
                )
            )
        # A drained producer (sentinel, not a cut) leaves a tail worth sending.
        if (
            pending_v is not None
            and not self._stop_requested
            and self.connected.is_set()
            and pending_v.shape[0] > 0
        ):
            await self._emit(pending_v, pending_a)

    # -------------------------------------------------------------- emitter

    async def _emit(self, video: np.ndarray, audio: np.ndarray) -> str:
        """Emit one video block as paced slices on the chain's 24 fps metronome.

        - Paced by FRAMES, not slices, and by one clock spanning the whole chain,
          so clip boundaries are gap-free rather than each clip re-anchoring.
        - Never bursts to catch up: if the transport held a slice back, it
          re-anchors instead, since a catch-up burst only overflows the queue.
        - Emits omit ``compute_time``, so every slice is tagged at the pinned
          24 fps — the rate the audio is already sample-clocked against.

        Returns ``"stopped"`` when `stop`/`reset` cut it, ``"gone"`` when the
        audience left, ``"done"`` when the whole block went out.
        """
        n = int(video.shape[0])
        if n == 0:
            return "done"
        for lo in range(0, n, EMIT_FRAMES):
            if self._stop_requested:
                return "stopped"
            if not self.connected.is_set():
                return "gone"
            hi = min(lo + EMIT_FRAMES, n)
            alo = lo * SAMPLES_PER_FRAME
            ahi = hi * SAMPLES_PER_FRAME

            now = asyncio.get_running_loop().time()
            if self._clock_start is None:
                self._clock_start = now
            content_pos = self._frames_paced / FRAME_RATE
            self._clock_start = max(self._clock_start, now - content_pos)
            delay = self._clock_start + content_pos - now
            if delay > 0:
                await asyncio.sleep(delay)

            self._frames_paced += hi - lo
            self._frames_sent += hi - lo
            self._seconds_sent = self._frames_sent / FRAME_RATE
            await self.emit(
                LiveH3Output(
                    main_video=np.ascontiguousarray(video[lo:hi]),
                    main_audio=np.ascontiguousarray(audio[:, alo:ahi]),
                )
            )
        return "done"
