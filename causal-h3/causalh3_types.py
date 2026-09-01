"""Client-facing types for the CausalH3 persistent Reactor model.

CausalH3 is MiniMax-H3 served as a *persistent* stream: one continuous
generation session whose KV cache and clean x0 state carry forward across
chunks, with no hard cuts at clip boundaries.  This is the wire contract --
everything a client sees on the Reactor transport.

Two family slugs share one model class:

- ``fl2va``: text-to-video, image-to-video, and first-last-frame conditioning.
  Optional first and last frame images anchor the generation.
- ``ref2va``: reference-to-video.  Image, video, or audio references ride
  as immutable prefix rows in the packed sequence.

The schema is product surface: every ``@event`` / ``InputField`` /
``MessageField`` description compiles into the published OpenAPI schema.
Describe only what a client observes on the wire; never internals.
"""

from __future__ import annotations

from dataclasses import dataclass

from reactor_runtime import (
    Audio,
    InputField,
    MessageField,
    ModelMessage,
    Output,
    Video,
    event,
)
from reactor_runtime.types import ClipInfo as _ClipInfo  # noqa: F401 — re-exported

MAX_PROMPT_CHARS = 800
MAX_METADATA_CHARS = 2000


class CausalH3Output(Output):
    """The generated video and its synchronized audio, streamed per chunk."""

    main_video: Video
    main_audio: Audio


@dataclass(frozen=True)
class ChunkInfo:
    """One media chunk, as every chunk-referencing message reports it.

    Mirrors ``ClipInfo`` from fast-h3 but adapted for the persistent stream:
    chunks are numbered sequentially within a session, not independent clips.
    """

    chunk_id: str
    chunk_index: int
    prompt: str
    metadata: str
    frames: int
    seed: int
    ready: bool


class StateUpdate(ModelMessage):
    """Emitted on connect and after every change to the session's state.

    Carries the conditions in force, what is generating, progress counters,
    and the commands valid right now.  A client reads everything it needs
    from this one message.
    """

    family: str = MessageField(description="The model family slug: `fl2va` or `ref2va`.")
    prompt: str = MessageField(description="The active prompt driving generation.")
    seed: int = MessageField(description="The session's seed.")
    aspect: str = MessageField(description="Aspect ratio of `main_video`.")
    width: int = MessageField(description="Width of every frame on `main_video` in pixels.")
    height: int = MessageField(description="Height of every frame on `main_video` in pixels.")
    generating: bool = MessageField(
        description="True when the session is actively generating media chunks."
    )
    current_chunk_index: int = MessageField(
        description="The zero-based index of the chunk currently generating or last emitted."
    )
    frames_generated: int = MessageField(
        description="Total H3 frames generated so far in this session."
    )
    frames_emitted: int = MessageField(
        description="Total frames sent on `main_video` so far in this session."
    )
    seconds_emitted: float = MessageField(
        description="Total seconds of media sent on the output tracks."
    )
    compute_time: float = MessageField(
        description="Wall-clock seconds the last chunk took to generate."
    )
    throughput_x: float = MessageField(
        description="Generation throughput as a multiple of realtime (>1 means faster than playback)."
    )
    cache_tokens: int = MessageField(
        description="Number of tokens currently held in the KV cache."
    )
    valid_commands: list[str] = MessageField(
        description="Commands the session accepts right now, for enabling or greying out controls."
    )


class SessionStarted(ModelMessage):
    """Emitted when a generation session begins after `start`."""

    family: str = MessageField(description="The model family slug in use.")
    prompt: str = MessageField(description="The prompt driving the session.")
    seed: int = MessageField(description="The session's seed.")
    aspect: str = MessageField(description="Aspect ratio of `main_video`.")
    width: int = MessageField(description="Frame width in pixels.")
    height: int = MessageField(description="Frame height in pixels.")
    target_seconds: float = MessageField(
        description="The session's target duration in seconds."
    )
    target_frames: int = MessageField(
        description="Total H3 frames the session will generate."
    )
    emit_frames: int = MessageField(
        description="Frames that will be emitted on `main_video` (target_frames minus trim)."
    )
    chunks: int = MessageField(
        description="Number of 17-frame causal chunks the session will run."
    )


class ChunkGenerated(ModelMessage):
    """Emitted when a chunk's build completes and its media is ready to emit."""

    chunk: dict = MessageField(description="The chunk's full `ChunkInfo` structure.")
    compute_time: float = MessageField(
        description="Wall-clock seconds the chunk took to generate."
    )
    throughput_x: float = MessageField(
        description="Generation throughput as a multiple of realtime for this chunk."
    )
    cache_tokens: int = MessageField(
        description="Tokens held in the KV cache after this chunk."
    )


class ChunkStarted(ModelMessage):
    """Emitted as a chunk begins streaming on the output tracks."""

    chunk: dict = MessageField(description="The chunk now playing.")


class ChunkFinished(ModelMessage):
    """Emitted when a chunk has been fully sent on the output tracks."""

    chunk: dict = MessageField(description="The chunk that finished.")
    frames_emitted: int = MessageField(
        description="Total frames sent on `main_video` so far in this session."
    )
    seconds_emitted: float = MessageField(
        description="Total seconds of media sent on the output tracks."
    )


class SessionFinished(ModelMessage):
    """Emitted when the entire session's media has been generated and emitted."""

    frames_generated: int = MessageField(
        description="Total H3 frames generated."
    )
    frames_emitted: int = MessageField(
        description="Total frames emitted on `main_video`."
    )
    seconds_emitted: float = MessageField(
        description="Total seconds of media sent."
    )
    generation_wall_time: float = MessageField(
        description="Wall-clock seconds from the first chunk to the last frame emitted."
    )
    generation_ratio: float = MessageField(
        description="Generation wall time divided by content duration (<1 means faster than realtime)."
    )


class PromptAccepted(ModelMessage):
    """Emitted when `set_prompt` is accepted; takes effect at the next chunk."""

    prompt: str = MessageField(description="The new prompt, active from the next chunk.")


class SeedAccepted(ModelMessage):
    """Emitted when `set_seed` is accepted."""

    seed: int = MessageField(description="The new seed.")


class CanvasAccepted(ModelMessage):
    """Emitted when `set_canvas` is accepted; only valid before `start`."""

    aspect: str = MessageField(description="The aspect ratio now in force.")
    width: int = MessageField(description="Frame width in pixels.")
    height: int = MessageField(description="Frame height in pixels.")


class FamilyAccepted(ModelMessage):
    """Emitted when `set_family` is accepted; only valid before `start`."""

    family: str = MessageField(description="The family slug now in force.")


class SessionReset(ModelMessage):
    """Emitted when `reset` is accepted."""

    was_generating: bool = MessageField(
        description="True if a generation was in progress when reset was called."
    )


class CommandError(ModelMessage):
    """Emitted when a command is rejected. The command had no effect."""

    command: str = MessageField(description="The command that was rejected.")
    reason: str = MessageField(description="Why it was rejected.")


__all__ = [
    "MAX_PROMPT_CHARS",
    "MAX_METADATA_CHARS",
    "CausalH3Output",
    "ChunkInfo",
    "StateUpdate",
    "SessionStarted",
    "ChunkGenerated",
    "ChunkStarted",
    "ChunkFinished",
    "SessionFinished",
    "PromptAccepted",
    "SeedAccepted",
    "CanvasAccepted",
    "FamilyAccepted",
    "SessionReset",
    "CommandError",
]
