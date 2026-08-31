"""Client-facing types for the live (continuous-chain) FastH3 model.

Everything a client can see lives here: the outbound video and audio tracks and
the typed messages the model sends. ``live_h3.py`` imports these; a frontend
developer reads this file to learn the whole API without opening the inference
code.

Unlike the hard-cut ``fast-h3`` channel, this model has no queue and no
per-clip identity a client addresses. One held prompt drives an open-ended
chain: ``start`` begins it, each generated clip is stitched onto the last and
streamed continuously, and ``stop`` ends it. So the messages here describe the
*stream*, not clips-as-objects — ``clip_complete`` is progress, not a handle.

The conditions behind the ``set_*`` commands are not here. A ``ReactorModel``
owns its session state itself, so they are plain attributes on ``LiveH3`` reset
in ``_reset_session_state``; their client-facing text lives on each handler's
own ``InputField`` declaration.
"""

from __future__ import annotations

from reactor_runtime import (
    Audio,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

MAX_PROMPT_CHARS = 800


class LiveH3Output(Output):
    """The generated video and its synchronized audio, streamed continuously."""

    main_video: Video
    main_audio: Audio


class StateUpdate(ModelMessage):
    """Emitted on connect and after every change to the session's state.

    One snapshot of everything observable, so a client can render its whole UI
    from this alone instead of accumulating the individual messages below.
    """

    prompt: str = MessageField(
        description=(
            "The held prompt the chain generates from. Empty until `set_prompt`; "
            "`start` needs it non-empty."
        )
    )
    streaming: bool = MessageField(
        description="A continuous chain is generating and playing on the output tracks."
    )
    clip_seconds: float = MessageField(
        description="Length of each clip the chain generates, before seam overlap."
    )
    clip_seconds_min: float = MessageField(
        description="Shortest clip length `set_clip_seconds` accepts."
    )
    clip_seconds_max: float = MessageField(
        description="Longest clip length `set_clip_seconds` accepts."
    )
    seed: int = MessageField(
        description=(
            "Seed the chain's first clip generates from; each continuation clip "
            "advances it by one so a run is reproducible."
        )
    )
    aspect: str = MessageField(description="Aspect ratio in effect, e.g. `16:9`.")
    width: int = MessageField(description="Width of every frame on `main_video`.")
    height: int = MessageField(description="Height of every frame on `main_video`.")
    clips_emitted: int = MessageField(
        description="Clips stitched into the stream since it started."
    )
    seconds_sent: float = MessageField(
        description="Seconds of video and audio sent since the stream started."
    )
    valid_commands: list[str] = MessageField(
        description=(
            "Names of the commands the session would accept right now. Use this "
            "to enable or grey out controls instead of re-deriving the state "
            "machine client-side; any command not listed would be rejected."
        )
    )


class StreamStarted(ModelMessage):
    """Emitted when `start` begins the continuous chain.

    The first clip is generated from the held prompt (text-to-video-and-audio);
    every clip after is anchored on the previous clip's last frame, so the
    stream continues off the one prompt with no further input.
    """

    prompt: str = MessageField(description="The held prompt the chain is generating from.")
    aspect: str = MessageField(description="Aspect ratio the chain is generating at.")
    width: int = MessageField(description="Width of every frame on `main_video`.")
    height: int = MessageField(description="Height of every frame on `main_video`.")
    clip_seconds: float = MessageField(description="Length of each generated clip, in seconds.")
    seed: int = MessageField(description="Seed the chain's first clip generated from.")


class ClipComplete(ModelMessage):
    """Emitted each time a clip has been stitched into the stream and sent.

    Progress, not a handle: the chain is one continuous video, so this reports
    how far it has run rather than naming a clip a client can act on.
    """

    index: int = MessageField(description="The clip's position in the chain, starting at 0.")
    frames: int = MessageField(description="Frames the clip contributed before seam overlap.")
    prompt: str = MessageField(
        description="The held prompt in force when this clip was generated."
    )
    seconds_sent: float = MessageField(
        description="Seconds of video and audio sent since the stream started, this clip included."
    )


class StreamStopped(ModelMessage):
    """Emitted when `stop` ends the chain, or it stops on a lost audience.

    The stream flushes to black. A later `start` begins a fresh chain from the
    held prompt's first clip.
    """

    clips_emitted: int = MessageField(description="Clips stitched into the stream before it ended.")
    seconds_sent: float = MessageField(
        description="Seconds of video and audio sent over the whole run."
    )


class PromptAccepted(ModelMessage):
    """Emitted when `set_prompt` is accepted.

    Held for the rest of the session unless changed. Set before `start` it is
    the chain's prompt; set during a run it swaps in for the next clip generated,
    steering the stream without interrupting it.
    """

    prompt: str = MessageField(description="The held prompt now in force.")
    applied_live: bool = MessageField(
        description="True when a chain was already running and the new prompt takes the next clip."
    )


class SeedAccepted(ModelMessage):
    """Emitted when `set_seed` is accepted."""

    seed: int = MessageField(description="Seed the next chain's first clip will generate from.")


class ClipLengthAccepted(ModelMessage):
    """Emitted when `set_clip_seconds` is accepted.

    The requested length is snapped to the nearest length the model can produce,
    so the value here may differ slightly from the one sent.
    """

    clip_seconds: float = MessageField(description="Clip length now in effect, in seconds.")
    frames: int = MessageField(description="Frames each generated clip will carry.")


class CanvasAccepted(ModelMessage):
    """Emitted when `set_canvas` is accepted."""

    aspect: str = MessageField(description="Aspect ratio now in effect.")
    width: int = MessageField(description="Width of every frame on `main_video`.")
    height: int = MessageField(description="Height of every frame on `main_video`.")


class SessionReset(ModelMessage):
    """Emitted when `reset` is accepted.

    Every condition is back to its default, any running chain is cut, and the
    output stream is cleared.
    """

    was_streaming: bool = MessageField(
        description="A chain was running and has been cut; a `stream_stopped` accompanies it."
    )


class CommandError(ModelMessage):
    """Emitted when a command is rejected. The command had no effect."""

    command: str = MessageField(description="Name of the command that was rejected.")
    reason: str = MessageField(description="Why it was rejected.")
