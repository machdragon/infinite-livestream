"""Live FastH3's seam math, chain control, playout, schema and manifest.

Everything here runs on a laptop: the GPU work sits behind the backend, which
the chain tests replace with a fake that builds instantly, so the real
producer/consumer, seam stitching, pacing, emission and teardown all run. The
seam module is pure numpy and is tested directly.

Run from the model folder: ``PYTHONPATH=. python -m pytest tests/ -q``.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

import live_h3_clip_plan as clip_plan
import live_h3_seam as seam
from live_h3 import EMIT_FRAMES, SAMPLES_PER_FRAME, LiveH3
from live_h3_assets import LiveH3Config, load_config
from live_h3_backend import OUTPUT_SAMPLE_RATE, ClipJob

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_DIR.parent
FAST_H3_DIR = REPO_ROOT / "fast-h3"


# ============================================================ shared-module parity
#
# Three files are copied from fast-h3 rather than imported, so the schema renders
# without torch and the two channels stay independently deployable. These tests
# are what stop a copy going stale.


def test_clip_plan_is_a_verbatim_copy_of_fast_h3():
    ours = (MODEL_DIR / "live_h3_clip_plan.py").read_text(encoding="utf-8")
    theirs = (FAST_H3_DIR / "fasth3_clip_plan.py").read_text(encoding="utf-8")
    assert ours == theirs, "live_h3_clip_plan.py has drifted from fast-h3/fasth3_clip_plan.py"


def test_sitecustomize_is_a_verbatim_copy_of_fast_h3():
    ours = (MODEL_DIR / "sitecustomize.py").read_text(encoding="utf-8")
    theirs = (FAST_H3_DIR / "sitecustomize.py").read_text(encoding="utf-8")
    assert ours == theirs, "sitecustomize.py has drifted from fast-h3/sitecustomize.py"


def test_requirements_is_a_verbatim_copy_of_fast_h3():
    ours = (MODEL_DIR / "requirements.txt").read_text(encoding="utf-8")
    theirs = (FAST_H3_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert ours == theirs, "requirements.txt has drifted from fast-h3/requirements.txt"


def test_clip_geometry_matches_fast_h3():
    """The copy is exercised, not just diffed: the geometry both channels rely on."""
    assert clip_plan.FPS == 24
    assert clip_plan.MIN_FRAMES == 124
    assert clip_plan.MAX_FRAMES == 345
    for frames in (1, 100, 124, 200, 345):
        aligned = clip_plan.align_frames(frames)
        assert aligned % 17 == 5 and aligned >= frames
    assert len(clip_plan.legal_frame_counts()) == 14


# ================================================================= the seam math
#
# Pure numpy: the color-match lock and the linear-light "linearfade" blend.


def _gradient_clip(frames: int, base: int) -> np.ndarray:
    """A small clip whose brightness rises frame to frame, offset by ``base``."""
    h, w = 8, 8
    clip = np.zeros((frames, h, w, 3), np.float32)
    for f in range(frames):
        clip[f] = base + f * 3
    return np.clip(clip, 0, 255).astype(np.uint8)


def test_reference_rgb_is_the_frames_mean():
    frame = np.full((4, 4, 3), 120, np.uint8)
    frame[..., 0] = 40  # a red-shifted frame
    assert seam.reference_rgb(frame).tolist() == pytest.approx([40.0, 120.0, 120.0])


def test_color_match_locks_a_clip_to_the_reference_mean():
    """A continuation clip's mean RGB is shifted onto clip 0's, once for the clip."""
    reference = np.array([100.0, 110.0, 120.0], np.float32)
    clip = _gradient_clip(6, base=200)  # far brighter than the reference
    matched = seam.color_match_to_reference(clip, reference)
    assert np.allclose(matched.reshape(-1, 3).mean(0), reference, atol=1.0)


def test_color_match_is_one_offset_so_intra_clip_variation_survives():
    """The per-frame brightness ramp is preserved; only the average moves."""
    clip = _gradient_clip(6, base=100)
    before = seam.luma(clip)
    matched = seam.color_match_to_reference(clip, np.array([50.0, 50.0, 50.0], np.float32))
    after = seam.luma(matched)
    # The frame-to-frame differences (the motion/variation) are unchanged.
    assert np.allclose(np.diff(before), np.diff(after), atol=1.0)


def test_color_match_does_not_ratchet_across_a_chain():
    """Locking every clip to ONE reference keeps the chain mean flat, not drifting.

    Re-deriving the reference from each corrected clip's own last frame (the bug
    that blew the bear video out to white) would let it climb; this asserts the
    lock holds it.
    """
    reference = seam.reference_rgb(_gradient_clip(6, base=90)[-1])
    means = []
    for base in (90, 160, 220, 255):  # successively brighter raw clips
        matched = seam.color_match_to_reference(_gradient_clip(6, base=base), reference)
        means.append(matched.reshape(-1, 3).mean(0))
    spread = np.ptp(np.array(means), axis=0)
    assert np.all(spread < 3.0), f"clip means drifted across the chain: {means}"


def test_linearfade_is_monotonic_with_no_midpoint_flash():
    """The linear-light complementary blend rises smoothly from tail to head.

    The sRGB equal-power blend it replaces overshoots at the midpoint for
    near-identical frames — the flash. Between two flat clips the fixed blend's
    luma must be monotonic and never exceed the brighter endpoint.
    """
    tail = np.full((12, 16, 16, 3), 60, np.uint8)  # darker clip tail
    head = np.full((12, 16, 16, 3), 200, np.uint8)  # brighter clip head
    blended = seam.blend_video_linear(tail, head)
    y = seam.luma(blended)
    assert blended.shape == tail.shape
    assert np.all(np.diff(y) >= -0.5), f"luma dipped inside the blend: {y}"
    assert y.max() <= seam.luma(head).max() + 1.0, "blend rose above the brighter endpoint (flash)"
    assert y.min() >= seam.luma(tail).min() - 1.0


def test_linearfade_endpoints_approach_each_clip():
    """The blend starts near the tail's level and ends near the head's."""
    tail = np.full((12, 8, 8, 3), 60, np.uint8)
    head = np.full((12, 8, 8, 3), 200, np.uint8)
    y = seam.luma(seam.blend_video_linear(tail, head))
    assert abs(y[0] - seam.luma(tail)[0]) < abs(y[0] - seam.luma(head)[0])
    assert abs(y[-1] - seam.luma(head)[-1]) < abs(y[-1] - seam.luma(tail)[-1])


def test_audio_overlap_is_equal_power_and_never_wraps():
    """The crossfade holds energy flat and stays inside int16 without wrapping."""
    fade_out, fade_in = seam.equal_power_ramps(64)
    assert np.allclose(fade_out**2 + fade_in**2, 1.0, atol=1e-5)
    tail = np.full((1, 64), 30000, np.int16)
    head = np.full((1, 64), 30000, np.int16)
    mixed = seam.blend_audio_equal_power(tail, head)
    assert mixed.dtype == np.int16
    assert mixed.shape == (1, 64)
    assert int(mixed.max()) <= 32767 and int(mixed.min()) >= -32768


# ==================================================================== the config


def test_the_shipped_config_parses():
    config = load_config(MODEL_DIR / "live_h3.yaml")
    assert config.aspect == "16:9"
    assert config.crossfade_frames == 12
    assert config.color_match == "per_clip"
    assert config.buffer_depth == 2
    assert config.clip_frames == clip_plan.frames_for_seconds(5.167)
    # "default" warms only the shipped clip length — the live channel's one shape.
    assert config.warmup_frames == (config.clip_frames,)


def test_a_crossfade_at_least_a_clip_long_is_rejected(tmp_path):
    bad = tmp_path / "xf.yaml"
    bad.write_text(
        "inference:\n  clip_seconds: 5.167\n  crossfade_frames: 999\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_config(bad)


def test_an_unknown_color_match_mode_is_rejected(tmp_path):
    bad = tmp_path / "cm.yaml"
    bad.write_text("inference:\n  color_match: sometimes\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_a_bad_aspect_or_buffer_depth_fails_startup(tmp_path):
    for body in ("inference:\n  aspect: '32:9'\n", "inference:\n  buffer_depth: 0\n"):
        path = tmp_path / "bad.yaml"
        path.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)


# ========================================================= command contract (state)
#
# The real handlers on a model whose ``load()`` never ran: everything they touch
# is session state and pure arithmetic, so the whole control surface is testable
# on a laptop.


def make_config(clip_frames=6, crossfade_frames=2, buffer_depth=2) -> LiveH3Config:
    return LiveH3Config(
        aspect="16:9",
        clip_frames=clip_frames,
        seed=1000,
        num_inference_steps=5,
        crossfade_frames=crossfade_frames,
        color_match="per_clip",
        buffer_depth=buffer_depth,
        warmup_aspects=("16:9",),
        warmup_frames=(clip_frames,),
        inference={},
        runtime={},
    )


def run(coro):
    return asyncio.run(coro)


def refusal(model):
    errors = [m for m in model.sent if type(m).__name__ == "CommandError"]
    return errors[-1] if errors else None


@pytest.fixture
def model():
    """A LiveH3 with the attributes ``load()`` would have set, and no engine."""
    instance = LiveH3()
    instance._on_loop_ready()
    instance.config = make_config()
    instance._reset_session_state()

    sent: list = []

    async def capture(message):
        sent.append(message)

    instance.send = capture
    instance.sent = sent
    return instance


def test_start_needs_a_held_prompt(model):
    assert run(model.start()) is None
    assert refusal(model).command == "start"
    assert model._streaming is False


def test_set_prompt_holds_the_prompt_and_enables_start(model):
    assert "start" not in run(model.get_state()).valid_commands
    reply = run(model.set_prompt(prompt="  a bear in a misty forest  "))
    assert reply.prompt == "a bear in a misty forest"
    assert reply.applied_live is False
    assert model._prompt == "a bear in a misty forest"
    assert "start" in run(model.get_state()).valid_commands


def test_an_empty_prompt_is_refused(model):
    assert run(model.set_prompt(prompt="   ")) is None
    assert refusal(model).command == "set_prompt"
    assert model._prompt == ""


def test_stop_needs_a_running_chain(model):
    assert run(model.stop()) is None
    assert refusal(model).command == "stop"


def test_set_seed_and_clip_seconds_take_the_next_chain(model):
    assert run(model.set_seed(seed=7)).seed == 7
    assert model._seed == 7
    reply = run(model.set_clip_seconds(seconds=8.0))
    assert reply.frames == clip_plan.frames_for_seconds(8.0)
    assert reply.frames % 17 == 5
    assert model._clip_frames == reply.frames


def test_canvas_is_locked_while_streaming(model):
    reply = run(model.set_canvas(aspect="9:16"))
    assert (reply.height, reply.width) == clip_plan.canvas_for_choice("9:16")
    model._streaming = True
    assert run(model.set_canvas(aspect="1:1")) is None
    assert refusal(model).command == "set_canvas"
    assert model._aspect == "9:16"


def test_reset_clears_the_prompt_and_restores_defaults(model):
    run(model.set_prompt(prompt="a bear"))
    run(model.set_seed(seed=7))
    run(model.set_clip_seconds(seconds=8.0))
    reply = run(model.reset())
    assert reply.was_streaming is False
    assert model._prompt == ""
    assert model._seed == model.config.seed
    assert model._clip_frames == model.config.clip_frames
    assert model._aspect == model.config.aspect


def test_get_state_publishes_the_live_command_set(model):
    snapshot = run(model.get_state())
    assert "start" not in snapshot.valid_commands
    assert "set_canvas" in snapshot.valid_commands
    assert snapshot.streaming is False
    assert snapshot.clips_emitted == 0
    run(model.set_prompt(prompt="a bear"))
    assert "start" in run(model.get_state()).valid_commands


# ============================================================ the chain, end to end
#
# ``_serve`` running a real producer/consumer against a fake backend: the whole
# held-prompt chain — FL2VA anchoring, the seam stitch, pacing, stop — all run.

FRAMES_PER_CLIP = 6


class FakeBackend:
    """Builds tiny clips on demand; records every submission, anchor included."""

    def __init__(self):
        self.built: list[dict] = []

    def submit(self, *, frames, prompt, seed, height, width, anchor_image=None) -> ClipJob:
        self.built.append(
            {"frames": frames, "prompt": prompt, "seed": seed, "anchored": anchor_image is not None}
        )
        job = ClipJob(None)
        # A distinct grey per clip so color-match has something to move.
        base = 40 + 30 * (len(self.built) % 4)
        video = [np.full((height, width, 3), base + f, np.uint8) for f in range(frames)]
        samples = np.zeros((1, round(frames / 24 * OUTPUT_SAMPLE_RATE)), np.int16)
        job.result = (video, samples)
        job.done.set()
        return job


@pytest.fixture
def live():
    """A LiveH3 wired to a fake backend, with a connected audience."""
    instance = LiveH3()
    instance._on_loop_ready()
    instance.connected.set()
    instance.config = make_config()
    instance._reset_session_state()
    instance.backend = FakeBackend()

    emitted: list = []

    async def fake_emit(output):
        emitted.append(output)

    instance.emit = fake_emit
    instance.emitted = emitted

    messages: list = []

    async def fake_send(message):
        messages.append(message)

    instance.send = fake_send
    instance.messages = messages

    instance.output.flush = lambda: None
    return instance


def names(messages) -> list[str]:
    return [type(m).__name__ for m in messages]


def drive(live, scenario):
    """Run ``_serve`` against a scenario coroutine that ends the session."""

    async def main():
        async def wrapped():
            try:
                await scenario()
            finally:
                live.connected.clear()

        await asyncio.gather(live._serve(), wrapped())

    asyncio.run(main())


async def eventually(predicate, timeout=4.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        await asyncio.sleep(0.01)


def test_one_prompt_drives_an_indefinite_fl2va_chain(live):
    async def scenario():
        await live.set_prompt(prompt="a bear in a misty forest")
        await live.start()
        await eventually(lambda: live._clips_emitted >= 3)
        await live.stop()
        await eventually(lambda: not live._streaming)

    drive(live, scenario)
    built = live.backend.built
    assert len(built) >= 3
    # One held prompt, no re-prompting: every build used the same prompt.
    assert {b["prompt"] for b in built} == {"a bear in a misty forest"}
    # The chain: clip 0 is T2VA (no anchor), every clip after is FL2VA-anchored.
    assert built[0]["anchored"] is False
    assert all(b["anchored"] for b in built[1:])
    # The seed advances by one per clip so a run reproduces.
    assert [b["seed"] for b in built[:3]] == [1000, 1001, 1002]
    assert "StreamStarted" in names(live.messages)
    assert "StreamStopped" in names(live.messages)
    assert live._streaming is False


def test_video_and_audio_stay_locked_slice_for_slice(live):
    async def scenario():
        await live.set_prompt(prompt="a bear")
        await live.start()
        await eventually(lambda: live._clips_emitted >= 2)
        await live.stop()
        await eventually(lambda: not live._streaming)

    drive(live, scenario)
    assert live.emitted, "the chain emitted no frames"
    for output in live.emitted:
        video_frames = output.main_video.shape[0]
        audio_samples = output.main_audio.shape[1]
        assert output.main_video.dtype == np.uint8
        assert output.main_audio.dtype == np.int16
        assert output.main_audio.ndim == 2 and output.main_audio.shape[0] == 1
        assert audio_samples == video_frames * SAMPLES_PER_FRAME


def test_set_prompt_steers_the_chain_mid_stream(live):
    async def scenario():
        await live.set_prompt(prompt="scene one")
        await live.start()
        await eventually(lambda: live._clips_emitted >= 1)
        reply = await live.set_prompt(prompt="scene two")
        assert reply.applied_live is True
        await eventually(
            lambda: any(b["prompt"] == "scene two" for b in live.backend.built), timeout=4.0
        )
        await live.stop()
        await eventually(lambda: not live._streaming)

    drive(live, scenario)
    prompts = [b["prompt"] for b in live.backend.built]
    assert prompts[0] == "scene one"
    assert "scene two" in prompts

def test_stop_cuts_the_chain_and_a_restart_begins_fresh(live):
    async def scenario():
        await live.set_prompt(prompt="a bear")
        await live.start()
        await eventually(lambda: live._clips_emitted >= 2)
        await live.stop()
        await eventually(lambda: not live._streaming)
        # A second start begins a new chain from clip 0 (T2VA) again.
        live.backend.built.clear()
        await live.start()
        await eventually(lambda: live.backend.built and live.backend.built[0]["anchored"] is False)
        await live.stop()
        await eventually(lambda: not live._streaming)

    drive(live, scenario)
    assert names(live.messages).count("StreamStarted") == 2


def test_reset_cuts_the_chain_without_a_stream_stopped(live):
    async def scenario():
        await live.set_prompt(prompt="a bear")
        await live.start()
        await eventually(lambda: live._clips_emitted >= 1)
        await live.reset()
        await eventually(lambda: not live._streaming)
        await asyncio.sleep(0.1)

    drive(live, scenario)
    # `reset` answers with its own `session_reset` reply (a returned value, not a
    # broadcast), so the run loop suppresses the `stream_stopped` it would send
    # for a plain `stop`.
    assert "StreamStopped" not in names(live.messages)
    assert live._prompt == ""
    assert live._streaming is False


# --------------------------------------------------------- the seam consumer alone
#
# Driving ``_consumer`` on a pre-filled queue proves the frame-count arithmetic:
# each seam merges the k-frame tail and head into one k-frame blend, so a C-clip
# chain of N-frame clips emits C*N - (C-1)*k frames.


def test_the_seam_removes_exactly_one_overlap_per_boundary():
    instance = LiveH3()
    instance._on_loop_ready()
    instance.config = make_config(clip_frames=6, crossfade_frames=2)
    instance._reset_session_state()

    emitted: list = []

    async def fake_emit(output):
        emitted.append(output)

    async def fake_send(message):
        pass

    instance.emit = fake_emit
    instance.send = fake_send
    instance.connected.set()

    n, k, clips = 6, 2, 3

    async def main():
        queue: asyncio.Queue = asyncio.Queue()
        for index in range(clips):
            base = 50 + 20 * index
            video = np.stack(
                [np.full((8, 8, 3), base + f, np.uint8) for f in range(n)]
            )
            audio = np.zeros((1, n * SAMPLES_PER_FRAME), np.int16)
            await queue.put((index, "p", video, audio))
        await queue.put(None)
        await instance._consumer(queue)

    asyncio.run(main())
    total = sum(o.main_video.shape[0] for o in emitted)
    assert total == clips * n - (clips - 1) * k
    assert instance._frames_sent == total
    # And every emitted frame is a real uint8 frame with locked audio.
    for output in emitted:
        assert output.main_video.dtype == np.uint8
        assert output.main_audio.shape[1] == output.main_video.shape[0] * SAMPLES_PER_FRAME


# ================================================================ published contract
#
# ``reactor schema`` compiles this document out of the model class. A change here
# is a change to every client, so these tests make an accidental one fail loudly.

EXPECTED_COMMANDS = {
    "get_state": "StateUpdate",
    "reset": "SessionReset",
    "set_canvas": "CanvasAccepted",
    "set_clip_seconds": "ClipLengthAccepted",
    "set_prompt": "PromptAccepted",
    "set_seed": "SeedAccepted",
    "start": None,
    "stop": None,
}

EXPECTED_MESSAGES = {
    "canvas_accepted",
    "clip_complete",
    "clip_length_accepted",
    "command_error",
    "prompt_accepted",
    "seed_accepted",
    "session_reset",
    "state_update",
    "stream_started",
    "stream_stopped",
}

EXPECTED_REJECTIONS = ("start", "stop", "set_prompt", "set_canvas")


@pytest.fixture(scope="module")
def schema():
    from reactor_runtime.schema import render

    return render(MODEL_DIR, version="v0.1.0")


def test_the_model_publishes_two_outbound_tracks(schema):
    tracks = schema["x-reactor"]["tracks"]
    assert [(t["name"], t["kind"], t["direction"]) for t in tracks] == [
        ("main_video", "video", "out"),
        ("main_audio", "audio", "out"),
    ]


def test_the_command_set_is_exactly_what_clients_expect(schema):
    published = {path.removeprefix("/events/") for path in schema["paths"]}
    assert published == set(EXPECTED_COMMANDS)


def test_every_command_answers_with_the_type_it_promises(schema):
    for name, message in EXPECTED_COMMANDS.items():
        operation = schema["paths"][f"/events/{name}"]["post"]
        responses = operation["responses"]
        if message is None:
            assert set(responses) == {"202"}, name
            continue
        body = responses["200"]["content"]["application/json"]["schema"]
        assert body["$ref"] == f"#/components/schemas/{message}", name


def test_every_message_is_published_once(schema):
    assert set(schema["webhooks"]) == EXPECTED_MESSAGES


def test_the_prompt_is_marked_for_moderation(schema):
    """`set_prompt` carries client free text into generated video and audio."""
    properties = schema["paths"]["/events/set_prompt"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]
    assert properties["prompt"]["x-reactor-moderate"] is True


def test_the_clip_length_bounds_a_client_reads_are_generatable(schema):
    seconds = schema["paths"]["/events/set_clip_seconds"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]["seconds"]
    assert seconds["minimum"] == clip_plan.MIN_SECONDS_PUBLISHED
    assert seconds["maximum"] == clip_plan.MAX_SECONDS_PUBLISHED
    for bound in (seconds["minimum"], seconds["maximum"]):
        assert clip_plan.frames_for_seconds(bound) % 17 == 5


def test_the_canvas_choices_are_published_as_an_enum(schema):
    aspect = schema["paths"]["/events/set_canvas"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]["aspect"]
    assert aspect["enum"] == list(clip_plan.ASPECT_CHOICES)


def test_every_client_facing_string_is_documented(schema):
    for path, operations in schema["paths"].items():
        assert operations["post"].get("summary"), f"{path} has no description"
        body = operations["post"].get("requestBody")
        if not body:
            continue
        properties = body["content"]["application/json"]["schema"]["properties"]
        for name, field in properties.items():
            assert field.get("description"), f"{path} parameter {name} has no description"
    for name, message in schema["webhooks"].items():
        assert message["post"].get("summary"), f"message {name} has no description"


def test_every_message_summary_says_when_it_is_emitted(schema):
    for name, message in schema["webhooks"].items():
        summary = message["post"]["summary"]
        assert summary.startswith("Emitted "), f"{name}: {summary!r}"


def test_no_message_name_repeats_the_model_name(schema):
    forbidden = ("liveh3", "live-h3", "live_h3", "fasth3", "fast-h3")
    for name in schema["webhooks"]:
        assert not name.lower().startswith(forbidden), name
    for name in schema["components"]["schemas"]:
        assert not name.lower().startswith(forbidden), name


def test_every_refusable_command_documents_its_failure(schema):
    for name in EXPECTED_REJECTIONS:
        summary = schema["paths"][f"/events/{name}"]["post"]["summary"]
        assert "`command_error`" in summary, f"{name} does not document its failure"


# ===================================================================== the manifest

WEIGHT_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text(encoding="utf-8"))


def test_the_model_name_matches_the_folder(manifest):
    assert manifest["model"]["name"] == MODEL_DIR.name


def test_the_version_is_bare_semver(manifest):
    version = manifest["model"]["version"]
    assert isinstance(version, str), "quote the version so it is not parsed as a number"
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), f"{version!r} must be bare semver"


def test_the_manifest_carries_a_complete_resource_spec(manifest):
    resources = manifest["model"]["resources"]
    assert resources["gpu"]["type"] and resources["gpu"]["count"] >= 1
    assert resources["cpu"]["request"] and resources["cpu"]["limit"]
    assert resources["memory"]["request"] and resources["memory"]["limit"]


def test_the_image_is_built_from_the_manifest_not_a_dockerfile(manifest):
    assert not (MODEL_DIR / "Dockerfile").exists()
    build = manifest["build"]
    assert build["python_requirements"] == "requirements.txt"
    assert (MODEL_DIR / build["python_requirements"]).is_file()


def test_the_config_the_runtime_hands_to_load_exists(manifest):
    config = manifest["runtime"]["config"]
    assert (MODEL_DIR / config).is_file(), f"runtime.config points at a missing {config}"


def test_the_runtime_pin_is_current(manifest):
    assert manifest["build"]["runtime_version"] == "3.2.6"


def test_the_runtime_import_resolves_to_the_model_class(manifest):
    module_name, _, class_name = manifest["runtime"]["import"].partition(":")
    module = __import__(module_name)
    assert getattr(module, class_name, None) is not None, f"{module_name}.py has no {class_name}"


def test_no_weights_are_committed_alongside_the_model():
    offenders = [
        path.relative_to(MODEL_DIR)
        for path in MODEL_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in WEIGHT_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(MODEL_DIR).parts)
    ]
    assert not offenders, f"weights never live in git: {offenders}"


def test_the_manifest_build_matches_fast_h3(manifest):
    """The engine and its kernels are identical, so the build block must be too."""
    theirs = yaml.safe_load((FAST_H3_DIR / "reactor.yaml").read_text(encoding="utf-8"))
    assert manifest["build"] == theirs["build"]
    assert manifest["model"]["resources"] == theirs["model"]["resources"]


# ------------------------------------------------------------------ sitecustomize


def test_the_built_capabilities_mirror_the_manifest_arch_list(manifest):
    import sitecustomize

    arch_list = manifest["build"]["build_env"]["TORCH_CUDA_ARCH_LIST"]
    from_manifest = {
        tuple(int(part) for part in entry.rstrip("a").split("."))
        for entry in arch_list.split(";")
    }
    assert set(sitecustomize._VSA_BUILT_CAPABILITIES) == from_manifest
