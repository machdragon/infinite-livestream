"""The low-level H3 engine behind CausalH3: persistent GPU work, nothing client-facing.

The persistent model differs from ``fast-h3`` in one fundamental way: the
KV cache and clean x0 state carry forward across chunks, so the stream is
one continuous generation rather than a sequence of independent clips.

Each 5-latent media chunk executes four read-only denoise forwards (the
distilled four-step schedule).  The first three forwards are *read-only*:
they attend to the existing KV cache but do not write back, via a
``ReadOnlyKVCacheView`` that intercepts ``append`` calls.  Only the final
forward's clean x0 prediction is cache-filled (the transformer appends its
new KV to the real ``SlidingWindowKVCache``), so the next chunk attends to
the denoised output.

The backend loads the actual H3 model components (transformer, video VAE,
audio VAE, text encoder) directly from the validated model package path,
not through FastVideo's high-level ``VideoGenerator``.  The denoise loop
calls ``MiniMaxH3Transformer.forward()`` with the persistent KV cache,
using the real packed-sequence geometry from ``build_packed_sequence`` and
the real sigma schedule from ``build_sigma_schedule`` / ``remap_sigma``.

The backend also handles:
- Incremental VisualVAE decode with a separate bounded left-latent buffer.
- AudioVAE overlap-save for continuous audio without boundary artifacts.
- Stereo 32 kHz internally, downmixed/resampled to Reactor mono 48 kHz output.
- Honest compute_time / throughput / cache metrics on every chunk.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

from reactor_runtime.log import get_logger

import causalh3_chunk_plan as chunk_plan
from causalh3_assets import CausalH3Config, ModelPackageManifest
from causalh3_cache import (
    AudioVAEOverlapSave,
    CacheConfig,
    CleanX0State,
    ReadOnlyKVCacheView,
    VisualVAEDecodeBuffer,
)

logger = get_logger(__name__)

FRAME_RATE = chunk_plan.FPS

# WebRTC-native rate every chunk's waveform is resampled to.  The checkpoint's
# audio decoder is 32 kHz stereo; the wire is 48 kHz mono.
OUTPUT_SAMPLE_RATE = 48_000
NATIVE_SAMPLE_RATE = chunk_plan.AUDIO_SAMPLE_RATE  # 32_000
NATIVE_CHANNELS = chunk_plan.AUDIO_CHANNELS  # 2 (stereo)

# How often a blocking wait on the worker re-checks.
_WORKER_POLL_SECONDS = 0.1


class ChunkJob:
    """The handle to one submitted chunk build: its inputs, outcome, and completion.

    The error is carried back rather than only logged, so the submitter can
    report the failed chunk to clients.  ``cancelled`` set before the worker
    reaches the job skips the build entirely.
    """

    def __init__(self, fn):
        self.fn = fn
        self.done = threading.Event()
        self.result: Optional[ChunkResult] = None
        self.error: Optional[BaseException] = None
        self.cancelled = False


class ChunkResult:
    """One chunk's built output: decoded frames and wire-ready audio.

    ``native_audio`` is the stereo 32 kHz waveform kept for overlap-save
    state between chunks; ``audio`` is the downmixed/resampled mono 48 kHz
    int16 waveform for the output track.
    """

    def __init__(
        self,
        *,
        chunk_index: int,
        frames: list,
        audio,
        compute_time: float,
        cache_tokens_after: int,
        native_audio=None,
    ):
        self.chunk_index = chunk_index
        self.frames = frames
        self.audio = audio
        self.compute_time = compute_time
        self.cache_tokens_after = cache_tokens_after
        self.native_audio = native_audio


class CausalH3Backend:
    """Build persistent causal H3 chunks on demand, serialised on one worker thread.

    The GPU work itself runs through the low-level ai-toolkit H3 components
    (transformer, VAE, audio VAE, text encoder) loaded directly from the
    validated model package.  The thread serialises submissions and gives
    teardown a single handle to wait on.
    """

    def __init__(
        self,
        config: CausalH3Config,
        model_path: Path,
        manifest: ModelPackageManifest,
    ) -> None:
        """Remember the recipe and the weights location; nothing loads yet."""
        self._config = config
        self._model_path = model_path
        self._manifest = manifest
        self._jobs: queue.Queue[ChunkJob] = queue.Queue()
        self._worker: threading.Thread | None = None

        # Model components -- loaded by load(), held as real torch modules.
        self._transformer: Any = None
        self._video_vae: Any = None
        self._audio_vae: Any = None
        self._text_encoder: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self._device: Any = None
        self._dtype: Any = None

        # Cache state -- created here so tests can inspect without loading.
        cache_cfg = CacheConfig(
            sink_tokens=int(config.inference.get("cache_sink_tokens", 256)),
            window_tokens=int(config.inference.get("cache_window_tokens", 512)),
            vae_left_latents=int(config.inference.get("vae_left_latents", 5)),
            audio_overlap_latents=int(
                config.inference.get("audio_overlap_latents", 10)
            ),
        )
        self._cache_config = cache_cfg
        self.kv_cache: Any = None  # SlidingWindowKVCache, created on load
        self.clean_x0 = CleanX0State()
        self.vae_buffer = VisualVAEDecodeBuffer(cache_cfg.vae_left_latents)
        self.audio_overlap = AudioVAEOverlapSave(
            overlap_samples=cache_cfg.audio_overlap_latents
            * NATIVE_SAMPLE_RATE // int(chunk_plan.AUDIO_LATENTS_PER_SECOND)
        )

        # Session state: whether the text/conditions have been prefilled.
        self._prefilled = False
        self._chunk_index = -1

        # Prefilled layout state: the packed layout and text embeds from
        # the prefill, reused for every chunk's forward.
        self._prefill_layout: Any = None
        self._prefill_text_embeds: Any = None
        self._prefill_cond_rows: Any = None
        self._prefill_cond_audio_rows: Any = None

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        """Load the H3 model components and warm the persistent pipeline.

        Runs once at startup.  The caller's ``load()`` returning is what
        marks the pod ready, so everything that can fail must fail here.
        """
        self._apply_profile_environment()
        self._validate_profile_dependencies()
        self._validate_runtime_compatibility()
        self._raise_dynamo_limits()

        runtime = self._config.runtime
        num_gpus = int(runtime.get("num_gpus", 8))
        logger.info(
            "loading causal-h3 model components",
            model_path=str(self._model_path),
            num_gpus=num_gpus,
            family=self._config.family,
        )

        self._load_model_components()

        self._worker = threading.Thread(
            target=self._worker_loop, name="causal-h3-generation", daemon=True
        )
        self._worker.start()
        self._preload_native_imports()
        self._run_blocking(self._warmup)
        logger.info("causal-h3 backend loaded")

    def _load_model_components(self) -> None:
        """Load the transformer, VAEs, and text encoder from the model package.

        This loads the actual merged H3 weights directly, not through
        FastVideo's ``VideoGenerator``.  The transformer, video VAE, audio
        VAE, and text encoder are loaded from the validated component
        directories in the model package path.
        """
        import torch
        from safetensors.torch import load_file

        from minimax_h3.src.transformer import (
            MiniMaxH3Transformer,
            MiniMaxH3TransformerParams,
        )
        from minimax_h3.src.vae import MiniMaxH3VideoVAE
        from minimax_h3.src.audio_vae import (
            MiniMaxH3AudioVAE,
            fold_audio_vae_weight_norm,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        self._device = device
        self._dtype = dtype

        # --- transformer ---
        transformer_dir = self._model_path / "transformer"
        transformer_sd = load_file(transformer_dir / "model.safetensors")
        params = MiniMaxH3TransformerParams()
        table = transformer_sd.get("adaln_t_table", None)
        if table is not None:
            params.adaln_t_table_size = table.shape[0]
        transformer = MiniMaxH3Transformer(params)
        transformer.load_state_dict(transformer_sd, assign=True, strict=False)
        transformer.to(device).eval()
        transformer.requires_grad_(False)
        self._transformer = transformer
        del transformer_sd

        # --- video VAE ---
        video_vae_dir = self._model_path / "video_vae"
        video_sd = load_file(video_vae_dir / "model.safetensors")
        video_vae = MiniMaxH3VideoVAE()
        video_vae.load_state_dict(video_sd, strict=True, assign=True)
        video_vae.eval().requires_grad_(False)
        self._video_vae = video_vae
        del video_sd

        # --- audio VAE ---
        audio_vae_dir = self._model_path / "audio_vae"
        audio_sd = load_file(audio_vae_dir / "model.safetensors")
        audio_sd = fold_audio_vae_weight_norm(audio_sd)
        audio_vae = MiniMaxH3AudioVAE()
        audio_vae.load_state_dict(audio_sd, strict=True, assign=True)
        audio_vae.to(torch.float32).eval().requires_grad_(False)
        self._audio_vae = audio_vae
        del audio_sd

        # --- text encoder ---
        self._load_text_encoder()

        # --- KV cache ---
        from minimax_h3.src.causal import SlidingWindowKVCache

        num_layers = len(transformer.blocks)
        self.kv_cache = SlidingWindowKVCache(
            window_size=self._cache_config.window_tokens,
            sink_size=self._cache_config.sink_tokens,
        )

        logger.info(
            "causal-h3 model components loaded",
            transformer_layers=num_layers,
            device=str(device),
        )

    def _load_text_encoder(self) -> None:
        """Load the Qwen3-VL text encoder and tokenizer from the model package.

        The text encoder produces the 5120-dim conditioning embeddings that
        the transformer's ``condition_proj`` projects into the hidden size.
        Only the hidden states at the configured layer are used; the LM
        head and decoder stack are truncated.
        """
        import torch
        from safetensors.torch import load_file

        te_dir = self._model_path / "text_encoder"
        if not te_dir.is_dir():
            raise FileNotFoundError(
                f"text_encoder directory is missing: {te_dir}"
            )

        from transformers import AutoConfig, AutoTokenizer, AutoProcessor

        ORIGINAL_REPO = "MiniMaxAI/MiniMax-H3"
        TEXT_ENCODER_LAYER = 50

        tokenizer = AutoTokenizer.from_pretrained(
            ORIGINAL_REPO, subfolder="FL2VA/tokenizer"
        )
        processor = AutoProcessor.from_pretrained(
            ORIGINAL_REPO, subfolder="FL2VA/processor"
        )

        config = AutoConfig.from_pretrained(
            ORIGINAL_REPO, subfolder="FL2VA/text_encoder"
        )
        config.text_config.num_hidden_layers = TEXT_ENCODER_LAYER

        te_file = te_dir / "model.safetensors"
        if te_file.is_file():
            state_dict = load_file(te_file)
        else:
            # Fall back to loading from a directory of safetensors shards.
            from safetensors.torch import load_file as _lf
            import glob
            shards = sorted(glob.glob(str(te_dir / "*.safetensors")))
            state_dict = {}
            for s in shards:
                state_dict.update(_lf(s))

        # The text encoder is a Qwen3VLForConditionalGeneration; load with
        # the truncated config so only the first TEXT_ENCODER_LAYER layers
        # are instantiated.
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError:
            raise RuntimeError(
                "transformers does not provide Qwen3VLForConditionalGeneration; "
                "install the pinned transformers version"
            )

        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            te_dir, config=config, torch_dtype=self._dtype
        )
        text_encoder.model.language_model.norm = torch.nn.Identity()
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        text_encoder.to(self._device)

        self._text_encoder = text_encoder
        self._tokenizer = tokenizer
        self._processor = processor

    def _validate_runtime_compatibility(self) -> None:
        """Fail when VSA is requested without a compatible runtime.

        Dense mode may exist only as an explicit non-realtime diagnostic
        profile, never a silent fallback.  If VSA is requested but the
        runtime cannot provide it, fail loudly.
        """
        cfg = self._config.inference
        vsa_enabled = bool(cfg.get("vsa_enabled", True))
        if not vsa_enabled:
            if not bool(cfg.get("dense_diagnostic", False)):
                raise RuntimeError(
                    "VSA disabled but dense_diagnostic is not set; dense mode "
                    "may exist only as an explicit non-realtime diagnostic "
                    "profile, never a silent fallback. Set inference.dense_diagnostic: "
                    "true to acknowledge, or re-enable VSA."
                )
            logger.warning(
                "causal-h3 running in dense diagnostic mode (non-realtime)"
            )
            return

        # VSA requested: verify the runtime can provide it.
        vsa_kernel = str(cfg.get("vsa_kernel", "sm100a"))
        if vsa_kernel == "sm100a":
            try:
                from fastvideo_kernel import block_sparse_attn_sm100a
            except ImportError:
                present = False
            else:
                present = bool(getattr(block_sparse_attn_sm100a, "_HAS_VSA_SM100A", False))
            if not present:
                raise RuntimeError(
                    "VSA requested but the fastvideo-kernel sm100a route is "
                    "not available. Install a matching wheel, or set "
                    "inference.vsa_kernel: triton for the portable fallback."
                )

    @staticmethod
    def _raise_dynamo_limits() -> None:
        """Raise torch dynamo's recompile limits in this (parent) process."""
        import torch._dynamo.config as dynamo_config

        limit = int(os.environ.get("CAUSALH3_DYNAMO_RECOMPILE_LIMIT", "64"))
        dynamo_config.recompile_limit = max(limit, dynamo_config.recompile_limit)
        dynamo_config.cache_size_limit = max(limit, dynamo_config.cache_size_limit)
        dynamo_config.accumulated_recompile_limit = max(
            512, dynamo_config.accumulated_recompile_limit
        )
        dynamo_config.accumulated_cache_size_limit = max(
            512, dynamo_config.accumulated_cache_size_limit
        )
        dynamo_config.fail_on_recompile_limit_hit = False
        logger.info(
            "dynamo recompile limits raised",
            recompile_limit=dynamo_config.recompile_limit,
        )

    @staticmethod
    def _preload_native_imports() -> None:
        """Touch every deferred native import the build path needs."""
        import numpy  # noqa: F401
        import torch
        import torchaudio.functional as AF

        AF.resample(
            torch.zeros(NATIVE_CHANNELS, NATIVE_SAMPLE_RATE // 10),
            NATIVE_SAMPLE_RATE,
            OUTPUT_SAMPLE_RATE,
        )

    # --------------------------------------------------------------- profile

    def _apply_profile_environment(self) -> None:
        """Set the CausalH3 profile environment, mirroring the reference CLI."""
        cfg = self._config.inference
        vsa_kernel = str(cfg.get("vsa_kernel", "sm100a"))
        fusions = "all" if bool(cfg.get("h3_fusions", True)) else "0"
        environment: dict[str, str | None] = {
            "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3",
            "FASTVIDEO_VSA_SM100A": "1" if vsa_kernel == "sm100a" else "0",
            "FASTVIDEO_VSA_CUTEDSL": "0",
            "FASTVIDEO_H3_VSA_PROBE": None,
            "FASTVIDEO_DISABLE_ATTENTION_COMPILE": "0",
            "FASTVIDEO_FA4": "1" if bool(cfg.get("fa4", True)) else "0",
            "FASTVIDEO_NVFP4_FA4": "0",
            "FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN": "0",
            "FASTVIDEO_MINIMAX_H3_FUSIONS": fusions,
            "FASTVIDEO_INFERENCE_TORCH_COMPILE": (
                "1" if bool(cfg.get("inference_torch_compile", True)) else "0"
            ),
            "FASTVIDEO_VAE_PARALLEL_DECODE": (
                "1" if bool(cfg.get("vae_parallel_decode", True)) else "0"
            ),
            "FASTVIDEO_VAE_PARALLEL_ENCODE": "0",
            "FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY": "gather",
            "FASTVIDEO_ULYSSES_A2A": str(cfg.get("ulysses_a2a", "off")),
            "FASTVIDEO_STAGE_LOGGING": "1",
        }
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        logger.info(
            "causal-h3 profile", **{k: (v or "<unset>") for k, v in environment.items()}
        )

    def _validate_profile_dependencies(self) -> None:
        """Fail before the load when the selected fast route is absent."""
        import importlib.util

        cfg = self._config.inference
        if bool(cfg.get("fa4", True)):
            try:
                present = importlib.util.find_spec("flash_attn.cute") is not None
            except (ImportError, ModuleNotFoundError):
                present = False
            if not present:
                raise RuntimeError(
                    "CausalH3's FA4 route needs the pinned flash-attn-4 package."
                )

    # ---------------------------------------------------------------- worker

    def _worker_loop(self) -> None:
        """Run submitted jobs, one at a time, forever."""
        logger.info("generation worker ready")
        while True:
            job = self._jobs.get()
            try:
                if not job.cancelled:
                    job.result = job.fn()
            except BaseException as error:  # noqa: BLE001
                job.error = error
                logger.exception("generation worker job raised")
            finally:
                job.done.set()

    def submit_chunk(
        self,
        *,
        chunk_index: int,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        frames: int,
    ) -> ChunkJob:
        """Queue one chunk build and hand back its job handle.

        Returns immediately; the caller polls ``job.done`` and reads
        ``job.result`` (a :class:`ChunkResult`) or ``job.error``.
        """
        job = ChunkJob(
            lambda: self._generate_chunk(
                chunk_index=chunk_index,
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                frames=frames,
            )
        )
        self._jobs.put(job)
        return job

    def submit_prefill(
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        first_frame: Any = None,
        last_frame: Any = None,
        references: list[Any] | None = None,
    ) -> ChunkJob:
        """Queue the one-time prefill of text and conditions.

        The prefill encodes the text prompt and any condition images
        (first/last frame for FL2VA, reference blocks for Ref2VA) into the
        KV cache's sink.  This runs once per session; subsequent chunks
        attend to the prefilled sink.
        """
        job = ChunkJob(
            lambda: self._prefill(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                first_frame=first_frame,
                last_frame=last_frame,
                references=references,
            )
        )
        self._jobs.put(job)
        return job

    def _run_blocking(self, fn) -> None:
        """Run work on the worker, block until it finishes, and re-raise its failure."""
        job = ChunkJob(fn)
        self._jobs.put(job)
        while not job.done.wait(timeout=_WORKER_POLL_SECONDS):
            pass
        if job.error is not None:
            raise job.error

    # --------------------------------------------------------------- warm-up

    def _warmup(self) -> None:
        """Build one throwaway prefix chunk before the pod reports ready."""
        height, width = chunk_plan.canvas_for_choice(self._config.aspect)
        started = time.monotonic()
        self._generate_chunk(
            chunk_index=-1,
            prompt="A slow cinematic shot of sunlight moving across a quiet room.",
            seed=self._config.seed,
            height=height,
            width=width,
            frames=chunk_plan.PREFIX_FRAMES,
        )
        # Reset cache state after warmup -- the real session starts fresh.
        self._reset_cache_state()
        logger.info(
            "causal-h3 warmed",
            seconds=round(time.monotonic() - started, 2),
        )

    def _reset_cache_state(self) -> None:
        """Reset all cache state for a new session."""
        from minimax_h3.src.causal import SlidingWindowKVCache

        self.kv_cache = SlidingWindowKVCache(
            window_size=self._cache_config.window_tokens,
            sink_size=self._cache_config.sink_tokens,
        )
        self.clean_x0.reset()
        self.vae_buffer.reset()
        self.audio_overlap.reset()
        self._prefilled = False
        self._chunk_index = -1
        self._prefill_layout = None
        self._prefill_text_embeds = None
        self._prefill_cond_rows = None
        self._prefill_cond_audio_rows = None

    # ------------------------------------------------------------ prefill

    def _prefill(
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        first_frame: Any = None,
        last_frame: Any = None,
        references: list[Any] | None = None,
    ) -> ChunkResult:
        """Encode text and conditions into the KV cache sink, once.

        After prefill, the sink tokens are immutable and never evicted.
        Returns a sentinel ChunkResult so the submitter's polling works.
        """
        import torch
        import numpy as np
        from PIL import Image

        from minimax_h3.src import packing
        from minimax_h3.src.packing import (
            build_packed_sequence,
            build_row_timesteps,
            patchify_video_latents,
        )

        started = time.monotonic()
        device = self._device
        dtype = self._dtype

        h_lat = height // 16
        w_lat = width // 16

        # --- encode text ---
        text_embeds, text_token_tags = self._encode_text(prompt)
        text_embeds = text_embeds.to(device, dtype)
        text_token_tags = text_token_tags.to("cpu", torch.long)

        # --- build condition rows ---
        cond_rows = None
        cond_audio_rows = None
        anchors = ()
        ref_blocks = ()

        if self._config.family == "fl2va":
            if first_frame is not None:
                anchors = ("first",)
                cond_rows = self._encode_condition_image(first_frame, device)
            # last_frame is handled as a second anchor if present
            if last_frame is not None:
                anchors = ("first", "last") if first_frame is not None else ("last",)
                last_rows = self._encode_condition_image(last_frame, device)
                if cond_rows is not None:
                    cond_rows = torch.cat([cond_rows, last_rows], dim=1)
                else:
                    cond_rows = last_rows
        elif self._config.family == "ref2va" and references:
            parts = []
            audio_parts = []
            for r in references:
                if isinstance(r, dict):
                    parts.append(self._noise_aug_rows(r["latent"][None], device))
                    if r.get("audio_rows") is not None:
                        audio_parts.append(r["audio_rows"][None].to(device))
                    ref_blocks = ref_blocks + (
                        (r["latent"].shape[1], r["latent"].shape[2], r["latent"].shape[3]),
                    )
                elif isinstance(r, torch.Tensor):
                    parts.append(self._noise_aug_rows(r[None], device))
                    ref_blocks = ref_blocks + ((r.shape[1], r.shape[2], r.shape[3]),)
                else:
                    # PIL Image
                    parts.append(self._encode_condition_image(r, device))
                    ref_blocks = ref_blocks + ((1, r.size[1] // 16, r.size[0] // 16),)
            cond_rows = torch.cat(parts, dim=1)
            if audio_parts:
                cond_audio_rows = torch.cat(audio_parts, dim=1).float()

        # --- build packed layout for prefill ---
        # Use the prefix frame count to determine latent geometry.
        t_lat = chunk_plan._LATENTS_PER_CHUNK  # 5 latents for a 17-frame chunk
        a_lat = packing.audio_latent_num_frames(chunk_plan.FRAMES_PER_CHUNK)

        layout = build_packed_sequence(
            text_token_tags=text_token_tags,
            num_latent_frames=t_lat,
            latent_height=h_lat,
            latent_width=w_lat,
            num_audio_latents=a_lat,
            keyframe_anchors=anchors,
            ref_blocks=ref_blocks,
        )

        # --- run the transformer once to prefill the KV cache sink ---
        # Build the full packed input (text + conditions + zero media) and
        # run one forward with the real KV cache to populate the sink.
        num_cond = layout.num_condition_video_rows

        # Create zero media rows for the prefill forward.
        rows_per_frame = (h_lat // 2) * (w_lat // 2)
        video_rows = torch.zeros(
            1, t_lat * rows_per_frame, 96, device=device, dtype=dtype
        )
        audio_rows = torch.zeros(
            1, a_lat * 2, 32, device=device, dtype=dtype
        )

        video_in = video_rows
        if cond_rows is not None:
            video_in = torch.cat([cond_rows, video_rows], dim=1)
        audio_in = audio_rows
        if cond_audio_rows is not None:
            audio_in = torch.cat([cond_audio_rows, audio_rows], dim=1)

        position_ids = layout.position_ids[None].to(device)
        tags = layout.token_tags[None].to(device)
        video_indices = layout.video_indices.to(device)
        audio_indices = layout.audio_indices.to(device)
        text_indices = layout.text_indices.to(device)

        # Use t=1.0 (pure noise) for the prefill forward.
        row_t = build_row_timesteps(layout, 1.0, 1.0)[None].to(device)

        self._transformer(
            hidden_states=video_in.to(dtype),
            audio_hidden_states=audio_in.to(dtype),
            encoder_hidden_states=text_embeds[None],
            row_timesteps=row_t,
            token_tags=tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            use_causal_mask=True,
            kv_cache=self.kv_cache,
            num_condition_rows=num_cond + layout.num_condition_audio_rows,
        )

        # Store the prefill state for reuse in chunk forwards.
        self._prefill_layout = layout
        self._prefill_text_embeds = text_embeds
        self._prefill_cond_rows = cond_rows
        self._prefill_cond_audio_rows = cond_audio_rows

        self._prefilled = True
        elapsed = time.monotonic() - started
        cache_tokens = len(self.kv_cache) if self.kv_cache else 0
        logger.info(
            "causal-h3 prefilled",
            sink_tokens=cache_tokens,
            seconds=round(elapsed, 2),
        )
        return ChunkResult(
            chunk_index=-1,
            frames=[],
            audio=None,
            compute_time=elapsed,
            cache_tokens_after=cache_tokens,
        )

    def _encode_text(self, prompt: str):
        """Encode a text prompt into conditioning embeddings and token tags.

        Runs the Qwen3-VL text encoder and extracts the hidden states at
        the configured layer.  Returns ``(text_embeds (L, 5120),
        text_token_tags (L,) long)``.
        """
        import torch

        inputs = self._tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(self._device)

        with torch.no_grad():
            outputs = self._text_encoder(
                **inputs, output_hidden_states=True
            )
            # Layer 50 hidden states are the conditioning embeddings.
            hidden = outputs.hidden_states[-1]
            # Remove any vision tokens; text-only conditioning.
            # The token tags are all 1 (text) for the text segment.
            text_embeds = hidden[0].float()  # (L, 5120)
            text_token_tags = torch.ones(
                text_embeds.shape[0], dtype=torch.long
            )
        return text_embeds, text_token_tags

    def _encode_condition_image(self, image, device):
        """Encode a condition image into patchified video rows.

        Uses the video VAE encoder with the released noise-aug recipe:
        ``x = t * clean + (1 - t) * noise`` at ``t = 0.999``.
        Returns ``(1, rows, 96)`` patchified latent rows.
        """
        import torch
        import numpy as np

        KEYFRAME_NOISE_AUG_T = 0.999

        frame = torch.from_numpy(np.array(image)).float()
        frame = (frame / 255.0) * 2.0 - 1.0  # (H, W, 3) -> [-1, 1]
        frame = frame.permute(2, 0, 1)[None, :, None]  # (1, 3, 1, H, W)

        with torch.no_grad():
            cond_latents = self._video_vae.encode(
                frame.to(self._video_vae.device), sample=False
            )  # (1, 24, 1, h, w)

        cond_noise = torch.randn(
            (1, 24, 1, cond_latents.shape[3], cond_latents.shape[4]),
            device=device,
            dtype=torch.float32,
        )
        mixed = (
            KEYFRAME_NOISE_AUG_T * cond_latents.to(device)
            + (1.0 - KEYFRAME_NOISE_AUG_T) * cond_noise
        )
        from minimax_h3.src.packing import patchify_video_latents
        return patchify_video_latents(mixed)

    def _noise_aug_rows(self, latents, device):
        """Apply noise augmentation to pre-encoded reference latents."""
        import torch
        from minimax_h3.src.packing import patchify_video_latents

        KEYFRAME_NOISE_AUG_T = 0.999
        cond_noise = torch.randn(latents.shape, device=device, dtype=torch.float32)
        mixed = (
            KEYFRAME_NOISE_AUG_T * latents.to(device, torch.float32)
            + (1.0 - KEYFRAME_NOISE_AUG_T) * cond_noise
        )
        return patchify_video_latents(mixed)

    # ------------------------------------------------------------ generation

    def _generate_chunk(
        self,
        *,
        chunk_index: int,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        frames: int,
    ) -> ChunkResult:
        """Build one media chunk with persistent clean-KV generation.

        The four-step distilled schedule runs four forwards.  The first
        three are read-only (attend to the KV cache but do not write back,
        via ``ReadOnlyKVCacheView``).  Only the final forward's clean x0
        is cache-filled (the transformer appends to the real
        ``SlidingWindowKVCache``), so the next chunk attends to the
        denoised output.

        After denoising, the chunk's latents are decoded incrementally:
        the VisualVAE uses the left-latent buffer for context, and the
        AudioVAE uses overlap-save for continuous audio.
        """
        import torch
        started = time.monotonic()

        # --- denoise: four forwards, first three read-only, final cache-fill ---
        video_x0, audio_x0 = self._denoise_chunk(
            chunk_index=chunk_index,
            seed=seed,
            height=height,
            width=width,
            frames=frames,
        )

        # --- carry the clean x0 forward ---
        self.clean_x0.update(video_x0, audio_x0, chunk_index)
        self._chunk_index = chunk_index

        # --- incremental VisualVAE decode with left-latent buffer ---
        video_frames = self._decode_video(video_x0, height, width, frames)

        # --- AudioVAE overlap-save decode ---
        native_audio, wire_audio = self._decode_audio(audio_x0, frames)

        elapsed = time.monotonic() - started
        content = len(video_frames) / FRAME_RATE
        throughput = content / elapsed if elapsed > 0 else 0.0

        cache_tokens = len(self.kv_cache) if self.kv_cache else 0
        logger.info(
            f"chunk built: {len(video_frames)}f ({content:.2f}s content) "
            f"in {elapsed:.2f}s = {throughput:.2f}x realtime, "
            f"cache={cache_tokens} tokens"
        )
        return ChunkResult(
            chunk_index=chunk_index,
            frames=video_frames,
            audio=wire_audio,
            compute_time=elapsed,
            cache_tokens_after=cache_tokens,
            native_audio=native_audio,
        )

    def _denoise_chunk(
        self,
        *,
        chunk_index: int,
        seed: int,
        height: int,
        width: int,
        frames: int,
    ):
        """Run the four-step distilled schedule with persistent KV cache.

        Steps 1-3 (sigma 999->749, 749->500, 500->250) are read-only:
        they attend to the existing KV cache through a
        ``ReadOnlyKVCacheView`` but do not write back.  Step 4 (250->0)
        produces the clean x0 that is cache-filled by running the
        transformer with the real ``SlidingWindowKVCache``.

        The previous chunk's clean x0 (from :attr:`clean_x0`) initializes
        the noisy latent.  For the first chunk (prefix), the clean x0 is
        zero (pure noise).

        Returns ``(video_x0, audio_x0)`` as real latent row tensors.
        """
        import torch
        from minimax_h3.src import packing
        from minimax_h3.src.packing import (
            build_packed_sequence,
            build_row_timesteps,
            build_sigma_schedule,
            patchify_video_latents,
            pack_audio_latents,
            remap_sigma,
            unpatchify_video_tokens,
            unpack_audio_tokens,
        )

        device = self._device
        dtype = self._dtype

        h_lat = height // 16
        w_lat = width // 16

        # Determine latent frame count for this chunk.
        if frames == chunk_plan.PREFIX_FRAMES:
            t_lat = 2  # 5 prefix frames -> 2 latent frames
        else:
            t_lat = chunk_plan._LATENTS_PER_CHUNK  # 5 latents per 17-frame chunk

        a_lat = packing.audio_latent_num_frames(frames)

        # --- build the packed layout for this chunk ---
        # The chunk_offset advances the temporal position for each chunk
        # after the prefix.
        chunk_offset = 0.0
        if chunk_index > 0:
            # Each chunk advances by the temporal span of 5 latents.
            chunk_offset = float(chunk_index) * packing._temporal_position_span(
                chunk_plan._LATENTS_PER_CHUNK
            )

        layout = build_packed_sequence(
            text_token_tags=self._prefill_layout.text_indices.new_ones(
                self._prefill_layout.text_indices.shape[0]
            ),
            num_latent_frames=t_lat,
            latent_height=h_lat,
            latent_width=w_lat,
            num_audio_latents=a_lat,
            keyframe_anchors=(),
            ref_blocks=(),
            chunk_offset=chunk_offset,
        )

        num_cond = layout.num_condition_video_rows

        # --- initialize noisy latents ---
        generator = torch.Generator(device=device).manual_seed(seed)

        # Start from the previous chunk's clean x0, or pure noise for the
        # first chunk.
        prev_video_x0 = self.clean_x0.video_x0
        prev_audio_x0 = self.clean_x0.audio_x0

        rows_per_frame = (h_lat // 2) * (w_lat // 2)

        if prev_video_x0 is not None:
            # Carry the clean x0 forward: the previous chunk's denoised
            # output seeds this chunk's noisy initialization.
            video_rows = prev_video_x0.clone()
            audio_rows = prev_audio_x0.clone()
        else:
            # First chunk: pure noise.
            video_noise = torch.randn(
                (1, 24, t_lat, h_lat, w_lat),
                generator=generator, device=device, dtype=torch.float32,
            )
            video_rows = patchify_video_latents(video_noise)
            audio_noise = torch.randn(
                (1, 2, 32, a_lat),
                generator=generator, device=device, dtype=torch.float32,
            )
            audio_rows = pack_audio_latents(audio_noise)

        # --- sigma schedule ---
        # The four-step distilled schedule: [999, 749, 500, 250, 0]
        # through the video shift (12) and audio shift (3).
        sigmas_v = build_sigma_schedule(
            chunk_plan.NUM_INFERENCE_STEPS, chunk_plan.VIDEO_SIGMA_SHIFT
        ).to(device)
        sigmas_a = remap_sigma(
            sigmas_v, chunk_plan.VIDEO_SIGMA_SHIFT, chunk_plan.AUDIO_SIGMA_SHIFT
        ).to(device)

        position_ids = layout.position_ids[None].to(device)
        tags = layout.token_tags[None].to(device)
        video_indices = layout.video_indices.to(device)
        audio_indices = layout.audio_indices.to(device)
        text_indices = layout.text_indices.to(device)

        text_embeds = self._prefill_text_embeds
        cond_rows = self._prefill_cond_rows
        cond_audio_rows = self._prefill_cond_audio_rows

        num_steps = sigmas_v.shape[0] - 1

        # --- denoise loop ---
        for i in range(num_steps):
            sv, sv_next = sigmas_v[i], sigmas_v[i + 1]
            sa, sa_next = sigmas_a[i], sigmas_a[i + 1]
            t_v = 1.0 - float(sv)
            t_a = 1.0 - float(sa)

            row_t = build_row_timesteps(layout, t_v, t_a)[None].to(device)

            video_in = video_rows
            if cond_rows is not None:
                video_in = torch.cat([cond_rows, video_rows], dim=1)
            audio_in = audio_rows
            if cond_audio_rows is not None:
                audio_in = torch.cat([cond_audio_rows, audio_rows], dim=1)

            # Steps 1..n-1: read-only (do not write to the KV cache).
            # Step n (the final step): cache-fill (write to the real KV cache).
            is_final_step = (i == num_steps - 1)
            cache_to_use = self.kv_cache if is_final_step else ReadOnlyKVCacheView(self.kv_cache)

            video_pred, audio_pred = self._transformer(
                hidden_states=video_in.to(dtype),
                audio_hidden_states=audio_in.to(dtype),
                encoder_hidden_states=text_embeds[None],
                row_timesteps=row_t,
                token_tags=tags,
                position_ids=position_ids,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
                use_causal_mask=True,
                kv_cache=cache_to_use,
                num_condition_rows=num_cond + layout.num_condition_audio_rows,
            )

            v_video = video_pred[:, num_cond:].float()
            v_audio = audio_pred[:, layout.num_condition_audio_rows:].float()

            denoised_v = video_rows + sv * v_video
            ratio_v = sv_next / sv
            video_rows = ratio_v * video_rows + (1.0 - ratio_v) * denoised_v

            denoised_a = audio_rows + sa * v_audio
            ratio_a = sa_next / sa if float(sa) != 0.0 else 0.0
            audio_rows = ratio_a * audio_rows + (1.0 - ratio_a) * denoised_a

        # The final video_rows and audio_rows are the clean x0.
        return video_rows, audio_rows

    def _decode_video(
        self,
        video_x0_rows,
        height: int,
        width: int,
        frames: int,
    ) -> list:
        """Incremental VisualVAE decode with left-latent buffer.

        Unpatchifies the clean x0 video rows into latent tensors, prepends
        the left-context latents from the previous chunk (held in
        :attr:`vae_buffer`), decodes via the real ``MiniMaxH3VideoVAE.decode``,
        and pushes the chunk's latents into the buffer for the next chunk.

        Returns a list of RGB uint8 frames ``(T, H, W, C)``.
        """
        import torch
        from minimax_h3.src.packing import unpatchify_video_tokens

        device = self._video_vae.device
        h_lat = height // 16
        w_lat = width // 16

        if frames == chunk_plan.PREFIX_FRAMES:
            t_lat = 2
        else:
            t_lat = chunk_plan._LATENTS_PER_CHUNK

        # Unpatchify the clean x0 rows into latent tensors.
        video_latents = unpatchify_video_tokens(
            video_x0_rows, t_lat, h_lat, w_lat
        )  # (1, 24, t_lat, h_lat, w_lat)

        # Prepend left-context latents from the previous chunk.
        left_ctx = self.vae_buffer.context()
        if left_ctx is not None:
            decode_latents = torch.cat([left_ctx, video_latents], dim=2)
        else:
            decode_latents = video_latents

        # Decode with the real VideoVAE.
        with torch.no_grad():
            decoded = self._video_vae.decode(decode_latents.to(device))
            # (1, 3, T, H, W) in [-1, 1]

        # Strip the left-context frames from the output.
        if left_ctx is not None:
            left_frames = left_ctx.shape[2] * 4  # 4x temporal compression
            decoded = decoded[:, :, left_frames:]

        # Trim to the requested frame count.
        if decoded.shape[2] > frames:
            decoded = decoded[:, :, :frames]

        # Convert to uint8 frames.
        video = ((decoded.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
        video = video[0].permute(1, 2, 3, 0).cpu()  # (T, H, W, C)

        # Push the chunk's latents into the buffer for the next decode.
        self.vae_buffer.push(video_latents.cpu())

        return [video[i].numpy() for i in range(video.shape[0])]

    def _decode_audio(self, audio_x0_rows, frames: int):
        """AudioVAE overlap-save decode: native stereo 32k -> wire mono 48k.

        Unpacks the clean x0 audio rows into latent tensors, decodes via
        the real ``MiniMaxH3AudioVAE.decode``, applies overlap-save for
        continuous audio, then downmixes and resamples to the wire format.

        Returns ``(native_audio, wire_audio)`` where native_audio is the
        stereo 32 kHz waveform (for overlap-save state) and wire_audio is
        the downmixed, resampled mono 48 kHz int16 waveform.
        """
        import torch
        import torchaudio.functional as AF
        from minimax_h3.src.packing import unpack_audio_tokens

        device = self._audio_vae.device
        a_lat = int(round(frames / FRAME_RATE * chunk_plan.AUDIO_LATENTS_PER_SECOND))

        # Unpack the clean x0 audio rows into latent tensors.
        audio_latents = unpack_audio_tokens(audio_x0_rows, a_lat)
        # audio_latents: (1, 2, 32, a_lat) -- stereo, 32 channels, a_lat frames

        # Decode each channel through the AudioVAE (mono decoder).
        with torch.no_grad():
            # The audio VAE is mono: decode each channel separately.
            ch0 = self._audio_vae.decode(audio_latents[0, 0:1].to(device))
            ch1 = self._audio_vae.decode(audio_latents[0, 1:2].to(device))
            # Each: (1, 1, samples)
            waveform = torch.cat([ch0, ch1], dim=0)  # (2, 1, samples)
            waveform = waveform[:, 0]  # (2, samples)

        waveform = waveform.detach().float().cpu()

        # Preserve stereo 32 kHz internally (native_audio).
        native_audio = waveform.clone()

        # Overlap-save: mix the saved overlap from the previous block.
        saved = self.audio_overlap.pop()
        if saved is not None:
            overlap_len = min(saved.shape[-1], waveform.shape[-1])
            if overlap_len > 0:
                waveform[:, :overlap_len] += saved[:, :overlap_len]

        # Save the tail for the next block's overlap.
        overlap_latents = self._cache_config.audio_overlap_latents
        overlap_samples = overlap_latents * NATIVE_SAMPLE_RATE // int(
            chunk_plan.AUDIO_LATENTS_PER_SECOND
        )
        if waveform.shape[-1] > overlap_samples:
            self.audio_overlap.save(
                waveform[:, -overlap_samples:].clone(),
                self._chunk_index,
            )

        # Downmix and resample to wire format (mono 48 kHz int16).
        wire = self._to_wire_audio(waveform, NATIVE_SAMPLE_RATE, frames)
        return native_audio, wire

    @staticmethod
    def _to_wire_audio(waveform, sample_rate: int, frames: int):
        """Resample, downmix and quantize one chunk's waveform for the wire.

        Mono at the source is deliberate: the transport mean-downmixes
        before the wire anyway, and the runtime recorder flattens two
        channels by concatenation, so a stereo emit only corrupts
        recordings.  Averaging here, in float and before the int16 scale,
        is the same downmix one step earlier.
        """
        import torch
        import torchaudio.functional as AF

        rate = int(sample_rate or NATIVE_SAMPLE_RATE)
        if rate != OUTPUT_SAMPLE_RATE:
            waveform = AF.resample(waveform, rate, OUTPUT_SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        want = round(frames / FRAME_RATE * OUTPUT_SAMPLE_RATE)
        if waveform.shape[-1] > want:
            waveform = waveform[:, :want]
        elif waveform.shape[-1] < want:
            pad = torch.zeros(
                (waveform.shape[0], want - waveform.shape[-1]), dtype=waveform.dtype
            )
            waveform = torch.cat([waveform, pad], dim=-1)
        return (waveform.clamp(-1, 1) * 32767).to(torch.int16).numpy()

    # ------------------------------------------------------------ session

    def reset_session(self) -> None:
        """Reset the cache and session state for a new session.

        Called by the model's ``_reset_session_state``.  The KV cache,
        clean x0, VAE buffer, and audio overlap are all cleared.
        """
        self._reset_cache_state()

    @property
    def prefilled(self) -> bool:
        """Whether the text/conditions have been prefilled for this session."""
        return self._prefilled

    @property
    def chunk_index(self) -> int:
        """The index of the last chunk generated, or -1."""
        return self._chunk_index


__all__ = [
    "OUTPUT_SAMPLE_RATE",
    "ChunkJob",
    "ChunkResult",
    "CausalH3Backend",
]
