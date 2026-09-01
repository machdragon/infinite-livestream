"""The low-level H3 engine behind CausalH3: persistent GPU work, nothing client-facing.

The persistent model differs from ``fast-h3`` in one fundamental way: the
KV cache and clean x0 state carry forward across chunks, so the stream is
one continuous generation rather than a sequence of independent clips.

The backend loads the actual H3 model components (transformer, video VAE,
audio VAE, text encoder) directly from the validated model package path.
The denoise loop calls ``MiniMaxH3Transformer.forward()`` (vendored from
ai-toolkit PR #17 head, which provides the ``update_cache`` flag) with a
persistent ``SlidingWindowKVCache``.

Per-chunk generation sequence (five forwards):
  1. Four read-only denoise forwards (update_cache=False) over the sigma
     ladder [999, 749, 500, 250, 0].  Each attends to the cached KV but
     does not write back.
  2. One clean x0 forward at t=0 (update_cache=True) that cache-fills
     the denoised output, so the next chunk attends to it.

Prefill caches text and immutable condition rows only (never zero media).
Subsequent chunk layouts contain only target media rows -- text and
condition rows are in the cache, not duplicated in the packed sequence.

Each chunk starts with fresh noise; the previous chunk's clean x0 is
history attended to through the KV cache, not cloned as the new input.

The backend also handles:
- Incremental VisualVAE decode with a separate bounded left-latent buffer.
- AudioVAE overlap-save for continuous audio without boundary artifacts.
- Stereo 32 kHz internally, downmixed/resampled to Reactor mono 48 kHz output.
- Honest compute_time / throughput / cache metrics on every chunk.

VSA / multi-GPU: if VSA is requested, the backend verifies the
fastvideo-kernel VSA route is importable at load time and fails loudly
if not.  Multi-GPU tensor parallelism is not yet wired; ``num_gpus > 1``
fails at load until implemented, rather than silently running on one GPU.
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
    """The handle to one submitted chunk build: its inputs, outcome, and completion."""

    def __init__(self, fn):
        self.fn = fn
        self.done = threading.Event()
        self.result: Optional[ChunkResult] = None
        self.error: Optional[BaseException] = None
        self.cancelled = False


class ChunkResult:
    """One chunk's built output: decoded frames and wire-ready audio."""

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
    """Build persistent causal H3 chunks on demand, serialised on one worker thread."""

    def __init__(
        self,
        config: CausalH3Config,
        model_path: Path,
        manifest: ModelPackageManifest,
    ) -> None:
        self._config = config
        self._model_path = model_path
        self._manifest = manifest
        self._jobs: queue.Queue[ChunkJob] = queue.Queue()
        self._worker: threading.Thread | None = None

        # Model components -- loaded by load().
        self._transformer: Any = None
        self._video_vae: Any = None
        self._audio_vae: Any = None
        self._text_encoder: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self._device: Any = None
        self._dtype: Any = None

        # Cache state.
        cache_cfg = CacheConfig(
            sink_tokens=int(config.inference.get("cache_sink_tokens", 256)),
            window_tokens=int(config.inference.get("cache_window_tokens", 512)),
            vae_left_latents=int(config.inference.get("vae_left_latents", 5)),
            audio_overlap_latents=int(
                config.inference.get("audio_overlap_latents", 10)
            ),
        )
        self._cache_config = cache_cfg
        self.kv_cache: Any = None
        self.clean_x0 = CleanX0State()
        self.vae_buffer = VisualVAEDecodeBuffer(cache_cfg.vae_left_latents)
        self.audio_overlap = AudioVAEOverlapSave(
            overlap_samples=cache_cfg.audio_overlap_latents
            * NATIVE_SAMPLE_RATE // int(chunk_plan.AUDIO_LATENTS_PER_SECOND)
        )

        # Session state.
        self._prefilled = False
        self._chunk_index = -1
        self._prefill_text_embeds: Any = None
        self._prefill_cond_rows: Any = None
        self._prefill_cond_audio_rows: Any = None
        self._prefill_num_cond: int = 0
        self._prefill_num_cond_audio: int = 0

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        """Load H3 model components and warm the persistent pipeline."""
        self._apply_profile_environment()
        self._validate_profile_dependencies()
        self._validate_runtime_compatibility()
        self._validate_multi_gpu()
        self._raise_dynamo_limits()

        logger.info(
            "loading causal-h3 model components",
            model_path=str(self._model_path),
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

    def _validate_multi_gpu(self) -> None:
        """Fail at load when multi-GPU is requested but not yet wired.

        Multi-GPU tensor parallelism is not yet implemented.  Rather than
        silently running on one GPU, fail loudly so the operator knows.
        """
        runtime = self._config.runtime
        num_gpus = int(runtime.get("num_gpus", 1))
        if num_gpus > 1:
            raise RuntimeError(
                f"Multi-GPU ({num_gpus} GPUs) is not yet wired for causal-h3. "
                "The transformer loads on a single device. Set runtime.num_gpus: 1 "
                "until tensor-parallel inference is implemented."
            )

    def _validate_runtime_compatibility(self) -> None:
        """Fail when VSA is requested without a compatible runtime.

        Dense mode may exist only as an explicit non-realtime diagnostic
        profile, never a silent fallback.
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
        import torch._dynamo.config as dynamo_config
        limit = int(os.environ.get("CAUSALH3_DYNAMO_RECOMPILE_LIMIT", "64"))
        dynamo_config.recompile_limit = max(limit, dynamo_config.recompile_limit)
        dynamo_config.cache_size_limit = max(limit, dynamo_config.cache_size_limit)
        dynamo_config.accumulated_recompile_limit = max(512, dynamo_config.accumulated_recompile_limit)
        dynamo_config.accumulated_cache_size_limit = max(512, dynamo_config.accumulated_cache_size_limit)
        dynamo_config.fail_on_recompile_limit_hit = False

    @staticmethod
    def _preload_native_imports() -> None:
        import numpy  # noqa: F401
        import torch
        import torchaudio.functional as AF
        AF.resample(
            torch.zeros(NATIVE_CHANNELS, NATIVE_SAMPLE_RATE // 10),
            NATIVE_SAMPLE_RATE, OUTPUT_SAMPLE_RATE,
        )

    # --------------------------------------------------------------- profile

    def _apply_profile_environment(self) -> None:
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

    def _validate_profile_dependencies(self) -> None:
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

    # ---------------------------------------------------------------- model

    def _load_model_components(self) -> None:
        """Load transformer, VAEs, and text encoder from the model package."""
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
        from minimax_h3.src.causal import SlidingWindowKVCache

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        self._device = device
        self._dtype = dtype

        # --- transformer (strict load) ---
        transformer_dir = self._model_path / "transformer"
        transformer_sd = load_file(transformer_dir / "model.safetensors")
        params = MiniMaxH3TransformerParams()
        table = transformer_sd.get("adaln_t_table", None)
        if table is not None:
            params.adaln_t_table_size = table.shape[0]
        transformer = MiniMaxH3Transformer(params)
        # strict=True: every weight must match.  VSA gates are validated
        # separately by the model package; if they are separate keys not
        # in the transformer state dict, they are loaded by the VSA
        # integration layer (not yet wired).
        result = transformer.load_state_dict(transformer_sd, assign=True, strict=True)
        transformer.to(device).eval()
        transformer.requires_grad_(False)
        self._transformer = transformer
        del transformer_sd

        # --- video VAE ---
        video_vae_dir = self._model_path / "video_vae"
        video_sd = load_file(video_vae_dir / "model.safetensors")
        video_vae = MiniMaxH3VideoVAE()
        video_vae.load_state_dict(video_sd, strict=True, assign=True)
        video_vae.to(device).eval().requires_grad_(False)
        self._video_vae = video_vae
        del video_sd

        # --- audio VAE (fp32) ---
        audio_vae_dir = self._model_path / "audio_vae"
        audio_sd = load_file(audio_vae_dir / "model.safetensors")
        audio_sd = fold_audio_vae_weight_norm(audio_sd)
        audio_vae = MiniMaxH3AudioVAE()
        audio_vae.load_state_dict(audio_sd, strict=True, assign=True)
        audio_vae.to(torch.float32).to(device).eval().requires_grad_(False)
        self._audio_vae = audio_vae
        del audio_sd

        # --- text encoder ---
        self._load_text_encoder()

        # --- KV cache ---
        num_layers = len(transformer.blocks)
        self.kv_cache = SlidingWindowKVCache(
            window_size=self._cache_config.window_tokens,
            sink_size=self._cache_config.sink_tokens,
        )
        logger.info("causal-h3 model loaded", transformer_layers=num_layers)

    def _load_text_encoder(self) -> None:
        """Load Qwen3-VL text encoder and tokenizer."""
        import torch
        from safetensors.torch import load_file

        te_dir = self._model_path / "text_encoder"
        if not te_dir.is_dir():
            raise FileNotFoundError(f"text_encoder directory missing: {te_dir}")

        from transformers import AutoConfig, AutoTokenizer, AutoProcessor

        ORIGINAL_REPO = "MiniMaxAI/MiniMax-H3"
        TEXT_ENCODER_LAYER = 50

        # Load tokenizer/processor from local files under offline mode,
        # or from the hub when online.
        offline = bool(self._config.runtime.get("offline", False))
        if offline:
            tokenizer = AutoTokenizer.from_pretrained(te_dir / "tokenizer")
            processor = AutoProcessor.from_pretrained(te_dir / "processor")
            config = AutoConfig.from_pretrained(te_dir / "config")
        else:
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
        text_encoder.eval().requires_grad_(False)
        text_encoder.to(self._device)
        self._text_encoder = text_encoder
        self._tokenizer = tokenizer
        self._processor = processor

    # ---------------------------------------------------------------- worker

    def _worker_loop(self) -> None:
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

    def submit_chunk(self, *, chunk_index, prompt, seed, height, width, frames) -> ChunkJob:
        job = ChunkJob(
            lambda: self._generate_chunk(
                chunk_index=chunk_index, prompt=prompt, seed=seed,
                height=height, width=width, frames=frames,
            )
        )
        self._jobs.put(job)
        return job

    def submit_prefill(self, *, prompt, seed, height, width,
                       first_frame=None, last_frame=None,
                       references=None) -> ChunkJob:
        job = ChunkJob(
            lambda: self._prefill(
                prompt=prompt, seed=seed, height=height, width=width,
                first_frame=first_frame, last_frame=last_frame,
                references=references,
            )
        )
        self._jobs.put(job)
        return job

    def _run_blocking(self, fn) -> None:
        job = ChunkJob(fn)
        self._jobs.put(job)
        while not job.done.wait(timeout=_WORKER_POLL_SECONDS):
            pass
        if job.error is not None:
            raise job.error

    # --------------------------------------------------------------- warm-up

    def _warmup(self) -> None:
        height, width = chunk_plan.canvas_for_choice(self._config.aspect)
        started = time.monotonic()
        self._generate_chunk(
            chunk_index=-1,
            prompt="A slow cinematic shot of sunlight moving across a quiet room.",
            seed=self._config.seed,
            height=height, width=width, frames=chunk_plan.PREFIX_FRAMES,
        )
        self._reset_cache_state()
        logger.info("causal-h3 warmed", seconds=round(time.monotonic() - started, 2))

    def _reset_cache_state(self) -> None:
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
        self._prefill_text_embeds = None
        self._prefill_cond_rows = None
        self._prefill_cond_audio_rows = None
        self._prefill_num_cond = 0
        self._prefill_num_cond_audio = 0

    # ------------------------------------------------------------ prefill

    def _prefill(self, *, prompt, seed, height, width,
                 first_frame=None, last_frame=None, references=None) -> ChunkResult:
        """Encode text and conditions into the KV cache sink, once.

        Only text and condition rows are prefilled -- never zero media.
        The packed layout for prefill contains text + condition rows only;
        no target video/audio rows are included.  The transformer forward
        with update_cache=True populates the cache sink.

        Subsequent chunk forwards contain only target media rows; text and
        condition rows are attended to through the cached prefix, not
        duplicated in the packed sequence.
        """
        import torch
        from minimax_h3.src import packing
        from minimax_h3.src.packing import (
            build_packed_sequence, build_row_timesteps, patchify_video_latents,
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
            if last_frame is not None:
                anchors = ("first", "last") if first_frame is not None else ("last",)
                last_rows = self._encode_condition_image(last_frame, device)
                cond_rows = torch.cat([cond_rows, last_rows], dim=1) if cond_rows is not None else last_rows
        elif self._config.family == "ref2va" and references:
            parts = []
            audio_parts = []
            for r in references:
                if isinstance(r, dict):
                    parts.append(self._noise_aug_rows(r["latent"][None], device))
                    if r.get("audio_rows") is not None:
                        audio_parts.append(r["audio_rows"][None].to(device))
                    ref_blocks = ref_blocks + ((r["latent"].shape[1], r["latent"].shape[2], r["latent"].shape[3]),)
                elif isinstance(r, torch.Tensor):
                    parts.append(self._noise_aug_rows(r[None], device))
                    ref_blocks = ref_blocks + ((r.shape[1], r.shape[2], r.shape[3]),)
                else:
                    parts.append(self._encode_condition_image(r, device))
                    ref_blocks = ref_blocks + ((1, r.size[1] // 16, r.size[0] // 16),)
            cond_rows = torch.cat(parts, dim=1)
            if audio_parts:
                cond_audio_rows = torch.cat(audio_parts, dim=1).float()

        # --- build prefill layout: text + conditions ONLY, no media ---
        # Use zero latent frames so the layout contains only text and
        # condition rows.  The transformer forward populates the cache
        # with these rows; no target media is cached.
        layout = build_packed_sequence(
            text_token_tags=text_token_tags,
            num_latent_frames=0,
            latent_height=h_lat,
            latent_width=w_lat,
            num_audio_latents=0,
            keyframe_anchors=anchors,
            ref_blocks=ref_blocks,
        )

        num_cond = layout.num_condition_video_rows
        num_cond_audio = layout.num_condition_audio_rows

        # Build the packed input: only text + condition rows.
        # The layout has zero target video/audio rows.
        rows_per_frame = (h_lat // 2) * (w_lat // 2)
        # video_indices includes condition rows only (no target).
        # Create the video input from condition rows.
        video_in = cond_rows if cond_rows is not None else torch.zeros(1, 0, 96, device=device, dtype=dtype)
        audio_in = cond_audio_rows if cond_audio_rows is not None else torch.zeros(1, 0, 32, device=device, dtype=dtype)

        position_ids = layout.position_ids[None].to(device)
        tags = layout.token_tags[None].to(device)
        video_indices = layout.video_indices.to(device)
        audio_indices = layout.audio_indices.to(device)
        text_indices = layout.text_indices.to(device)

        # t=1.0 (clean) for condition rows in prefill.
        row_t = build_row_timesteps(layout, 1.0, 1.0)[None].to(device)

        # Run the transformer with update_cache=True to populate the sink.
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
            num_condition_rows=num_cond + num_cond_audio,
            update_cache=True,
        )

        # Store prefill state for chunk forwards.
        self._prefill_text_embeds = text_embeds
        self._prefill_cond_rows = cond_rows
        self._prefill_cond_audio_rows = cond_audio_rows
        self._prefill_num_cond = num_cond
        self._prefill_num_cond_audio = num_cond_audio

        self._prefilled = True
        elapsed = time.monotonic() - started
        cache_tokens = len(self.kv_cache) if self.kv_cache else 0
        logger.info("causal-h3 prefilled", sink_tokens=cache_tokens, seconds=round(elapsed, 2))
        return ChunkResult(
            chunk_index=-1, frames=[], audio=None,
            compute_time=elapsed, cache_tokens_after=cache_tokens,
        )

    def _encode_text(self, prompt: str):
        """Encode text prompt into conditioning embeddings and token tags."""
        import torch
        inputs = self._tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        with torch.no_grad():
            outputs = self._text_encoder(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            text_embeds = hidden[0].float()
            text_token_tags = torch.ones(text_embeds.shape[0], dtype=torch.long)
        return text_embeds, text_token_tags

    def _encode_condition_image(self, image, device):
        """Encode a condition image into patchified video rows with noise aug."""
        import torch
        import numpy as np
        from minimax_h3.src.packing import patchify_video_latents

        KEYFRAME_NOISE_AUG_T = 0.999
        frame = torch.from_numpy(np.array(image)).float()
        frame = (frame / 255.0) * 2.0 - 1.0
        frame = frame.permute(2, 0, 1)[None, :, None]
        with torch.no_grad():
            cond_latents = self._video_vae.encode(frame.to(self._video_vae.device), sample=False)
        cond_noise = torch.randn(
            (1, 24, 1, cond_latents.shape[3], cond_latents.shape[4]),
            device=device, dtype=torch.float32,
        )
        mixed = KEYFRAME_NOISE_AUG_T * cond_latents.to(device) + (1.0 - KEYFRAME_NOISE_AUG_T) * cond_noise
        return patchify_video_latents(mixed)

    def _noise_aug_rows(self, latents, device):
        import torch
        from minimax_h3.src.packing import patchify_video_latents
        KEYFRAME_NOISE_AUG_T = 0.999
        cond_noise = torch.randn(latents.shape, device=device, dtype=torch.float32)
        mixed = KEYFRAME_NOISE_AUG_T * latents.to(device, torch.float32) + (1.0 - KEYFRAME_NOISE_AUG_T) * cond_noise
        return patchify_video_latents(mixed)

    # ------------------------------------------------------------ generation

    def _generate_chunk(self, *, chunk_index, prompt, seed, height, width, frames) -> ChunkResult:
        """Build one media chunk with persistent clean-KV generation.

        Five forwards:
          1-4. Read-only denoise (update_cache=False) over [999,749,500,250,0].
          5. Clean x0 forward at t=0 (update_cache=True) for cache-fill.

        Each chunk starts with fresh noise.  The previous chunk's clean x0
        is history attended to through the KV cache, not cloned as input.
        """
        started = time.monotonic()

        video_x0, audio_x0 = self._denoise_chunk(
            chunk_index=chunk_index, seed=seed,
            height=height, width=width, frames=frames,
        )

        self.clean_x0.update(video_x0, audio_x0, chunk_index)
        self._chunk_index = chunk_index

        video_frames = self._decode_video(video_x0, height, width, frames)
        native_audio, wire_audio = self._decode_audio(audio_x0, frames)

        elapsed = time.monotonic() - started
        content = len(video_frames) / FRAME_RATE
        throughput = content / elapsed if elapsed > 0 else 0.0
        cache_tokens = len(self.kv_cache) if self.kv_cache else 0
        logger.info(
            f"chunk {chunk_index}: {len(video_frames)}f ({content:.2f}s) "
            f"in {elapsed:.2f}s = {throughput:.2f}x, cache={cache_tokens}"
        )
        return ChunkResult(
            chunk_index=chunk_index, frames=video_frames, audio=wire_audio,
            compute_time=elapsed, cache_tokens_after=cache_tokens,
            native_audio=native_audio,
        )

    def _denoise_chunk(self, *, chunk_index, seed, height, width, frames):
        """Run the four-step denoise plus clean x0 cache-fill.

        Forwards 1-4: read-only denoise with update_cache=False.
        Forward 5: clean x0 at t=0 with update_cache=True (cache-fill).

        Each chunk starts with fresh noise.  The packed layout contains
        only target media rows -- no text or condition rows (they are in
        the cached prefix).
        """
        import torch
        from minimax_h3.src import packing
        from minimax_h3.src.packing import (
            build_packed_sequence, build_row_timesteps, build_sigma_schedule,
            patchify_video_latents, pack_audio_latents, remap_sigma,
        )

        device = self._device
        dtype = self._dtype
        h_lat = height // 16
        w_lat = width // 16

        if frames == chunk_plan.PREFIX_FRAMES:
            t_lat = 2
        else:
            t_lat = chunk_plan._LATENTS_PER_CHUNK

        a_lat = packing.audio_latent_num_frames(frames)

        # Chunk offset: advance the temporal position for each chunk.
        chunk_offset = 0.0
        if chunk_index > 0:
            chunk_offset = float(chunk_index) * packing._temporal_position_span(t_lat)

        # Build layout with NO text/condition rows -- only target media.
        # The text and condition rows are in the cached prefix.
        layout = build_packed_sequence(
            text_token_tags=torch.zeros(0, dtype=torch.long),
            num_latent_frames=t_lat,
            latent_height=h_lat,
            latent_width=w_lat,
            num_audio_latents=a_lat,
            keyframe_anchors=(),
            ref_blocks=(),
            chunk_offset=chunk_offset,
        )

        # Fresh noise for each chunk.
        generator = torch.Generator(device=device).manual_seed(seed)
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

        # Sigma schedule.
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

        num_steps = sigmas_v.shape[0] - 1  # 4 forwards

        # Forwards 1-4: read-only denoise (update_cache=False).
        for i in range(num_steps):
            sv, sv_next = sigmas_v[i], sigmas_v[i + 1]
            sa, sa_next = sigmas_a[i], sigmas_a[i + 1]
            t_v = 1.0 - float(sv)
            t_a = 1.0 - float(sa)
            row_t = build_row_timesteps(layout, t_v, t_a)[None].to(device)

            video_pred, audio_pred = self._transformer(
                hidden_states=video_rows.to(dtype),
                audio_hidden_states=audio_rows.to(dtype),
                encoder_hidden_states=self._prefill_text_embeds[None],
                row_timesteps=row_t,
                token_tags=tags,
                position_ids=position_ids,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
                use_causal_mask=True,
                kv_cache=self.kv_cache,
                num_condition_rows=0,  # no condition rows in chunk layout
                update_cache=False,  # read-only
            )

            v_video = video_pred.float()
            v_audio = audio_pred.float()
            denoised_v = video_rows + sv * v_video
            ratio_v = sv_next / sv
            video_rows = ratio_v * video_rows + (1.0 - ratio_v) * denoised_v
            denoised_a = audio_rows + sa * v_audio
            ratio_a = sa_next / sa if float(sa) != 0.0 else 0.0
            audio_rows = ratio_a * audio_rows + (1.0 - ratio_a) * denoised_a

        # Forward 5: clean x0 at t=0 with update_cache=True (cache-fill).
        # At t=0 (sigma=0), the model sees the fully denoised input and
        # its K/V is appended to the cache for the next chunk to attend to.
        row_t_clean = build_row_timesteps(layout, 0.0, 0.0)[None].to(device)
        self._transformer(
            hidden_states=video_rows.to(dtype),
            audio_hidden_states=audio_rows.to(dtype),
            encoder_hidden_states=self._prefill_text_embeds[None],
            row_timesteps=row_t_clean,
            token_tags=tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            use_causal_mask=True,
            kv_cache=self.kv_cache,
            num_condition_rows=0,
            update_cache=True,  # cache-fill
        )

        return video_rows, audio_rows

    def _decode_video(self, video_x0_rows, height, width, frames) -> list:
        """Incremental VisualVAE decode with bounded left-latent buffer.

        Concatenates prior latent context from the buffer, decodes via
        the real VideoVAE, discards the context frames from the output,
        and updates the buffer with the chunk's latents.

        The VAE's temporal compression is 4x with a 3-frame pre-padding
        per clip.  The exact frame count for a given number of latents
        depends on the VAE's internal chunking (clip_length=17,
        token_drop=3).  Rather than computing the frame count from a
        formula, we use the previous chunk's actual decoded frame count
        (stored in the buffer) to discard the correct number of context
        frames.
        """
        import torch
        from minimax_h3.src.packing import unpatchify_video_tokens

        device = self._video_vae.device
        h_lat = height // 16
        w_lat = width // 16
        t_lat = 2 if frames == chunk_plan.PREFIX_FRAMES else chunk_plan._LATENTS_PER_CHUNK

        video_latents = unpatchify_video_tokens(video_x0_rows, t_lat, h_lat, w_lat)

        # Concatenate left-context latents from the previous chunk.
        left_ctx = self.vae_buffer.context()
        if left_ctx is not None:
            decode_latents = torch.cat([left_ctx.to(device), video_latents.to(device)], dim=2)
        else:
            decode_latents = video_latents.to(device)

        with torch.no_grad():
            decoded = self._video_vae.decode(decode_latents)

        # Discard context frames from the output.  Use the previous
        # chunk's actual decoded frame count, not a formula, because the
        # VAE's internal chunking (clip_length=17, token_drop=3) makes
        # the latent-to-frame mapping non-trivial.
        if left_ctx is not None and self.vae_buffer.prev_frame_count > 0:
            left_frames = self.vae_buffer.prev_frame_count
            decoded = decoded[:, :, left_frames:]

        if decoded.shape[2] > frames:
            decoded = decoded[:, :, :frames]

        video = ((decoded.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
        video = video[0].permute(1, 2, 3, 0).cpu()

        # Update buffer with this chunk's latents and frame count.
        actual_frames = video.shape[0]
        self.vae_buffer.push(video_latents.cpu(), actual_frames)

        return [video[i].numpy() for i in range(video.shape[0])]

    def _decode_audio(self, audio_x0_rows, frames):
        """AudioVAE overlap-save decode: stereo 32k -> wire mono 48k."""
        import torch
        from minimax_h3.src.packing import unpack_audio_tokens

        device = self._audio_vae.device
        a_lat = int(round(frames / FRAME_RATE * chunk_plan.AUDIO_LATENTS_PER_SECOND))

        audio_latents = unpack_audio_tokens(audio_x0_rows, a_lat)

        with torch.no_grad():
            ch0 = self._audio_vae.decode(audio_latents[0, 0:1].to(device))
            ch1 = self._audio_vae.decode(audio_latents[0, 1:2].to(device))
            waveform = torch.cat([ch0, ch1], dim=0)[:, 0]

        waveform = waveform.detach().float().cpu()
        native_audio = waveform.clone()

        # Overlap-save.
        saved = self.audio_overlap.pop()
        if saved is not None:
            overlap_len = min(saved.shape[-1], waveform.shape[-1])
            if overlap_len > 0:
                waveform[:, :overlap_len] += saved[:, :overlap_len]

        overlap_samples = self._cache_config.audio_overlap_latents * NATIVE_SAMPLE_RATE // int(
            chunk_plan.AUDIO_LATENTS_PER_SECOND
        )
        if waveform.shape[-1] > overlap_samples:
            self.audio_overlap.save(waveform[:, -overlap_samples:].clone(), self._chunk_index)

        wire = self._to_wire_audio(waveform, NATIVE_SAMPLE_RATE, frames)
        return native_audio, wire

    @staticmethod
    def _to_wire_audio(waveform, sample_rate, frames):
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
            pad = torch.zeros((waveform.shape[0], want - waveform.shape[-1]), dtype=waveform.dtype)
            waveform = torch.cat([waveform, pad], dim=-1)
        return (waveform.clamp(-1, 1) * 32767).to(torch.int16).numpy()

    # ------------------------------------------------------------ session

    def reset_session(self) -> None:
        self._reset_cache_state()

    @property
    def prefilled(self) -> bool:
        return self._prefilled

    @property
    def chunk_index(self) -> int:
        return self._chunk_index


__all__ = [
    "OUTPUT_SAMPLE_RATE",
    "ChunkJob",
    "ChunkResult",
    "CausalH3Backend",
]
