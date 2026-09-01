"""Vendored MiniMax-H3 runtime from ai-toolkit PR #17 head (a472c2e).

This is a pinned subset of the ai-toolkit H3 diffusion model runtime,
vendored to make the causal-h3 serving model self-contained.  The
transformer, causal KV cache, packing geometry, VAEs, text encoder,
pipeline, and artifacts are copies of the ai-toolkit source at commit
a472c2e44937c88cbf34f89ef1d465e2cd0ab71c (PR #17 head), which adds the
``update_cache`` flag to ``MiniMaxH3Transformer.forward()`` that the
persistent causal serving model needs for read-only denoise forwards.

License: ai-toolkit's license terms apply to the vendored code.
See LICENSE and NOTICE in this directory for details.
"""
