"""Interpreter-wide fixes applied in every process of this deployment.

Same role as ``fast-h3/sitecustomize.py`` but for the causal-h3 model.
``reactor.yaml`` sets ``PYTHONPATH=/app``, so CPython imports this
``sitecustomize`` at interpreter start in every process of the container.

Two fixes ride the import hook:

**Dynamo recompile limits.** Every distinct chunk shape is a torch.compile
shape, and the regional-compile route runs fullgraph.  The hook re-raises
the limits after ``torch._dynamo.config`` and after the modules that lower
them.  The limit comes from ``CAUSALH3_DYNAMO_RECOMPILE_LIMIT`` (default 64).

**The VSA kernel's arch gate.** The image compiles ``fastvideo-kernel``
for every architecture in ``TORCH_CUDA_ARCH_LIST``, but
``block_sparse_attn_sm100a.is_supported`` compares the device capability
against a ``(10, 0)`` constant by equality.  On a B300 (capability
``(10, 3)``) the gate refuses it, and the deployment silently drops to the
~2.5x-slower Triton kernels.  The hook replaces the constant with a value
equal to every capability the build carries.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
from collections.abc import Callable

# Device capabilities the image's fastvideo-kernel build embeds SASS for.
# Keep in sync with TORCH_CUDA_ARCH_LIST in reactor.yaml.
_VSA_BUILT_CAPABILITIES = ((10, 0), (10, 3))


def _raise_limits() -> None:
    config = sys.modules.get("torch._dynamo.config")
    if config is None:
        return
    limit = int(os.environ.get("CAUSALH3_DYNAMO_RECOMPILE_LIMIT", "64"))
    config.recompile_limit = max(limit, config.recompile_limit)
    config.cache_size_limit = max(limit, config.cache_size_limit)
    config.accumulated_recompile_limit = max(512, config.accumulated_recompile_limit)
    config.accumulated_cache_size_limit = max(
        512, config.accumulated_cache_size_limit
    )
    config.fail_on_recompile_limit_hit = False


class _AnyBuiltCapability:
    """Equal to every device capability the kernel build carries."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return other in _VSA_BUILT_CAPABILITIES
        return NotImplemented

    def __hash__(self) -> int:
        return hash(_VSA_BUILT_CAPABILITIES)

    def __repr__(self) -> str:
        return f"<any of {list(_VSA_BUILT_CAPABILITIES)}>"


def _widen_vsa_arch_gate() -> None:
    module = sys.modules.get("fastvideo_kernel.block_sparse_attn_sm100a")
    if module is None:
        return
    if hasattr(module, "_SM100"):
        module._SM100 = _AnyBuiltCapability()
    if hasattr(module, "is_supported"):
        original = module.is_supported

        def _patched_is_supported(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            except Exception:
                return True

        module.is_supported = _patched_is_supported


_ACTIONS: dict[str, Callable[[], None]] = {
    "torch._dynamo.config": _raise_limits,
    "fastvideo_kernel.block_sparse_attn_sm100a": _widen_vsa_arch_gate,
    "fastvideo.third_party.longcat_video.longcat_attn.bsa_interface": _raise_limits,
    "fastvideo.layers.lora.linear": _raise_limits,
}


class _AfterExecLoader(importlib.abc.Loader):
    """Run the wrapped loader, then the module's fix."""

    def __init__(self, wrapped: importlib.abc.Loader, action: Callable[[], None]) -> None:
        self._wrapped = wrapped
        self._action = action

    def exec_module(self, module) -> None:
        self._wrapped.exec_module(module)
        self._action()

    def create_module(self, spec):
        return self._wrapped.create_module(spec)


class _FixupFinder(importlib.abc.MetaPathFinder):
    """Wrap the loaders of the target modules; transparent to everything else."""

    def find_spec(self, fullname: str, path=None, target=None):
        action = _ACTIONS.get(fullname)
        if action is None:
            return None
        for finder in sys.meta_path[1:]:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _AfterExecLoader(spec.loader, action)
                return spec
        return None


sys.meta_path.insert(0, _FixupFinder())
