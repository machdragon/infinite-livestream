"""Interpreter-wide fixes applied in every process of this deployment.

``reactor.yaml`` sets ``PYTHONPATH=/app``, so CPython imports this
``sitecustomize`` at interpreter start in every process of the container —
the runtime and each engine worker FastVideo spawns (never forks —
``force_spawn``) alike. It installs one import hook that runs a fix right
after specific third-party modules execute, which is the only point where
these settings can be corrected: nothing set in the parent carries into a
spawned worker, and the fixes must land *after* the module code that writes
the value being fixed. Two fixes ride the hook:

**Dynamo recompile limits.** Every distinct clip length is a torch.compile
shape, and the regional-compile route runs fullgraph, where exceeding
dynamo's recompile limit is a hard failure (``FailOnRecompileLimitHit``)
that kills the engine workers and the serving process with them. torch 2.12
maps no environment variable onto the limit, and FastVideo's own imports
lower it (``layers/lora/linear.py`` sets ``recompile_limit = 16``,
``third_party/longcat_video/.../bsa_interface.py`` sets
``cache_size_limit = 32``), so the hook re-raises the limits after
``torch._dynamo.config`` and after each of those two modules execute — the
highest setting wins no matter the import order. The limit comes from
``FASTH3_DYNAMO_RECOMPILE_LIMIT`` (default 64 — 14 legal clip lengths times
the warmed canvases, with room to spare).

**The VSA kernel's arch gate.** The image compiles ``fastvideo-kernel``
for every architecture in ``TORCH_CUDA_ARCH_LIST`` (reactor.yaml), but
``block_sparse_attn_sm100a.is_supported`` compares the device capability
against a ``(10, 0)`` constant by equality — written when B200 was the only
Blackwell. On a B300 (capability ``(10, 3)``) the embedded sm_103a binary
is present and correct, yet the gate refuses it, and because the same
predicate decides regional compile, the deployment silently drops to the
~2.5x-slower Triton kernels *in eager mode* (measured 0.74x realtime on
eight B300s against >1x with the fast route). The hook replaces the
constant with a value equal to every capability the build carries.

Nothing here imports torch: interpreters that never touch these modules
(build tooling, small subprocesses) pay only for registering the hook.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
from collections.abc import Callable

# Device capabilities the image's fastvideo-kernel build embeds SASS for —
# the tuple form of TORCH_CUDA_ARCH_LIST in reactor.yaml (10.0a → (10, 0)).
# Keep the two in sync: listing an arch the build lacks would send the
# kernel launches of that device to a missing binary instead of Triton.
_VSA_BUILT_CAPABILITIES = ((10, 0), (10, 3))


def _raise_limits() -> None:
    config = sys.modules.get("torch._dynamo.config")
    if config is None:
        return
    limit = int(os.environ.get("FASTH3_DYNAMO_RECOMPILE_LIMIT", "64"))

    def lift(name: str, floor: int) -> None:
        current = getattr(config, name, None)
        if isinstance(current, int) and current < floor:
            setattr(config, name, floor)

    lift("recompile_limit", limit)
    lift("cache_size_limit", limit)
    # The cross-code-object total; generous so it never binds before the
    # per-object limit does.
    lift("accumulated_recompile_limit", max(512, limit * 8))
    lift("accumulated_cache_size_limit", max(512, limit * 8))
    config.fail_on_recompile_limit_hit = False


class _AnyBuiltCapability:
    """Equal to every device capability the kernel build carries.

    ``is_supported`` tests the device with ``get_device_capability(...) !=
    _SM100``; a plain tuple's ``__ne__`` against this object returns
    ``NotImplemented``, so Python falls through to the reflected operators
    below, and the gate passes exactly the capabilities whose binaries are
    embedded in the extension.
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, tuple) and other in _VSA_BUILT_CAPABILITIES

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return f"<any of {list(_VSA_BUILT_CAPABILITIES)}>"


def _widen_vsa_arch_gate() -> None:
    module = sys.modules.get("fastvideo_kernel.block_sparse_attn_sm100a")
    if module is None:
        return
    # The isinstance keeps the patch idempotent, and stands aside if a
    # future kernel release reshapes the constant.
    if isinstance(getattr(module, "_SM100", None), tuple):
        module._SM100 = _AnyBuiltCapability()


# Modules whose import must be followed by a fix; the hook fires after each.
_ACTIONS: dict[str, Callable[[], None]] = {
    "torch._dynamo.config": _raise_limits,
    "fastvideo.layers.lora.linear": _raise_limits,
    "fastvideo.third_party.longcat_video.block_sparse_attention.bsa_interface": _raise_limits,
    "fastvideo_kernel.block_sparse_attn_sm100a": _widen_vsa_arch_gate,
}


class _AfterExecLoader(importlib.abc.Loader):
    """Run the wrapped loader, then the module's fix."""

    def __init__(self, inner, action: Callable[[], None]) -> None:
        self._inner = inner
        self._action = action

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module) -> None:
        self._inner.exec_module(module)
        self._action()


class _FixupFinder(importlib.abc.MetaPathFinder):
    """Wrap the loaders of the target modules; transparent to everything else."""

    _resolving = False

    def find_spec(self, fullname, path=None, target=None):
        action = _ACTIONS.get(fullname)
        if action is None or _FixupFinder._resolving:
            return None
        # Resolving the real spec re-enters the import system (parent
        # packages get imported); the guard keeps that re-entry out of here.
        _FixupFinder._resolving = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            _FixupFinder._resolving = False
        if fullname in sys.modules:
            # Importing the parents pulled the target in as a side effect;
            # it is already executed, so patch now and stand aside.
            action()
            return None
        if spec is None or spec.loader is None:
            return None
        spec.loader = _AfterExecLoader(spec.loader, action)
        return spec


sys.meta_path.insert(0, _FixupFinder())
