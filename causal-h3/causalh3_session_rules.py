"""Which commands the session would accept right now.

Kept out of ``causalh3.py`` so the state machine is readable on its own and
can be tested without a GPU.  ``StateUpdate.valid_commands`` carries the
result to clients.
"""

from __future__ import annotations

# Always available: they only ever record a value or report one.
_ALWAYS = ("set_seed", "get_state", "reset")


def valid_commands(
    *,
    generating: bool,
    started: bool,
    family_locked: bool,
) -> list[str]:
    """Name every command the session would accept in this state.

    Args:
        generating: The session is actively generating media chunks.
        started: The session has been started (``start`` was called).
        family_locked: The family has been locked by ``start`` and cannot
            change until ``reset``.
    """
    commands = list(_ALWAYS)
    if not started:
        # Before start: configure the session.
        commands.append("set_family")
        commands.append("set_canvas")
        commands.append("set_prompt")
        commands.append("start")
    else:
        # After start: prompt can change at the next chunk boundary.
        commands.append("set_prompt")
        if generating:
            commands.append("stop")
        else:
            # Generation finished or paused; can restart.
            commands.append("start")
    return sorted(commands)


__all__ = ["valid_commands"]
