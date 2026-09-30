"""Wire the builders together, run the time loop, write results and restarts."""

import logging
from typing import Any, Callable

from .config import ConfigError

logger = logging.getLogger(__name__)


class SolverFailure(RuntimeError):
    """The simulation failed at runtime (exit code 2)."""


def _on_rank0(comm, error_type: type[Exception], fn: Callable[[], Any]) -> None:
    """Run ``fn`` on rank 0 only, and make any failure raise on *every* rank.

    Rank-0-only filesystem work followed by a barrier/collective otherwise deadlocks when rank 0
    raises. A :class:`ConfigError` stays a ConfigError; anything else is raised as
    ``error_type`` with the same message (chained to the original on rank 0).
    """
    original: Exception | None = None
    error: tuple[bool, str] | None = None
    if comm.rank == 0:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - re-raised on every rank below
            original = e
            error = (isinstance(e, ConfigError), str(e) or repr(e))
    error = comm.bcast(error, root=0)
    if error is None:
        return
    is_config, message = error
    exc: Exception = ConfigError(message) if is_config else error_type(message)
    raise exc from original
