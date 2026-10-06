"""Shared data for the CLI tests: a tiny, fast, valid config."""

from pathlib import Path
from typing import Any

from mpi4py import MPI

import toml


def minimal_config_dict(tmp_path, **overrides: Any) -> dict[str, Any]:
    """A 2x2x2 box, compressible neo-Hookean, clamped at X0, pressure ramp on X1, 3 steps."""
    data: dict[str, Any] = {
        "geometry": {"type": "box", "nx": 2, "ny": 2, "nz": 2, "unit": "m"},
        "material": {"type": "neo_hookean", "mu": "15 kPa"},
        "compressibility": {"type": "compressible"},
        "bcs": {"dirichlet": [{"marker": "X0"}]},
        "load": [
            {
                "target": "pressure",
                "marker": "X1",
                "profile": {
                    "type": "ramp",
                    "start": "0 s",
                    "end": "0.3 s",
                    "from_value": "0 kPa",
                    "to_value": "0.3 kPa",
                },
            },
        ],
        "time": {"end_time": "0.3 s", "dt": "0.1 s"},
        "problem": {"u_space": "P_1"},
        "output": {"folder": str(Path(tmp_path) / "output")},
    }
    for key, value in overrides.items():
        base = data.get(key, {})
        if not isinstance(value, dict):
            data[key] = value
        elif (
            isinstance(base, dict)
            and "type" in base
            and "type" in value
            and base["type"] != value["type"]
        ):
            # Switching a discriminated-union section (e.g. material) to another variant:
            # a shallow merge would leak fields from the old variant that the new one forbids.
            data[key] = value
        else:
            data[key] = {**base, **value}
    return data


def write_cfg(tmp_path, **overrides: Any) -> Path:
    return write_file(
        Path(tmp_path) / "config.toml",
        toml.dumps(minimal_config_dict(tmp_path, **overrides)),
    )


def write_file(path, text: str) -> Path:
    """Write ``text`` to ``path`` on rank 0 only (all ranks share it).

    Synchronizes before the write, too: tests overwrite files (configs, CSVs) that another rank
    may still be reading from the previous step.
    """
    path = Path(path)
    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    MPI.COMM_WORLD.barrier()
    return path
