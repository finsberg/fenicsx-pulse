"""Shared data for the CLI tests: a tiny, fast, valid config."""

from pathlib import Path
from typing import Any

from mpi4py import MPI

import toml

WINDKESSEL = Path(__file__).parents[1] / "data" / "windkessel.ode"

# A coarse LV ellipsoid in metres (the same shape as the library's coupling tests).
LV_GEOMETRY: dict[str, Any] = {
    "type": "lv_ellipsoid",
    "unit": "m",
    "r_short_endo": 0.025,
    "r_short_epi": 0.035,
    "r_long_endo": 0.09,
    "r_long_epi": 0.097,
    "psize_ref": 0.05,
    "fiber_space": "P_1",
    "quadrature_degree": 4,
}

# howto/restart.py's compressed LV cycle: PRELOAD ends after two 2 ms steps.
CYCLE_CAVITY: dict[str, Any] = {
    "marker": "ENDO",
    "period": "0.8 s",
    "t_zero": "2 ms",
    "t_end_diastole": "4 ms",
    "preload_pressure": "500 Pa",
    "p_end_diastole": "1000 Pa",
    "p_fill": "500 Pa",
    "filling_rate": "0.046 mL/ms",
    "windkessel": {
        "p_init": "9 kPa",
        "compliance": "1.5 mL/mmHg",
        "resistance": "1.1 mmHg*s/mL",
        "characteristic_impedance": "0.03 mmHg*s/mL",
    },
}


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


def lv_sections(geometry_folder, **overrides: Any) -> dict[str, Any]:
    """Sections for minimal_config_dict: the coarse LV, HO + active stress, fixed base."""
    sections: dict[str, Any] = {
        "geometry": {**LV_GEOMETRY, "folder": str(geometry_folder)},
        "material": {"type": "holzapfel_ogden"},
        "active": {"type": "active_stress"},
        "bcs": {"base_bc": "fixed", "dirichlet": []},
        "load": [],
        "time": {"end_time": "6 ms", "dt": "2 ms"},
    }
    sections.update(overrides)
    return sections


def ode_section(ode_file, kind: str = "split", **extra: Any) -> dict[str, Any]:
    """[circulation] for the test circuit tests/data/windkessel.ode, its LV on ENDO."""
    return {
        "type": kind,
        "ode_file": str(ode_file),
        "drop_components": ["timing", "LV"],
        "chamber": [{"marker": "ENDO", "volume_state": "V_LV", "pressure_missing": "p_LV"}],
        "inputs": {"beat_phase": {"type": "phase", "period": "1 s"}},
        **extra,
    }
