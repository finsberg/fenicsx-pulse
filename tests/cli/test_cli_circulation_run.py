"""Coupled runs from a config: cycle, split and monolithic on a coarse LV ellipsoid."""

import csv
import json

from mpi4py import MPI

import numpy as np
import pytest
from cli_helpers import (
    CYCLE_CAVITY,
    WINDKESSEL,
    lv_sections,
    ode_section,
    write_cfg,
    write_file,
)

import pulse
from pulse.cli.config import ConfigError
from pulse.cli.overrides import load_config
from pulse.cli.runner import LOADS, RESTART_META, RUN_META, SolverFailure, build_simulation, run

pytest.importorskip("cardiac_geometries")


@pytest.fixture(scope="module")
def lv_folder(tmp_path_factory):
    """One geometry cache for the module: the LV is generated once, then reused by hash."""
    return MPI.COMM_WORLD.bcast(tmp_path_factory.mktemp("lv_geometry"), root=0)


def _style(style, ode):
    if style == "cycle":
        return {
            "circulation": {"type": "cycle", "cavity": [CYCLE_CAVITY]},
            "problem": {"type": "dynamic", "u_space": "P_1"},
            "viscoelasticity": {"type": "viscous"},
        }
    pytest.importorskip("gotranx")
    return {"circulation": ode_section(ode, style, record=["Q_in"]), "problem": {"u_space": "P_1"}}


def _config(tmp_path, lv_folder, style, **overrides):
    ode = write_file(tmp_path / "circuit.ode", WINDKESSEL.read_text())
    sections = lv_sections(lv_folder, **_style(style, ode))
    sections.update(overrides)
    return load_config(write_cfg(tmp_path, **sections), environ={})


def _rows(folder):
    with open(folder / LOADS) as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("style", ["cycle", "split", "monolithic"])
def test_coupled_run_writes_cavity_columns(tmp_path, lv_folder, style):
    conf = _config(tmp_path, lv_folder, style)
    run(conf)
    rows = _rows(conf.output.folder)
    assert [float(r["t"]) for r in rows] == pytest.approx([0.0, 0.002, 0.004, 0.006])
    assert {"volume_ENDO", "pressure_ENDO"} <= set(rows[0])
    assert all(float(r["volume_ENDO"]) > 0 for r in rows)
    if style == "cycle":
        assert {"phase_ENDO", "Pc_ENDO", "Q_ENDO"} <= set(rows[0])
        # PRELOAD for the steps ending at 2 and 4 ms, then isovolumic contraction
        assert [int(float(r["phase_ENDO"])) for r in rows] == [0, 0, 0, 1]
    else:
        assert {"circ_V_LV", "circ_p_AR", "circ_Q_in"} <= set(rows[0])
        for r in rows:
            assert float(r["volume_ENDO"]) == pytest.approx(float(r["circ_V_LV"]) * 1e-6, rel=1e-12)


def _final_state(conf):
    sim = build_simulation(conf)
    sim.restore()
    return {name: f.x.array.copy() for name, f in sim.restart_functions()}


@pytest.mark.parametrize("style", ["cycle", "split", "monolithic"])
def test_restart_matches_continuous_run(tmp_path, lv_folder, style):
    full = _config(tmp_path / "a", lv_folder, style, time={"end_time": "8 ms", "dt": "2 ms"})
    run(full)
    first = _config(tmp_path / "b", lv_folder, style, time={"end_time": "4 ms", "dt": "2 ms"})
    run(first)
    second = load_config(
        tmp_path / "b" / "config.toml",
        sets=['time.end_time="8 ms"'],
        environ={},
    )
    run(second, restart=True)
    restarted, continuous = _final_state(second), _final_state(full)
    assert list(restarted) == list(continuous)
    for name, values in continuous.items():
        if MPI.COMM_WORLD.size == 1:
            assert np.array_equal(restarted[name], values), name  # bit for bit
        else:
            # Parallel MUMPS is not bitwise reproducible (two identical restarts differ by
            # ~1e-20 in about one run in eight), so under MPI compare to round-off.
            scale = MPI.COMM_WORLD.allreduce(float(np.abs(values).max(initial=0.0)), op=MPI.MAX)
            np.testing.assert_allclose(restarted[name], values, rtol=0, atol=1e-12 * scale)
    meta = [
        json.loads((c.output.folder / RESTART_META).read_text())["mechanics"]["coupling"]
        for c in (full, second)
    ]
    assert meta[0] == meta[1]
    t_rows = [float(r["t"]) for r in _rows(second.output.folder)]
    np.testing.assert_allclose(t_rows, np.arange(5) * 0.002)


@pytest.mark.parametrize(
    "extra, bad",
    [
        ({"parameters": {"R_outt": 1.0}}, "R_outt"),
        ({"initial_state": {"p_ARR": 1.0}}, "p_ARR"),
        ({"record": ["Q_nope"]}, "Q_nope"),
        (
            {"chamber": [{"marker": "ENDO", "volume_state": "V_XX", "pressure_missing": "p_LV"}]},
            "V_XX",
        ),
        ({"inputs": {}}, "beat_phase"),
        ({"drop_components": ["timing", "LV", "nope"]}, "nope"),
    ],
)
def test_ode_name_typos_are_config_errors(tmp_path, lv_folder, extra, bad):
    pytest.importorskip("gotranx")
    ode = write_file(tmp_path / "circuit.ode", WINDKESSEL.read_text())
    sections = lv_sections(lv_folder, circulation=ode_section(ode, **extra))
    conf = load_config(write_cfg(tmp_path, **sections), environ={})
    with pytest.raises(ConfigError, match=bad):
        build_simulation(conf)


def test_unknown_coupled_marker(tmp_path, lv_folder):
    cavity = {**CYCLE_CAVITY, "marker": "NOPE"}
    conf = load_config(
        write_cfg(
            tmp_path, **lv_sections(lv_folder, circulation={"type": "cycle", "cavity": [cavity]})
        ),
        environ={},
    )
    with pytest.raises(ConfigError, match="NOPE"):
        build_simulation(conf)


def test_cycle_step_halves_with_the_dynamic_dt(tmp_path, lv_folder, monkeypatch):
    """A failed cycle step is retried as two halves, each a cycle step of dt/2 with the
    dynamic problem's dt Constant at dt/2; afterwards the Constant is back at dt."""
    conf = _config(tmp_path, lv_folder, "cycle")
    sim = build_simulation(conf)
    sim.start()
    real = sim.coupling.advance
    seen = []

    def flaky(t, dt):
        seen.append((round(t, 12), round(dt, 12), float(sim.dt_constant.value)))
        return False if len(seen) == 1 else real(t, dt)

    monkeypatch.setattr(sim.coupling, "advance", flaky)
    sim.step(0.002)
    assert seen == [(0.0, 0.002, 0.002), (0.0, 0.001, 0.001), (0.001, 0.001, 0.001)]
    assert float(sim.dt_constant.value) == pytest.approx(0.002)
    assert sim.coupling.record()["volume_ENDO"] > 0


def test_split_initial_solve_failure_is_a_solver_failure(tmp_path, lv_folder, monkeypatch):
    conf = _config(tmp_path, lv_folder, "split")
    monkeypatch.setattr(pulse.StaticProblem, "solve", lambda self, *a, **k: False)
    with pytest.raises(SolverFailure, match="t=0"):
        run(conf)
    assert json.loads((conf.output.folder / RUN_META).read_text())["status"] == "failed"
