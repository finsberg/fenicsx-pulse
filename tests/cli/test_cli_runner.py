import csv
import json

from mpi4py import MPI

import numpy as np
import pytest
import toml
from cli_helpers import write_cfg

import pulse
from pulse.cli.config import ConfigError
from pulse.cli.overrides import load_config
from pulse.cli.runner import (
    LOADS,
    RESULTS,
    RUN_META,
    SolverFailure,
    build_simulation,
    read_result_times,
    run,
)


def _rows(folder):
    with open(folder / LOADS) as f:
        return list(csv.DictReader(f))


def test_run_writes_results_and_metadata(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    out = run(conf)
    assert (out / RESULTS).exists()
    assert json.loads((out / RUN_META).read_text())["status"] == "finished"
    assert (out / "config.resolved.toml").exists()
    times = read_result_times(out / RESULTS, MPI.COMM_WORLD)
    np.testing.assert_allclose(times, [0.0, 0.1, 0.2, 0.3])
    rows = _rows(out)
    assert [float(r["t"]) for r in rows] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert float(rows[-1]["pressure_X1"]) == pytest.approx(300.0)


def test_step_api_advances_time_and_state(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    sim = build_simulation(conf)
    assert sim.t == pytest.approx(0.0)
    sim.step(0.1)
    sim.step(0.1)
    assert sim.t == pytest.approx(0.2)
    assert sim.step_index == 2
    norm = sim.geo.mesh.comm.allreduce(float(np.abs(sim.problem.u.x.array).sum()), op=MPI.SUM)
    assert norm > 0


def test_newton_failure_halves_the_step(tmp_path, monkeypatch):
    conf = load_config(write_cfg(tmp_path), environ={})
    sim = build_simulation(conf)
    original = sim.problem.solve
    outcomes = iter([False, True, True])
    calls = []

    def flaky(*args, **kwargs):
        ok = next(outcomes)
        calls.append(float(sim.loads.loads[0].variable.value.value))
        return original(*args, **kwargs) and ok

    monkeypatch.setattr(sim.problem, "solve", flaky)
    sim.step(0.1)
    # first attempt at t=0.1, then the two halves at t=0.05 and t=0.1
    assert calls == pytest.approx([100.0, 50.0, 100.0])
    assert sim.t == pytest.approx(0.1)


def test_failed_halving_leaves_state_and_dt_intact(tmp_path, monkeypatch):
    conf = load_config(
        write_cfg(
            tmp_path,
            problem={"type": "dynamic", "u_space": "P_1"},
            solver={"max_halvings": 2},
        ),
        environ={},
    )
    sim = build_simulation(conf)
    sim.step(0.1)
    u, v, a = (f.x.array.copy() for f in (sim.problem.u, sim.problem.v_old, sim.problem.a_old))

    def always_fail(*args, **kwargs):
        # what a diverged Newton solve does: old states updated first, then garbage in u
        sim.problem.update_old_states()
        sim.problem.u.x.array[:] += 1.0
        return False

    monkeypatch.setattr(sim.problem, "solve", always_fail)
    with pytest.raises(SolverFailure, match="t=0.2"):
        sim.step(0.1)
    np.testing.assert_array_equal(sim.problem.u.x.array, u)
    np.testing.assert_array_equal(sim.problem.v_old.x.array, v)
    np.testing.assert_array_equal(sim.problem.a_old.x.array, a)
    assert float(sim.dt_constant.value) == pytest.approx(0.1)
    assert sim.t == pytest.approx(0.1) and sim.step_index == 1


def test_max_halvings_zero_fails_with_exit_2_semantics(tmp_path, monkeypatch):
    conf = load_config(write_cfg(tmp_path, solver={"max_halvings": 0}), environ={})
    monkeypatch.setattr(pulse.StaticProblem, "solve", lambda self, *a, **k: False)
    with pytest.raises(SolverFailure):
        run(conf)
    assert json.loads((conf.output.folder / RUN_META).read_text())["status"] == "failed"


def test_activation_load_requires_active_model(tmp_path):
    load = {"target": "activation", "profile": {"type": "constant", "value": "1 kPa"}}
    conf = load_config(write_cfg(tmp_path, load=[load]), environ={})
    with pytest.raises(ConfigError, match="passive"):
        build_simulation(conf)
    conf = load_config(
        write_cfg(tmp_path, load=[load], active={"type": "active_stress"}),
        environ={},
    )
    with pytest.raises(ConfigError, match="injected"):
        build_simulation(conf, active_model=pulse.Passive())


def test_performance_summary_is_saved(tmp_path):
    conf = load_config(write_cfg(tmp_path, output={"performance": True}), environ={})
    out = run(conf)
    data = json.loads((out / "performance.json").read_text())
    assert data["total_steps"] == 3
    assert data["newton"]["total_iterations"] >= 3
    assert {"step", "save", "newton_solve", "loads", "volumes"} <= set(data["timings"])


def test_halvings_are_counted(tmp_path, monkeypatch):
    from pulse.telemetry import PerformanceMonitor

    conf = load_config(write_cfg(tmp_path), environ={})
    monitor = PerformanceMonitor()
    sim = build_simulation(conf, monitor=monitor)
    assert sim.problem.monitor is monitor
    original = sim.problem.solve
    outcomes = iter([False, True, True])
    monkeypatch.setattr(sim.problem, "solve", lambda *a, **k: original(*a, **k) and next(outcomes))
    sim.step(0.1)
    assert monitor.counters["halvings"] == 1
    assert monitor.step_counter == 1


def test_dynamic_run_and_cavity_volume_columns(tmp_path):
    conf = load_config(
        write_cfg(tmp_path, problem={"type": "dynamic", "u_space": "P_1"}),
        environ={},
    )
    out = run(conf)
    rows = _rows(out)
    assert len(rows) == 4
    assert not any(k.startswith("volume_") for k in rows[0])  # box: no cavity markers


def test_existing_output_requires_overwrite(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    run(conf)
    with pytest.raises(ConfigError, match="--overwrite"):
        run(conf)
    run(conf, overwrite=True)


def test_overwrite_only_removes_pulse_artifacts(tmp_path):
    cfg = write_cfg(tmp_path, output={"folder": str(tmp_path)})
    conf = load_config(cfg, environ={})
    run(conf)
    if MPI.COMM_WORLD.rank == 0:
        (tmp_path / "notes.txt").write_text("keep me")
    MPI.COMM_WORLD.barrier()
    run(conf, overwrite=True)
    assert (tmp_path / "notes.txt").read_text() == "keep me"
    assert cfg.exists()


def test_invalid_config_never_wipes(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    run(conf)
    bad = load_config(write_cfg(tmp_path, bcs={"dirichlet": [{"marker": "NOPE"}]}), environ={})
    with pytest.raises(ConfigError, match="NOPE"):
        run(bad, overwrite=True)
    assert (conf.output.folder / RESULTS).exists()


def test_failure_after_a_converged_half_rolls_back_to_step_start(tmp_path, monkeypatch):
    conf = load_config(
        write_cfg(
            tmp_path,
            problem={"type": "dynamic", "u_space": "P_1"},
            solver={"max_halvings": 1},
        ),
        environ={},
    )
    sim = build_simulation(conf)
    sim.step(0.1)
    u, v, a = (f.x.array.copy() for f in (sim.problem.u, sim.problem.v_old, sim.problem.a_old))
    original = sim.problem.solve
    # full step fails, first half converges, second half fails at the deepest level
    outcomes = iter([False, True, False])
    monkeypatch.setattr(sim.problem, "solve", lambda *a, **k: original(*a, **k) and next(outcomes))
    with pytest.raises(SolverFailure, match="t=0.2"):
        sim.step(0.1)
    np.testing.assert_array_equal(sim.problem.u.x.array, u)
    np.testing.assert_array_equal(sim.problem.v_old.x.array, v)
    np.testing.assert_array_equal(sim.problem.a_old.x.array, a)
    assert float(sim.loads.loads[0].variable.value.value) == pytest.approx(100.0)
    assert float(sim.dt_constant.value) == pytest.approx(0.1)
    assert sim.t == pytest.approx(0.1) and sim.step_index == 1


def test_unexpected_error_mid_step_rolls_back_and_propagates(tmp_path, monkeypatch):
    conf = load_config(
        write_cfg(
            tmp_path,
            problem={"type": "dynamic", "u_space": "P_1"},
            solver={"max_halvings": 1},
        ),
        environ={},
    )
    sim = build_simulation(conf)
    sim.step(0.1)
    u, v, a = (f.x.array.copy() for f in (sim.problem.u, sim.problem.v_old, sim.problem.a_old))
    original = sim.problem.solve
    calls = iter([False, True])  # full step fails, first half converges, then a PETSc-like error

    def solve(*args, **kwargs):
        outcome = next(calls, None)
        if outcome is None:
            raise RuntimeError("PETSc error")
        return original(*args, **kwargs) and outcome

    monkeypatch.setattr(sim.problem, "solve", solve)
    with pytest.raises(RuntimeError, match="PETSc error") as info:
        sim.step(0.1)
    assert not isinstance(info.value, SolverFailure)
    np.testing.assert_array_equal(sim.problem.u.x.array, u)
    np.testing.assert_array_equal(sim.problem.v_old.x.array, v)
    np.testing.assert_array_equal(sim.problem.a_old.x.array, a)
    assert float(sim.loads.loads[0].variable.value.value) == pytest.approx(100.0)
    assert float(sim.dt_constant.value) == pytest.approx(0.1)
    assert sim.t == pytest.approx(0.1) and sim.step_index == 1


def _doc_snippet(heading: str) -> str:
    from pathlib import Path

    text = (Path(__file__).parents[2] / "docs" / "cli.md").read_text()
    section = text.split(heading, 1)[1]
    return section.split("```python\n", 1)[1].split("```", 1)[0]


def test_step_api_doc_snippet_runs_in_a_fresh_directory(tmp_path, monkeypatch):
    from cli_helpers import write_file

    data = toml.loads(write_cfg(tmp_path / "src").read_text())
    data["output"]["folder"] = "output"
    write_file(tmp_path / "fresh" / "config.toml", toml.dumps(data))
    monkeypatch.chdir(tmp_path / "fresh")
    namespace: dict = {}
    exec(_doc_snippet("## Using pulse from Python"), namespace)
    out = tmp_path / "fresh" / "output"
    np.testing.assert_allclose(read_result_times(out / RESULTS, MPI.COMM_WORLD), [0.1, 0.2, 0.3])
    assert (out / "restart.json").is_file()
    assert isinstance(namespace["sim"].geo.geometry, pulse.HeartGeometry)  # as documented


def test_vertex_tags_are_checked_before_wipe(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    run(conf)
    for marker in ("NOPE", "X0"):  # unknown, and a facet (not a vertex) marker
        bad = load_config(
            write_cfg(tmp_path, postprocess={"vertex_tags": {"apex": marker}}),
            environ={},
        )
        with pytest.raises(ConfigError, match="vertex_tags"):
            run(bad, overwrite=True)
        assert (conf.output.folder / RESULTS).exists()


def test_load_and_bc_markers_must_be_facet_markers(tmp_path):
    from pulse.cli.geometry import build_geometry

    for over in (
        {
            "load": [
                {
                    "target": "pressure",
                    "marker": "CELLS",
                    "profile": {"type": "constant", "value": "1 kPa"},
                },
            ],
        },
        {"bcs": {"dirichlet": [{"marker": "CELLS"}]}},
        {"bcs": {"robin": [{"marker": "CELLS", "value": "1 Pa/m"}]}},
        {"bcs": {"base_bc": "fixed", "base_marker": "CELLS"}},
    ):
        conf = load_config(write_cfg(tmp_path, **over), environ={})
        geo = build_geometry(conf.geometry)
        geo.geometry.markers["CELLS"] = (1, 3)
        with pytest.raises(ConfigError, match=r"CELLS.*dimension 3"):
            build_simulation(conf, geometry=geo)


def test_performance_summary_is_saved_on_failure(tmp_path, monkeypatch):
    conf = load_config(
        write_cfg(tmp_path, output={"performance": True}, solver={"max_halvings": 0}),
        environ={},
    )
    monkeypatch.setattr(pulse.StaticProblem, "solve", lambda self, *a, **k: False)
    with pytest.raises(SolverFailure):
        run(conf)
    assert (conf.output.folder / "performance.json").is_file()


def test_failed_performance_summary_write_does_not_mask_outcome(tmp_path, monkeypatch, caplog):
    from pulse.telemetry import PerformanceMonitor

    def fail(self, path):
        raise OSError("disk full")

    monkeypatch.setattr(PerformanceMonitor, "save_summary", fail)
    conf = load_config(write_cfg(tmp_path, output={"performance": True}), environ={})
    out = run(conf)
    assert json.loads((out / RUN_META).read_text())["status"] == "finished"
    assert "disk full" in caplog.text

    conf = load_config(
        write_cfg(tmp_path, output={"performance": True}, solver={"max_halvings": 0}),
        environ={},
    )
    monkeypatch.setattr(pulse.StaticProblem, "solve", lambda self, *a, **k: False)
    with pytest.raises(SolverFailure, match="(?i)newton|converge|fail"):
        run(conf, overwrite=True)
