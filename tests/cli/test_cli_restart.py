import csv
import json

from mpi4py import MPI

import numpy as np
import pytest
from cli_helpers import write_cfg

from pulse.cli.config import ConfigError
from pulse.cli.overrides import load_config
from pulse.cli.runner import LOADS, RESTART_META, RESULTS, build_simulation, read_result_times, run


def _final_u(conf):
    sim = build_simulation(conf)
    sim.restore()
    return sim.problem.u.x.array.copy()


@pytest.mark.parametrize("problem_type", ["static", "dynamic"])
def test_restart_matches_continuous_run(tmp_path, problem_type):
    problem = {"type": problem_type, "u_space": "P_1"}
    full = load_config(
        write_cfg(tmp_path / "a", problem=problem, time={"end_time": "0.6 s", "dt": "0.1 s"}),
        environ={},
    )
    run(full)
    first = load_config(
        write_cfg(tmp_path / "b", problem=problem, time={"end_time": "0.3 s", "dt": "0.1 s"}),
        environ={},
    )
    run(first)
    second = load_config(
        tmp_path / "b" / "config.toml",
        sets=['time.end_time="0.6 s"'],
        environ={},
    )
    run(second, restart=True)
    np.testing.assert_allclose(_final_u(second), _final_u(full), rtol=1e-8, atol=1e-12)
    times = read_result_times(second.output.folder / RESULTS, MPI.COMM_WORLD)
    np.testing.assert_allclose(times, np.arange(7) * 0.1)
    with open(second.output.folder / LOADS) as f:
        t_rows = [float(r["t"]) for r in csv.DictReader(f)]
    np.testing.assert_allclose(t_rows, np.arange(7) * 0.1)  # no duplicated rows


def test_restart_json_is_namespaced(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    run(conf)
    meta = json.loads((conf.output.folder / RESTART_META).read_text())
    assert set(meta) == {"mechanics"}
    assert meta["mechanics"]["step"] == 3
    assert meta["mechanics"]["functions"][0] == "mechanics_u"


def test_restart_rejects_changed_physics(tmp_path):
    cfg = write_cfg(tmp_path)
    run(load_config(cfg, environ={}))
    changed = load_config(cfg, sets=['material.mu="16 kPa"'], environ={})
    with pytest.raises(ConfigError, match="physics"):
        run(changed, restart=True)


def test_restart_without_checkpoint_errors(tmp_path):
    with pytest.raises(ConfigError, match="no restart checkpoint"):
        run(load_config(write_cfg(tmp_path), environ={}), restart=True)


def test_checkpoint_every_then_restart_truncates_loads(tmp_path):
    cfg = write_cfg(tmp_path, output={"checkpoint_every": "0.1 s", "folder": str(tmp_path / "o")})
    conf = load_config(cfg, environ={})
    run(conf)
    # pretend the run died after the checkpoint at t=0.1: point restart.json there
    folder = conf.output.folder
    if MPI.COMM_WORLD.rank == 0:
        meta = json.loads((folder / RESTART_META).read_text())
        meta["mechanics"].update(t=0.1, step=1)
        (folder / RESTART_META).write_text(json.dumps(meta))
    MPI.COMM_WORLD.barrier()
    run(conf, restart=True)
    with open(folder / LOADS) as f:
        t_rows = [float(r["t"]) for r in csv.DictReader(f)]
    np.testing.assert_allclose(t_rows, [0.0, 0.1, 0.2, 0.3])


def test_unknown_marker_is_config_error_before_wipe(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    run(conf)
    for over in (
        {
            "load": [
                {
                    "target": "pressure",
                    "marker": "ENDO",
                    "profile": {"type": "constant", "value": "1 kPa"},
                },
            ],
        },
        {"bcs": {"robin": [{"marker": "EPI", "value": "1 Pa/m"}]}},
        {"bcs": {"base_bc": "fixed"}},
    ):
        bad = load_config(write_cfg(tmp_path, **over), environ={})
        with pytest.raises(ConfigError, match="available"):
            run(bad, overwrite=True)
        assert (conf.output.folder / RESULTS).exists()


def _loads_times(folder):
    with open(folder / LOADS) as f:
        return [float(r["t"]) for r in csv.DictReader(f)]


def test_restart_with_finer_save_every_keeps_loads_in_sync(tmp_path, monkeypatch):
    import pulse
    from pulse.cli.runner import SolverFailure

    output = {"folder": str(tmp_path / "o"), "save_every": "0.2 s", "checkpoint_every": "0.1 s"}
    cfg = write_cfg(
        tmp_path,
        time={"end_time": "0.6 s", "dt": "0.1 s"},
        output=output,
        solver={"max_halvings": 0},
    )
    conf = load_config(cfg, environ={})
    # die in the 4th step: checkpoint at t=0.3, which is not on the save_every = 0.2 s grid
    original = pulse.StaticProblem.solve
    calls = {"n": 0}

    def solve(self, *args, **kwargs):
        calls["n"] += 1
        return calls["n"] < 4 and original(self, *args, **kwargs)

    monkeypatch.setattr(pulse.StaticProblem, "solve", solve)
    with pytest.raises(SolverFailure):
        run(conf)
    monkeypatch.setattr(pulse.StaticProblem, "solve", original)
    finer = load_config(cfg, sets=['output.save_every="0.1 s"'], environ={})
    run(finer, restart=True)
    folder = finer.output.folder
    times = read_result_times(folder / RESULTS, MPI.COMM_WORLD)
    rows = _loads_times(folder)
    assert len(rows) == len(set(rows))  # no duplicated rows
    np.testing.assert_allclose(sorted(rows), rows)
    for t in times:
        assert np.min(np.abs(np.asarray(rows) - t)) < 1e-9, (t, rows)


def test_restart_with_empty_loads_csv_rewrites_header(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    run(conf)
    folder = conf.output.folder
    with open(folder / LOADS) as f:
        header = f.readline()
    if MPI.COMM_WORLD.rank == 0:
        (folder / LOADS).write_text("")
    MPI.COMM_WORLD.barrier()
    longer = load_config(write_cfg(tmp_path), sets=['time.end_time="0.5 s"'], environ={})
    run(longer, restart=True)
    with open(folder / LOADS) as f:
        assert f.readline() == header
    np.testing.assert_allclose(_loads_times(folder), [0.3, 0.4, 0.5])
