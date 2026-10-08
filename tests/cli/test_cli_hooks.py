"""MechanicsSimulation drives a coupling and step hooks; halving and rollback cover both."""

import json

import dolfinx
import numpy as np
import pytest
from cli_helpers import write_cfg, write_file

from pulse.cli.config import ConfigError
from pulse.cli.geometry import build_geometry
from pulse.cli.overrides import load_config
from pulse.cli.runner import RESTART_META, SolverFailure, build_simulation, run
from pulse.coupling import NoCoupling, StepHook


class CountingHook:
    """Counts solve attempts in a Function and converged ones in an int, so rollbacks show."""

    def __init__(self, mesh):
        V = dolfinx.fem.functionspace(mesh, ("DG", 0))
        self.f = dolfinx.fem.Function(V, name="hook_f")
        self.committed = 0
        self.calls: list[tuple[str, float, float]] = []

    def before_solve(self, t, dt):
        self.calls.append(("before", round(t, 12), round(dt, 12)))
        self.f.x.array[:] += 1.0

    def after_solve(self, t, dt):
        self.calls.append(("after", round(t, 12), round(dt, 12)))
        self.committed += 1

    def state_dict(self):
        return {"committed": self.committed}

    def load_state_dict(self, state):
        self.committed = int(state["committed"])

    def restart_functions(self):
        return [("hook_f", self.f)]


class FlakyCoupling(NoCoupling):
    """NoCoupling whose advances fail as `outcomes` says (then succeed), counting commits."""

    def __init__(self, outcomes=()):
        super().__init__()
        self.outcomes = list(outcomes)
        self.commits = 0

    def advance(self, t, dt):
        if self.outcomes and not self.outcomes.pop(0):
            return False
        ok = super().advance(t, dt)
        self.commits += int(ok)
        return ok

    def state_dict(self):
        return {"commits": self.commits}

    def load_state_dict(self, state):
        self.commits = int(state["commits"])


def _sim(tmp_path, coupling, max_halvings=4):
    conf = load_config(write_cfg(tmp_path, solver={"max_halvings": max_halvings}), environ={})
    geo = build_geometry(conf.geometry)
    hook = CountingHook(geo.mesh)
    sim = build_simulation(conf, geometry=geo, coupling=coupling, hooks=[hook])
    sim.start()
    return sim, hook


def test_counting_hook_is_a_step_hook(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    assert isinstance(CountingHook(build_geometry(conf.geometry).mesh), StepHook)


def test_v1_behaviour_without_a_coupling(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    sim = build_simulation(conf)
    assert isinstance(sim.coupling, NoCoupling)
    assert sim.coupling.problem is sim.problem
    sim.start()
    sim.step(0.1)
    assert sim.t == pytest.approx(0.1)


def test_halving_rolls_the_hook_back_before_the_retry(tmp_path):
    coupling = FlakyCoupling([False])
    sim, hook = _sim(tmp_path, coupling)
    sim.step(0.1)
    assert hook.calls == [
        ("before", 0.0, 0.1),
        ("before", 0.0, 0.05),
        ("after", 0.0, 0.05),
        ("before", 0.05, 0.05),
        ("after", 0.05, 0.05),
    ]
    assert hook.committed == 2
    assert np.all(hook.f.x.array == 2.0)  # the failed attempt's +1 was rolled back
    assert coupling.commits == 2
    assert sim.t == pytest.approx(0.1)


def test_failed_step_rolls_back_coupling_hook_and_converged_substeps(tmp_path):
    coupling = FlakyCoupling()
    sim, hook = _sim(tmp_path, coupling, max_halvings=1)
    sim.step(0.1)
    f, committed, commits = hook.f.x.array.copy(), hook.committed, coupling.commits
    u = sim.problem.u.x.array.copy()
    # full step fails, first half converges, second half fails at the deepest level
    coupling.outcomes = [False, True, False]
    with pytest.raises(SolverFailure, match="t=0.2"):
        sim.step(0.1)
    np.testing.assert_array_equal(hook.f.x.array, f)
    assert hook.committed == committed
    assert coupling.commits == commits  # the converged half's commit is gone too
    np.testing.assert_array_equal(sim.problem.u.x.array, u)
    assert sim.t == pytest.approx(0.1) and sim.step_index == 1


def test_checkpoint_restores_coupling_and_hook(tmp_path):
    sim, hook = _sim(tmp_path, FlakyCoupling())
    sim.step(0.1)
    sim.step(0.1)
    sim.checkpoint()
    meta = json.loads((sim.folder / RESTART_META).read_text())["mechanics"]
    assert meta["coupling"] == {"type": "FlakyCoupling", "state": {"commits": 2}}
    assert meta["hooks"] == [{"committed": 2}]
    assert "hook_f" in meta["functions"]

    geo = build_geometry(sim.conf.geometry)
    other_hook, other_coupling = CountingHook(geo.mesh), FlakyCoupling()
    other = build_simulation(sim.conf, geometry=geo, coupling=other_coupling, hooks=[other_hook])
    other.restore()
    assert other_hook.committed == 2 and other_coupling.commits == 2
    np.testing.assert_array_equal(other_hook.f.x.array, hook.f.x.array)
    assert other.t == pytest.approx(0.2)


def test_restore_refuses_another_coupling_type(tmp_path):
    sim, _ = _sim(tmp_path, FlakyCoupling())
    sim.step(0.1)
    sim.checkpoint()
    geo = build_geometry(sim.conf.geometry)
    other = build_simulation(sim.conf, geometry=geo, hooks=[CountingHook(geo.mesh)])
    with pytest.raises(ConfigError, match="FlakyCoupling"):
        other.restore()


def test_v1_checkpoint_restores_into_no_coupling(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    run(conf)
    path = conf.output.folder / RESTART_META
    data = json.loads(path.read_text())
    del data["mechanics"]["coupling"], data["mechanics"]["hooks"]
    write_file(path, json.dumps(data))
    sim = build_simulation(conf)
    sim.restore()
    assert sim.t == pytest.approx(0.3)

    with pytest.raises(ConfigError, match="no coupling state"):
        build_simulation(conf, coupling=FlakyCoupling()).restore()
