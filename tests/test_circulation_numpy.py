"""`GotranxNumpyCirculation`: the numpy twin of `GotranxCirculation`, used by split coupling."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gotranx")

from pulse.circulation import GotranxNumpyCirculation  # noqa: E402

WINDKESSEL = Path(__file__).parent / "data" / "windkessel.ode"
SPLIT = ("timing", "LV")


def test_names():
    model = GotranxNumpyCirculation(ode_file=WINDKESSEL, drop_components=SPLIT)
    assert set(model.state_names) == {"V_LV", "p_AR"}
    assert set(model.missing_names) == {"beat_phase", "p_LV"}
    assert {"Q_in", "Q_out"} <= set(model.monitor_names)


def test_rhs_and_monitor_match_the_hand_written_circuit():
    model = GotranxNumpyCirculation(
        ode_file=WINDKESSEL,
        drop_components=SPLIT,
        parameters={"R_out": 0.4},
    )
    y = model.initial_states_with({"p_AR": 70.0})
    missing = np.zeros(len(model.missing_names))
    missing[model.missing_index("p_LV")] = 90.0
    missing[model.missing_index("beat_phase")] = 0.25
    dy = model.rhs(0.0, y, missing)
    p_ven = 8.0 + 2.0 * 0.25 / 1.0
    q_in = (p_ven - 90.0) / 0.05
    q_out = (90.0 - 70.0) / 0.4
    assert dy[model.state_index("V_LV")] == pytest.approx(q_in - q_out, rel=1e-12)
    assert dy[model.state_index("p_AR")] == pytest.approx((q_out - 70.0 / 1.1) / 1.5, rel=1e-12)
    monitors = model.monitor(0.0, y, missing)
    assert monitors[model.monitor_index("Q_out")] == pytest.approx(q_out, rel=1e-12)


def test_full_model_needs_no_missing_values():
    model = GotranxNumpyCirculation(ode_file=WINDKESSEL)
    assert model.missing_names == ()
    dy = model.rhs(0.5, model.initial_states, np.zeros(0))
    assert np.all(np.isfinite(dy))


def test_unknown_names():
    model = GotranxNumpyCirculation(
        ode_file=WINDKESSEL,
        drop_components=SPLIT,
        parameters={"R_out": 0.4, "E_LV": 0.2, "not_a_parameter": 1.0},
    )
    # E_LV belonged to the dropped LV component
    assert model.ignored_parameters == ("E_LV", "not_a_parameter")
    with pytest.raises(KeyError, match="nope"):
        model.initial_states_with({"nope": 1.0})


def test_matches_regazzoni2020_class(tmp_path):
    """The spike behind the spec's `.ode`-only decision, kept as a test."""
    pytest.importorskip("circulation")
    from circulation import base, regazzoni2020

    HR = 1.0
    flat = regazzoni2020.flat_ode_parameters(
        base.remove_units(regazzoni2020.Regazzoni2020.default_parameters() | {"HR": HR}),
    )
    model = GotranxNumpyCirculation(
        ode_file=regazzoni2020.ODE_FILE,
        parameters=flat,
        drop_components=("timing", "LV", "RV"),
    )
    supplied = {}
    reference = regazzoni2020.Regazzoni2020(
        parameters={"HR": HR},
        add_units=False,
        outdir=tmp_path,
        p_BiV=lambda V_LV, V_RV, t: supplied["p"],
    )
    names = regazzoni2020.Regazzoni2020.state_names()
    order = [model.state_index(n) for n in names]
    default = np.array(
        [v.magnitude for v in regazzoni2020.Regazzoni2020.default_initial_conditions().values()],
    )
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = rng.uniform(0.0, 5.0)
        y = default * rng.uniform(0.5, 1.5, size=12)
        supplied["p"] = (rng.uniform(0, 130), rng.uniform(0, 30))
        y_model = np.zeros(12)
        y_model[order] = y
        missing = np.zeros(3)
        missing[model.missing_index("beat_phase")] = t % (1.0 / HR)
        missing[model.missing_index("p_LV")] = supplied["p"][0]
        missing[model.missing_index("p_RV")] = supplied["p"][1]
        got = model.rhs(t, y_model, missing)[order]
        expected = np.asarray(reference.rhs(t, y))
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)
