from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

from pulse.cli.config import (
    BestelActivationProfile,
    BestelPressureProfile,
    ConfigError,
    ConstantProfile,
    LoadConfig,
    RampProfile,
    TableProfile,
    TimeConfig,
)
from pulse.cli.loads import (
    build_loads,
    build_profile,
    make_load_variables,
    require_optional_packages,
)


def test_constant_profile_is_si():
    f = build_profile(ConstantProfile(value="2 kPa"), 0.0, 1.0, 0.1)
    assert f(0.0) == pytest.approx(2000.0)
    assert f(5.0) == pytest.approx(2000.0)


def test_ramp_holds_outside_its_window():
    f = build_profile(
        RampProfile(start="0.5 s", end="1.5 s", from_value="1 kPa", to_value="3 kPa"),
        0.0,
        2.0,
        0.1,
    )
    assert f(0.0) == pytest.approx(1000.0)
    assert f(0.5) == pytest.approx(1000.0)
    assert f(1.0) == pytest.approx(2000.0)
    assert f(1.5) == pytest.approx(3000.0)
    assert f(9.0) == pytest.approx(3000.0)


def test_inline_table_interpolates_and_holds():
    f = build_profile(TableProfile(times=[0, 1], values=[0, 10], value_unit="kPa"), 0.0, 2.0, 0.1)
    assert f(0.25) == pytest.approx(2500.0)
    assert f(-1.0) == pytest.approx(0.0)
    assert f(3.0) == pytest.approx(10000.0)


def test_table_units_and_period(tmp_path):
    csv = tmp_path / "ta.csv"
    csv.write_text("t_ms,Ta\n0,0\n500,60\n1000,0\n")
    profile = TableProfile(
        file=csv,
        time_column="t_ms",
        value_column="Ta",
        time_unit="ms",
        value_unit="kPa",
        period="1 s",
    )
    f = build_profile(profile, 0.0, 3.0, 0.1)
    assert f(0.25) == pytest.approx(30000.0)
    assert f(1.25) == pytest.approx(30000.0)  # wrapped by the period


def test_table_file_errors_are_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="missing.csv"):
        build_profile(TableProfile(file=tmp_path / "missing.csv"), 0.0, 1.0, 0.1)
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n0,1\n")
    with pytest.raises(ConfigError, match="column"):
        build_profile(TableProfile(file=bad), 0.0, 1.0, 0.1)


def test_load_set_updates_variables():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    loads = [
        LoadConfig(target="pressure", marker="ENDO", profile=ConstantProfile(value="1 kPa")),
        LoadConfig(
            target="activation",
            profile=RampProfile(end="1 s", to_value="10 kPa"),
        ),
    ]
    variables = make_load_variables(loads, mesh)
    assert sorted(variables) == ["activation", "pressure_ENDO"]
    load_set = build_loads(loads, variables, TimeConfig(end_time="1 s", dt="0.1 s"))
    load_set.update(0.5)
    assert float(variables["pressure_ENDO"].value.value) == pytest.approx(1000.0)
    assert float(variables["activation"].value.value) == pytest.approx(5000.0)
    assert load_set.values(0.5) == pytest.approx({"pressure_ENDO": 1000.0, "activation": 5000.0})
    assert load_set.names == ["pressure_ENDO", "activation"]
    assert load_set.pressure_markers == ["ENDO"]


def test_missing_circulation_gives_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("circulation"):
            raise ImportError("No module named 'circulation'", name="circulation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    loads = [LoadConfig(target="activation", profile=BestelActivationProfile())]
    with pytest.raises(ConfigError, match="pip install"):
        require_optional_packages(loads)


def test_bestel_matches_direct_solve_ivp():
    circulation = pytest.importorskip("circulation.bestel")
    solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp
    times = np.arange(0.0, 0.5 + 1e-12, 0.01)
    reference = solve_ivp(
        circulation.BestelPressure(),
        [0.0, 0.5],
        [0.0],
        t_eval=times,
        method="Radau",
        rtol=1e-8,
        atol=1e-6,
    ).y[0]
    f = build_profile(BestelPressureProfile(), 0.0, 0.5, 0.01)
    np.testing.assert_allclose([f(t) for t in times], reference, rtol=1e-4, atol=1.0)


def test_bestel_prefix_independent_of_end_time():
    pytest.importorskip("circulation.bestel")
    pytest.importorskip("scipy.integrate")
    short = build_profile(BestelActivationProfile(), 0.0, 0.4, 0.01)
    long = build_profile(BestelActivationProfile(), 0.0, 1.0, 0.01)
    for t in np.arange(0.0, 0.4, 0.01):
        assert short(t) == long(t)  # bit-identical: needed for restart == continuous
