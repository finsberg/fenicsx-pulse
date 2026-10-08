"""[circulation] and [prestress]: validation without dolfinx, and the physics hash."""

from pathlib import Path

import pytest
from cli_helpers import write_cfg, write_file

from pulse.cli import TEMPLATES_DIR
from pulse.cli.config import ConfigError, si
from pulse.cli.overrides import load_config, physics_hash

WINDKESSEL = Path(__file__).parents[1] / "data" / "windkessel.ode"
LV = {"type": "lv_ellipsoid", "unit": "m"}
V1_HASHES = {
    "minimal": "e475bb4f3d3846a43f3ce823818e2f541a4f8a2f0fc292ea7609db78ce26051a",
    "bestel_lv": "2eb34118e755001cd5a82958195d9c3479d644b241b7861459db6712e0304f14",
    "ukb_bcs": "2d7d953beea570ba806419a09d159ce370300e6a66c36b1ea90f73e427814738",
    "spatial_material": "55dc0c713912e087a8f4163af71e636e8f059f4a45c48c5afd26028db9f6884c",
}
CAVITY = {
    "marker": "ENDO",
    "period": "0.8 s",
    "t_zero": "50 ms",
    "t_end_diastole": "120 ms",
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
CHAMBER = {"marker": "ENDO", "volume_state": "V_LV", "pressure_missing": "p_LV"}


def _ode_section(ode_file, kind="split", **extra):
    return {
        "type": kind,
        "ode_file": str(ode_file),
        "drop_components": ["timing", "LV"],
        "chamber": [CHAMBER],
        "inputs": {"beat_phase": {"type": "phase", "period": "1 s"}},
        **extra,
    }


def _load(tmp_path, **sections):
    sections.setdefault("geometry", LV)
    sections.setdefault("load", [])
    return load_config(write_cfg(tmp_path, **sections), environ={})


@pytest.fixture
def ode(tmp_path):
    return write_file(tmp_path / "circuit.ode", WINDKESSEL.read_text())


def test_v1_physics_hashes_are_unchanged(tmp_path):
    assert physics_hash(load_config(write_cfg(tmp_path), environ={})) == V1_HASHES["minimal"]
    for name in ("bestel_lv", "ukb_bcs", "spatial_material"):
        conf = load_config(TEMPLATES_DIR / name / "config.toml", environ={})
        assert physics_hash(conf) == V1_HASHES[name], name


def test_defaults(tmp_path):
    conf = _load(tmp_path)
    assert conf.circulation.type == "none"
    assert conf.prestress is None
    assert conf.solver.preconditioner_lag is None


def test_cycle_section_is_si(tmp_path):
    conf = _load(tmp_path, circulation={"type": "cycle", "cavity": [CAVITY]})
    cavity = conf.circulation.cavity[0]
    assert si(cavity.windkessel.compliance) == pytest.approx(1.5e-6 / 133.322387415)
    assert si(cavity.filling_rate) == pytest.approx(0.046e-3)
    assert si(cavity.min_ejection_duration) == pytest.approx(0.01)


def test_cycle_timing_order(tmp_path):
    bad = {**CAVITY, "t_end_diastole": "40 ms"}
    with pytest.raises(ConfigError, match="t_zero <= t_end_diastole"):
        _load(tmp_path, circulation={"type": "cycle", "cavity": [bad]})


def test_ode_file_resolves_relative_to_the_config(tmp_path, ode):
    conf = _load(tmp_path, circulation=_ode_section("circuit.ode"))
    assert conf.circulation.ode_file == (tmp_path / "circuit.ode").resolve()


def test_missing_ode_file(tmp_path):
    with pytest.raises(ConfigError, match="nope.ode"):
        _load(tmp_path, circulation=_ode_section("nope.ode"))


@pytest.mark.parametrize(
    "sections, match",
    [
        ({"geometry": {**LV, "unit": "mm"}}, "geometry.unit"),
        ({"time": {"start_time": "0.1 s", "end_time": "0.3 s", "dt": "0.1 s"}}, "start_time"),
        (
            {
                "load": [
                    {
                        "target": "pressure",
                        "marker": "ENDO",
                        "profile": {"type": "constant", "value": "1 kPa"},
                    },
                ],
            },
            "ENDO",
        ),
        ({"solver": {"preconditioner_lag": 0}}, "preconditioner_lag"),
    ],
)
def test_cycle_cross_checks(tmp_path, sections, match):
    with pytest.raises(ConfigError, match=match):
        _load(tmp_path, circulation={"type": "cycle", "cavity": [CAVITY]}, **sections)


def test_duplicate_cycle_cavities(tmp_path):
    with pytest.raises(ConfigError, match="unique"):
        _load(tmp_path, circulation={"type": "cycle", "cavity": [CAVITY, CAVITY]})


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"chamber": [CHAMBER, CHAMBER]}, "unique"),
        ({"initial_state": {"V_LV": 100.0}}, "V_LV"),
        ({"inputs": {"p_LV": {"type": "phase", "period": "1 s"}}}, "p_LV"),
        ({"record": ["Q_in", "Q_in"]}, "record"),
    ],
)
def test_ode_section_checks(tmp_path, ode, extra, match):
    with pytest.raises(ConfigError, match=match):
        _load(tmp_path, circulation=_ode_section(ode, **extra))


def test_split_needs_a_static_problem(tmp_path, ode):
    with pytest.raises(ConfigError, match="static"):
        _load(tmp_path, circulation=_ode_section(ode), problem={"type": "dynamic"})


def test_preconditioner_lag_is_cycle_only(tmp_path, ode):
    with pytest.raises(ConfigError, match="preconditioner_lag"):
        _load(tmp_path, circulation=_ode_section(ode), solver={"preconditioner_lag": 5})


@pytest.mark.parametrize(
    "circulation, prestress, extra, match",
    [
        ("cycle", {"target": [{"marker": "ENDO", "pressure": "1 kPa"}]}, {}, "p_end_diastole"),
        ("split", {}, {}, "target"),
        (
            "none",
            {"inflate_steps": 3, "target": [{"marker": "ENDO", "pressure": "1 kPa"}]},
            {},
            "inflate_steps",
        ),
        (
            "split",
            {"inflate_steps": 3, "target": [{"marker": "EPI", "pressure": "1 kPa"}]},
            {},
            "ENDO",
        ),
        (
            "split",
            {"target": [{"marker": "ENDO", "pressure": "1 kPa"}]},
            {"problem": {"rigid_body_constraint": True}},
            "rigid_body",
        ),
        (
            "none",
            {"target": [{"marker": "X1", "pressure": "1 kPa"}]},
            {"geometry": {"type": "box"}},
            "box",
        ),
    ],
)
def test_prestress_checks(tmp_path, ode, circulation, prestress, extra, match):
    sections = {
        "none": {},
        "cycle": {"circulation": {"type": "cycle", "cavity": [CAVITY]}},
        "split": {"circulation": _ode_section(ode)},
    }[circulation]
    with pytest.raises(ConfigError, match=match):
        _load(tmp_path, prestress=prestress, **sections, **extra)


def test_prestress_cache_folder_resolves(tmp_path):
    conf = _load(tmp_path, prestress={"target": [{"marker": "ENDO", "pressure": "1 kPa"}]})
    assert conf.prestress.cache_folder == (tmp_path / "prestress").resolve()


def test_physics_hash_follows_ode_contents_not_path(tmp_path, ode):
    a = physics_hash(_load(tmp_path / "a", circulation=_ode_section(ode)))
    other = write_file(tmp_path / "copy.ode", WINDKESSEL.read_text())
    b = physics_hash(_load(tmp_path / "b", circulation=_ode_section(other)))
    assert a == b
    write_file(other, WINDKESSEL.read_text().replace("R_in = 0.05", "R_in = 0.06"))
    c = physics_hash(_load(tmp_path / "c", circulation=_ode_section(other)))
    assert c != a


def test_physics_hash_covers_prestress_but_not_its_cache_or_solver(tmp_path):
    target = [{"marker": "ENDO", "pressure": "1 kPa"}]
    base = physics_hash(_load(tmp_path / "a", prestress={"target": target}))
    moved = physics_hash(
        _load(tmp_path / "b", prestress={"target": target, "cache_folder": "elsewhere"}),
    )
    other = physics_hash(
        _load(tmp_path / "c", prestress={"target": [{"marker": "ENDO", "pressure": "2 kPa"}]}),
    )
    assert base == moved and base != other
