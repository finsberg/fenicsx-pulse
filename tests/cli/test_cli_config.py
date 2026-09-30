import pytest
from cli_helpers import minimal_config_dict
from pydantic import ValidationError

from pulse.cli.config import Config, si


def _config(tmp_path, **over):
    return Config.model_validate(minimal_config_dict(tmp_path, **over))


def test_minimal_config_is_valid_and_defaults_are_validated(tmp_path):
    conf = _config(tmp_path)
    assert conf.geometry.type == "box"
    assert conf.active.type == "passive"
    assert conf.problem.type == "static"
    assert si(conf.problem.rho) == pytest.approx(1000.0)  # default parsed into a Quantity
    assert conf.time.n_steps() == 3
    assert conf.time.dt_s() == pytest.approx(0.1)
    assert conf.output.save_stride(0.1) == 1  # save_every unset -> every step


def test_bare_numbers_are_rejected(tmp_path):
    with pytest.raises(ValidationError, match="mu"):
        _config(tmp_path, material={"type": "neo_hookean", "mu": 15})


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(ValidationError, match="Extra inputs"):
        _config(tmp_path, problem={"u_space": "P_1", "typo": 1})


def test_exactly_one_of_dt_or_num_steps(tmp_path):
    with pytest.raises(ValidationError, match="exactly one of"):
        _config(tmp_path, time={"end_time": "1 s", "dt": "0.1 s", "num_steps": 10})
    conf = _config(tmp_path, time={"end_time": "1 s", "dt": None, "num_steps": 4})
    assert conf.time.dt_s() == pytest.approx(0.25)
    assert conf.time.n_steps() == 4


def test_ramp_needs_end_after_start(tmp_path):
    load = minimal_config_dict(tmp_path)["load"][0]
    load["profile"]["end"] = "0 s"
    with pytest.raises(ValidationError, match="end must be after start"):
        _config(tmp_path, load=[load])


def test_pressure_load_needs_marker_and_activation_forbids_it(tmp_path):
    ramp = minimal_config_dict(tmp_path)["load"][0]["profile"]
    with pytest.raises(ValidationError, match="needs a marker"):
        _config(tmp_path, load=[{"target": "pressure", "profile": ramp}])
    with pytest.raises(ValidationError, match="no marker"):
        _config(tmp_path, load=[{"target": "activation", "marker": "X1", "profile": ramp}])


def test_duplicate_loads_are_rejected(tmp_path):
    load = minimal_config_dict(tmp_path)["load"][0]
    with pytest.raises(ValidationError, match="pressure_X1"):
        _config(tmp_path, load=[load, load])


def test_load_values_must_be_pressures(tmp_path):
    load = minimal_config_dict(tmp_path)["load"][0]
    load["profile"]["to_value"] = "3 m"
    with pytest.raises(ValidationError, match="to_value"):
        _config(tmp_path, load=[load])


def test_table_profile_inline_or_file(tmp_path):
    base = {"target": "activation"}
    ok = {**base, "profile": {"type": "table", "times": [0, 1], "values": [0, 5]}}
    assert _config(tmp_path, active={"type": "active_stress"}, load=[ok])
    both = {**base, "profile": {"type": "table", "times": [0, 1], "values": [0, 5], "file": "a"}}
    with pytest.raises(ValidationError, match="either"):
        _config(tmp_path, load=[both])
    decreasing = {**base, "profile": {"type": "table", "times": [1, 0], "values": [0, 5]}}
    with pytest.raises(ValidationError, match="increasing"):
        _config(tmp_path, load=[decreasing])


def test_bestel_parameters_are_checked(tmp_path):
    good = {
        "type": "bestel_pressure",
        "parameters": {"sigma_pre": "12000 Pa", "t_sys_pre": "0.17 s"},
    }
    assert _config(tmp_path, load=[{"target": "pressure", "marker": "X1", "profile": good}])
    bad_unit = {"type": "bestel_pressure", "parameters": {"sigma_pre": "12 s"}}
    with pytest.raises(ValidationError, match="sigma_pre"):
        _config(tmp_path, load=[{"target": "pressure", "marker": "X1", "profile": bad_unit}])
    unknown = {"type": "bestel_activation", "parameters": {"nope": "1 s"}}
    with pytest.raises(ValidationError, match="nope"):
        _config(tmp_path, load=[{"target": "activation", "profile": unknown}])


def test_stretch_formulation_requires_eta_zero(tmp_path):
    with pytest.raises(ValidationError, match="eta"):
        _config(tmp_path, active={"type": "active_stress", "formulation": "stretch", "eta": 0.3})


def test_robin_units_depend_on_damping(tmp_path):
    ok = {
        "robin": [
            {"marker": "X1", "value": "1e3 Pa/m"},
            {"marker": "X1", "value": "5e3 Pa*s/m", "damping": True},
        ]
    }
    assert _config(tmp_path, bcs=ok)
    with pytest.raises(ValidationError, match="Pa\\*s/m"):
        _config(tmp_path, bcs={"robin": [{"marker": "X1", "value": "1e3 Pa/m", "damping": True}]})


def test_material_region_keys_and_units(tmp_path):
    ho = {"type": "holzapfel_ogden", "region": [{"marker": "10", "a": "22.8 kPa", "b_f": 20.0}]}
    conf = _config(tmp_path, material=ho)
    assert conf.material.region[0].values() == {"a": "22.8 kPa", "b_f": 20.0}
    with pytest.raises(ValidationError, match="unknown"):
        _config(
            tmp_path, material={"type": "holzapfel_ogden", "region": [{"marker": "1", "zz": 1.0}]}
        )
    with pytest.raises(ValidationError, match="pressure"):
        _config(
            tmp_path, material={"type": "holzapfel_ogden", "region": [{"marker": "1", "a": 2.0}]}
        )


def test_save_every_must_not_be_smaller_than_dt(tmp_path):
    with pytest.raises(ValidationError, match="save_every"):
        _config(tmp_path, output={"folder": "o", "save_every": "0.01 s"})


def test_dirichlet_components_unique(tmp_path):
    with pytest.raises(ValidationError, match="unique"):
        _config(tmp_path, bcs={"dirichlet": [{"marker": "X0", "components": ["x", "x"]}]})
