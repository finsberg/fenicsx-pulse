import logging

from mpi4py import MPI

from cli_helpers import write_cfg

import pulse.cli
from pulse.cli import EXIT_CONFIG, EXIT_OK, EXIT_RUNTIME, _available_templates, main


def test_version_before_and_after_flags(caplog):
    caplog.set_level(logging.INFO)
    assert main(["version"]) == EXIT_OK
    assert main(["-v", "version"]) == EXIT_OK
    assert main(["version", "-v"]) == EXIT_OK


def test_usage_error_is_exit_1():
    assert main(["no-such-command"]) == EXIT_CONFIG
    assert main(["run"]) == EXIT_CONFIG  # missing config argument


def test_init_writes_template(tmp_path):
    target = tmp_path / "case" / "config.toml"
    assert "unit_cube" in _available_templates()
    assert main(["init", str(target), "--template", "unit_cube"]) == EXIT_OK
    assert target.exists()
    assert main(["init", str(target), "--template", "unit_cube"]) == EXIT_CONFIG  # exists
    assert main(["init", str(target), "--template", "unit_cube", "--force"]) == EXIT_OK
    assert main(["init", str(target), "--template", "nope"]) == EXIT_CONFIG


def test_validate_config_and_set(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert main(["validate-config", str(cfg)]) == EXIT_OK
    assert main(["validate-config", str(cfg), "--set", 'time.dt="1 m"']) == EXIT_CONFIG


def test_validate_config_checks_table_files(tmp_path):
    cfg = write_cfg(
        tmp_path,
        active={"type": "active_stress"},
        load=[{"target": "activation", "profile": {"type": "table", "file": "missing.csv"}}],
    )
    assert main(["validate-config", str(cfg)]) == EXIT_CONFIG


def test_env_override(tmp_path, monkeypatch):
    cfg = write_cfg(tmp_path)
    monkeypatch.setenv("PULSE_TIME__DT", '"1 m"')
    assert main(["validate-config", str(cfg)]) == EXIT_CONFIG


def test_geometry_and_run_and_exit_codes(tmp_path, monkeypatch):
    cfg = write_cfg(tmp_path)
    assert main(["geometry", str(cfg)]) == EXIT_OK
    assert main(["run", str(cfg)]) == EXIT_OK
    assert main(["run", str(cfg)]) == EXIT_CONFIG  # needs --overwrite/--restart
    assert main(["run", str(cfg), "--restart", "--set", 'time.end_time="0.4 s"']) == EXIT_OK

    import pulse

    monkeypatch.setattr(pulse.StaticProblem, "solve", lambda self, *a, **k: False)
    assert main(["run", str(cfg), "--overwrite", "--set", "solver.max_halvings=0"]) == EXIT_RUNTIME


def test_install_hint_without_cli_extra(monkeypatch, caplog):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "pydantic_pint" else real(name, *a),
    )
    assert main(["validate-config", "x.toml"]) == EXIT_CONFIG
    assert "fenicsx-pulse[cli]" in caplog.text
    assert main(["version"]) == EXIT_OK


def test_unit_cube_template_runs(tmp_path):
    tmp = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    cfg = tmp / "config.toml"
    assert main(["init", str(cfg), "--template", "unit_cube"]) == EXIT_OK
    assert main(["run", str(cfg)]) == EXIT_OK
    assert pulse.cli is not None
