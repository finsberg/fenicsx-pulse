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
    import sys

    # Make the cli extra genuinely impossible to import (not just missing from find_spec):
    # pydantic_pint itself, plus pulse.cli.config/pulse.cli.runner which import it (and
    # io4dolfinx) at module level. `None` in sys.modules makes the next `import`/`from ... import`
    # of that name raise ImportError immediately. This proves `version` returns before any of
    # those imports are attempted -- a weaker fake (just patching find_spec) would pass even if
    # `version` still unconditionally ran `from .config import ConfigError` /
    # `from .runner import SolverFailure` ahead of the dispatch, which crashes with an unhandled
    # ImportError instead of EXIT_OK.
    monkeypatch.setitem(sys.modules, "pydantic_pint", None)
    monkeypatch.setitem(sys.modules, "pulse.cli.config", None)
    monkeypatch.setitem(sys.modules, "pulse.cli.runner", None)

    assert main(["version"]) == EXIT_OK
    assert main(["validate-config", "x.toml"]) == EXIT_CONFIG
    assert "fenicsx-pulse[cli]" in caplog.text


def test_unit_cube_template_runs(tmp_path):
    tmp = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    cfg = tmp / "config.toml"
    assert main(["init", str(cfg), "--template", "unit_cube"]) == EXIT_OK
    assert main(["run", str(cfg)]) == EXIT_OK
    assert pulse.cli is not None
