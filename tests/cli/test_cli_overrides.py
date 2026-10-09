from pathlib import Path

from mpi4py import MPI

import pytest
import toml
from cli_helpers import minimal_config_dict, write_cfg, write_file

from pulse.cli.config import ConfigError
from pulse.cli.overrides import (
    dump_config,
    env_overrides,
    load_config,
    parse_petsc_options,
    physics_hash,
)


def test_precedence_file_env_set_flag(tmp_path):
    cfg = write_cfg(tmp_path)
    env = {"PULSE_SOLVER__MAX_HALVINGS": "2", "PULSE_OUTPUT__LOG_EVERY": "7"}
    conf = load_config(cfg, sets=["solver.max_halvings=3"], environ=env)
    assert conf.solver.max_halvings == 3  # --set beats env
    assert conf.output.log_every == 7  # env beats file
    conf = load_config(cfg, environ={}, output_folder=tmp_path / "elsewhere")
    assert conf.output.folder == (tmp_path / "elsewhere").resolve()


def test_env_prefix_is_a_parameter():
    assert env_overrides({"SIM_TIME__DT": '"1 s"'}, prefix="SIM_") == [("time.dt", "1 s")]
    assert env_overrides({"PULSE_TIME__DT": '"1 s"'}) == [("time.dt", "1 s")]


def test_keys_are_case_insensitive(tmp_path):
    conf = load_config(write_cfg(tmp_path), sets=["SOLVER.MAX_HALVINGS=1"], environ={})
    assert conf.solver.max_halvings == 1


def test_list_items_can_be_overridden(tmp_path):
    conf = load_config(write_cfg(tmp_path), sets=['load.0.marker="X0"'], environ={})
    assert conf.load[0].marker == "X0"


def test_relative_paths_resolve_against_config_dir(tmp_path):
    csv = write_file(tmp_path / "ta.csv", "time,value\n0,0\n1,5\n")
    cfg = write_cfg(
        tmp_path,
        active={"type": "active_stress"},
        load=[{"target": "activation", "profile": {"type": "table", "file": "ta.csv"}}],
        output={"folder": "out"},
    )
    conf = load_config(cfg, environ={})
    assert conf.load[0].profile.file == csv.resolve()
    assert conf.output.folder == (tmp_path / "out").resolve()
    assert conf.geometry.folder == (tmp_path / "geometry").resolve()


def test_petsc_options_flag_merges(tmp_path):
    conf = load_config(
        write_cfg(tmp_path),
        petsc_options="-ksp_type cg -snes_rtol -1e-8",
        environ={},
    )
    assert conf.solver.petsc_options == {"ksp_type": "cg", "snes_rtol": "-1e-8"}
    assert parse_petsc_options("-snes_monitor") == {"snes_monitor": True}


def test_invalid_config_is_config_error(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.toml", environ={})
    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_config(write_cfg(tmp_path), sets=['time.dt="1 m"'], environ={})


def test_dump_round_trips_and_keeps_the_hash(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    out = tmp_path / "resolved.toml"
    # Every rank shares tmp_path: dump on rank 0 only, between barriers, or another rank can
    # read the file while it is being rewritten (empty -> "geometry: Field required").
    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        dump_config(conf, out)
    MPI.COMM_WORLD.barrier()
    again = load_config(out, environ={})
    assert physics_hash(again) == physics_hash(conf)


def test_physics_hash_ignores_run_length_and_output(tmp_path):
    base = physics_hash(load_config(write_cfg(tmp_path), environ={}))
    longer = load_config(
        write_cfg(tmp_path, time={"end_time": "0.9 s", "dt": "0.1 s"}, output={"folder": "x"}),
        environ={},
    )
    assert physics_hash(longer) == base
    other = load_config(
        write_cfg(tmp_path, material={"type": "neo_hookean", "mu": "16 kPa"}),
        environ={},
    )
    assert physics_hash(other) != base


def test_physics_hash_uses_effective_dt(tmp_path):
    a = load_config(
        write_cfg(tmp_path, time={"end_time": "1 s", "dt": None, "num_steps": 10}),
        environ={},
    )
    b = load_config(
        write_cfg(tmp_path, time={"end_time": "2 s", "dt": None, "num_steps": 10}),
        environ={},
    )
    c = load_config(write_cfg(tmp_path, time={"end_time": "2 s", "dt": "0.1 s"}), environ={})
    assert physics_hash(a) != physics_hash(b)  # same num_steps, different dt
    assert physics_hash(a) == physics_hash(c)  # same dt, different spelling/run length


def test_physics_hash_uses_csv_contents_not_path(tmp_path):
    for sub in ("a", "b"):
        write_file(tmp_path / sub / "ta.csv", "time,value\n0,0\n1,5\n")
    hashes = []
    for sub in ("a", "b"):
        cfg = write_cfg(
            tmp_path / sub,
            active={"type": "active_stress"},
            load=[{"target": "activation", "profile": {"type": "table", "file": "ta.csv"}}],
        )
        hashes.append(physics_hash(load_config(cfg, environ={})))
    assert hashes[0] == hashes[1]
    write_file(tmp_path / "b" / "ta.csv", "time,value\n0,0\n1,6\n")
    cfg = tmp_path / "b" / "config.toml"
    assert physics_hash(load_config(cfg, environ={})) != hashes[0]


def test_geometry_folder_only_hashed_for_folder_type(tmp_path):
    a = load_config(write_cfg(tmp_path, geometry={"folder": "g1"}), environ={})
    b = load_config(write_cfg(tmp_path, geometry={"folder": "g2"}), environ={})
    assert physics_hash(a) == physics_hash(b)
    data = minimal_config_dict(tmp_path)
    data["geometry"] = {"type": "folder", "folder": "g1"}
    path = tmp_path / "f.toml"
    write_file(path, toml.dumps(data))
    h1 = physics_hash(load_config(path, environ={}))
    data["geometry"]["folder"] = "g2"
    write_file(path, toml.dumps(data))
    assert physics_hash(load_config(path, environ={})) != h1


def test_missing_csv_is_config_error_in_hash(tmp_path):
    cfg = write_cfg(
        tmp_path,
        active={"type": "active_stress"},
        load=[{"target": "activation", "profile": {"type": "table", "file": "nope.csv"}}],
    )
    conf = load_config(cfg, environ={})
    with pytest.raises(ConfigError, match="nope.csv"):
        physics_hash(conf)
    assert isinstance(conf.load[0].profile.file, Path)


def test_physics_hash_ignores_solver_section(tmp_path):
    base = physics_hash(load_config(write_cfg(tmp_path), environ={}))
    for sets, petsc in (
        (["solver.max_halvings=8"], None),
        ([], "-ksp_type cg -pc_type hypre"),
    ):
        conf = load_config(write_cfg(tmp_path), sets=sets, petsc_options=petsc, environ={})
        assert physics_hash(conf) == base
    for sets in (['time.dt="0.05 s"'], ['load.0.profile.to_value="0.5 kPa"']):
        conf = load_config(write_cfg(tmp_path), sets=sets, environ={})
        assert physics_hash(conf) != base
