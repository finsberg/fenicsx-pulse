"""[prestress]: unload the imaged mesh once, cache it by hash, deform it and remap the fibres."""

import csv
import json
import shutil

from mpi4py import MPI

import numpy as np
import pytest
from cli_helpers import WINDKESSEL, lv_sections, ode_section, write_cfg, write_file

import pulse
from pulse.cli.geometry import build_geometry
from pulse.cli.overrides import load_config
from pulse.cli.prestress import cavity_volume, prestress_hash
from pulse.cli.runner import LOADS, SolverFailure, build_simulation, run

pytest.importorskip("cardiac_geometries")

TARGET = [{"marker": "ENDO", "pressure": "1 kPa"}]


@pytest.fixture(scope="module")
def lv_folder(tmp_path_factory):
    return MPI.COMM_WORLD.bcast(tmp_path_factory.mktemp("lv_geometry"), root=0)


def _conf(tmp_path, lv_folder, sets=(), **prestress):
    sections = lv_sections(
        lv_folder,
        prestress={"target": TARGET, "ramp_steps": 3, **prestress},
        problem={"u_space": "P_1"},
    )
    return load_config(write_cfg(tmp_path, **sections), sets=list(sets), environ={})


def _entries(conf):
    folder = conf.prestress.cache_folder
    return sorted(p.name for p in folder.iterdir() if not p.name.startswith("."))


def test_prestress_unloads_caches_and_remaps(tmp_path, lv_folder):
    conf = _conf(tmp_path, lv_folder)
    imaged = cavity_volume(conf, build_geometry(conf.geometry), "ENDO")
    sim = build_simulation(conf)
    assert cavity_volume(conf, sim.geo, "ENDO") < imaged
    h = prestress_hash(conf)
    assert _entries(conf) == [h[:16]]
    meta = json.loads((conf.prestress.cache_folder / h[:16] / "meta.json").read_text())
    assert meta["hash"] == h
    assert meta["targets_Pa"] == {"ENDO": pytest.approx(1000.0)}
    assert meta["imaged_volumes_m3"]["ENDO"] == pytest.approx(imaged)
    assert sim.prestress.hash == h
    f0 = sim.geo.f0.x.array.reshape(-1, 3)
    np.testing.assert_allclose(np.linalg.norm(f0, axis=1), 1.0, rtol=1e-10)


def test_cached_result_is_reused_bit_for_bit(tmp_path, lv_folder, monkeypatch):
    conf = _conf(tmp_path, lv_folder)
    first = build_simulation(conf).geo.mesh.geometry.x.copy()

    def recomputed(self):
        raise AssertionError("the cached prestress was recomputed")

    monkeypatch.setattr(pulse.unloading.PrestressProblem, "unload", recomputed)
    second_sim = build_simulation(conf)
    assert np.array_equal(first, second_sim.geo.mesh.geometry.x)
    # the remapped fibres too: a restart rebuilds them from the cached u_pre
    first_f0 = build_simulation(conf).geo.f0.x.array
    assert np.array_equal(first_f0, second_sim.geo.f0.x.array)


def test_new_target_gets_a_new_cache_entry(tmp_path, lv_folder):
    cache = str(tmp_path / "cache")
    a = _conf(tmp_path / "a", lv_folder, cache_folder=cache)
    b = _conf(
        tmp_path / "b",
        lv_folder,
        cache_folder=cache,
        target=[{"marker": "ENDO", "pressure": "1.2 kPa"}],
    )
    build_simulation(a)
    build_simulation(b)
    assert len(_entries(a)) == 2


def test_overwrite_keeps_the_cache(tmp_path, lv_folder):
    conf = _conf(tmp_path, lv_folder)
    run(conf)
    run(conf, overwrite=True)
    assert _entries(conf) == [prestress_hash(conf)[:16]]


def test_restart_recomputes_a_deleted_cache_with_a_warning(tmp_path, lv_folder, caplog):
    conf = _conf(tmp_path, lv_folder)
    run(conf)
    x = build_simulation(conf).geo.mesh.geometry.x.copy()
    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        shutil.rmtree(conf.prestress.cache_folder)
    MPI.COMM_WORLD.barrier()
    longer = _conf(tmp_path, lv_folder, sets=['time.end_time="8 ms"'])
    with caplog.at_level("WARNING"):
        run(longer, restart=True)
    if MPI.COMM_WORLD.rank == 0:
        assert "recomputing" in caplog.text
    again = build_simulation(conf).geo.mesh.geometry.x
    if MPI.COMM_WORLD.size == 1:
        assert np.array_equal(again, x)
    else:  # parallel MUMPS is not bit-reproducible: the prestress was solved again
        scale = MPI.COMM_WORLD.allreduce(float(np.max(np.abs(x))), op=MPI.MAX)
        np.testing.assert_allclose(again, x, rtol=0, atol=1e-12 * scale)


def test_corrupt_cache_meta_is_recomputed_and_replaced(tmp_path, lv_folder):
    conf = _conf(tmp_path, lv_folder)
    build_simulation(conf)
    meta = conf.prestress.cache_folder / prestress_hash(conf)[:16] / "meta.json"
    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        meta.write_bytes(b"\xff\xfe garbage")
    MPI.COMM_WORLD.barrier()
    build_simulation(conf)
    assert json.loads(meta.read_text())["hash"] == prestress_hash(conf)


def test_failed_prestress_caches_nothing(tmp_path, lv_folder, monkeypatch):
    conf = _conf(tmp_path, lv_folder)

    def fails(self):
        raise RuntimeError("no convergence")

    monkeypatch.setattr(pulse.unloading.PrestressProblem, "unload", fails)
    with pytest.raises(SolverFailure):
        build_simulation(conf)
    folder = conf.prestress.cache_folder
    assert not folder.exists() or _entries(conf) == []


def _inflated_conf(tmp_path, lv_folder, end_time="2 ms"):
    pytest.importorskip("gotranx")
    ode = write_file(tmp_path / "circuit.ode", WINDKESSEL.read_text())
    sections = lv_sections(
        lv_folder,
        circulation=ode_section(ode, "monolithic"),
        prestress={"target": TARGET, "ramp_steps": 3, "inflate_steps": 8},
        problem={"u_space": "P_1"},
        time={"end_time": end_time, "dt": "2 ms"},
    )
    return load_config(write_cfg(tmp_path, **sections), environ={})


def test_reinflation_returns_to_the_imaged_volume(tmp_path, lv_folder):
    conf = _inflated_conf(tmp_path, lv_folder)
    imaged = cavity_volume(conf, build_geometry(conf.geometry), "ENDO")
    run(conf)
    with open(conf.output.folder / LOADS) as f:
        first = next(csv.DictReader(f))
    assert float(first["volume_ENDO"]) == pytest.approx(imaged, rel=1e-6)
    assert float(first["circ_V_LV"]) * 1e-6 == pytest.approx(imaged, rel=1e-6)
    # re-inflating the unloaded mesh to the imaged volume takes about the prestress pressure
    assert float(first["pressure_ENDO"]) == pytest.approx(1000.0, rel=0.2), (
        f"measured {first['pressure_ENDO']} Pa, target 1000 Pa"
    )


def test_restart_does_not_reinflate(tmp_path, lv_folder, monkeypatch):
    conf = _inflated_conf(tmp_path, lv_folder)
    run(conf)

    def no_inflation(*args, **kwargs):
        raise AssertionError("a restart re-inflated")

    monkeypatch.setattr("pulse.cli.runner.inflate", no_inflation)
    run(_inflated_conf(tmp_path, lv_folder, end_time="4 ms"), restart=True)


def test_failed_reinflation_is_a_solver_failure(tmp_path, lv_folder, monkeypatch):
    conf = _inflated_conf(tmp_path, lv_folder)
    monkeypatch.setattr(pulse.StaticProblem, "solve", lambda self, *a, **k: False)
    with pytest.raises(SolverFailure, match="Re-inflation failed at 0.12"):
        build_simulation(conf, fresh=True)
