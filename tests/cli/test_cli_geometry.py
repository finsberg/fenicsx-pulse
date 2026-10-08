from mpi4py import MPI

import numpy as np
import pytest

from pulse.cli import TEMPLATES_DIR
from pulse.cli.config import BoxGeometry, ConfigError, LDRBAngles, LVEllipsoidGeometry, UKBGeometry
from pulse.cli.geometry import (
    _geometry_hash,
    build_geometry,
    cache_folder,
    check_markers,
    ldrb_fibers,
)
from pulse.cli.overrides import load_config


def test_box_markers_fibres_and_scale():
    geo = build_geometry(BoxGeometry(lx=2.0, nx=2, ny=1, nz=1, scale=0.5))
    assert sorted(geo.markers) == ["X0", "X1", "Y0", "Y1", "Z0", "Z1"]
    x = geo.mesh.geometry.x
    xmax = geo.mesh.comm.allreduce(x[:, 0].max() if len(x) else 0.0, op=MPI.MAX)
    assert xmax == pytest.approx(1.0)  # 2.0 * 0.5
    assert np.allclose(geo.f0.value, [1, 0, 0])
    assert np.allclose(geo.s0.value, [0, 1, 0])
    assert np.allclose(geo.n0.value, [0, 0, 1])
    # the X1 facets were tagged at the *scaled* coordinate
    n_x1 = geo.geometry.facet_tags.find(geo.markers["X1"][0]).size
    assert geo.mesh.comm.allreduce(n_x1, op=MPI.SUM) > 0


def test_box_axis_direction_and_no_fibres():
    geo = build_geometry(BoxGeometry(nx=1, ny=1, nz=1, fibers={"type": "axis", "direction": "z"}))
    assert np.allclose(geo.f0.value, [0, 0, 1])
    assert np.allclose(geo.s0.value, [1, 0, 0])
    geo = build_geometry(BoxGeometry(nx=1, ny=1, nz=1, fibers={"type": "none"}))
    assert geo.f0 is None


def test_check_markers_lists_available():
    geo = build_geometry(BoxGeometry(nx=1, ny=1, nz=1))
    check_markers(geo, ["X0"], "bcs")
    with pytest.raises(ConfigError, match=r"\['ENDO'\].*available.*X0"):
        check_markers(geo, ["X0", "ENDO"], "load")


def test_cache_folder_ignores_unit_scale_and_quadrature(tmp_path):
    a = LVEllipsoidGeometry(folder=tmp_path, unit="mm", scale=1.0, quadrature_degree=4)
    b = LVEllipsoidGeometry(folder=tmp_path, unit="m", scale=1e-3, quadrature_degree=6)
    c = LVEllipsoidGeometry(folder=tmp_path, psize_ref=2.0)
    assert cache_folder(a) == cache_folder(b)
    assert cache_folder(a) != cache_folder(c)
    assert cache_folder(a).parent == tmp_path


@pytest.mark.slow
def test_lv_ellipsoid_is_generated_once_and_reused(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    conf = LVEllipsoidGeometry(folder=tmp_path, psize_ref=6.0, unit="mm", fiber_space="P_1")
    geo = build_geometry(conf)
    assert {"ENDO", "EPI", "BASE"} <= set(geo.markers)
    assert geo.f0 is not None
    stamp = (cache_folder(conf) / "pulse_geometry.json").stat().st_mtime
    build_geometry(conf)
    assert (cache_folder(conf) / "pulse_geometry.json").stat().st_mtime == stamp
    assert [p.name for p in tmp_path.iterdir()] == [cache_folder(conf).name]


def test_missing_folder_is_config_error(tmp_path):
    from pulse.cli.config import FolderGeometry

    with pytest.raises(ConfigError, match="does not exist"):
        build_geometry(FolderGeometry(folder=tmp_path / "nope"))


UKB_BCS_GEOMETRY_HASH = "cf99cab62d1dc2f4a1c4d7605f61b7c33583750491bc9510702e22254cce76ef"


def test_ldrb_angles_are_part_of_the_geometry_hash():
    plain = UKBGeometry()
    angled = UKBGeometry(ldrb=LDRBAngles())
    assert _geometry_hash(plain) != _geometry_hash(angled)
    assert "ldrb" not in plain.generator_kwargs()
    conf = load_config(TEMPLATES_DIR / "ukb_bcs" / "config.toml", environ={})
    assert _geometry_hash(conf.geometry) == UKB_BCS_GEOMETRY_HASH  # unchanged without ldrb


@pytest.mark.skip_in_parallel
def test_ldrb_fibers_on_a_biv_ellipsoid_survive_save_and_load(tmp_path):
    pytest.importorskip("ldrb")
    import cardiac_geometries as cg

    g = cg.mesh.biv_ellipsoid(
        outdir=tmp_path / "raw",
        char_length=1.5,
        create_fibers=False,
        comm=MPI.COMM_SELF,
    )
    angled = ldrb_fibers(g, LDRBAngles(), fiber_space="Quadrature_4", clipped=False)
    assert angled.f0 is not None and angled.f0 is not g.f0
    angled.save_folder(folder=tmp_path / "saved")
    back = cg.geometry.Geometry.from_folder(comm=MPI.COMM_SELF, folder=tmp_path / "saved")
    np.testing.assert_allclose(np.sort(back.f0.x.array), np.sort(angled.f0.x.array))
