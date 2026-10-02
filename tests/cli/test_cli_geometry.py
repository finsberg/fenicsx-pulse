from mpi4py import MPI

import numpy as np
import pytest

from pulse.cli.config import BoxGeometry, ConfigError, LVEllipsoidGeometry
from pulse.cli.geometry import build_geometry, cache_folder, check_markers


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
