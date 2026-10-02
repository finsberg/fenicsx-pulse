import dolfinx
import numpy as np
import pytest

import pulse
from pulse.cli.bcs import build_bcs
from pulse.cli.config import (
    ActiveStressConfig,
    BCsConfig,
    BoxGeometry,
    ConfigError,
    GuccioneMaterial,
    HolzapfelOgdenMaterial,
    NeoHookeanMaterial,
    PassiveConfig,
)
from pulse.cli.geometry import build_geometry
from pulse.cli.model import build_active, build_material


@pytest.fixture(scope="module")
def box():
    return build_geometry(BoxGeometry(nx=2, ny=1, nz=1))


def test_holzapfel_preset_with_override(box):
    material = build_material(
        HolzapfelOgdenMaterial(preset="transversely_isotropic", a_f="3 kPa", b_f=10.0),
        box,
    )
    assert isinstance(material, pulse.HolzapfelOgden)
    assert material.a.to_base_units() == pytest.approx(2280.0)  # from the preset
    assert material.a_f.to_base_units() == pytest.approx(3000.0)
    assert material.b_f.to_base_units() == pytest.approx(10.0)


def test_holzapfel_without_fibres_is_config_error():
    geo = build_geometry(BoxGeometry(nx=1, ny=1, nz=1, fibers={"type": "none"}))
    with pytest.raises(ConfigError, match="fibre"):
        build_material(HolzapfelOgdenMaterial(), geo)
    assert isinstance(build_material(NeoHookeanMaterial(), geo), pulse.NeoHookean)
    iso = GuccioneMaterial(C="10 kPa", bf=1.0, bt=1.0, bfs=1.0)
    assert isinstance(build_material(iso, geo), pulse.Guccione)


def test_region_override_builds_piecewise_constant(box):
    # tag the cells in x > 0.5 with value 7 and use them as a region
    mesh = box.mesh
    tdim = mesh.topology.dim
    n_local = mesh.topology.index_map(tdim).size_local
    cells = np.arange(n_local, dtype=np.int32)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
    values = np.where(midpoints[:, 0] > 0.5, 7, 1).astype(np.int32)
    box.cfun = dolfinx.mesh.meshtags(mesh, tdim, cells, values)
    material = build_material(
        NeoHookeanMaterial(mu="10 kPa", region=[{"marker": "7", "mu": "20 kPa"}]),
        box,
    )
    mu = material.mu.value
    assert isinstance(mu, dolfinx.fem.Function)
    assert sorted(np.unique(mu.x.array * material.mu.factor)) == pytest.approx([1e4, 2e4])
    box.cfun = None
    with pytest.raises(ConfigError, match="cell marker"):
        build_material(NeoHookeanMaterial(region=[{"marker": "7", "mu": "20 kPa"}]), box)


def test_active_models(box):
    ta = pulse.Variable(dolfinx.fem.Constant(box.mesh, 0.0), "Pa")
    assert isinstance(build_active(PassiveConfig(), box, None), pulse.Passive)
    active = build_active(ActiveStressConfig(eta=0.3), box, ta)
    assert isinstance(active, pulse.ActiveStress)
    assert active.activation is ta


def test_bcs(box):
    pressure = pulse.Variable(dolfinx.fem.Constant(box.mesh, 0.0), "Pa")
    conf = BCsConfig(
        dirichlet=[{"marker": "X0"}, {"marker": "Y0", "components": ["y"]}],
        robin=[
            {"marker": "Z1", "value": "1e3 Pa/m"},
            {"marker": "Z1", "value": "5 Pa*s/m", "damping": True},
        ],
    )
    bcs = build_bcs(conf, box, {"X1": pressure})
    assert len(bcs.neumann) == 1 and bcs.neumann[0].traction is pressure
    assert bcs.neumann[0].marker == box.markers["X1"][0]
    assert len(bcs.robin) == 2 and bcs.robin[1].damping
    V = dolfinx.fem.functionspace(box.mesh, ("Lagrange", 1, (3,)))
    assert len(bcs.dirichlet[0](V)) == 1
    assert len(bcs.dirichlet[1](V)) == 1
