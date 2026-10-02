"""Build a ``pulse.CardiacModel`` from ``[material]``, ``[active]``, ``[compressibility]`` and
``[viscoelasticity]``."""

import logging
from typing import TYPE_CHECKING, Any

from mpi4py import MPI

import dolfinx
import numpy as np
from pint import Quantity

import pulse

from ..units import ureg
from .config import (
    ActiveConfig,
    CompressibilityConfig,
    ConfigError,
    MaterialConfig,
    ViscoelasticityConfig,
    si,
)
from .geometry import CLIGeometry

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

_MATERIAL_CLASSES = {
    "holzapfel_ogden": ("HolzapfelOgden", ("f0", "s0")),
    "guccione": ("Guccione", ("f0", "s0", "n0")),
    "neo_hookean": ("NeoHookean", ()),
    "usyk": ("Usyk", ("f0", "s0", "n0")),
    "saint_venant_kirchhoff": ("SaintVenantKirchhoff", ()),
}


def _base_values(conf: MaterialConfig) -> dict[str, Quantity | float]:
    """Every parameter of the material: a kPa-compatible Quantity or a dimensionless float."""
    values: dict[str, Quantity | float] = {}
    if conf.type == "holzapfel_ogden":
        if conf.preset is not None:
            preset = getattr(pulse.HolzapfelOgden, f"{conf.preset}_parameters")()
            for name, var in preset.items():
                if name in conf.PRESSURE_FIELDS:
                    values[name] = ureg.Quantity(var.value, var.original_unit)
                else:
                    values[name] = float(var.value)
        for name in conf.PRESSURE_FIELDS + conf.FLOAT_FIELDS:
            explicit = getattr(conf, name)
            if explicit is not None:
                values[name] = explicit
            elif name not in values:
                values[name] = ureg.Quantity(0.0, "kPa") if name in conf.PRESSURE_FIELDS else 0.0
        return values
    for name in conf.PRESSURE_FIELDS + conf.FLOAT_FIELDS:
        values[name] = getattr(conf, name)
    return values


def _as_kpa(value: Any) -> float:
    q = value if isinstance(value, Quantity) else ureg.Quantity(value)
    return float(q.to("kPa").magnitude)


def _cell_tag(geo: CLIGeometry, marker: str) -> int:
    tdim = geo.mesh.topology.dim
    if marker in geo.markers and geo.markers[marker][1] == tdim:
        return int(geo.markers[marker][0])
    if marker.lstrip("-").isdigit():
        return int(marker)
    raise ConfigError(
        f"material.region: {marker!r} is not a cell marker; available: "
        f"{sorted(k for k, (_, d) in geo.markers.items() if d == tdim)} or an integer cell tag",
    )


def _region_function(geo: CLIGeometry, conf: MaterialConfig, name: str, base: float) -> Any:
    """A piecewise-constant Function: ``base`` everywhere, region values on their cells."""
    import scifem

    if geo.cfun is None:
        raise ConfigError("material.region needs a geometry with cell markers (cfun)")
    regions = [r for r in conf.region if name in r.values()]
    tags = [_cell_tag(geo, r.marker) for r in regions]
    # The simple-function space needs a patch for every local cell, ghosts included, but cfun may
    # tag only some cells (e.g. only owned ones in parallel): untagged cells keep ``base``.
    cell_map = geo.mesh.topology.index_map(geo.mesh.topology.dim)
    n_owned = cell_map.size_local
    indices = np.arange(n_owned + cell_map.num_ghosts, dtype=np.int32)
    values = np.zeros_like(indices)
    tagged, owned = geo.cfun.indices, geo.cfun.indices < n_owned
    for k, tag in enumerate(tags, start=1):
        values[tagged[geo.cfun.values == tag]] = k
    local_hits = [int(np.sum((geo.cfun.values == tag) & owned)) for tag in tags]
    hits = geo.mesh.comm.allreduce(np.array(local_hits), op=MPI.SUM)
    for region, count in zip(regions, np.atleast_1d(hits)):
        if count == 0:
            raise ConfigError(f"material.region: no cells carry marker {region.marker!r}")
    region_tags = dolfinx.mesh.meshtags(geo.mesh, geo.mesh.topology.dim, indices, values)
    V = scifem.create_space_of_simple_functions(geo.mesh, region_tags, list(range(len(tags) + 1)))
    f = dolfinx.fem.Function(V)
    f.x.array[0] = base
    for k, region in enumerate(regions, start=1):
        value = region.values()[name]
        f.x.array[k] = _as_kpa(value) if name in conf.PRESSURE_FIELDS else float(value)
    return f


def build_material(conf: MaterialConfig, geo: CLIGeometry) -> Any:
    cls_name, fibre_args = _MATERIAL_CLASSES[conf.type]
    kwargs: dict[str, Any] = {}
    for arg in fibre_args:
        kwargs[arg] = getattr(geo, arg)
    if "f0" in fibre_args and geo.f0 is None:
        iso = conf.type == "guccione" and conf.bf == conf.bt == conf.bfs
        if not iso:
            raise ConfigError(
                f"material.type = {conf.type!r} needs a fibre field; set geometry.fibers",
            )
    regional = {name for r in conf.region for name in r.values()}
    for name, value in _base_values(conf).items():
        is_pressure = name in conf.PRESSURE_FIELDS
        base = _as_kpa(value) if is_pressure else float(value)  # type: ignore[arg-type]
        unit = "kPa" if is_pressure else "dimensionless"
        raw = _region_function(geo, conf, name, base) if name in regional else base
        kwargs[name] = pulse.Variable(raw, unit)
    return getattr(pulse, cls_name)(**kwargs)


def build_active(conf: ActiveConfig, geo: CLIGeometry, activation: pulse.Variable | None) -> Any:
    if conf.type == "passive":
        return pulse.Passive()
    if geo.f0 is None:
        raise ConfigError("active.type = 'active_stress' needs a fibre field; set geometry.fibers")
    if activation is None:
        activation = pulse.Variable(
            dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0)),
            "Pa",
        )
    return pulse.ActiveStress(
        geo.f0,
        activation=activation,
        eta=conf.eta,
        formulation=pulse.ActiveStressFormulation(conf.formulation),
    )


def build_compressibility(conf: CompressibilityConfig) -> Any:
    if conf.type == "incompressible":
        return pulse.Incompressible()
    classes = {
        "compressible": pulse.compressibility.Compressible,
        "compressible2": pulse.compressibility.Compressible2,
        "compressible3": pulse.compressibility.Compressible3,
    }
    return classes[conf.type](kappa=pulse.Variable(si(conf.kappa), "Pa"))


def build_viscoelasticity(conf: ViscoelasticityConfig) -> Any:
    if conf.type == "none":
        return pulse.viscoelasticity.NoneViscoElasticity()
    return pulse.viscoelasticity.Viscous(eta=pulse.Variable(si(conf.eta), "Pa*s"))


def build_model(
    conf: "Config",
    geo: CLIGeometry,
    *,
    activation: pulse.Variable | None = None,
    active_model: Any = None,
) -> pulse.CardiacModel:
    """``active_model`` (e.g. simcardemsx's) replaces whatever ``[active]`` would build."""
    active = (
        active_model if active_model is not None else build_active(conf.active, geo, activation)
    )
    return pulse.CardiacModel(
        material=build_material(conf.material, geo),
        active=active,
        compressibility=build_compressibility(conf.compressibility),
        viscoelasticity=build_viscoelasticity(conf.viscoelasticity),
    )
