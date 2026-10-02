"""Build ``pulse.BoundaryConditions`` from ``[bcs]`` and the pressure loads."""

from typing import Any

import dolfinx

import pulse

from ..units import ureg
from .config import BCsConfig, DirichletConfig
from .geometry import CLIGeometry, get_marker


def _dirichlet(geo: CLIGeometry, conf: DirichletConfig) -> Any:
    marker = get_marker(geo, conf.marker)[0]
    components = ["xyz".index(c) for c in conf.components]
    fdim = geo.mesh.topology.dim - 1

    def bc(V: dolfinx.fem.FunctionSpace) -> list[dolfinx.fem.DirichletBC]:
        facet_tags = geo.geometry.facet_tags
        assert facet_tags is not None, "geometry has no facet_tags"
        facets = facet_tags.find(marker)
        if len(components) == 3:
            dofs = dolfinx.fem.locate_dofs_topological(V, fdim, facets)
            return [dolfinx.fem.dirichletbc(dolfinx.fem.Function(V), dofs)]
        out = []
        for c in components:
            Vc = V.sub(c)
            dofs = dolfinx.fem.locate_dofs_topological(Vc, fdim, facets)
            out.append(dolfinx.fem.dirichletbc(float(dolfinx.default_scalar_type(0.0)), dofs, Vc))
        return out

    return bc


def build_bcs(
    conf: BCsConfig,
    geo: CLIGeometry,
    pressures: dict[str, pulse.Variable],
) -> pulse.BoundaryConditions:
    """``pressures`` maps a facet marker to its pressure load's Variable (the Neumann BC)."""
    neumann = [
        pulse.NeumannBC(traction=variable, marker=get_marker(geo, marker)[0])
        for marker, variable in pressures.items()
    ]
    robin = [
        pulse.RobinBC(
            value=pulse.Variable.from_quantity(ureg.Quantity(r.value)),
            marker=get_marker(geo, r.marker)[0],
            damping=r.damping,
            perpendicular=r.perpendicular,
        )
        for r in conf.robin
    ]
    dirichlet = [_dirichlet(geo, d) for d in conf.dirichlet]
    return pulse.BoundaryConditions(neumann=neumann, dirichlet=dirichlet, robin=robin)
