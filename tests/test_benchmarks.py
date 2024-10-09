from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
from dolfinx import log

import fenicsx_pulse


@pytest.mark.benchmark
def test_problem1():
    L = 10.0
    W = 1.0
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], [L, W, W]],
        [30, 3, 3],
        dolfinx.mesh.CellType.hexahedron,
    )

    left = 1
    bottom = 2
    boundaries = [
        fenicsx_pulse.Marker(
            name="left",
            marker=left,
            dim=2,
            locator=lambda x: np.isclose(x[0], 0),
        ),
        fenicsx_pulse.Marker(
            name="bottom",
            marker=bottom,
            dim=2,
            locator=lambda x: np.isclose(x[2], 0),
        ),
    ]

    geo = fenicsx_pulse.Geometry(
        mesh=mesh,
        boundaries=boundaries,
        metadata={"quadrature_degree": 4},
    )

    material_params = {
        "C": dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(2.0)),
        "bf": dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(8.0)),
        "bt": dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(2.0)),
        "bfs": dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(4.0)),
    }
    f0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((1.0, 0.0, 0.0)))
    s0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 1.0, 0.0)))
    n0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 0.0, 1.0)))
    material = fenicsx_pulse.Guccione(f0=f0, s0=s0, n0=n0, **material_params)

    active_model = fenicsx_pulse.active_model.Passive()
    comp_model = fenicsx_pulse.Incompressible()

    model = fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    def dirichlet_bc(
        V: dolfinx.fem.FunctionSpace,
    ) -> list[dolfinx.fem.bcs.DirichletBC]:
        facets = geo.facet_tags.find(left)  # Specify the marker used on the boundary
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs)]

    traction = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=bottom)
    bcs = fenicsx_pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))
    problem = fenicsx_pulse.StaticProblem(model=model, geometry=geo, bcs=bcs)

    log.set_log_level(log.LogLevel.INFO)

    for t in [0.0, 0.001, 0.002, 0.003, 0.004]:
        print(f"Solving problem for traction={t}")
        traction.value = t
        problem.solve()

    # Now let us turn off the logging again

    log.set_log_level(log.LogLevel.WARNING)
