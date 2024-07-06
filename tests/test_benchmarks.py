import math

from mpi4py import MPI

import dolfinx
import fenicsx_pulse
import numpy as np
import pytest
from dolfinx import log

import cardiac_geometries


@pytest.fixture(scope="session")
def geo(tmp_path_factory):
    geodir = tmp_path_factory.mktemp("lv_ellipsoid")

    return cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        r_short_endo=7.0,
        r_short_epi=10.0,
        r_long_endo=17.0,
        r_long_epi=20.0,
        mu_apex_endo=-math.pi,
        mu_base_endo=-math.acos(5 / 17),
        mu_apex_epi=-math.pi,
        mu_base_epi=-math.acos(5 / 20),
        fiber_space="Quadrature_6",
        create_fibers=True,
        fiber_angle_epi=-90,
        fiber_angle_endo=90,
    )


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
        state_space: dolfinx.fem.FunctionSpace,
    ) -> list[dolfinx.fem.bcs.DirichletBC]:
        V, _ = state_space.sub(0).collapse()
        facets = geo.facet_tags.find(left)  # Specify the marker used on the boundary
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        dofs = dolfinx.fem.locate_dofs_topological((state_space.sub(0), V), 2, facets)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs, state_space.sub(0))]

    traction = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=bottom)
    bcs = fenicsx_pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))
    problem = fenicsx_pulse.MechanicsProblemMixed(model=model, geometry=geo, bcs=bcs)

    log.set_log_level(log.LogLevel.INFO)

    for t in [0.0, 0.001, 0.002, 0.003, 0.004]:
        print(f"Solving problem for traction={t}")
        traction.value = t
        problem.solve()

    # Now let us turn off the logging again

    log.set_log_level(log.LogLevel.WARNING)


@pytest.mark.benchmark
def test_problem2(geo):
    geometry = fenicsx_pulse.Geometry.from_cardiac_geometries(
        geo,
        metadata={"quadrature_degree": 4},
    )
    material_params = {
        "C": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(10.0)),
        "bf": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0)),
        "bt": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0)),
        "bfs": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0)),
    }
    material = fenicsx_pulse.Guccione(**material_params)
    active_model = fenicsx_pulse.active_model.Passive()
    comp_model = fenicsx_pulse.Incompressible()

    model = fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    def dirichlet_bc(
        state_space: dolfinx.fem.FunctionSpace,
    ) -> list[dolfinx.fem.bcs.DirichletBC]:
        V, _ = state_space.sub(0).collapse()
        facets = geometry.facet_tags.find(
            geo.markers["BASE"][0],
        )  # Specify the marker used on the boundary
        geometry.mesh.topology.create_connectivity(
            geometry.mesh.topology.dim - 1,
            geometry.mesh.topology.dim,
        )
        dofs = dolfinx.fem.locate_dofs_topological((state_space.sub(0), V), 2, facets)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs, state_space.sub(0))]

    traction = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
    neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=geo.markers["ENDO"][0])
    bcs = fenicsx_pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))
    problem = fenicsx_pulse.MechanicsProblemMixed(model=model, geometry=geometry, bcs=bcs)

    log.set_log_level(log.LogLevel.INFO)

    problem.solve()
    target_value = 10.0
    incr = 1.0

    use_continuation = True

    old_states = [problem.state.copy()]
    old_tractions = [traction.value.copy()]

    while traction.value < target_value:
        value = min(traction.value + incr, target_value)
        print(f"Solving problem for traction={value}")

        if use_continuation and len(old_tractions) > 1:
            d = (value - old_tractions[-2]) / (old_tractions[-1] - old_tractions[-2])
            problem.state.x.array[:] = (1 - d) * old_states[-2].x.array + d * old_states[-1].x.array

        traction.value = value

        try:
            nit, converged = problem.solve()
        except RuntimeError:
            # Reset state and half the increment
            traction.value = old_tractions[-1]
            problem.state.x.array[:] = old_states[-1].x.array
            incr *= 0.5
        else:
            if nit < 3:
                # Increase increment
                incr *= 1.5
            old_states.append(problem.state.copy())
            old_tractions.append(traction.value.copy())

    log.set_log_level(log.LogLevel.INFO)


@pytest.mark.benchmark
def test_problem3(geo):
    geometry = fenicsx_pulse.Geometry.from_cardiac_geometries(
        geo,
        metadata={"quadrature_degree": 6},
    )

    material_params = {
        "C": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2.0)),
        "bf": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(8.0)),
        "bt": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2.0)),
        "bfs": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(4.0)),
    }
    material = fenicsx_pulse.Guccione(f0=geo.f0, s0=geo.s0, n0=geo.n0, **material_params)

    Ta = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
    active_model = fenicsx_pulse.ActiveStress(geo.f0, activation=Ta)
    comp_model = fenicsx_pulse.Incompressible()
    model = fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    def dirichlet_bc(
        state_space: dolfinx.fem.FunctionSpace,
    ) -> list[dolfinx.fem.bcs.DirichletBC]:
        V, _ = state_space.sub(0).collapse()
        facets = geometry.facet_tags.find(
            geo.markers["BASE"][0],
        )  # Specify the marker used on the boundary
        geometry.mesh.topology.create_connectivity(
            geometry.mesh.topology.dim - 1,
            geometry.mesh.topology.dim,
        )
        dofs = dolfinx.fem.locate_dofs_topological((state_space.sub(0), V), 2, facets)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs, state_space.sub(0))]

    traction = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
    neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=geo.markers["ENDO"][0])
    bcs = fenicsx_pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

    problem = fenicsx_pulse.MechanicsProblemMixed(model=model, geometry=geometry, bcs=bcs)

    log.set_log_level(log.LogLevel.INFO)
    problem.solve()

    target_pressure = 15.0
    target_Ta = 60.0
    N = 40

    for Ta_value, traction_value in zip(
        np.linspace(0, target_Ta, N),
        np.linspace(0, target_pressure, N),
    ):
        print(f"Solving problem for traction={traction_value} and active contraction={Ta_value}")
        Ta.value = Ta_value
        traction.value = traction_value
        problem.solve()

    log.set_log_level(log.LogLevel.INFO)
