from petsc4py import PETSc

import dolfinx
import numpy as np

import fenicsx_pulse


def test_IncompressibleProblem_and_boundary_conditions(mesh):
    boundaries = [
        ("X0", 1, 2, lambda x: np.isclose(x[0], 0)),
        ("X1", 2, 2, lambda x: np.isclose(x[0], 1)),
        ("Y0", 3, 2, lambda x: np.isclose(x[1], 0)),
        ("Y1", 4, 2, lambda x: np.isclose(x[1], 1)),
    ]
    geo = fenicsx_pulse.Geometry(
        mesh=mesh,
        boundaries=boundaries,
        metadata={"quadrature_degree": 4},
    )

    material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters()
    f0 = dolfinx.fem.Constant(mesh, PETSc.ScalarType((1.0, 0.0, 0.0)))
    s0 = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 1.0, 0.0)))
    material = fenicsx_pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)

    Ta = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))
    active_model = fenicsx_pulse.ActiveStress(f0, activation=Ta)
    comp_model = fenicsx_pulse.Incompressible()

    model = fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    def dirichlet_bc(
        V: dolfinx.fem.FunctionSpace,
    ) -> list[dolfinx.fem.bcs.DirichletBC]:
        facets = geo.facet_tags.find(1)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs)]

    traction = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))
    neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=2)

    robin_value = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))
    robin = fenicsx_pulse.RobinBC(value=robin_value, marker=3)

    body_force = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))

    bcs = fenicsx_pulse.BoundaryConditions(
        dirichlet=(dirichlet_bc,),
        neumann=(neumann,),
        robin=(robin,),
        body_force=(body_force,),
    )

    problem = fenicsx_pulse.StaticProblem(model=model, geometry=geo, bcs=bcs)
    problem.solve()

    u = problem.u
    p = problem.p

    # With the HolzapfelOgden model the hydrostatic pressure
    # should equal the negative of the material parameter a
    assert np.allclose(p.x.array, -material_params["a"].to_base_units())
    # And with no external forces, there should be no displacement
    assert np.allclose(u.x.array, 0.0)

    # Update traction
    traction.value = -1.0
    problem.solve()
    # Now the displacement should be non zero
    assert not np.allclose(problem.u.x.array, 0.0)

    # Put on a similar opposite active stress
    Ta.value = 1.0
    problem.solve()
    # Now the displacement should be almost zero again
    assert np.allclose(problem.u.x.array, 0.0)

    # Put on a body force
    body_force.value[1] = 1.0
    problem.solve()
    # This should also change the displacement
    u_body = problem.u.x.array.copy()
    assert not np.allclose(u_body, 0.0)

    # Now add a robin condition
    robin_value.value = 100.0
    problem.solve()
    u_robin = problem.u.x.array
    # This should again change the displacement
    assert not np.allclose(u_body - u_robin, 0.0)


def test_CompressibleProblem_and_boundary_conditions(mesh):
    boundaries = [
        ("X0", 1, 2, lambda x: np.isclose(x[0], 0)),
        ("X1", 2, 2, lambda x: np.isclose(x[0], 1)),
        ("Y0", 3, 2, lambda x: np.isclose(x[1], 0)),
        ("Y1", 4, 2, lambda x: np.isclose(x[1], 1)),
    ]
    geo = fenicsx_pulse.Geometry(
        mesh=mesh,
        boundaries=boundaries,
        metadata={"quadrature_degree": 4},
    )

    f0 = dolfinx.fem.Constant(mesh, PETSc.ScalarType((1.0, 0.0, 0.0)))
    material = fenicsx_pulse.NeoHookean(mu=dolfinx.fem.Constant(mesh, PETSc.ScalarType(15.0)))

    Ta = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))
    active_model = fenicsx_pulse.ActiveStress(f0, activation=Ta)
    comp_model = fenicsx_pulse.Compressible()

    model = fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
        decouple_deviatoric_volumetric=True,
    )

    def dirichlet_bc(
        state_space: dolfinx.fem.FunctionSpace,
    ) -> list[dolfinx.fem.bcs.DirichletBC]:
        facets = geo.facet_tags.find(1)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        dofs = dolfinx.fem.locate_dofs_topological(state_space, 2, facets)
        u_fixed = dolfinx.fem.Function(state_space)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs)]

    traction = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))
    neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=2)

    robin_value = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))
    robin = fenicsx_pulse.RobinBC(value=robin_value, marker=3)

    body_force = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))

    bcs = fenicsx_pulse.BoundaryConditions(
        dirichlet=(dirichlet_bc,),
        neumann=(neumann,),
        robin=(robin,),
        body_force=(body_force,),
    )

    problem = fenicsx_pulse.StaticProblem(model=model, geometry=geo, bcs=bcs)
    problem.solve()

    # And with no external forces, there should be no displacement
    assert np.allclose(problem.u.x.array, 0.0)

    # Update traction
    traction.value = -1.0
    problem.solve()
    # Now the displacement should be non zero
    assert not np.allclose(problem.u.x.array, 0.0)

    # Put on a similar opposite active stress
    Ta.value = 1.0
    problem.solve()
    # Now the displacement should be almost zero again
    # However for the compressible model this is not really the case
    # so add ad quite high tolerance
    assert np.allclose(problem.u.x.array, 0.0, atol=1e-3)

    # Put on a body force
    body_force.value[1] = 1.0
    problem.solve()
    # This should also change the displacement
    u_body = problem.u.x.array.copy()
    assert not np.allclose(u_body, 0.0)

    # Now add a robin condition
    robin_value.value = 100.0
    problem.solve()
    u_robin = problem.u.x.array
    assert not np.allclose(u_body - u_robin, 0.0)
