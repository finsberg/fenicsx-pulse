import dolfinx
import fenicsx_pulse
import numpy as np
from petsc4py import PETSc


def test_MechanicsProblem_and_boundary_conditions(mesh):
    boundaries = [
        (1, 2, lambda x: np.isclose(x[0], 0)),
        (2, 2, lambda x: np.isclose(x[0], 1)),
        (3, 2, lambda x: np.isclose(x[1], 0)),
        (4, 2, lambda x: np.isclose(x[1], 1)),
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
        state_space: dolfinx.fem.FunctionSpace,
    ) -> list[dolfinx.fem.bcs.DirichletBC]:
        V, _ = state_space.sub(0).collapse()
        facets = geo.facet_tags.find(1)
        dofs = dolfinx.fem.locate_dofs_topological((state_space.sub(0), V), 2, facets)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs, state_space.sub(0))]

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

    problem = fenicsx_pulse.MechanicsProblem(model=model, geometry=geo, bcs=bcs)
    problem.solve()

    u = problem.state.sub(0).collapse()
    p = problem.state.sub(1).collapse()

    # With the HolzapfelOgden model the hydrostatic pressure
    # should equal the negative of the material parameter a
    assert np.allclose(p.x.array, -material_params["a"])
    # And with no external forces, there should be no displacement
    assert np.allclose(u.x.array, 0.0)

    # Update traction
    traction.value = -1.0
    problem.solve()
    # Now the displacement should be non zero
    assert not np.allclose(problem.state.sub(0).collapse().x.array, 0.0)

    # Put on a similar opposite active stress
    Ta.value = 1.0
    problem.solve()
    # Now the displacement should be almost zero again
    assert np.allclose(problem.state.sub(0).collapse().x.array, 0.0)

    # Put on a body force
    body_force.value[1] = 1.0
    problem.solve()
    # This should also change the displacement
    u_body = problem.state.sub(0).collapse().x.array
    assert not np.allclose(u_body, 0.0)

    # Now add a robin condition
    robin_value.value = 1.0
    problem.solve()
    u_robin = problem.state.sub(0).collapse().x.array
    # This should again change the displacement
    assert not np.allclose(u_body - u_robin, 0.0)
