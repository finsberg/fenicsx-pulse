import dolfinx
import numpy as np
import pulsex
from mpi4py import MPI
from petsc4py import PETSc


def main():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)

    boundaries = [
        (1, 2, lambda x: np.isclose(x[0], 0)),
        (2, 2, lambda x: np.isclose(x[0], 1)),
        (3, 2, lambda x: np.isclose(x[1], 0)),
        (4, 2, lambda x: np.isclose(x[1], 1)),
    ]
    geo = pulsex.Geometry(
        mesh=mesh,
        boundaries=boundaries,
        metadata={"quadrature_degree": 4},
    )

    material_params = pulsex.HolzapfelOgden.transversely_isotropic_parameters()
    f0 = dolfinx.fem.Constant(mesh, PETSc.ScalarType((1.0, 0.0, 0.0)))
    s0 = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 1.0, 0.0)))
    material = pulsex.HolzapfelOgden(f0=f0, s0=s0, **material_params)

    Ta = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))
    active_model = pulsex.ActiveStress(f0, activation=Ta)
    comp_model = pulsex.Incompressible()

    model = pulsex.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    def dirichlet_bc(
        state_space: dolfinx.fem.FunctionSpace,
    ) -> list[dolfinx.fem.bcs.DirichletBCMetaClass]:
        V, _ = state_space.sub(0).collapse()
        facets = geo.facet_tags.find(1)
        dofs = dolfinx.fem.locate_dofs_topological((state_space.sub(0), V), 2, facets)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.set(0.0)
        return [dolfinx.fem.dirichletbc(u_fixed, dofs, state_space.sub(0))]

    traction = dolfinx.fem.Constant(mesh, PETSc.ScalarType(-1.0))
    neumann = pulsex.NeumannBC(traction=traction, marker=2)
    bcs = pulsex.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

    problem = pulsex.MechanicsProblem(model=model, geometry=geo, bcs=bcs)
    problem.solve()

    xdmf = dolfinx.io.XDMFFile(mesh.comm, "displacement.xdmf", "w")
    xdmf.write_mesh(mesh)

    u, _ = problem.state.split()
    xdmf.write_function(u, 0.0)

    Ta.value = 1.0

    problem.solve()
    u, _ = problem.state.split()
    xdmf.write_function(u, 1.0)


if __name__ == "__main__":
    main()
