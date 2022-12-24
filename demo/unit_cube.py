import dolfinx
import numpy as np
import pulsex
from mpi4py import MPI
from petsc4py import PETSc


def main():
    # Create unit cube mesh
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)

    # Specific list of boundary markers
    boundaries = [
        pulsex.Marker(marker=1, dim=2, locator=lambda x: np.isclose(x[0], 0)),
        pulsex.Marker(marker=2, dim=2, locator=lambda x: np.isclose(x[0], 1)),
    ]
    # Create geometry
    geo = pulsex.Geometry(
        mesh=mesh,
        boundaries=boundaries,
        metadata={"quadrature_degree": 4},
    )

    # Create passive material model
    material_params = pulsex.HolzapfelOgden.transversely_isotropic_parameters()
    f0 = dolfinx.fem.Constant(mesh, PETSc.ScalarType((1.0, 0.0, 0.0)))
    s0 = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 1.0, 0.0)))
    material = pulsex.HolzapfelOgden(f0=f0, s0=s0, **material_params)

    # Create model for active contraction
    Ta = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))
    active_model = pulsex.ActiveStress(f0, activation=Ta)

    # Create model for compressibility
    comp_model = pulsex.Incompressible()

    # Create Cardiac Model
    model = pulsex.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    # Specific dirichlet boundary conditions on the boundary
    def dirichlet_bc(
        state_space: dolfinx.fem.FunctionSpace,
    ) -> list[dolfinx.fem.bcs.DirichletBCMetaClass]:
        V, _ = state_space.sub(0).collapse()
        facets = geo.facet_tags.find(1)  # Specify the marker used on the boundary
        dofs = dolfinx.fem.locate_dofs_topological((state_space.sub(0), V), 2, facets)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.set(0.0)
        return [dolfinx.fem.dirichletbc(u_fixed, dofs, state_space.sub(0))]

    # Use a traction on the opposite boundary
    traction = dolfinx.fem.Constant(mesh, PETSc.ScalarType(-1.0))
    neumann = pulsex.NeumannBC(traction=traction, marker=2)

    # Collect all boundary conditions
    bcs = pulsex.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

    # Create mechanics problem
    problem = pulsex.MechanicsProblem(model=model, geometry=geo, bcs=bcs)

    # Set a value for the active stress
    Ta.value = 2.0

    # Solve the problem
    problem.solve()

    # Get the solution
    u, p = problem.state.split()

    # And save to XDMF
    xdmf = dolfinx.io.XDMFFile(mesh.comm, "results.xdmf", "w")
    xdmf.write_mesh(mesh)
    xdmf.write_function(u, 0.0)
    xdmf.write_function(p, 0.0)


if __name__ == "__main__":
    main()
