# # Unit Cube
#
# In this demo we will use `fenicsx_pulse` to solve a simple contracting cube with one fixed
# side and with the opposite side having a traction force.
#
# First let us do the necessary imports

from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import log
import fenicsx_pulse
import numpy as np

# Then we can create unit cube mesh

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)

# Next let up specify a list of boundary markers where we will set the different boundary conditions

boundaries = [
    fenicsx_pulse.Marker(name="X0", marker=1, dim=2, locator=lambda x: np.isclose(x[0], 0)),
    fenicsx_pulse.Marker(name="X1", marker=2, dim=2, locator=lambda x: np.isclose(x[0], 1)),
]

# Now collect the boundaries and mesh in to a geometry object
#

geo = fenicsx_pulse.Geometry(
    mesh=mesh,
    boundaries=boundaries,
    metadata={"quadrature_degree": 4},
)

# We would also need to to create a passive material model.
# Here we will used the Holzapfel and Ogden material model

material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters()
f0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((1.0, 0.0, 0.0)))
s0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 1.0, 0.0)))
material = fenicsx_pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)  # type: ignore

# We also need to create a model for the active contraction. Here we use an active stress model

Ta = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
active_model = fenicsx_pulse.ActiveStress(f0, activation=Ta)

# We also need to specify whether the model what type of compressibility we want for our model.
# Here we use a full incompressible model

comp_model = fenicsx_pulse.Incompressible()

# Finally we collect all the models into a cardiac model
model = fenicsx_pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)


# Now we need to specify the different boundary conditions.
#
# We can specify the dirichlet boundary conditions using a function that takes the state
# space as input and return a list of dirichlet boundary conditions. Since we are using
# the an incompressible formulation the state space have two subspaces where the first
# subspace represents the displacement. Here we set the displacement to zero on the
# boundary with marker 1


def dirichlet_bc(
    V: dolfinx.fem.FunctionSpace,
) -> list[dolfinx.fem.bcs.DirichletBC]:
    facets = geo.facet_tags.find(1)  # Specify the marker used on the boundary
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
    u_fixed = dolfinx.fem.Function(V)
    u_fixed.x.array[:] = 0.0
    return [dolfinx.fem.dirichletbc(u_fixed, dofs)]


# We als set a traction on the opposite boundary

traction = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(-1.0))
neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=2)

# Finally we collect all the boundary conditions

bcs = fenicsx_pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

# and create a mechanics problem

problem = fenicsx_pulse.StaticProblem(model=model, geometry=geo, bcs=bcs)

# We also set a value for the active stress

Ta.value = 2.0

# And solve the problem

log.set_log_level(log.LogLevel.INFO)

problem.solve()

log.set_log_level(log.LogLevel.WARNING)


# We can get the solution (displacement)

u = problem.u
outdir = Path("unit_cube")
outdir.mkdir(exist_ok=True)

with dolfinx.io.VTXWriter(mesh.comm, outdir / "unit_cube_displacement.bp", [u], engine="BP4") as vtx:
    vtx.write(0.0)

# and visualize it using pyvista

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:

    pyvista.start_xvfb()

    # Create plotter and pyvista grid
    p = pyvista.Plotter()

    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(problem.u_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach vector values to grid and warp grid by vectora
    grid["u"] = u.x.array.reshape((geometry.shape[0], 3))
    actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=1.5)
    actor_1 = p.add_mesh(warped, show_edges=True)
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        figure_as_array = p.screenshot(outdir / "unit_cube_displacement.png")
