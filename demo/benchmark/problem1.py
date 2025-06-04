# # Problem 1: deformation of a beam
#
# In the first problem we will solve the deformation of a beam. First we import the necessary libraries

from mpi4py import MPI
import numpy as np
import dolfinx
from dolfinx import log
import pulse

# Next we create the beam, which should have a length og 10 mm and a width of 1 mm

L = 10.0
W = 1.0
mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, W, W]], [30, 3, 3], dolfinx.mesh.CellType.hexahedron)

# There will be two boundaries. On the left boundary ($x = 0$) we will have a fixed Dirichlet condition and on the bottom boundary we will have a traction force pushing the beam upwards.
#
# We create markers for the two boundaries
left = 1
bottom = 2
boundaries = [
    pulse.Marker(name="left", marker=left, dim=2, locator=lambda x: np.isclose(x[0], 0)),
    pulse.Marker(name="bottom", marker=bottom, dim=2, locator=lambda x: np.isclose(x[2], 0)),
]

# and assemble the geometry

geo = pulse.Geometry(
    mesh=mesh,
    boundaries=boundaries,
    metadata={"quadrature_degree": 4},
)

# The material model used in this benchmark is the {py:class}`Guccione <pulse.material_models.guccione.Guccione>` model.

material_params = {
    "C": pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(2.0)), "kPa"),
    "bf": dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(8.0)),
    "bt": dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(2.0)),
    "bfs": dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(4.0)),
}
f0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((1.0, 0.0, 0.0)))
s0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 1.0, 0.0)))
n0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 0.0, 1.0)))
material = pulse.Guccione(f0=f0, s0=s0, n0=n0, **material_params)

# There are now active contraction, so we choose a pure passive model

active_model = pulse.active_model.Passive()

# and the model should be incompressible

comp_model = pulse.Incompressible()

# We can now assemble the `CardiacModel`
#

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)


# Next we define the Dirichlet BC

def dirichlet_bc(
    V: dolfinx.fem.FunctionSpace,
) -> list[dolfinx.fem.bcs.DirichletBC]:
    facets = geo.facet_tags.find(left)  # Specify the marker used on the boundary
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
    u_fixed = dolfinx.fem.Function(V)
    u_fixed.x.array[:] = 0.0
    return [dolfinx.fem.dirichletbc(u_fixed, dofs)]


# and the traction force

traction = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
neumann = pulse.NeumannBC(traction=traction, marker=bottom)

# We assemble all the boundary conditions

bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

# and create a mechanics problem

problem = pulse.StaticProblem(model=model, geometry=geo, bcs=bcs)

# Now let us turn on some more logging

log.set_log_level(log.LogLevel.INFO)

# and step up the traction to the target value (which is 0.004 kPa)

for t in [0.0, 0.001, 0.002, 0.003, 0.004]:
    print(f"Solving problem for traction={t}")
    traction.value = t
    problem.solve()

# Now let us turn off the logging again

log.set_log_level(log.LogLevel.WARNING)

# Save the displacement to a file
with dolfinx.io.VTXWriter(mesh.comm, "problem1_displacement.bp", [problem.u], engine="BP4") as vtx:
    vtx.write(0.0)

# and we find the deflection of the given point in the benchmark

point = np.array([10.0, 0.5, 1.0])
tree = dolfinx.geometry.bb_tree(mesh, 3)
cell_candidates = dolfinx.geometry.compute_collisions_points(tree, point)
cell = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, point)
uz = mesh.comm.allreduce(problem.u.eval(point, cell.array)[2], op=MPI.MAX)
result = point[2] + uz
print(f"Get z-position of point {point}: {result:.2f} mm")
assert np.isclose(result, 4.17, atol=1.0e-2)
# Finally, let us plot the deflected beam using pyvista

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    pyvista.start_xvfb()
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    uh = dolfinx.fem.Function(V)
    uh.interpolate(problem.u)
    # Create plotter and pyvista grid
    p = pyvista.Plotter()
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
    actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=1.5)
    actor_1 = p.add_mesh(warped, show_edges=True)
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        figure_as_array = p.screenshot("deflection.png")
