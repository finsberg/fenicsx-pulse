# # Problem 1: Deformation of a Beam
#
# This example implements Problem 1 from the cardiac mechanics benchmark suite [Land et al. 2015].
#
# ## Problem Description
#
# **Geometry**:
# A cuboid beam with dimensions $10 \times 1 \times 1$ mm.
# The domain is defined as $\Omega = [0, 10] \times [0, 1] \times [0, 1]$.
#
# **Material**:
# Transversely isotropic Guccione material.
# * Fiber direction $\mathbf{f}_0 = (1, 0, 0)$ (aligned with the long axis).
# * Constitutive parameters: $C = 2.0$ kPa, $b_f = 8.0$, $b_t = 2.0$, $b_{fs} = 4.0$.
# * The material is incompressible.
#
# **Boundary Conditions**:
# * **Dirichlet**: The face at $X=0$ is fully clamped ($\mathbf{u} = \mathbf{0}$).
# * **Neumann**: A pressure load $P$ is applied to the bottom face ($Z=0$). The pressure increases linearly from 0 to 0.004 kPa.
#
# **Target Quantity**:
# The deflection (vertical displacement) of the point $(10, 0.5, 1.0)$ at the maximum load.
# The benchmark reference value is approximately **4.0 - 4.2 mm**.
#
# ---

from mpi4py import MPI
import numpy as np
import dolfinx
import pulse

# ## 1. Geometry and Mesh
# We create a box mesh of dimensions $10 \times 1 \times 1$.

L = 10.0
W = 1.0
mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, W, W]], [30, 3, 3], dolfinx.mesh.CellType.hexahedron,
)


# We define markers for the boundary conditions.
# * `left` (Marker 1): Face at X=0.
# * `bottom` (Marker 2): Face at Z=0.

left = 1
bottom = 2
boundaries = [
    pulse.Marker(name="left", marker=left, dim=2, locator=lambda x: np.isclose(x[0], 0)),
    pulse.Marker(name="bottom", marker=bottom, dim=2, locator=lambda x: np.isclose(x[2], 0)),
]

geo = pulse.Geometry(
    mesh=mesh,
    boundaries=boundaries,
    metadata={"quadrature_degree": 4},
)

# ## 2. Constitutive Model
#
# We use the **Guccione** model as specified in the benchmark.
#
# $$
# \Psi = \frac{C}{2} (e^Q - 1), \quad Q = b_f E_{ff}^2 + b_t (E_{ss}^2 + E_{nn}^2 + E_{sn}^2 + E_{ns}^2) + b_{fs} (E_{fs}^2 + E_{sf}^2 + E_{fn}^2 + E_{nf}^2)
# $$
#
# Since the problem defines a fixed fiber direction along $x$, we set $\mathbf{f}_0, \mathbf{s}_0, \mathbf{n}_0$ to align with the coordinate axes.

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

# The problem is purely passive (no active contraction) and incompressible.

active_model = pulse.active_model.Passive()
comp_model = pulse.Incompressible()

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# ## 3. Boundary Conditions
#
# **Dirichlet BC**: Fix all displacement components on the 'left' face.


def dirichlet_bc(V: dolfinx.fem.FunctionSpace):
    facets = geo.facet_tags.find(left)
    dofs = dolfinx.fem.locate_dofs_topological(V, geo.facet_dimension, facets)
    u_fixed = dolfinx.fem.Function(V)
    u_fixed.x.array[:] = 0.0
    return [dolfinx.fem.dirichletbc(u_fixed, dofs)]


# **Neumann BC**: Apply a pressure load to the 'bottom' face.
# Note: In `fenicsx-pulse`, a `NeumannBC` with a scalar traction represents a pressure load $P$ acting normal to the deformed surface: $\mathbf{t} = P \mathbf{n}$.
# Since the pressure is pushing *up* against the bottom face (normal pointing down), we define a positive pressure.

pressure = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "kPa")
neumann = pulse.NeumannBC(traction=pressure, marker=bottom)

bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

# ## 4. Solving the Problem
#
# We initialize the static problem and solve it incrementally up to the target pressure of 0.004 kPa.

problem = pulse.StaticProblem(model=model, geometry=geo, bcs=bcs)

target_pressure = 0.004
steps = [0.0005, 0.001, 0.002, 0.003, 0.004]

for p in steps:
    print(f"Solving for pressure = {p} kPa")
    pressure.assign(p)
    problem.solve()

# ## 5. Post-processing
#
# We save the final displacement and evaluate the deflection at the specific target point $(10, 0.5, 1)$.

with dolfinx.io.VTXWriter(mesh.comm, "problem1_displacement.bp", [problem.u], engine="BP4") as vtx:
    vtx.write(0.0)

# Evaluate displacement at the target point
point = np.array([10.0, 0.5, 1.0])

# Use bounding box tree for point collision
tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
cell_candidates = dolfinx.geometry.compute_collisions_points(tree, point)
cell = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, point)

# Get the z-displacement
if len(cell.array) > 0:
    # Evaluate only if the point is found on this process
    u_val = problem.u.eval(point, cell.array)[2]
else:
    u_val = -np.inf  # Placeholder for reduction

# Reduce across all processes (take the max to get the actual value)
uz = mesh.comm.allreduce(u_val, op=MPI.MAX)
result = point[2] + uz

print(f"Target Point: {point}")
print(f"Vertical Deflection (Uz): {uz:.4f} mm")
print(f"Final Z Position: {result:.4f} mm")

# Verify against benchmark tolerance (approx 4.0 - 4.2 mm deflection)
# Note: Result is typically around 4.17 mm for the settings described.
if mesh.comm.rank == 0:
    print(f"Benchmark Ref: ~4.17 mm")

# Visualization with PyVista
try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    uh = dolfinx.fem.Function(V)
    uh.interpolate(problem.u)

    p = pyvista.Plotter()
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))

    p.add_mesh(grid, style="wireframe", color="k", opacity=0.5, label="Reference")
    warped = grid.warp_by_vector("u", factor=1.0)
    p.add_mesh(warped, show_edges=True, label="Deformed")

    p.add_legend()
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot("deflection.png")
