# # Unit Cube Simulation
#
# This demo illustrates how to setup and solve a simple cardiac mechanics problem using `fenicsx-pulse`.
# We will simulate a contracting unit cube of cardiac tissue. One face of the cube is fixed, while the
# opposite face is subject to a traction force.
#
# ## Problem Definition
#
# **Geometry**: A unit cube $\Omega = [0,1]^3$.
#
# **Physics**:
# We solve the balance of linear momentum for a hyperelastic material:
#
# $$
# \nabla \cdot \mathbf{P} + \mathbf{B} = \mathbf{0} \quad \text{in } \Omega
# $$
#
# where $\mathbf{P}$ is the First Piola-Kirchhoff stress tensor and $\mathbf{B}$ is the body force.
#
# **Material Model**:
# The material is modeled as:
# 1. **Passive**: Anisotropic Holzapfel-Ogden material (orthotropic myocardium).
# 2. **Active**: Active stress generated along the fiber direction.
# 3. **Incompressible**: Volume is conserved ($J = \det \mathbf{F} = 1$).
#
# **Boundary Conditions**:
# * **Dirichlet**: Fixed displacement ($\mathbf{u} = \mathbf{0}$) on the face $X=0$.
# * **Neumann**: A constant traction force $\mathbf{t}$ on the face $X=1$.
#
# ---

# ## Imports
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import log
import pulse
import numpy as np

# ## Geometry and Mesh
# We start by creating a unit cube mesh with hexahedral elements.
# The mesh defines the reference configuration $\Omega_0$.

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)

# ### Boundary Markers
# We define markers to identify specific boundaries of the domain.
# * Marker `1` ("X0"): The face where $x=0$.
# * Marker `2` ("X1"): The face where $x=1$.

boundaries = [
    pulse.Marker(name="X0", marker=1, dim=2, locator=lambda x: np.isclose(x[0], 0)),
    pulse.Marker(name="X1", marker=2, dim=2, locator=lambda x: np.isclose(x[0], 1)),
]

# We collect the mesh and boundaries into a `Geometry` object.
# This object handles the integration measures (`dx`, `ds`) and facet tags.

geo = pulse.Geometry(
    mesh=mesh,
    boundaries=boundaries,
    metadata={"quadrature_degree": 4},
)

# ## Material Constitutive Models
#
# ### 1. Passive Material
# We use the **Holzapfel-Ogden** law, a standard constitutive model for ventricular myocardium.
# It captures the orthotropic behavior using fiber ($\mathbf{f}_0$) and sheet ($\mathbf{s}_0$) directions.
#
# The strain energy density function $\Psi_{pass}$ is composed of isotropic and anisotropic terms:
#
# $$
# \Psi_{pass} = \frac{a}{2b} (e^{b(I_1-3)} - 1) + \frac{a_f}{2b_f} (e^{b_f(I_{4f}-1)^2} - 1) + \dots
# $$
#
# For this simple cube, we define constant fiber directions aligned with the X-axis.

material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
f0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((1.0, 0.0, 0.0)))
s0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 1.0, 0.0)))
material = pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)

# ### 2. Active Contraction
# We model the active contraction using an **Active Stress** approach.
# An active stress $T_a$ is added to the stress tensor, usually along the fiber direction.
#
# In `fenicsx-pulse`, this is implemented by adding an active term to the strain energy potential:
#
# $$
# \Psi_{act} = \frac{1}{2} T_a (I_{4f} - 1)
# $$
#
# Differentiating this yields an active stress $\mathbf{S}_{act} = T_a \mathbf{f}_0 \otimes \mathbf{f}_0$.

Ta = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
active_model = pulse.ActiveStress(f0, activation=Ta)

# ### 3. Compressibility
# Myocardium is nearly incompressible. We enforce full **Incompressibility** ($J=1$)
# using a Lagrange multiplier $p$ (hydrostatic pressure).
#
# $$
# \Psi_{comp} = p (J - 1)
# $$

comp_model = pulse.Incompressible()

# ### Cardiac Model Assembly
# We combine the passive, active, and compressibility components into a single `CardiacModel`.
# The total strain energy is $\Psi = \Psi_{pass} + \Psi_{act} + \Psi_{comp}$.

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# ## Boundary Conditions
#
# ### Dirichlet BC
# We fix the displacement on the left face (Marker 1).
# This function identifies the degrees of freedom (DOFs) on the boundary and sets them to zero.

def dirichlet_bc(
    V: dolfinx.fem.FunctionSpace,
) -> list[dolfinx.fem.bcs.DirichletBC]:
    facets = geo.facet_tags.find(1)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
    u_fixed = dolfinx.fem.Function(V)
    u_fixed.x.array[:] = 0.0
    return [dolfinx.fem.dirichletbc(u_fixed, dofs)]

# ### Neumann BC
# We apply a traction $\mathbf{t}$ on the opposite face (Marker 2).
# A negative value indicates compression/pushing, while positive would be tension/pulling.

traction = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(-1.0))
neumann = pulse.NeumannBC(traction=traction, marker=2)

# We collect all boundary conditions into a `BoundaryConditions` object.

bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

# ## Solving the Problem
# We initialize the `StaticProblem`. This sets up the function spaces (Mixed elements for displacement and pressure if incompressible), the variational form, and the nonlinear solver.

problem = pulse.StaticProblem(model=model, geometry=geo, bcs=bcs)

# ### Apply Activation
# We set the active stress parameter $T_a$ to 2.0 kPa. This will cause the "fibers" (aligned with X) to shorten, pulling the cube's free end.

Ta.value = 2.0

# Solve the nonlinear system of equations.

problem.solve()

# ## Post-processing
# The solution displacement field `u` is extracted from the problem.
# We save the results to `BP4` format (ADIOS2) for visualization in Paraview or similar tools.

u = problem.u
outdir = Path("unit_cube")
outdir.mkdir(exist_ok=True)

with dolfinx.io.VTXWriter(mesh.comm, outdir / "unit_cube_displacement.bp", [u], engine="BP4") as vtx:
    vtx.write(0.0)

# ### Visualization (Optional)
# If `pyvista` is installed, we can render the deformed mesh directly in the notebook.

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    # Create plotter and pyvista grid
    p = pyvista.Plotter()

    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(problem.u_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = u.x.array.reshape((geometry.shape[0], 3))
    actor_0 = p.add_mesh(grid, style="wireframe", color="k")

    # Warp the mesh by the displacement vector to visualize deformation
    warped = grid.warp_by_vector("u", factor=1.0)
    actor_1 = p.add_mesh(warped, show_edges=True, color="red", opacity=0.5)

    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        figure_as_array = p.screenshot(outdir / "unit_cube_displacement.png")
