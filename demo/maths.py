# %% [markdown]
# # Mathematical Background & Implementation Details
#
# This document outlines the mathematical theory of finite hyperelasticity used in `fenicsx-pulse`
# and demonstrates how these concepts are mapped to specific functions and classes in the library.
#
# We will assume a standard continuum mechanics framework where a body $\mathcal{B}$ is identified
# with a reference configuration $\Omega_0$. The motion is described by the map
# $\mathbf{x} = \chi(\mathbf{X}, t)$, where $\mathbf{X} \in \Omega_0$ is the reference position
# and $\mathbf{x}$ is the current position.

# %%
from pathlib import Path
import logging
import dolfinx
import ufl
import pulse
from mpi4py import MPI
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pulse")
for lib in ["trame_server", "wslink"]:
    logging.getLogger(lib).setLevel(logging.WARNING)
logger.setLevel(logging.DEBUG)
# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

# %% [markdown]
# ## 1. Geometry and Mesh
#
# The first step in any Finite Element simulation is defining the domain.
# Here, we create a simple unit cube mesh to serve as our reference configuration $\Omega_0$.
# In a realistic cardiac simulation, this would be replaced by a patient-specific geometry.

# %%
comm = MPI.COMM_WORLD
mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)

# %% [markdown]
# ### Visualization
# We can visualize the mesh using `pyvista`.

# %%
try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    p = pyvista.Plotter()
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    p.add_mesh(grid, show_edges=True)
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        # Save screenshot if running in CI/headless
        p.screenshot("maths_mesh.png")

# %% [markdown]
# ### Boundary Markers
# To solve the boundary value problem, we need to identify specific parts of the boundary $\partial \Omega_0$.
# We define markers using geometric locators:
#
# * **Marker `1` ("X0")**: The face where $X=0$ (to be fixed).
# * **Marker `2` ("X1")**: The face where $X=1$ (to apply traction).

# %%
boundaries = [
    pulse.Marker(name="X0", marker=1, dim=2, locator=lambda x: np.isclose(x[0], 0)),
    pulse.Marker(name="X1", marker=2, dim=2, locator=lambda x: np.isclose(x[0], 1)),
]

# %% [markdown]
# We wrap the mesh and markers into a `pulse.Geometry` object. This object manages the integration measures
# (`dx` for volume, `ds` for surface) and ensures they are set up with the correct quadrature degree.

# %%
geo = pulse.Geometry(
    mesh=mesh,
    boundaries=boundaries,
    metadata={"quadrature_degree": 4},
)


# %% [markdown]
# We can also visualize the boundary markers

# %%
geo.mesh.topology.create_connectivity(mesh.topology.dim-1 , mesh.topology.dim)
vtk_bmesh = dolfinx.plot.vtk_mesh(geo.mesh, geo.facet_tags.dim, geo.facet_tags.indices)
bgrid = pyvista.UnstructuredGrid(*vtk_bmesh)
bgrid.cell_data["Facet tags"] = geo.facet_tags.values
bgrid.set_active_scalars("Facet tags")
p = pyvista.Plotter(window_size=[800, 800])
p.add_mesh(bgrid, show_edges=True)
p.add_mesh(grid, show_edges=True, style="wireframe", color="k")
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("facet_tags.png")

# %% [markdown]
# ## 2. Constitutive Equations
#
# The material behavior is governed by a **Strain Energy Density Function** $\Psi(\mathbf{C})$.
# In `fenicsx-pulse`, the total energy is composed of three parts:
#
# $$
# \Psi = \Psi_{\text{passive}} + \Psi_{\text{active}} + \Psi_{\text{vol}}
# $$
#
# The stress tensors are derived from $\Psi$ via automatic differentiation:
# * **Second Piola-Kirchhoff stress**: $\mathbf{S} = 2 \frac{\partial \Psi}{\partial \mathbf{C}}$
# * **First Piola-Kirchhoff stress**: $\mathbf{P} = \mathbf{F} \mathbf{S}$

# %% [markdown]
# ### A. Passive Material ($\Psi_{\text{passive}}$)
#
# We use the **Holzapfel-Ogden** model ({py:class}`pulse.HolzapfelOgden`), a standard for ventricular myocardium.
#
# $$
# \Psi_{HO} = \frac{a}{2b} (e^{b(I_1-3)} - 1)
# + \sum_{i=f,s} \frac{a_i}{2b_i} \mathcal{H}(I_{4i}-1) (e^{b_i(I_{4i}-1)^2} - 1)
# + \frac{a_{fs}}{2b_{fs}} (e^{b_{fs}I_{8fs}^2} - 1)
# $$
#
# Here $\mathcal{H}(\cdot)$ is the Heaviside function, ensuring fibers only stiffen in tension.

# %%
# Retrieve default parameters (which are wrapped in pulse.units.Variable)
material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
print(f"Parameter 'a': {material_params['a']}")

# Define constant fiber/sheet fields for this simple cube
f0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((1.0, 0.0, 0.0)))
s0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 1.0, 0.0)))

material = pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)

# %% [markdown]
# ### B. Active Contraction ($\Psi_{\text{active}}$)
#
# We model contraction using the **Active Stress** approach ({py:class}`pulse.ActiveStress`).
# An active component is added to the energy:
#
# $$
# \Psi_{\text{active}} = \frac{1}{2} T_a (I_{4f} - 1)
# $$
#
# Differentiating this yields the active stress contribution $\mathbf{S}_{\text{active}} = T_a \mathbf{f}_0 \otimes \mathbf{f}_0$.

# %%
Ta = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "kPa")
active_model = pulse.ActiveStress(f0, activation=Ta)

# %% [markdown]
# ### C. Compressibility ($\Psi_{\text{vol}}$)
#
# Myocardium is nearly incompressible ($J \approx 1$). We enforce this using the **Incompressible**
# model ({py:class}`pulse.Incompressible`), which uses a Lagrange multiplier $p$ (hydrostatic pressure).
#
# $$
# \Psi_{\text{vol}} = p (J - 1)
# $$

# %%
comp_model = pulse.Incompressible()

# %% [markdown]
# ### Assembly: The Cardiac Model
#
# The {py:class}`pulse.CardiacModel` class aggregates these components into a single object that
# provides the total $\mathbf{S}$ and $\mathbf{P}$ tensors.

# %%
model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)


# %% [markdown]
# ## 3. Balance Laws & Boundary Value Problem
#
# We solve the balance of linear momentum in the reference configuration:
#
# $$
# \nabla \cdot \mathbf{P} + \rho_0 \mathbf{B} = \mathbf{0} \quad \text{in } \Omega_0
# $$
#
# The weak (variational) form used in {py:class}`pulse.StaticProblem` is derived by multiplying by a
# test function $\delta \mathbf{u}$ and integrating by parts:
#
# $$
# \int_{\Omega_0} \mathbf{P} : \nabla \delta \mathbf{u} \, \text{d}X
# - \int_{\partial \Omega_N} \mathbf{t} \cdot \delta \mathbf{u} \, \text{d}S
# = 0
# $$
#
# For the incompressible case, we also add the constraint equation:
# $$
# \int_{\Omega_0} (J - 1) \delta p \, \text{d}X = 0
# $$

# %% [markdown]
# ### Boundary Conditions
#
# We define the specific conditions for our cube:
# 1.  **Dirichlet BC**: Fix displacement at $X=0$.
# 2.  **Neumann BC**: Apply traction at $X=1$.

# %%
# 1. Dirichlet: Fix X0
def dirichlet_bc(V: dolfinx.fem.FunctionSpace):
    facets = geo.facet_tags.find(1)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
    u_fixed = dolfinx.fem.Function(V)
    u_fixed.x.array[:] = 0.0
    return [dolfinx.fem.dirichletbc(u_fixed, dofs)]

# 2. Neumann: Traction on X1 (Marker 2)
traction = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), "kPa")
neumann_bc = pulse.NeumannBC(traction=traction, marker=2)

# Collect BCs
bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann_bc,))

# %% [markdown]
# ## 4. Solving the Problem
#
# The `StaticProblem` class handles the assembly of the mixed function space (for $\mathbf{u}$ and $p$),
# the construction of the variational forms, and the Newton solver configuration.

# %%
problem = pulse.StaticProblem(model=model, geometry=geo, bcs=bcs)

# Apply active tension to simulate contraction
Ta.value = 2.0

# Solve the system
problem.solve()

# %% [markdown]
# ### Visualization of Result
# Finally, we can visualize the deformed configuration.

# %%
try:
    import pyvista
except ImportError:
    pass
else:
    # Interpolate solution to a standard space for plotting
    p = pyvista.Plotter()
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(problem.u_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Add reference mesh (wireframe)
    p.add_mesh(grid, style="wireframe", color="black", opacity=0.3, label="Reference")

    # Warp by displacement
    grid["u"] = problem.u.x.array.reshape((-1, 3))
    warped = grid.warp_by_vector("u", factor=1.0)

    # Add deformed mesh
    p.add_mesh(warped, show_edges=True, label="Deformed")
    p.add_legend()
    p.show_axes()

    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot("maths_deformed.png")

# %% [markdown]
# ## 5. Kinematics definitions
#
# The primary unknown in our problem is the **Displacement field** $\mathbf{u}(\mathbf{X})$.
# We define a standard Lagrange function space and the function $\mathbf{u}$.
#
# The **Deformation Gradient** $\mathbf{F}$ is defined as:
#
# $$
# \mathbf{F} = \frac{\partial \mathbf{x}}{\partial \mathbf{X}} = \mathbf{I} + \nabla \mathbf{u}
# $$
#
# In `fenicsx-pulse`, this is computed via {py:func}`pulse.kinematics.DeformationGradient`.

# %%
u = problem.u
F = pulse.kinematics.DeformationGradient(u)
print(f"Shape of F: {F.ufl_shape}")

# %% [markdown]
# The volume change is measured by the Jacobian $J = \det \mathbf{F}$.

# %%
J = pulse.kinematics.Jacobian(F)

# %% [markdown]
# Since this is an incompressible problem we expect $J$ to be equal to 1.0

# %%
mesh_volume = comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * geo.dx)), op=MPI.SUM)
comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(J * geo.dx)), op=MPI.SUM) / mesh_volume

# %% [markdown]
# Here we first compiles form
# ```python
# dolfinx.fem.form(J * ufl.dx)
# ```
# then we assemble the form
# ```python
# dolfinx.fem.assemble_scalar(dolfinx.fem.form(J * ufl.dx))
# ```
# which will assemble the form locally on each process, and finally we perform an allreduce using the MPI communicator to sum up the contributions from all the processors.
# Note that we also divide by the volume (which is 1.0 in the case of the Unit Cube) which is computed the same fashion.

# %% [markdown]
# ### Strain Tensors
# To define material laws that are independent of rigid body rotations, we use strain tensors derived from $\mathbf{F}$.
# `pulse.kinematics` provides standard tensors:
#
# 1.  **Right Cauchy-Green tensor**: $\mathbf{C} = \mathbf{F}^T \mathbf{F}$
# 2.  **Left Cauchy-Green tensor**: $\mathbf{B} = \mathbf{F} \mathbf{F}^T$
# 3.  **Green-Lagrange strain**: $\mathbf{E} = \frac{1}{2}(\mathbf{C} - \mathbf{I})$

# %%
C = pulse.kinematics.RightCauchyGreen(F)
B = pulse.kinematics.LeftCauchyGreen(F)
E = pulse.kinematics.GreenLagrangeStrain(F)

# %% [markdown]
# Now, say you are interested in the strain in a given direction, e.g the fiber strain ($E_{ff}$). Then one can get the by computing the inner product

# %%
Eff_ufl_expr = ufl.inner(E * f0, f0)

# %% [markdown]
# If we now would like to visualize this in Pyvista or Paraview then we need to first interpolate this into a function space. Since we have $\mathbf{u}$ being $\mathbb{P}_2$, i.e second order polynomial but only continous and not continously differentiable that dofs, the graient $\nabla \mathbf{u}$ belongs to a discountinous first order space. Since $\mathbf{E}$ is a function of the $\nabla \mathbf{u}^T \nabla \mathbf{u}$ a reasonable space would be a second order discontinous space

# %%
V_strain = dolfinx.fem.functionspace(mesh, ("DG", 2))
Eff = dolfinx.fem.Function(V_strain)

# %% [markdown]
# We can now interpolate the fiber strain into this space

# %%
Eff_expr = dolfinx.fem.Expression(Eff_ufl_expr, V_strain.element.interpolation_points)
Eff.interpolate(Eff_expr)

# %% [markdown]
# We can now visualize the the strain

# %%
try:
    import pyvista
except ImportError:
    pass
else:
    # Interpolate solution to a standard space for plotting
    p = pyvista.Plotter()
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V_strain)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    grid["Eff"] = Eff.x.array
    # Add reference mesh (wireframe)
    p.add_mesh(grid, show_edges=True, cmap="inferno")
    p.show_axes()

    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot("Eff.png")

# %% [markdown]
# ## 6. Material Invariants
#
# Hyperelastic constitutive laws are typically expressed in terms of the invariants of $\mathbf{C}$.
# `pulse.invariants` provides helper functions for these.
#
# ### Isotropic Invariants
# For isotropic materials, the strain energy $\Psi$ depends on:
#
# $$
# I_1 = \text{tr}(\mathbf{C}), \quad I_2 = \frac{1}{2}(I_1^2 - \text{tr}(\mathbf{C}^2)), \quad I_3 = \det \mathbf{C} = J^2
# $$

# %%
I1 = pulse.invariants.I1(C)
I2 = pulse.invariants.I2(C)
I3 = pulse.invariants.I3(C)

# %% [markdown]
# ### Anisotropic Invariants
# Cardiac tissue is orthotropic. Its behavior depends on the local microstructure defined by
# fiber ($\mathbf{f}_0$), sheet ($\mathbf{s}_0$), and normal ($\mathbf{n}_0$) directions.
#
# We define pseudo-invariants to capture stretch along these directions and shear between them:
#
# $$
# I_{4f} = \mathbf{f}_0 \cdot (\mathbf{C} \mathbf{f}_0) \quad (\text{Fiber stretch squared})
# $$
# $$
# I_{8fs} = \mathbf{f}_0 \cdot (\mathbf{C} \mathbf{s}_0) \quad (\text{Fiber-sheet shear coupling})
# $$

# %%
I4f = pulse.invariants.I4(C, f0)
I8fs = pulse.invariants.I8(C, f0, s0)


# %% [markdown]
# ## 7. Stress tensors
# We can also compute the stress tensors directly from the cardiac model.
# Note that we need to wrap the deformation gradient and right Cauchy-Green tensor into a `ufl.variable` in order to be able to differentiate the strain energy function.

# %%
S = model.S(ufl.variable(C))
P = model.P(ufl.variable(F))
T = model.sigma(ufl.variable(F))

# %% [markdown]
# Similar to the strain case, one might also be interested in visualizing the fiber stress
#
# $$\sigma_{ff} =  \mathbf{f} \cdot (\mathbf{\sigma} \mathbf{f}),$$
#
# where
#
# $$\mathbf{f} = \frac{\mathbf{F} \mathbf{f}_0}{\| \mathbf{F} \mathbf{f}_0 \|} $$

# %%
f = F * f0
f /= ufl.sqrt(f**2)
Tff_ufl_expr = ufl.inner(T * f, f)
Tff = dolfinx.fem.Function(V_strain)
Tff_expr = dolfinx.fem.Expression(Tff_ufl_expr, V_strain.element.interpolation_points)
Tff.interpolate(Tff_expr)
try:
    import pyvista
except ImportError:
    pass
else:
    # Interpolate solution to a standard space for plotting
    p = pyvista.Plotter()
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V_strain)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    grid["Tff"] = Tff.x.array
    # Add reference mesh (wireframe)
    p.add_mesh(grid, show_edges=True, cmap="inferno", clim=(0, 1000))
    p.show_axes()

    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot("Eff.png")

# %% [markdown]
# ## Learn more
#
# To learn more you could check out Holzapfel's Nonlinear continuum mechanics book {cite}`holzapfel2002nonlinear`

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
#
