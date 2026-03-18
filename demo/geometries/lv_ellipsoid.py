# # Left Ventricular Ellipsoid Simulation
#
# This demo demonstrates how to simulate the mechanics of an idealized Left Ventricle (LV) represented
# as a truncated ellipsoid. We will cover geometry generation, fiber definition, passive and active
# material modeling, and boundary conditions.
#
# ## Problem Setup
#
# **Geometry**:
# An idealized LV geometry is generated using the [`cardiac-geometries`](https://computationalphysiology.github.io/cardiac-geometriesx/) library.
# The shape is defined by a truncated prolate spheroid.
#
# **Microstructure (Fibers)**:
# Cardiac tissue is anisotropic. We define a local coordinate system with:
# * **Fibers ($\mathbf{f}_0$)**: The primary direction of muscle cells.
# * **Sheets ($\mathbf{s}_0$)**: The direction of myocardial sheets.
# * **Sheet-normals ($\mathbf{n}_0$)**: Orthogonal to fibers and sheets.
#
# The fiber angle typically varies transmurally (from endocardium to epicardium), e.g., from $+60^\circ$ to $-60^\circ$.
#
# **Physics**:
# We solve the static balance of linear momentum for a hyperelastic material:
#
# $$
# \nabla \cdot \mathbf{P} = \mathbf{0} \quad \text{in } \Omega_0
# $$
#
# where $\mathbf{P}$ is the First Piola-Kirchhoff stress tensor.
#
# ---

# ## Imports

from pathlib import Path
from mpi4py import MPI
import dolfinx
import cardiac_geometries
import cardiac_geometries.geometry
import pulse

# ## Geometry Generation
#
# We define the output directory and generate the mesh.
# `cardiac_geometries.mesh.lv_ellipsoid` creates the mesh and tags the boundaries:
# * **ENDO**: Endocardium (inner surface)
# * **EPI**: Epicardium (outer surface)
# * **BASE**: Top basal plane
#
# It also generates the fiber fields based on the specified angles.

outdir = Path("lv_ellipsoid")
outdir.mkdir(parents=True, exist_ok=True)
geodir = outdir / "geometry"

if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        create_fibers=True,
        fiber_space="P_2",  # Second-order polynomial space for fibers
    )

# We load the geometry into a `pulse.Geometry` object.
# This object wraps the mesh, markers, and integration measures.

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)

# We convert the geometry to a `pulse.Geometry` object and specify the quadrature degree
# for numerical integration.

geometry = pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})

# ## Constitutive Model
#
# ### 1. Passive Material Properties
# We use the **Holzapfel-Ogden** constitutive model [Holzapfel & Ogden 2009].
# This model captures the exponential stiffening and orthotropic nature of myocardium.
#
# The strain energy density $\Psi_{pass}$ depends on the invariants $I_1, I_{4f}, I_{4s}, I_{8fs}$:
#
# $$
# \Psi_{pass} = \frac{a}{2b} (e^{b(I_1-3)} - 1)
# + \sum_{i=f,s} \frac{a_i}{2b_i} \mathcal{H}(I_{4i}-1) (e^{b_i(I_{4i}-1)^2} - 1)
# + \frac{a_{fs}}{2b_{fs}} (e^{b_{fs}I_{8fs}^2} - 1)
# $$
#
# In this example, we use the **Transversely Isotropic** parameter set (where sheet properties are zeroed out),
# focusing on the isotropic matrix and fiber direction.

material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)

# ### 2. Active Contraction
# We model contraction using the **Active Stress** approach.
# An active stress component is added to the total Second Piola-Kirchhoff stress $\mathbf{S}$.
#
# $$
# \mathbf{S}_{active} = T_a \left( \mathbf{f}_0 \otimes \mathbf{f}_0 + \eta (\mathbf{I} - \mathbf{f}_0 \otimes \mathbf{f}_0) \right)
# $$
#
# * $T_a$: Magnitude of active tension (scalar).
# * $\mathbf{f}_0$: Fiber direction.
# * $\eta$: Parameter governing transverse active stress (coupling factor).
#
# Here we set $\eta=0.3$, meaning 30% of the active tension is applied transversely to the fibers.

Ta = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
active_model = pulse.ActiveStress(geo.f0, activation=Ta, eta=0.3)

# ### 3. Incompressibility
# The heart muscle is treated as fully incompressible ($J=1$).
# This is enforced via a Lagrange multiplier $p$ (hydrostatic pressure).

comp_model = pulse.Incompressible()

# ### Assembly
# We combine these components into the final `CardiacModel`.

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# ## Boundary Conditions
#
# ### Neumann BC: Cavity Pressure
# We apply a pressure load on the endocardium. In the variational formulation, this appears as a traction term:
#
# $$
# \int_{\Gamma_{endo}} -p_{cavity} \mathbf{n} \cdot \mathbf{v} \, ds
# $$
#
# Note: `traction` here represents the magnitude of the pressure.

traction = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
neumann = pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])

# ### Dirichlet BC: Fixed Base
# We hold the base of the ellipsoid fixed ($\mathbf{u} = \mathbf{0}$ on $\Gamma_{base}$).
# This is handled via the `parameters` argument in the problem definition.

bcs = pulse.BoundaryConditions(neumann=(neumann,))

# ## Solving the Problem
#
# We define a `StaticProblem`.
# * `base_bc=pulse.BaseBC.fixed`: Automatically applies Dirichlet BCs to the boundary marked as 'BASE'.

problem = pulse.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs,
    parameters={"base_bc": pulse.BaseBC.fixed},
)

# ### Phase 1: Passive Inflation
# We first solve the passive mechanics by increasing the endocardial pressure.
# We initialize a VTX writer to save the displacement field for visualization.

vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, outdir / "lv_displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)

pressures = [0.1] # kPa. Add more steps for a smoother ramp, e.g. [0.1, 0.5, 1.0]
for i, plv in enumerate(pressures, start=1):
    print(f"Solving for pressure: {plv} kPa")
    traction.assign(plv)
    problem.solve()
    vtx.write(float(i))

# #### Visualization (Passive)
# We can visualize the inflated state using PyVista if available.

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    # Interpolate solution to a standard CG-1 space for plotting
    V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 1, (geometry.mesh.geometry.dim,)))
    uh = dolfinx.fem.Function(V)
    uh.interpolate(problem.u)

    # Create plotter
    p = pyvista.Plotter()
    topology, cell_types, geometry_data = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry_data)

    # Warp grid by displacement
    grid["u"] = uh.x.array.reshape((geometry_data.shape[0], 3))
    p.add_mesh(grid, style="wireframe", color="k", opacity=0.3, label="Reference")
    warped = grid.warp_by_vector("u", factor=1.0)
    p.add_mesh(warped, show_edges=True, color="firebrick", label="Inflated")

    p.add_legend()
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot(outdir / "lv_ellipsoid_pressure.png")

# ### Phase 2: Active Contraction
# Now we keep the pressure constant and increase the active tension $T_a$.
# This simulates the systole phase (isovolumetric contraction/ejection).

active_tensions = [0.1] # kPa. Add steps like [0.5, 1.0, 2.0] for full contraction
for i, ta in enumerate(active_tensions, start=len(pressures) + 1):
    print(f"Solving for active tension: {ta} kPa")
    Ta.assign(ta)
    problem.solve()
    vtx.write(float(i))

vtx.close()

# #### Visualization (Active)

try:
    import pyvista
except ImportError:
    pass
else:
    uh.interpolate(problem.u)
    grid["u"] = uh.x.array.reshape((geometry_data.shape[0], 3))

    p = pyvista.Plotter()
    p.add_mesh(grid, style="wireframe", color="k", opacity=0.3, label="Reference")

    warped = grid.warp_by_vector("u", factor=1.0)
    p.add_mesh(warped, show_edges=True, color="red", label="Contracted")

    p.add_legend()
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot(outdir / "lv_ellipsoid_active.png")
