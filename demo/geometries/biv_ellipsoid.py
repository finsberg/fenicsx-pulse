# # Bi-Ventricular Ellipsoid Simulation
#
# This demo illustrates how to simulate the mechanics of an idealized Bi-Ventricular (BiV) geometry.
# Unlike the Left Ventricle (LV) only model, this includes both the LV and Right Ventricle (RV),
# allowing us to study interventricular interactions.
#
# ## Problem Setup
#
# **Geometry**:
# An idealized BiV geometry is generated using [`cardiac-geometries`](https://computationalphysiology.github.io/cardiac-geometriesx/).
# It consists of two truncated ellipsoids joined together.
#
# **Microstructure (Fibers)**:
# For BiV geometries, analytical fiber definitions are complex. We use the **Laplace-Dirichlet Rule-Based (LDRB)** algorithm
# [Bayer et al. 2012] implemented in [`fenicsx-ldrb`](https://github.com/finsberg/fenicsx-ldrb) to generate
# realistic fiber ($\mathbf{f}_0$), sheet ($\mathbf{s}_0$), and sheet-normal ($\mathbf{n}_0$) fields.
#
# **Physics**:
# We solve the static balance of linear momentum:
#
# $$
# \nabla \cdot \mathbf{P} = \mathbf{0} \quad \text{in } \Omega_0
# $$
#
# **Material Model**:
# * **Passive**: Neo-Hookean (or Holzapfel-Ogden).
# * **Active**: Active stress along fibers.
# * **Compressibility**: Either Incompressible or Compressible formulation.
#
# ---

# ## Imports

from pathlib import Path
from mpi4py import MPI
import numpy as np
import dolfinx
import ldrb
import cardiac_geometries
import cardiac_geometries.geometry
import pulse

# ## Geometry and Microstructure
#
# We generate the mesh and fibers if they don't already exist.
#
# ### 2. Fiber Generation (LDRB)
# The LDRB algorithm solves Laplace equations to define transmural and apicobasal coordinates.
# Based on these coordinates, it assigns fiber angles (e.g., +60 to -60 degrees transmurally).

outdir = Path("biv_ellipsoid")
outdir.mkdir(parents=True, exist_ok=True)
geodir = outdir / "geometry"
if not geodir.exists():
    # Generate mesh
    geo = cardiac_geometries.mesh.biv_ellipsoid(outdir=geodir)

    # Run LDRB algorithm
    system = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=geo.markers,
        alpha_endo_lv=60,  # Fiber angle at LV endocardium
        alpha_epi_lv=-60,  # Fiber angle at LV epicardium
        beta_endo_lv=0,    # Sheet angle (0 for now)
        beta_epi_lv=0,
        fiber_space="P_2",
        create_fibers=True,
    )

    # Save microstructure to XDMF/H5
    cardiac_geometries.fibers.utils.save_microstructure(
        mesh=geo.mesh,
        functions=[system.f0, system.s0, system.n0],
        path=geodir / "geometry.bp",
    )

# Load the geometry with fibers
geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)

# Scale the geometry from mm to meters (approximate scale factor)
geo.mesh.geometry.x[:] *= 1.4e-2

# Convert to `pulse.Geometry`
geometry = pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})

# ## Constitutive Model
#
# ### 1. Passive Material
# We use a **Neo-Hookean** model for simplicity in this demo, though Holzapfel-Ogden is also available.
#
# $$
# \Psi_{NH} = \frac{\mu}{2} (I_1 - 3)
# $$
#

material = pulse.NeoHookean(mu=dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(15.0)))

# ### 2. Active Contraction
# We use the **Active Stress** model.
#
# $$
# \mathbf{S}_{active} = T_a (\mathbf{f}_0 \otimes \mathbf{f}_0)
# $$
#
# Here $T_a$ acts only along the fiber direction ($\eta=0$).

Ta = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
active_model = pulse.ActiveStress(geo.f0, activation=Ta)

# ### 3. Compressibility
# We can choose between an **Incompressible** formulation (using a Lagrange multiplier $p$)
# or a **Compressible** formulation (using a penalty function).

incompressible = False

if incompressible:
    comp_model: pulse.Compressibility = pulse.Incompressible()
else:
    # Compressible model with bulk modulus kappa (default in class)
    comp_model = pulse.Compressible()

# ### Assembly
model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# ## Boundary Conditions
#
# ### Neumann BCs: Ventricular Pressures
# We apply separate pressures to the LV and RV endocardiums.
#
# * $P_{LV}$ on $\Gamma_{endo, LV}$
# * $P_{RV}$ on $\Gamma_{endo, RV}$

lvp = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
neumann_lv = pulse.NeumannBC(traction=lvp, marker=geometry.markers["LV"][0])

rvp = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
neumann_rv = pulse.NeumannBC(traction=rvp, marker=geometry.markers["RV"][0])

# ### Robin BC: Pericardial Constraint
# We model the pericardium as a spring-like boundary condition on the epicardium.
#
# $$
# \mathbf{P}\mathbf{N} + k_{per} \mathbf{u} \cdot \mathbf{N} = 0 \quad \text{on } \Gamma_{epi}
# $$
#
# This prevents rigid body motion and mimics the surrounding tissue support.

pericardium_stiffness = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0))
robin_per = pulse.RobinBC(value=pericardium_stiffness, marker=geometry.markers["EPI"][0])

# We collect all BCs.

bcs = pulse.BoundaryConditions(neumann=(neumann_lv, neumann_rv), robin=(robin_per,))

# ## Solving the Problem
#
# We initialize the `StaticProblem`.
# Note: `base_bc=pulse.BaseBC.fixed` will fix the base, preventing movement in all directions.
# Combined with the pericardial spring, this fully constrains the model.

problem = pulse.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs,
    parameters={"base_bc": pulse.BaseBC.fixed},
)

# Initial solve (zero load) to initialize system.
problem.solve()

# ### Phase 1: Passive Inflation
# We inflate both ventricles. Typically $P_{LV} > P_{RV}$.

vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, outdir / "biv_displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)

# Pressures in kPa
lv_pressures = [0.1, 0.5, 1.0]
for i, plv in enumerate(lv_pressures, start=1):
    prv = plv * 0.2  # RV pressure is typically lower
    print(f"Solving: P_LV = {plv:.2f} kPa, P_RV = {prv:.2f} kPa")

    lvp.value = plv
    rvp.value = prv
    problem.solve()
    vtx.write(float(i))

# ### Phase 2: Active Contraction
# We keep the pressure constant and increase active tension $T_a$.

active_tensions = [1.0, 5.0, 10.0] # kPa
start_step = len(lv_pressures) + 1

for i, ta in enumerate(active_tensions, start=start_step):
    print(f"Solving: Ta = {ta:.2f} kPa")
    Ta.value = ta
    problem.solve()
    vtx.write(float(i))

vtx.close()

# ## Visualization
#
# We visualize the final state using PyVista.

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    # Interpolate to CG-1 for plotting
    V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 1, (geometry.mesh.geometry.dim,)))
    uh = dolfinx.fem.Function(V)
    uh.interpolate(problem.u)

    # Setup plotter
    p = pyvista.Plotter()
    topology, cell_types, geometry_data = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry_data)

    # Add reference mesh
    grid["u"] = uh.x.array.reshape((geometry_data.shape[0], 3))
    p.add_mesh(grid, style="wireframe", color="k", opacity=0.3, label="Reference")

    # Add deformed mesh
    warped = grid.warp_by_vector("u", factor=1.0)
    p.add_mesh(warped, show_edges=False, color="blue", opacity=1.0, label="Deformed")

    p.add_legend()
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot(outdir / "biv_ellipsoid_pressure.png")
