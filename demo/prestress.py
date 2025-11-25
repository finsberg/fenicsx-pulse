# # Pre-stressing of a Left Ventricle Ellipsoid
#
# In cardiac mechanics simulations, we often start from a geometry acquired from medical imaging (e.g., MRI or CT)
# at a specific point in the cardiac cycle, typically end-diastole. At this point, the ventricle is pressurized
# and thus already deformed. However, standard finite element mechanics assumes the initial geometry is stress-free.
#
# To correctly simulate the mechanics, we need to find the **unloaded reference configuration** corresponding to the
# acquired geometry and the known end-diastolic pressure. This process is often called "pre-stressing" or
# "inverse mechanics".
#
# In this demo, we solve the **Inverse Elasticity Problem (IEP)** as formulated in {cite}`barnafi2024reconstructing`.
# We formulate the equilibrium equations directly on the known target configuration and solve for the
# "inverse displacement" that maps points back to the stress-free state.
#
# ## Mathematical Formulation
#
# Let $\Omega_t$ be the known **target** (loaded) configuration and $\Omega_0$ be the unknown **reference** (unloaded) configuration.
# We seek a mapping $\boldsymbol{\chi}^{-1}: \Omega_t \to \Omega_0$.
#
# We define the **inverse displacement** field $\mathbf{u}$ on $\Omega_t$ such that:
#
# $$
# \mathbf{X} = \mathbf{x} + \mathbf{u}(\mathbf{x})
# $$
#
# where $\mathbf{x} \in \Omega_t$ are the current coordinates and $\mathbf{X} \in \Omega_0$ are the reference coordinates.
#
# ### Kinematics
# The inverse deformation gradient $\mathbf{f}$ is defined as:
#
# $$
# \mathbf{f} = \frac{\partial \mathbf{X}}{\partial \mathbf{x}} = \mathbf{I} + \nabla_{\mathbf{x}} \mathbf{u}
# $$
#
# The physical deformation gradient $\mathbf{F}$ (mapping reference to target) is the inverse of $\mathbf{f}$:
#
# $$
# \mathbf{F} = \frac{\partial \mathbf{x}}{\partial \mathbf{X}} = \mathbf{f}^{-1} = (\mathbf{I} + \nabla_{\mathbf{x}} \mathbf{u})^{-1}
# $$
#
# The Jacobian is $J = \det \mathbf{F} = (\det \mathbf{f})^{-1}$.
#
# ### Equilibrium
# We solve the balance of linear momentum. The weak form is pulled back from the reference configuration to the
# target configuration $\Omega_t$ (where the mesh is defined).
#
# $$
# \int_{\Omega_t} \sigma : \nabla_{\mathbf{x}} \mathbf{v} \, dx - \int_{\partial \Omega_t} \mathbf{t} \cdot \mathbf{v} \, ds = 0
# $$
#
# Here $\sigma = J^{-1} \mathbf{P} \mathbf{F}^T$ is the Cauchy stress, and $\mathbf{P}$ is the First Piola-Kirchhoff stress
# computed from the material model using $\mathbf{F}$.
#
# In `fenicsx-pulse`, the `PrestressProblem` class automates this specific formulation.
#
# ---

# ## Imports

from pathlib import Path
from mpi4py import MPI
import dolfinx
import logging
import math
import numpy as np
import pulse
import pulse.prestress
import cardiac_geometries
import cardiac_geometries.geometry

# We set up logging to monitor the process.

comm = MPI.COMM_WORLD
logging.basicConfig(level=logging.INFO)
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

# ## 1. Geometry Generation (Target Configuration)
#
# We generate an idealized LV ellipsoid which represents our **target** geometry (e.g., the end-diastolic state).
# We also generate the fiber architecture.

geodir = Path("lv_ellipsoid-prestress")
if not geodir.exists():
    comm.barrier()
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        create_fibers=True,
        fiber_space="P_2",
        r_short_endo=0.025,
        r_short_epi=0.035,
        r_long_endo=0.09,
        r_long_epi=0.097,
        psize_ref=0.03,
        mu_apex_endo=-math.pi,
        mu_base_endo=-math.acos(5 / 17),
        mu_apex_epi=-math.pi,
        mu_base_epi=-math.acos(5 / 20),
        comm=comm,
        fiber_angle_epi=-60,
        fiber_angle_endo=60,
    )

# Load the geometry and convert it to `pulse.HeartGeometry`.

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)

geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

# ## 2. Constitutive Model
#
# We define the material properties. For this example, we use the **Usyk** model {cite}`usyk2002computational`,
# which is a transversely isotropic hyperelastic model.
#
# Note that we use a compressible formulation with a high bulk modulus.

material = pulse.material_models.Usyk(f0=geo.f0, s0=geo.s0, n0=geo.n0)
comp = pulse.compressibility.Compressible3(kappa=pulse.Variable(5e4, "Pa"))

model = pulse.CardiacModel(
    material=material,
    compressibility=comp,
    active=pulse.active_model.Passive(),
)

# ## 3. Boundary Conditions
#
# We apply the loading and constraints that define the target state.
#
# * **Neumann (Pressure)**: The target end-diastolic pressure $P_{target} = 2000$ Pa applied to the endocardium.
# * **Robin (Springs)**: To prevent rigid body motion and mimic the pericardium.

target_pressure = 2000.0
pressure = pulse.Variable(dolfinx.fem.Constant(geo.mesh, 0.0), "Pa") # Variable to ramp up pressure

# Epicardial springs
alpha_epi = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2e5)),
    "Pa / m",
)
robin_epi = pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])

alpha_epi_perp = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2e5 / 10)),
    "Pa / m",
)
robin_epi_perp = pulse.RobinBC(
    value=alpha_epi_perp, marker=geometry.markers["EPI"][0], perpendicular=True,
)

# Endocardial Pressure
neumann = pulse.NeumannBC(traction=pressure, marker=geometry.markers["ENDO"][0])

bcs = pulse.BoundaryConditions(neumann=(neumann,), robin=(robin_epi, robin_epi_perp))

# ## 4. Solving the Inverse Elasticity Problem (IEP)
#
# The `PrestressProblem` class sets up the inverse formulation.
#
# We solve the problem incrementally using **continuation (load stepping)**. We ramp the pressure
# from 0 to the target pressure.
#
# **Note**: The geometry `geometry.mesh` does *not* change during this loop. It remains the target configuration.
# The solution `prestress_problem.u` is updated to reflect the displacement required to map
# from this fixed target back to the corresponding reference state for the *current* pressure.

prestress_problem = pulse.prestress.PrestressProblem(
    geometry=geometry,
    model=model,
    bcs=bcs,
    parameters={"u_space": "P_2"},
)

ramp_steps = 5
print(f"Starting prestress algorithm with target pressure {target_pressure} Pa")

for ramp in np.linspace(0.0, 1.0, ramp_steps):
    current_p = target_pressure * ramp
    pressure.assign(current_p)
    print(f"Solving for pressure fraction: {ramp:.2f} (P = {current_p:.2f} Pa)")
    prestress_problem.solve()

# We save the computed inverse displacement field $\mathbf{u}$.
# This field maps: **Target Geometry** ($\mathbf{x}$) $\to$ **Reference Geometry** ($\mathbf{X}$).

with dolfinx.io.VTXWriter(
    comm, "prestress_backward.bp", [prestress_problem.u], engine="BP4",
) as vtx:
    vtx.write(0.0)

# We can also visualize comparison between the original target geometry and the recovered unloaded geometry.

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    # Create plotter and pyvista grid
    p = pyvista.Plotter()

    topology, cell_types, vtk_geometry = dolfinx.plot.vtk_mesh(prestress_problem.u_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, vtk_geometry)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = prestress_problem.u.x.array.reshape((vtk_geometry.shape[0], 3))
    actor_0 = p.add_mesh(grid, style="wireframe", color="k")

    # Warp the mesh by the displacement vector to visualize deformation
    warped = grid.warp_by_vector("u", factor=1.0)
    actor_1 = p.add_mesh(warped, color="red", opacity=0.8)

    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        figure_as_array = p.screenshot("prestress_inverse_displacement.png")


# ## 5. Verification (Forward Problem)
#
# To verify the result, we now explicitly deform the geometry to the recovered reference configuration.
#
# $$
# \mathbf{X}_{ref} = \mathbf{X}_{target} + \mathbf{u}_{inverse}
# $$
#
# *Note: `geometry.deform(u)` adds `u` to the nodal coordinates. Since we defined $\mathbf{X} = \mathbf{x} + \mathbf{u}$, this correctly moves the mesh to $\Omega_0$.*

print("\nDeforming mesh to recovered reference configuration...")
geometry.deform(prestress_problem.u)

# We then solve the **Forward** mechanics problem starting from this new reference configuration.
# If we apply the target pressure to this reference geometry, the resulting deformed state
# should match our original target geometry.

forward_problem = pulse.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs,
    parameters={"u_space": "P_2"},
)

print("Solving forward problem to recover target geometry...")
# Ramp pressure again for robust forward solution
for ramp in np.linspace(0.0, 1.0, ramp_steps):
    current_p = target_pressure * ramp
    pressure.assign(current_p)
    forward_problem.solve()

# Save the forward displacement. This maps **Reference Geometry** $\to$ **Recovered Target**.
with dolfinx.io.VTXWriter(comm, "prestress_forward.bp", [forward_problem.u], engine="BP4") as vtx:
    vtx.write(0.0)

print("Done. You can now compare the original geometry with the recovered geometry in Paraview.")

# We can also visualize comparison between the original target geometry and the recovered unloaded geometry with the target pressure applied.

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    # Create plotter and pyvista grid
    p = pyvista.Plotter()

    topology, cell_types, vtk_geometry = dolfinx.plot.vtk_mesh(prestress_problem.u_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, vtk_geometry)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = forward_problem.u.x.array.reshape((vtk_geometry.shape[0], 3))
    actor_0 = p.add_mesh(grid, style="wireframe", color="k")

    # Warp the mesh by the displacement vector to visualize deformation
    warped = grid.warp_by_vector("u", factor=1.0)
    actor_1 = p.add_mesh(warped, color="red", opacity=0.8)

    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        figure_as_array = p.screenshot("prestress_forward_displacement.png")


# ## References
# ```{bibliography}
# :filter: docname in docnames
#
