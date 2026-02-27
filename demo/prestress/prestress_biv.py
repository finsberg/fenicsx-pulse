# # Pre-stressing of a Bi-Ventricular Geometry
#
# In cardiac mechanics simulations, we often start from a geometry acquired from medical imaging (e.g., MRI or CT)
# at a specific point in the cardiac cycle, typically end-diastole. At this point, the ventricles are pressurized
# and thus already deformed. However, standard finite element mechanics assumes the initial geometry is stress-free.
#
# To correctly simulate the mechanics, we need to find the **unloaded reference configuration** corresponding to the
# acquired geometry and the known end-diastolic pressures. This process is often called "pre-stressing" or
# "inverse mechanics".
#
# In this demo, we solve the **Inverse Elasticity Problem (IEP)** as formulated in {cite}`barnafi2024reconstructing`
# for a Bi-Ventricular (BiV) geometry. We formulate the equilibrium equations directly on the known target configuration
# and solve for the "inverse displacement" that maps points back to the stress-free state.
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
import numpy as np
import pulse
import io4dolfinx
import cardiac_geometries
import cardiac_geometries.geometry

# We set up logging to monitor the process.

comm = MPI.COMM_WORLD
logging.basicConfig(level=logging.INFO)
logging.getLogger("scifem").setLevel(logging.WARNING)

# ## 1. Geometry Generation (Target Configuration)
#
# We generate an Bi-Ventricular (BiV) geometry using `cardiac-geometries` which represents our **target**
# geometry (e.g., the end-diastolic state). This geometry is generated from the mean shape of an
# [atlas from the UK Biobank](https://github.com/ComputationalPhysiology/ukb-atlas).
# We also generate the fiber architecture using a [rule-based method](https://github.com/finsberg/fenicsx-ldrb).

mode = -1
std = 0
char_length = 10.0

outdir = Path("biv-prestress")
outdir.mkdir(parents=True, exist_ok=True)
geodir = outdir / "geometry"
if not geodir.exists():
    comm.barrier()
    geo = cardiac_geometries.mesh.ukb(
        outdir=geodir,
        comm=comm,
        mode=mode,
        std=std,
        case="ED",
        create_fibers=True,
        char_length_max=char_length,
        char_length_min=char_length,
        clipped=True,
        fiber_angle_endo=60,
        fiber_angle_epi=-60,
        fiber_space="Quadrature_6",
    )

# Load the geometry and convert it to `pulse.HeartGeometry`.

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)

geometry = pulse.HeartGeometry.from_cardiac_geometries(
    geo, metadata={"quadrature_degree": 6},
)


# ## 2. Constitutive Model
#
# We define the material properties. For this example, we use the **Usyk** model {cite}`usyk2002computational`,
# which is a transversely isotropic hyperelastic model.
#
# We use an incompressible formulation.

# ## 3. Boundary Conditions
#
# We apply the loading and constraints that define the target state.
#
# * **Neumann (Pressure)**: The target end-diastolic pressure is applied to the endocardial surfaces.
#   We apply $P_{LV} = 2000$ Pa to the LV and assume $P_{RV} = 0.5 P_{LV}$.
# * **Robin (Springs)**: To prevent rigid body motion and mimic the pericardial constraint, we apply spring
#   conditions to the epicardium and the base.


def setup_problem(geometry, f0, s0, target_pressure=2000.0):
    material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material = pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)

    comp = pulse.compressibility.Incompressible()

    model = pulse.CardiacModel(
        material=material,
        compressibility=comp,
        active=pulse.active_model.Passive(),
    )
    pressure_lv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "Pa")
    pressure_rv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "Pa")

    # Epicardial and Basal springs
    alpha_epi = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e3)),
        "Pa / m",
    )
    robin_epi = pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])
    alpha_base = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2e5)),
        "Pa / m",
    )
    robin_base = pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])

    # Endocardial Pressures
    neumann_lv = pulse.NeumannBC(traction=pressure_lv, marker=geometry.markers["LV"][0])
    neumann_rv = pulse.NeumannBC(traction=pressure_rv, marker=geometry.markers["RV"][0])

    bcs = pulse.BoundaryConditions(
        neumann=(neumann_lv, neumann_rv), robin=(robin_epi, robin_base),
    )
    return model, bcs, pressure_lv, pressure_rv, target_pressure


model, bcs, pressure_lv, pressure_rv, target_pressure = setup_problem(
    geometry, geo.f0, geo.s0,
)

# ## 4. Solving the Inverse Elasticity Problem (IEP)
#
# The `PrestressProblem` class sets up the inverse formulation.
#
# We solve the problem incrementally using **continuation (load stepping)**. We ramp the pressures
# from 0 to the target pressures.
#
# **Note**: The geometry `geometry.mesh` does *not* change during this loop. It remains the target configuration.
# The solution `prestress_problem.u` is updated to reflect the inverse displacement required to map
# from this fixed target back to the corresponding reference state for the *current* pressure load.

prestress_fname = outdir / "prestress_biv_inverse.bp"
if not prestress_fname.exists():
    prestress_problem = pulse.unloading.PrestressProblem(
        geometry=geometry,
        model=model,
        bcs=bcs,
        parameters={"u_space": "P_2"},
        targets=[
            pulse.unloading.TargetPressure(
                traction=pressure_lv, target=target_pressure, name="LV",
            ),
            pulse.unloading.TargetPressure(
                traction=pressure_rv, target=target_pressure * 0.5, name="RV",
            ),
        ],
        ramp_steps=20,
    )

    u_pre = prestress_problem.unload()

    # We save the computed inverse displacement field $\mathbf{u}_{pre}$.
    # This field maps: **Target Geometry** ($\mathbf{x}$) $\to$ **Reference Geometry** ($\mathbf{X}$).
    io4dolfinx.write_function_on_input_mesh(
        prestress_fname,
        u_pre,
        time=0.0,
        name="u_pre",
    )

    with dolfinx.io.VTXWriter(
        comm,
        outdir / "prestress_biv_backward.bp",
        [u_pre],
        engine="BP4",
    ) as vtx:
        vtx.write(0.0)

    # We can visualize the inverse displacement field.
    try:
        import pyvista
    except ImportError:
        print("Pyvista is not installed")
    else:
        # Create plotter and pyvista grid
        p = pyvista.Plotter()

        topology, cell_types, vtk_geometry = dolfinx.plot.vtk_mesh(u_pre.function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, vtk_geometry)

        # Attach vector values to grid
        grid["u"] = u_pre.x.array.reshape((vtk_geometry.shape[0], 3))
        actor_0 = p.add_mesh(grid, style="wireframe", color="k")

        # Warp the mesh by the inverse displacement vector to visualize the unloaded Reference Configuration
        warped = grid.warp_by_vector("u", factor=1.0)
        actor_1 = p.add_mesh(
            warped, color="red", opacity=0.8, label="Recovered Reference",
        )

        p.add_legend()
        p.show_axes()
        if not pyvista.OFF_SCREEN:
            p.show()
        else:
            figure_as_array = p.screenshot("prestress_inverse_displacement.png")


V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 2, (3,)))
u_pre = dolfinx.fem.Function(V)
io4dolfinx.read_function(
    prestress_fname,
    u_pre,
    time=0.0,
    name="u_pre",
)

# ## 5. Verification (Forward Problem)
#
# To verify the result, we perform a full explicit deformation to the reference configuration.
#
# 1.  **Deform Mesh**: We apply the inverse displacement $\mathbf{u}_{pre}$ to the mesh nodes: $\mathbf{X} = \mathbf{x} + \mathbf{u}_{pre}$.
#     The mesh now represents the **Reference (Unloaded)** configuration.
# 2.  **Map Fibers**: The fiber fields must be mapped (pulled back) to this new reference configuration to ensure correct material orientation.
# 3.  **Forward Solve**: We assume this new geometry is stress-free (pressure=0). We then ramp the pressure back up to the target value.
#
# If the prestressing was successful, the deformed geometry at full load should match the original target mesh.

print("\nDeforming mesh to recovered reference configuration...")
geometry.deform(u_pre)

# Map fiber fields to the new configuration
print("Mapping fiber fields...")
f0 = pulse.utils.map_vector_field(f=geo.f0, u=u_pre, normalize=True, name="f0_unloaded")
s0 = pulse.utils.map_vector_field(f=geo.s0, u=u_pre, normalize=True, name="s0_unloaded")

# Setup the problem on the REFERENCE geometry with MAPPED fibers
(
    model_unloaded,
    bcs_unloaded,
    pressure_lv_unloaded,
    pressure_biv_unloaded,
    target_pressure_unloaded,
) = setup_problem(geometry, f0, s0)

# Initialize Forward Problem using the UNLOADED model and BCs
forward_problem = pulse.StaticProblem(
    model=model_unloaded,
    geometry=geometry,
    bcs=bcs_unloaded,
    parameters={"u_space": "P_2"},
)

import shutil

shutil.rmtree(outdir / "prestress_biv.bp", ignore_errors=True)
vtx = dolfinx.io.VTXWriter(
    comm, outdir / "prestress_biv.bp", [forward_problem.u], engine="BP4",
)

# Step 1: Initial State (Zero Pressure)
# Since the mesh is now the Reference configuration, zero pressure means zero displacement.
print("\nSolving forward problem: Initial state (Reference)...")
pressure_lv_unloaded.assign(0.0)
pressure_biv_unloaded.assign(0.0)
forward_problem.solve()
vtx.write(0.0)

# Step 2: Reload to Target Pressure
# We apply the original target pressures. The mesh should deform from Reference -> Target.
print("Solving forward problem: Reloading to target pressure...")
ramp_steps = 20
for ramp in np.linspace(0.0, 1.0, ramp_steps):
    current_p = target_pressure_unloaded * ramp
    pressure_lv_unloaded.assign(current_p)
    pressure_biv_unloaded.assign(current_p * 0.5)

    print(f"Solving for pressure fraction: {ramp:.2f} (P_LV = {current_p:.2f} Pa)")
    forward_problem.solve()

    vtx.write(ramp * ramp_steps + 1)

vtx.close()

# Save final state. This displacement maps **Reference** -> **Recovered Target**.
with dolfinx.io.VTXWriter(
    comm, outdir / "prestress_biv_forward.bp", [forward_problem.u], engine="BP4",
) as vtx:
    vtx.write(0.0)

print(
    "Done. You can now verify that the final geometry matches the original target geometry in Paraview.",
)

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    # Create plotter
    p = pyvista.Plotter()

    topology, cell_types, vtk_geometry = dolfinx.plot.vtk_mesh(forward_problem.u_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, vtk_geometry)

    # Attach the final displacement result
    grid["u"] = forward_problem.u.x.array.reshape((vtk_geometry.shape[0], 3))

    # Reference Configuration (Wireframe) - This is the mesh we started the forward solve with
    actor_0 = p.add_mesh(grid, style="wireframe", color="k", label="Reference Config")

    # Recovered Target (Deformed by final u)
    # Ideally, we should also load the original target mesh for comparison, but here we show the deformed result.
    warped = grid.warp_by_vector("u", factor=1.0)
    actor_1 = p.add_mesh(warped, color="blue", opacity=0.5, label="Recovered Target")

    p.add_legend()
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        figure_as_array = p.screenshot("prestress_forward_displacement.png")

# ## References
# ```{bibliography}
# :filter: docname in docnames
# ```
