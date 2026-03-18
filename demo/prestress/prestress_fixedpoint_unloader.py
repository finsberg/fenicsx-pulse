# # Pre-stressing of a Left Ventricle Geometry using Fixed-Point Unloading
#
# In cardiac mechanics simulations, we often start from a geometry acquired from medical imaging (e.g., MRI or CT)
# at a specific point in the cardiac cycle, typically end-diastole. At this point, the ventricle is pressurized
# and thus already deformed. However, standard finite element mechanics assumes the initial geometry is stress-free.
#
# To correctly simulate the mechanics, we need to find the **unloaded reference configuration** corresponding to the
# acquired geometry and the known end-diastolic pressure. This process is often called "pre-stressing" or
# "inverse mechanics".
#
# In this demo, we solve the inverse problem using a **Fixed-Point Iteration** (also known as the Backward Displacement Method o
# Sellier's method) {cite}`SELLIER20111461`.
# Unlike the Inverse Elasticity Problem (IEP) which formulates equilibrium on the target configuration, this method
# iteratively updates the reference coordinates $\mathbf{X}$ by subtracting the displacement $\mathbf{u}$ computed from a
# forward solve.
#
# ## Mathematical Formulation
#
# Let $\Omega_t$ be the known **target** (loaded) configuration with coordinates $\mathbf{x}_{target}$.
# We seek the **reference** (unloaded) configuration $\Omega_0$ with coordinates $\mathbf{X}$.
#
# The algorithm proceeds as follows:
# 1. Initialize reference geometry: $\mathbf{X}_0 = \mathbf{x}_{target}$.
# 2. For iteration $k=0, 1, \dots$:
#    a. Solve the **Forward** mechanics problem on the geometry defined by $\mathbf{X}_k$ to get displacement $\mathbf{u}_k$.
#    b. Update the reference geometry:
#       $ \mathbf{X}_{k+1} = \mathbf{x}_{target} - \mathbf{u}_k $
#    c. Check convergence: $||\mathbf{X}_{k+1} - \mathbf{X}_k|| < \text{tol}$.
#
# In `fenicsx-pulse`, the `FixedPointUnloader` class automates this iterative process.
#
# ---

# ## Imports

from pathlib import Path
import math
from mpi4py import MPI
import dolfinx
import logging
import numpy as np
import pulse
import pulse.unloading
import io4dolfinx
import cardiac_geometries
import cardiac_geometries.geometry

# We set up logging to monitor the process.

comm = MPI.COMM_WORLD
logging.basicConfig(level=logging.INFO)
logging.getLogger("scifem").setLevel(logging.WARNING)

# ## 1. Geometry Generation (Target Configuration)
#
# We generate an idealized **Left Ventricular (LV)** geometry using `cardiac-geometries` which represents our
# **target** geometry (e.g., the end-diastolic state).
# We also generate the fiber architecture.

outdir = Path("lv_fixedpoint_unloader")
outdir.mkdir(parents=True, exist_ok=True)
geodir = outdir / "geometry"
if not geodir.exists():
    comm.barrier()
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        create_fibers=True,
        fiber_space="Quadrature_6",
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

geometry = pulse.HeartGeometry.from_cardiac_geometries(
    geo, metadata={"quadrature_degree": 6},
)

# ## 2. Constitutive Model & Boundary Conditions
#
# We define a helper function to setup the mechanical model.
#
# * **Material**: Usyk model (transversely isotropic).
# * **Compressibility**: Compressible formulation with high bulk modulus.
# * **BCs**:
#     * **Neumann**: Endocardial pressure (Target = 2000 Pa).
#     * **Robin**: Epicardial and Basal springs to mimic tissue support.


def setup_problem(geometry, f0, s0, n0, target_pressure=2000.0):
    material = pulse.material_models.Usyk(f0=f0, s0=s0, n0=n0)
    comp = pulse.compressibility.Compressible3(kappa=pulse.Variable(5e4, "Pa"))

    model = pulse.CardiacModel(
        material=material,
        compressibility=comp,
        active=pulse.active_model.Passive(),
    )
    pressure = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "Pa")

    # Epicardial and Basal springs
    alpha_epi = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2e5)),
        "Pa / m",
    )
    robin_epi = pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])
    alpha_base = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2e5)),
        "Pa / m",
    )
    robin_base = pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])

    # Endocardial Pressure
    neumann = pulse.NeumannBC(traction=pressure, marker=geometry.markers["ENDO"][0])

    bcs = pulse.BoundaryConditions(neumann=(neumann,), robin=(robin_epi, robin_base))
    return model, bcs, pressure, target_pressure


# Initialize model with fibers from the target geometry

model, bcs, pressure, target_pressure = setup_problem(geometry, geo.f0, geo.s0, geo.n0)

# Define the loading target for the unloader
target_pressure_obj = pulse.unloading.TargetPressure(
    traction=pressure, target=target_pressure,
)

# ## 3. Solving the Inverse Problem (Fixed Point Iteration)
#
# The `FixedPointUnloader` automates the iterative process of finding the stress-free reference configuration.
#
# * `ramp_steps`: Number of load steps for the forward solver within each iteration (improves stability).
# * `max_iter`: Maximum number of fixed-point iterations.
# * `tol`: Geometric convergence tolerance.

prestress_fname = outdir / "prestress_lv_inverse.bp"

if not prestress_fname.exists():
    prestress_problem = pulse.unloading.FixedPointUnloader(
        model=model,
        geometry=geometry,
        bcs=bcs,
        problem_parameters={"u_space": "P_2"},
        targets=[target_pressure_obj],
        unload_parameters={"ramp_steps": 20, "max_iter": 15, "tol": 1e-4},
    )

    # Execute unloading
    # This returns the inverse displacement u_pre (mapping Target -> Reference)
    u_pre = prestress_problem.unload()

    # Save the result
    io4dolfinx.write_function_on_input_mesh(
        prestress_fname,
        u_pre,
        time=0.0,
        name="u_pre",
    )

    with dolfinx.io.VTXWriter(
        comm,
        outdir / "prestress_lv_backward.bp",
        [u_pre],
        engine="BP4",
    ) as vtx:
        vtx.write(0.0)

    # Visualization
    try:
        import pyvista
    except ImportError:
        print("Pyvista is not installed")
    else:
        p = pyvista.Plotter()

        topology, cell_types, vtk_geometry = dolfinx.plot.vtk_mesh(u_pre.function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, vtk_geometry)

        grid["u"] = u_pre.x.array.reshape((vtk_geometry.shape[0], 3))
        actor_0 = p.add_mesh(
            grid, style="wireframe", color="k", label="Target (Loaded)",
        )

        # Warp by u_pre to show the recovered Reference (Unloaded) configuration
        warped = grid.warp_by_vector("u", factor=1.0)
        actor_1 = p.add_mesh(
            warped, color="red", opacity=0.8, label="Reference (Unloaded)",
        )

        p.add_legend()
        p.show_axes()
        if not pyvista.OFF_SCREEN:
            p.show()
        else:
            figure_as_array = p.screenshot("lv_prestress_inverse_displacement.png")


# ## 4. Verification (Forward Problem)
#
# To verify the result, we perform a explicit deformation to the reference configuration and solve the forward problem.
#
# 1.  **Deform Mesh**: Apply `u_pre` to the mesh nodes. The mesh now represents the **Reference** configuration.
# 2.  **Map Fibers**: Pull back the fiber fields to the reference configuration.
# 3.  **Forward Solve**: Ramp pressure from 0 to `target_pressure`.
#
# If successful, the final deformed geometry should match the original target mesh.

# Reload u_pre
V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 2, (3,)))
u_pre = dolfinx.fem.Function(V)
io4dolfinx.read_function(
    prestress_fname,
    u_pre,
    time=0.0,
    name="u_pre",
)

print("\nDeforming mesh to recovered reference configuration...")
geometry.deform(u_pre)

# Map fiber fields
print("Mapping fiber fields...")
f0 = pulse.utils.map_vector_field(f=geo.f0, u=u_pre, normalize=True, name="f0_unloaded")
s0 = pulse.utils.map_vector_field(f=geo.s0, u=u_pre, normalize=True, name="s0_unloaded")
n0 = pulse.utils.map_vector_field(f=geo.n0, u=u_pre, normalize=True, name="n0_unloaded")

# Setup Forward Problem on Reference Mesh
model_unloaded, bcs_unloaded, pressure_unloaded, target_pressure_unloaded = (
    setup_problem(geometry, f0, s0, n0)
)

forward_problem = pulse.StaticProblem(
    model=model_unloaded,
    geometry=geometry,
    bcs=bcs_unloaded,
    parameters={"u_space": "P_2"},
)

# Solve
import shutil

shutil.rmtree(outdir / "prestress_lv.bp", ignore_errors=True)
vtx = dolfinx.io.VTXWriter(
    comm, outdir / "prestress_lv.bp", [forward_problem.u], engine="BP4",
)

print("\nSolving forward problem: Initial state (Reference)...")
pressure_unloaded.assign(0.0)
forward_problem.solve()
vtx.write(0.0)

print("Solving forward problem: Reloading to target pressure...")
ramp_steps = 20
for ramp in np.linspace(0.0, 1.0, ramp_steps):
    current_p = target_pressure_unloaded * ramp
    pressure_unloaded.assign(current_p)

    print(f"Solving for pressure fraction: {ramp:.2f} (P = {current_p:.2f} Pa)")
    forward_problem.solve()
    vtx.write(ramp * ramp_steps + 1)

vtx.close()

print(
    "Done. You can now verify that the final geometry matches the original target geometry.",
)

# Visualization of Forward Result
try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    p = pyvista.Plotter()
    topology, cell_types, vtk_geometry = dolfinx.plot.vtk_mesh(forward_problem.u_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, vtk_geometry)

    grid["u"] = forward_problem.u.x.array.reshape((vtk_geometry.shape[0], 3))

    # Reference (Wireframe)
    actor_0 = p.add_mesh(grid, style="wireframe", color="k", label="Reference Config")

    # Recovered Target (Deformed)
    warped = grid.warp_by_vector("u", factor=1.0)
    actor_1 = p.add_mesh(warped, color="blue", opacity=0.5, label="Recovered Target")

    p.add_legend()
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        figure_as_array = p.screenshot("lv_prestress_forward_displacement.png")

# ## References
# ```{bibliography}
# :filter: docname in docnames
# ```
