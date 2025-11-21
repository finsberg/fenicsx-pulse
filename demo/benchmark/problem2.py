# # Problem 2: Passive Inflation of a Ventricle
#
# This example implements Problem 2 from the cardiac mechanics benchmark suite [Land et al. 2015].
#
# ## Problem Description
#
# **Geometry**:
# A truncated thick-walled ellipsoid representing an idealized Left Ventricle (LV).
#
# **Material**:
# Isotropic Guccione material.
# * Constitutive parameters: $C = 10.0$ kPa, $b_f = b_t = b_{fs} = 1.0$.
# * The material is incompressible.
#
# **Boundary Conditions**:
# * **Base**: The basal plane is fixed ($u_x = u_y = u_z = 0$).
#     *Note: The original benchmark specifies a sliding boundary condition in the plane, but fixing it is a common variation for stability in simple tests. We use a fixed base here.*
# * **Neumann**: A pressure load $P$ is applied to the endocardial surface. The pressure increases to 10 kPa.
#
# **Target Quantity**:
# The inflation of the ventricle and the corresponding volume change.
#
# ---

from pathlib import Path
from mpi4py import MPI
from dolfinx import log
import logging
import dolfinx
import numpy as np
import math
import cardiac_geometries
import pulse

logging.basicConfig(level=logging.INFO)

# ## 1. Geometry
#
# We generate the truncated ellipsoid using `cardiac_geometries`.
# * $r_{short}^{endo} = 7$ mm, $r_{short}^{epi} = 10$ mm
# * $r_{long}^{endo} = 17$ mm, $r_{long}^{epi} = 20$ mm
# * Base cut at specific coordinate.

comm = MPI.COMM_WORLD
geodir = Path("lv_ellipsoid-problem2")

if not geodir.exists():
    comm.barrier()
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        r_short_endo=7.0,
        r_short_epi=10.0,
        r_long_endo=17.0,
        r_long_epi=20.0,
        mu_apex_endo=-math.pi,
        mu_base_endo=-math.acos(5 / 17),
        mu_apex_epi=-math.pi,
        mu_base_epi=-math.acos(5 / 20),
        comm=comm,
        create_fibers=False,  # Isotropic problem, fibers not strictly needed
    )

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)

# We convert to `pulse.Geometry`. We assume the mesh units are mm (consistent with parameters).
geometry = pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})

# ## 2. Constitutive Model
#
# **Isotropic Guccione Model**:
# By setting $b_f = b_t = b_{fs} = 1.0$, the exponent $Q$ becomes $Q = (E_{11}^2 + E_{22}^2 + E_{33}^2 + 2E_{12}^2 + \dots) = \text{tr}(\mathbf{E}^2)$, making the model isotropic.

material_params = {
    "C": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(10.0)),
    "bf": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0)),
    "bt": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0)),
    "bfs": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0)),
}

# For an isotropic material, the fiber direction vectors don't affect the energy (as b parameters are equal),
# but the class requires them. We can use dummy fields or the ones from the mesh.
material = pulse.Guccione(**material_params)

active_model = pulse.active_model.Passive()
comp_model = pulse.Incompressible()

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# ## 3. Boundary Conditions
#
# * **Pressure**: Applied to the `ENDO` surface.
# * **Base**: Fixed (Dirichlet).

pressure = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
neumann = pulse.NeumannBC(
    traction=pulse.Variable(pressure, "kPa"),
    marker=geo.markers["ENDO"][0],
)

bcs = pulse.BoundaryConditions(neumann=(neumann,))

# ## 4. Solving
#
# We specify `base_bc=pulse.BaseBC.fixed` to clamp the base.

problem = pulse.StaticProblem(
    model=model, geometry=geometry, bcs=bcs, parameters={"base_bc": pulse.BaseBC.fixed},
)

log.set_log_level(log.LogLevel.INFO)

# ### Continuation Solver
# We ramp the pressure up to 10 kPa. We use a custom continuation loop to handle the nonlinearity.

target_pressure = 10.0
d_pressure = 1.0
current_pressure = 0.0

# Initial solve
problem.solve()

while current_pressure < target_pressure:
    current_pressure += d_pressure
    if current_pressure > target_pressure:
        current_pressure = target_pressure

    print(f"Solving for Pressure: {current_pressure:.2f} kPa")
    pressure.value = current_pressure

    try:
        num_iters = problem.solve()
        print(f"  Converged in {num_iters} iterations.")
    except RuntimeError:
        print("  Solver failed. Retrying with smaller step could be implemented here.")
        break

# ## 5. Post-processing

with dolfinx.io.VTXWriter(geometry.mesh.comm, "problem2.bp", [problem.u], engine="BP4") as vtx:
    vtx.write(0.0)

# Visualization
try:
    import pyvista
except ImportError:
    pass
else:
    V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 1, (geometry.mesh.geometry.dim,)))
    uh = dolfinx.fem.Function(V)
    uh.interpolate(problem.u)

    p = pyvista.Plotter()
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))

    p.add_mesh(grid, style="wireframe", color="k", opacity=0.3, label="Reference")
    warped = grid.warp_by_vector("u", factor=1.0)
    p.add_mesh(warped, show_edges=True, color="firebrick", label="Inflated")

    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot("problem2.png")

# Check apex displacement
U = dolfinx.fem.Function(problem.u.function_space)
U.interpolate(lambda x: (x[0], x[1], x[2]))

# Locate apex (assumed to be the point with min z? or max z depending on orientation)
# In this mesh generation, the apex is typically at the bottom.
endo_apex_coord = pulse.utils.evaluate_at_vertex_tag(U, geo.ffun, geo.markers["ENDOPT"][0])
u_endo_apex = pulse.utils.evaluate_at_vertex_tag(problem.u, geo.ffun, geo.markers["ENDOPT"][0])
endo_apex_pos = pulse.utils.gather_broadcast_array(geo.mesh.comm, endo_apex_coord + u_endo_apex)

print(f"\nEndocardial Apex Position: {endo_apex_pos}")
