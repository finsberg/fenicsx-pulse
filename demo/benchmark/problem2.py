# # Problem 2: Inflation of a ventricle
# In the second problem we will solve the inflation of a ventricle. First we import the necessary libraries
#

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
# Next we will create the geometry and save it in the folder called `lv_ellipsoid`.

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
    )

# If the folder already exist, then we just load the geometry

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)

# Now, lets convert the geometry to a `pulse.Geometry` object.

geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})


# The material model used in this benchmark is the {py:class}`Guccione <pulse.material_models.guccione.Guccione>` model.

material_params = {
    "C": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(10.0)),
    "bf": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0)),
    "bt": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0)),
    "bfs": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0)),
}
material = pulse.Guccione(**material_params)


# There are now active contraction, so we choose a pure passive model

active_model = pulse.active_model.Passive()

# and the model should be incompressible

comp_model = pulse.Incompressible()


# and assembles the `CardiacModel`

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)


# We apply a traction in endocardium

traction = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
neumann = pulse.NeumannBC(
    traction=pulse.Variable(traction, "kPa"),
    marker=geo.markers["ENDO"][0],
)

# and finally combine all the boundary conditions

bcs = pulse.BoundaryConditions(neumann=(neumann,))

# and create a Mixed problem

problem = pulse.StaticProblem(
    model=model, geometry=geometry, bcs=bcs, parameters={"base_bc": pulse.BaseBC.fixed},
)

# Now we can solve the problem
log.set_log_level(log.LogLevel.INFO)

problem.solve()

# Now step up the pressure to 10 kPa starting with an increment of 1 kPa
target_value = 10.0
incr = 1.0

# Here we use a continuation strategy to speed up the convergence

use_continuation = True

old_u = [problem.u.copy()]
old_p = [problem.p.copy()]
old_tractions = [traction.value.copy()]

while traction.value < target_value:
    value = min(traction.value + incr, target_value)
    print(f"Solving problem for traction={value}")

    if use_continuation and len(old_tractions) > 1:
        # Better initial guess
        d = (value - old_tractions[-2]) / (old_tractions[-1] - old_tractions[-2])
        problem.u.x.array[:] = (1 - d) * old_u[-2].x.array + d * old_u[-1].x.array
        problem.p.x.array[:] = (1 - d) * old_p[-2].x.array + d * old_p[-1].x.array

    traction.value = value

    try:
        nit = problem.solve()
    except RuntimeError:
        print("Convergence failed, reducing increment")

        # Reset state and half the increment
        traction.value = old_tractions[-1]
        problem.u.x.array[:] = old_u[-1].x.array
        problem.p.x.array[:] = old_p[-1].x.array
        incr *= 0.5
        problem._init_forms()
    else:
        print(f"Converged in {nit} iterations")
        if nit < 3:
            print("Increasing increment")
            # Increase increment
            incr *= 1.5
        old_u.append(problem.u.copy())
        old_p.append(problem.p.copy())
        old_tractions.append(traction.value.copy())

log.set_log_level(log.LogLevel.INFO)

# Now save the displacement to a file that we can view in Paraview

with dolfinx.io.VTXWriter(geometry.mesh.comm, "problem2.bp", [problem.u], engine="BP4") as vtx:
    vtx.write(0.0)


try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    pyvista.start_xvfb()
    V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 1, (geometry.mesh.geometry.dim,)))
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
        figure_as_array = p.screenshot("problem2.png")


# Finally we can extract the longitudinal position of the endocardial and epicardial apex
# First we create a function to get the coordinates of the apex

U = dolfinx.fem.Function(problem.u.function_space)
U.interpolate(lambda x: (x[0], x[1], x[2]))

endo_apex_coord = pulse.utils.evaluate_at_vertex_tag(U, geo.vfun, geo.markers["ENDOPT"][0])
u_endo_apex = pulse.utils.evaluate_at_vertex_tag(problem.u, geo.vfun, geo.markers["ENDOPT"][0])
endo_apex_pos = pulse.utils.gather_broadcast_array(geo.mesh.comm, endo_apex_coord + u_endo_apex)
print(f"\nGet longitudinal position of endocardial apex: {endo_apex_pos[0, 0]:4f} mm")

epi_apex_coord = pulse.utils.evaluate_at_vertex_tag(U, geo.vfun, geo.markers["EPIPT"][0])
u_epi_apex = pulse.utils.evaluate_at_vertex_tag(problem.u, geo.vfun, geo.markers["EPIPT"][0])
epi_apex_pos = pulse.utils.gather_broadcast_array(geo.mesh.comm, epi_apex_coord + u_epi_apex)
print(f"\nGet longitudinal position of epicardial apex: {epi_apex_pos[0, 0]:4f} mm")
