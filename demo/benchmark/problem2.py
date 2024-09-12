# # Problem 2: Inflation of a ventricle
# In the second problem we will solve the inflation of a ventricle. First we import the necessary libraries
#

from pathlib import Path
from mpi4py import MPI
from dolfinx import log
import dolfinx
import numpy as np
import math
import cardiac_geometries
import fenicsx_pulse



# Next we will create the geometry and save it in the folder called `lv_ellipsoid`.

comm = MPI.COMM_WORLD
geodir = Path("lv_ellipsoid")
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

# Now, lets convert the geometry to a `fenicsx_pulse.Geometry` object.

geometry = fenicsx_pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})


# The material model used in this benchmark is the {py:class}`Guccione <fenicsx_pulse.material_models.guccione.Guccione>` model.

material_params = {
    "C": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(10.0)),
    "bf": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0)),
    "bt": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0)),
    "bfs": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0)),
}
material = fenicsx_pulse.Guccione(**material_params)


# There are now active contraction, so we choose a pure passive model

active_model = fenicsx_pulse.active_model.Passive()

# and the model should be incompressible

comp_model = fenicsx_pulse.Incompressible()


# and assembles the `CardiacModel`

model = fenicsx_pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)


# Next we need to apply some boundary conditions. For the Dirichlet BC we can supply a function that takes as input the state space and returns a list of `DirichletBC`. We will fix the base in all directions. Note that the displacement is in the first subspace since we use an incompressible model. The hydrostatic pressure is in the second subspace

def dirichlet_bc(
    state_space: dolfinx.fem.FunctionSpace,
) -> list[dolfinx.fem.bcs.DirichletBC]:
    V, _ = state_space.sub(0).collapse()
    facets = geometry.facet_tags.find(
        geo.markers["BASE"][0],
    )  # Specify the marker used on the boundary
    geometry.mesh.topology.create_connectivity(
        geometry.mesh.topology.dim - 1,
        geometry.mesh.topology.dim,
    )
    dofs = dolfinx.fem.locate_dofs_topological((state_space.sub(0), V), 2, facets)
    u_fixed = dolfinx.fem.Function(V)
    u_fixed.x.array[:] = 0.0
    return [dolfinx.fem.dirichletbc(u_fixed, dofs, state_space.sub(0))]


# We apply a traction in endocardium

traction = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
neumann = fenicsx_pulse.NeumannBC(
    traction=fenicsx_pulse.Variable(traction, "kPa"),
    marker=geo.markers["ENDO"][0],
)

# and finally combine all the boundary conditions

bcs = fenicsx_pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

# and create a Mixed problem

problem = fenicsx_pulse.MechanicsProblemMixed(model=model, geometry=geometry, bcs=bcs)

# Now we can solve the problem
log.set_log_level(log.LogLevel.INFO)

problem.solve()

# Now step up the pressure to 10 kPa starting with an increment of 1 kPa
target_value = 10.0
incr = 1.0

# Here we use a continuation strategy to speed up the convergence

use_continuation = True

old_states = [problem.state.copy()]
old_tractions = [traction.value.copy()]

while traction.value < target_value:
    value = min(traction.value + incr, target_value)
    print(f"Solving problem for traction={value}")

    if use_continuation and len(old_tractions) > 1:
        d = (value - old_tractions[-2]) / (old_tractions[-1] - old_tractions[-2])
        problem.state.x.array[:] = (1 - d) * old_states[-2].x.array + d * old_states[-1].x.array

    traction.value = value

    try:
        nit, converged = problem.solve()
    except RuntimeError:

        # Reset state and half the increment
        traction.value = old_tractions[-1]
        problem.state.x.array[:] = old_states[-1].x.array
        incr *= 0.5
    else:
        if nit < 3:
            # Increase increment
            incr *= 1.5
        old_states.append(problem.state.copy())
        old_tractions.append(traction.value.copy())

log.set_log_level(log.LogLevel.INFO)

# Now save the displacement to a file that we can view in Paraview

u = problem.state.sub(0).collapse()
with dolfinx.io.VTXWriter(geometry.mesh.comm, "problem2.bp", [u], engine="BP4") as vtx:
    vtx.write(0.0)


try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    pyvista.start_xvfb()
    V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 1, (geometry.mesh.geometry.dim,)))
    uh = dolfinx.fem.Function(V)
    uh.interpolate(u)
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

U = dolfinx.fem.Function(u.function_space)
U.interpolate(lambda x: (x[0], x[1], x[2]))

endo_apex_coord = fenicsx_pulse.utils.evaluate_at_vertex_tag(U, geo.vfun, geo.markers["ENDOPT"][0])
u_endo_apex = fenicsx_pulse.utils.evaluate_at_vertex_tag(u, geo.vfun, geo.markers["ENDOPT"][0])
endo_apex_pos = fenicsx_pulse.utils.gather_broadcast_array(geo.mesh.comm, endo_apex_coord + u_endo_apex)
print(f"\nGet longitudinal position of endocardial apex: {endo_apex_pos[0, 0]:4f} mm")

epi_apex_coord = fenicsx_pulse.utils.evaluate_at_vertex_tag(U, geo.vfun, geo.markers["EPIPT"][0])
u_epi_apex = fenicsx_pulse.utils.evaluate_at_vertex_tag(u, geo.vfun, geo.markers["EPIPT"][0])
epi_apex_pos = fenicsx_pulse.utils.gather_broadcast_array(geo.mesh.comm, epi_apex_coord + u_epi_apex)
print(f"\nGet longitudinal position of epicardial apex: {epi_apex_pos[0, 0]:4f} mm")
