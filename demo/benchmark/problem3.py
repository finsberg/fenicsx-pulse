# # Problem 3: inflation and active contraction of a ventricle

# In the third problem we will solve the inflation and active contraction of a ventricle. First we import the necessary libraries
#

from pathlib import Path
from mpi4py import MPI
from dolfinx import log
import dolfinx
import numpy as np
import math
import cardiac_geometries
import pulse

# Next we will create the geometry and save it in the folder called `lv_ellipsoid`. Now we will also generate fibers and use a sixth order quadrature space for the fibers

geodir = Path("lv_ellipsoid-problem3")
comm = MPI.COMM_WORLD
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
        fiber_space="Quadrature_6",
        create_fibers=True,
        fiber_angle_epi=-90,
        fiber_angle_endo=90,
        comm=comm,
    )
    print("Done creating geometry.")

# If the folder already exist, then we just load the geometry

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)

# Now, lets convert the geometry to a `pulse.Geometry` object.

geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})


# The material model used in this benchmark is the {py:class}`Guccione <pulse.material_models.guccione.Guccione>` model.

material_params = {
    "C": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2.0)),
    "bf": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(8.0)),
    "bt": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2.0)),
    "bfs": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(4.0)),
}
material = pulse.Guccione(f0=geo.f0, s0=geo.s0, n0=geo.n0, **material_params)


# We use an active stress approach with 60% transverse active stress

Ta = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
active_model = pulse.ActiveStress(geo.f0, activation=pulse.Variable(Ta, "kPa"))

# and the model should be incompressible

comp_model = pulse.Incompressible()


# and assembles the `CardiacModel`

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
    decouple_deviatoric_volumetric=True,
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

# Now we will solve the problem for a range of active contraction and traction values

target_pressure = 15.0
target_Ta = 60.0

# Let us just gradually increase the active contraction and traction with a linear ramp of 40 steps

N = 40

for Ta_value, traction_value in zip(np.linspace(0, target_Ta, N), np.linspace(0, target_pressure, N)):
    print(f"Solving problem for traction={traction_value} and active contraction={Ta_value}")
    Ta.value = Ta_value
    traction.value = traction_value
    problem.solve()

log.set_log_level(log.LogLevel.INFO)

# Now save the displacement to a file that we can view in Paraview

with dolfinx.io.VTXWriter(geometry.mesh.comm, "problem3.bp", [problem.u], engine="BP4") as vtx:
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
