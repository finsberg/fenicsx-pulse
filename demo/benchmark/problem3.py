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
import fenicsx_pulse
import shutil
# Next we will create the geometry and save it in the folder called `lv_ellipsoid`.

geodir = Path("lv_ellipsoid-problem3")
if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        r_short_endo=7.0,
        r_short_epi=10.0,
        r_long_endo=17.0,
        r_long_epi=20.0,
        mu_apex_endo = -math.pi,
        mu_base_endo = -math.acos(5 / 17),
        mu_apex_epi = -math.pi,
        mu_base_epi = -math.acos(5 / 20),
        fiber_space="Quadrature_6",
        create_fibers=True,
        fiber_angle_epi=-90,
        fiber_angle_endo=90,
    )
    print("Done creating geometry.")
# If the folder already exist, then we just load the geometry

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)

# Now, lets convert the geometry to a `fenicsx_pulse.Geometry` object.

geometry = fenicsx_pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})


# The material model used in this benchmark is the {py:class}`Guccione <fenicsx_pulse.material_models.guccione.Guccione>` model.

material_params = {
    "C": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2.0)),
    "bf": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(8.0)),
    "bt": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2.0)),
    "bfs": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(4.0)),
}
material = fenicsx_pulse.Guccione(f0=geo.f0, s0=geo.s0, n0=geo.n0, **material_params)


Ta = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
active_model = fenicsx_pulse.ActiveStress(geo.f0, activation=Ta)

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
neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=geo.markers["ENDO"][0])

# and finally combine all the boundary conditions

bcs = fenicsx_pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

# and create a Mixed problem

problem = fenicsx_pulse.MechanicsProblemMixed(model=model, geometry=geometry, bcs=bcs)

# Now we can solve the problem
log.set_log_level(log.LogLevel.INFO)

problem.solve()


target_pressure = 15.0
target_Ta = 60.0

N = 40

for Ta_value, traction_value in zip(np.linspace(0, target_Ta, N), np.linspace(0, target_pressure, N)):
    print(f"Solving problem for traction={traction_value} and active contraction={Ta_value}")
    Ta.value = Ta_value
    traction.value = traction_value
    problem.solve()

log.set_log_level(log.LogLevel.INFO)

# Now save the displacement to a file that we can view in Paraview

u = problem.state.sub(0).collapse()
with dolfinx.io.VTXWriter(geometry.mesh.comm, "problem3.bp", [u], engine="BP4") as vtx:
    vtx.write(0.0)


# try:
#     import pyvista
# except ImportError:
#     print("Pyvista is not installed")
# else:
#     pyvista.start_xvfb()
#     V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 1, (geometry.mesh.geometry.dim,)))
#     uh = dolfinx.fem.Function(V)
#     uh.interpolate(u)
#     # Create plotter and pyvista grid
#     p = pyvista.Plotter()
#     topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
#     grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

#     # Attach vector values to grid and warp grid by vector
#     grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
#     actor_0 = p.add_mesh(grid, style="wireframe", color="k")
#     warped = grid.warp_by_vector("u", factor=1.5)
#     actor_1 = p.add_mesh(warped, show_edges=True)
#     p.show_axes()
#     if not pyvista.OFF_SCREEN:
#         p.show()
#     else:
#         figure_as_array = p.screenshot("problem2.png")


# # FIXME: Need to figure out how to evaluate the displacement at the apex
# # geometry.mesh.topology.create_connectivity(0, geometry.mesh.topology.dim)
# # apex_endo = geo.vfun.find(geo.markers["ENDOPT"][0])
# # endo_apex_coord = geo.mesh.geometry.x[apex_endo]

# # dofs_endo_apex = dolfinx.fem.locate_dofs_topological(problem.state_space.sub(0), 0, apex_endo)
# # u_endo_apex = u.x.array[dofs_endo_apex]

# # endo_apex_pos = endo_apex_coord + u_endo_apex

# # print(f"\nGet longitudinal position of endocardial apex: {endo_apex_pos[0, 0]:4f} mm")

# # apex_epi = geo.vfun.find(geo.markers["EPIPT"][0])
# # epi_apex_coord = geo.mesh.geometry.x[apex_epi]

# # geometry.mesh.topology.create_connectivity(0, geometry.mesh.topology.dim)
# # dofs_epi_apex = dolfinx.fem.locate_dofs_topological(problem.state_space.sub(0), 0, apex_epi)
# # u_epi_apex = u.x.array[dofs_epi_apex]

# # epi_apex_pos = epi_apex_coord + u_epi_apex

# # print(f"\nGet longitudinal position of epicardial apex: {epi_apex_pos[0, 0]:.4f} mm")
