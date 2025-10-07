# # LV Ellipsoid
#
# In this demo we will show how to simulate an idealized left ventricular geometry. For this we will use [`cardiac-geometries`](https://github.com/ComputationalPhysiology/cardiac-geometriesx) to generate an idealized LV ellipsoid.
#
# First we will make the necessary imports.

from pathlib import Path
from mpi4py import MPI
import dolfinx
from dolfinx import log
import cardiac_geometries
import cardiac_geometries.geometry
import pulse

# Next we will create the geometry and save it in the folder called `lv_ellipsoid`. We also make sure to generate fibers which can be done analytically and use a second order Lagrange space for the fibers

outdir = Path("lv_ellipsoid")
outdir.mkdir(parents=True, exist_ok=True)
geodir = outdir / "geometry"
if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid(outdir=geodir, create_fibers=True, fiber_space="P_2")

# If the folder already exist, then we just load the geometry

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)

# In order to use the geometry with `pulse` we need to convert it to a `pulse.Geometry` object. We can do this by using the `from_cardiac_geometries` method. We also specify that we want to use a quadrature degree of 4
#

geometry = pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})

# Next we create the material object, and we will use the transversely isotropic version of the {py:class}`Holzapfel Ogden model <pulse.holzapfelogden.HolzapfelOgden>`

material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore

# We use an active stress approach with 30% transverse active stress (see {py:meth}`pulse.active_stress.transversely_active_stress`)

Ta = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
active_model = pulse.ActiveStress(geo.f0, activation=Ta, eta=0.3)

# We use an incompressible model

comp_model = pulse.Incompressible()

# and assembles the `CardiacModel`

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# We apply a traction in endocardium

traction = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
neumann = pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])

# and finally combine all the boundary conditions

bcs = pulse.BoundaryConditions(neumann=(neumann,))

# and create a Mixed problem

problem = pulse.StaticProblem(model=model, geometry=geometry, bcs=bcs, parameters={"base_bc": pulse.BaseBC.fixed})

# Now we can solve the problem

log.set_log_level(log.LogLevel.INFO)
problem.solve()


# And save the displacement to a file that we can view in Paraview

vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, outdir / "lv_displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)

i = 1
for plv in [0.1]: #, 0.5, 1.0]:
    print(f"plv: {plv}")
    traction.value = plv
    problem.solve()

    vtx.write(float(i))
    i += 1


# Now plot with pyvista

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
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
        figure_as_array = p.screenshot(outdir / "lv_ellipsoid_pressure.png")


for ta in [0.1]: #, 0.5, 1.0]:
    print(f"ta: {ta}")
    Ta.value = ta
    problem.solve()
    vtx.write(float(i))
    i += 1

log.set_log_level(log.LogLevel.WARNING)
vtx.close()


try:
    import pyvista
except ImportError:
    pass
else:
    # Attach vector values to grid and warp grid by vector
    grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
    actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=1.5)
    actor_1 = p.add_mesh(warped, show_edges=True)
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        figure_as_array = p.screenshot(outdir / "lv_ellipsoid_active.png")
