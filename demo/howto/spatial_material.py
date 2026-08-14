# # Spatial material parameters
#
# This demo shows how to assign spatially varying material properties.
# Specifically, we create a left ventricular ellipsoid and assign different
# material parameters to different AHA segments {cite}`cerqueira2002standardized`.
# We simulate an infarct in one segment by making the material 10 times stiffer
# in that region. We then inflate the ventricle and visualize the resulting displacement.

from pathlib import Path
from mpi4py import MPI
import dolfinx
import ufl
import cardiac_geometries
import cardiac_geometries.geometry
import scifem
import pulse

# Create output directory

outdir = Path("lv_ellipsoid_spatial_material")
outdir.mkdir(parents=True, exist_ok=True)
geodir = outdir / "geometry"

# Create geometry if it does not exist

if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        create_fibers=True,
        fiber_space="Quadrature_4",
        aha=True, dmu_factor=1/4,
    )

# Load geometry
geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)

geometry = pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})

# Setup material parameters
material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()

# Here we use the AHA segments to define a material region.
# We set the value to 1 in segment 10, and 0 everywhere else.
values = geo.cfun.values.copy()
values[geo.cfun.values != 10] = 0
values[geo.cfun.values == 10] = 1
material_regions = dolfinx.mesh.meshtags(geo.mesh, 3, geo.cfun.indices, values)
material_regions.name = "material_regions"

# Optionally save the material regions to file
# with dolfinx.io.XDMFFile(geo.mesh.comm, outdir / "material_regions.xdmf", "w") as xdmf:
#     xdmf.write_mesh(geo.mesh)
#     xdmf.write_meshtags(material_regions, geo.mesh.geometry)

# Plot the material regions using PyVista if available

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    vtk_mesh = dolfinx.plot.vtk_mesh(geo.mesh, geo.mesh.topology.dim)
    p = pyvista.Plotter(window_size=[800, 800])
    grid = pyvista.UnstructuredGrid(*vtk_mesh)
    grid.cell_data["materials"] = material_regions.values
    grid.set_active_scalars("materials")
    p.add_mesh(grid, show_edges=True)
    if pyvista.OFF_SCREEN:
        figure = p.screenshot(outdir / "material_regions.png")
    p.show()

# Create a function space for the material parameter 'a'
# We use scifem to create a function that takes the value material_params["a"].value
# where material_regions is 0, and material_params["a"].value * 10.0 where it is 1.

V = scifem.create_space_of_simple_functions(geo.mesh, material_regions, [0, 1])
a_func = dolfinx.fem.Function(V)
a_func.x.array[0] = material_params["a"].value
a_func.x.array[1] = material_params["a"].value * 10.0
a = pulse.Variable(a_func, "kPa")
material_params["a"] = a

# Define the material model

material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)

# Define the active model (passive in this case) and compressibility

comp_model = pulse.Incompressible()

# Define the cardiac model
model = pulse.CardiacModel(
    material=material,
    active=pulse.Passive(),
    compressibility=comp_model,
)

# Define boundary conditions: Neumann on endocardium for pressure

traction = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
neumann = pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])
bcs = pulse.BoundaryConditions(neumann=(neumann,))

# Define the static problem, fixing the base

problem = pulse.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs,
    parameters={"base_bc": pulse.BaseBC.fixed},
)

# Setup VTX writer for displacement

vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, outdir / "lv_displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)

# Solve the problem for a given pressure

pressures =  [0.1] # kPa. Add more steps for a smoother ramp, e.g. [0.1, 0.5, 1.0]
for i, plv in enumerate(pressures, start=1):
    print(f"Solving for pressure: {plv} kPa")
    traction.assign(plv)
    problem.solve()
    vtx.write(float(i))

# Plot the inflated geometry using PyVista

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    # Interpolate solution to a standard CG-1 space for plotting
    V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 1, (geometry.mesh.geometry.dim,)))
    uh = dolfinx.fem.Function(V)
    uh.interpolate(problem.u)

    # Create plotter
    p = pyvista.Plotter()
    topology, cell_types, geometry_data = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry_data)

    grid["u"] = uh.x.array.reshape((geometry_data.shape[0], 3))
    warped = grid.warp_by_vector("u", factor=1.0)
    p.add_mesh(warped, show_edges=True, label="Inflated")

    p.add_legend()
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot(outdir / "lv_ellipsoid_pressure.png")

# ## References
# ```{bibliography}
# :filter: docname in docnames
# ```
