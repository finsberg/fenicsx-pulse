# # BiV Mesh from UK Biobank Atlas
#
# This demo illustrates how to set up a simulation using a patient-specific Bi-Ventricular (BiV) geometry
# from the UK Biobank Atlas.
#
# Key features of this example:
# 1.  **Geometry**: Loading a BiV mesh from the UK Biobank statistical atlas.
# 2.  **Microstructure**: Generating fiber and sheet orientations using the **Laplace-Dirichlet Rule-Based (LDRB)**
#     algorithm via the external library `fenicsx-ldrb`. This provides more flexibility than standard built-in methods,
#     allowing for distinct fiber angles in the LV and RV.
# 3.  **Boundary Conditions**: Fixing the **outflow tracts** (Aortic, Pulmonary, Mitral, and Tricuspid valves)
#     to prevent rigid body motion, instead of fixing the entire base.
#
# ---

# ## Imports

from pathlib import Path
from mpi4py import MPI
import dolfinx
import ldrb
import numpy as np
import cardiac_geometries
import cardiac_geometries.geometry
import pulse

# ## 1. Geometry Generation
#
# We generate the mesh using `cardiac_geometries`. We set `create_fibers=False` because we will
# generate a more advanced fiber architecture using `ldrb` in the next step.

outdir = Path("ukb_atlas")
outdir.mkdir(parents=True, exist_ok=True)
geodir = outdir / "geometry"
comm = MPI.COMM_WORLD
mode = -1
std = 0
char_length = 10.0

# Generate base UKB mesh

geo = cardiac_geometries.mesh.ukb(
    outdir=geodir,
    comm=comm,
    mode=mode,
    std=std,
    case="ED",
    create_fibers=False,
    char_length_max=char_length,
    char_length_min=char_length,
    clipped=False,
)

# ## 2. Fiber Generation with LDRB
#
# We use `fenicsx-ldrb` to generate the fiber field. This algorithm solves Laplace problems
# to define transmural and apicobasal directions.
#
# We specify distinct angles for the Left Ventricle (LV) and Right Ventricle (RV), which
# allows for a more physiologically accurate representation (e.g., $\alpha_{endo}^{RV} = 90^\circ$).

system = ldrb.dolfinx_ldrb(
    mesh=geo.mesh,
    ffun=geo.ffun,
    markers=cardiac_geometries.mesh.transform_markers(geo.markers, clipped=True),
    alpha_endo_lv=60,
    alpha_epi_lv=-60,
    alpha_endo_rv=90,
    alpha_epi_rv=-25,
    beta_endo_lv=-20,
    beta_epi_lv=20,
    beta_endo_rv=0,
    beta_epi_rv=20,
    fiber_space="Quadrature_6",
)

# We save the generated microstructure to the geometry folder so it can be automatically
# loaded by the `cardiac_geometries.geometry.Geometry` class later. We need to also save
# the existing geometry information (mesh, markers, etc), in order for the dofmap to be
# consistent. We first delete any existing `geometry.bp` file to avoid conflicts.

import shutil
shutil.rmtree(geodir / "geometry.bp")
cardiac_geometries.geometry.save_geometry(
    path=geodir / "geometry.bp",
    mesh=geo.mesh,
    markers=geo.markers,
    info=geo.info,
    ffun=geo.ffun,
    efun=geo.efun,
    vfun=geo.vfun,
    f0=system.f0,
    s0=system.s0,
    n0=system.n0,
)

# ## 3. Load Geometry and Visualization
#
# We reload the geometry (now including the fibers we just saved).

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)


# ### Visualizing the Mesh

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    p = pyvista.Plotter()
    topology, cell_types, geometry_pts = dolfinx.plot.vtk_mesh(geo.mesh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry_pts)
    p.add_mesh(grid, show_edges=True, color="lightblue")
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot("ukb_mesh.png")


# ### Visualizing Fibers
# We create a glyph plot to visualize the generated fiber vectors $\mathbf{f}_0$.

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    plotter = pyvista.Plotter()
    N = 5  # Only show every 5th point to avoid clutter
    points = geo.f0.function_space.tabulate_dof_coordinates()
    point_cloud = pyvista.PolyData(points[::N, :])
    f0_arr = geo.f0.x.array.reshape((points.shape[0], 3))
    point_cloud["fibers"] = f0_arr[::N, :]
    fibers = point_cloud.glyph(
        orient="fibers",
        scale=False,
        factor=5.0,
    )
    plotter.add_mesh(fibers, color="red")

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        fig_as_array = plotter.screenshot("fiber_ukb.png")


# ### Visualizing Boundary Markers
# The UK Biobank mesh comes with detailed surface markers, including specific tags for the
# outflow tracts (valves). We can visualize these to verify their location.

print("Available markers:", geo.markers)

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    p = pyvista.Plotter()
    vtk_bmesh = dolfinx.plot.vtk_mesh(
        geo.mesh, geo.ffun.dim, geo.ffun.indices,
    )
    bgrid = pyvista.UnstructuredGrid(*vtk_bmesh)
    bgrid.cell_data["Facet tags"] = geo.ffun.values
    bgrid.set_active_scalars("Facet tags")
    p = pyvista.Plotter(window_size=[800, 800])
    p.add_mesh(bgrid, show_edges=True, cmap="tab20", show_scalar_bar=False)
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        figure = p.screenshot("facet_tags_ukb.png")

# Convert to `pulse.Geometry` for simulation.

geometry = pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

# ## 4. Constitutive Model
#
# We use the **Holzapfel-Ogden** passive law and an **Active Stress** model.

material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)

Ta = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
active_model = pulse.ActiveStress(geo.f0, activation=Ta, eta=0.3)
comp_model = pulse.Incompressible()

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# ## 5. Boundary Conditions
#
# ### Neumann BCs (Cavity Pressures)
# We apply pressure loads to the LV and RV endocardial surfaces.

traction_lv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
neumann_lv = pulse.NeumannBC(traction=traction_lv, marker=geometry.markers["LV"][0])

traction_rv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
neumann_rv = pulse.NeumannBC(traction=traction_rv, marker=geometry.markers["RV"][0])

neumann = [neumann_lv, neumann_rv]

# ### Robin BCs (Spring Support)
# Elastic springs on the Epicardium to mimic the pericardial constraint.

robin_epi = pulse.RobinBC(
    value=pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e3)),
        "Pa / m",
    ),
    marker=geometry.markers["EPI"][0],
)

# ### Dirichlet BC (Fixed Outflow Tracts)
#
# Instead of fixing the entire base plane (which might be unphysical if the base moves),
# we fix the displacement at the four valves:
# * **PV**: Pulmonary Valve
# * **TV**: Tricuspid Valve
# * **AV**: Aortic Valve
# * **MV**: Mitral Valve
#
# This mimics the anatomical anchoring of the heart to the great vessels.


def dirichlet_bc(V: dolfinx.fem.FunctionSpace):
    # Find facets for the outflow tracts
    bcs = []
    u_bc = dolfinx.fem.Function(V)
    u_bc.x.array[:] = 0
    # Iterate over the valve markers provided by the UKB mesh
    for marker in ["PV", "TV", "AV", "MV"]:
        facets = geometry.facet_tags.find(geometry.markers[marker][0])
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
        bcs.append(dolfinx.fem.dirichletbc(u_bc, dofs))
    return bcs


bcs = pulse.BoundaryConditions(
    neumann=neumann,
    dirichlet=(dirichlet_bc,),
    robin=(robin_epi,),
)

# ## 6. Solving the Problem
#
# We set up the `StaticProblem` and run the simulation phases.

problem = pulse.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs,
)

# ### Phase 1: Passive Inflation
# We ramp up the pressure in both ventricles.

vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, outdir / "displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)
pressures = [0.5, 1.0, 1.5]  #  kPa
for i, plv in enumerate(pressures, start=1):
    print(f"Solving for pressure: {plv} kPa")
    traction_lv.assign(plv)
    traction_rv.assign(plv * 0.5)  # Assume RV pressure is half of LV
    problem.solve()
    vtx.write(float(i))

# Visualize Passive Inflation

try:
    import pyvista
except ImportError:
    pass
else:
    V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 1, (geometry.mesh.geometry.dim,)))
    uh = dolfinx.fem.Function(V)
    uh.interpolate(problem.u)

    p = pyvista.Plotter()
    topology, cell_types, geometry_data = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry_data)
    grid["u"] = uh.x.array.reshape((geometry_data.shape[0], 3))

    p.add_mesh(grid, style="wireframe", color="k", opacity=0.3, label="Reference")
    warped = grid.warp_by_vector("u", factor=1.0)
    p.add_mesh(warped, show_edges=False, opacity=0.5, color="firebrick", label="Inflated")
    p.add_legend()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot(outdir / "pressure.png")

# ### Phase 2: Active Contraction
# We hold the pressure constant and ramp up the active tension $T_a$.

active_tensions = [3.0, 5.0, 10.0]  # kPa
for i, ta in enumerate(active_tensions, start=len(pressures) + 1):
    print(f"Solving for active tension: {ta} kPa")
    Ta.assign(ta)
    problem.solve()
    vtx.write(float(i))

vtx.close()

# Visualize Active Contraction
try:
    import pyvista
except ImportError:
    pass
else:
    uh.interpolate(problem.u)
    grid["u"] = uh.x.array.reshape((geometry_data.shape[0], 3))

    p = pyvista.Plotter()
    p.add_mesh(grid, style="wireframe", color="k", opacity=0.3, label="Reference")
    warped = grid.warp_by_vector("u", factor=1.0)
    p.add_mesh(warped, show_edges=False, opacity=0.5, color="red", label="Contracted")
    p.add_legend()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot(outdir / "active.png")
