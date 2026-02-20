# # Rotated UK Biobank Atlas Geometry
#
# This demo shows how to load a patient-specific geometry from the UK Biobank Atlas,
# rotate it to align the basal plane with a coordinate axis, and perform a cardiac cycle simulation.
#
# ## Motivation: Geometry Alignment
#
# Patient-specific meshes often come in arbitrary orientations based on the imaging coordinate system.
# However, for defining boundary conditions—especially at the base—it is often convenient to align
# the geometry such that the base normal points along a principal axis (e.g., the X-axis).
#
# In this example:
# 1.  We generate a Bi-Ventricular (BiV) mesh from the UK Biobank atlas using `cardiac-geometries`.
# 2.  We **rotate** the mesh so the base normal aligns with $\mathbf{e}_1 = (1, 0, 0)$.
# 3.  We apply a **Dirichlet boundary condition** fixing $u_x = 0$ on the base. Since the base is
#     now perpendicular to X, this acts as a "sliding" condition, allowing the base to expand/contract
#     in the Y-Z plane while preventing longitudinal rigid body motion.
#
# ---

# ## Imports

from pathlib import Path
from mpi4py import MPI
import dolfinx
import cardiac_geometries
import cardiac_geometries.geometry
import pulse

# ## 1. Geometry Generation and Rotation
#
# We generate the mesh and then apply a rotation. The `rotate` method in `cardiac_geometries`
# takes a `target_normal` vector. It computes the rotation matrix required to align the
# average normal of the surface marked by `base_marker` with this target vector.

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
    create_fibers=True,
    char_length_max=char_length,
    char_length_min=char_length,
    clipped=True,
    fiber_angle_endo=60,
    fiber_angle_epi=-60,
    fiber_space="Quadrature_6",
)

# Rotate the geometry so the Base normal points in the X-direction (1, 0, 0)
rotated_geo = geo.rotate(target_normal=[1.0, 0.0, 0.0], base_marker="BASE")
rot_geodir = outdir / "geometry_rotated"
rotated_geo.save_folder(folder=rot_geodir)

# ## 2. Load Geometry and Visualization
#
# We load the original and rotated geometries to compare them.

geo_orig = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)
geo_rot = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=rot_geodir,
)

# Visualize using PyVista (Optional)
try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    p = pyvista.Plotter()

    # Original Mesh (Red)
    topology, cell_types, geometry_pts = dolfinx.plot.vtk_mesh(geo_orig.mesh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry_pts)
    p.add_mesh(grid, show_edges=True, color="red", opacity=0.3, label="Original")

    # Rotated Mesh (Yellow)
    topology_rot, cell_types_rot, geometry_rot = dolfinx.plot.vtk_mesh(rotated_geo.mesh)
    grid_rot = pyvista.UnstructuredGrid(topology_rot, cell_types_rot, geometry_rot)
    p.add_mesh(grid_rot, show_edges=True, color="yellow", opacity=0.5, label="Rotated (Base -> X)")

    p.add_legend()
    axes = pyvista.CubeAxesActor(camera=p.camera)
    axes.bounds = grid_rot.bounds
    p.add_actor(axes)
    p.view_zy() # View along the X-axis

    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot("rotated_mesh.png")

# Convert the rotated geometry to a pulse.Geometry object for simulation
geometry = pulse.Geometry.from_cardiac_geometries(geo_rot, metadata={"quadrature_degree": 4})

# ## 3. Constitutive Model
#
# We use the **Holzapfel-Ogden** passive law and an **Active Stress** model.
# Note that we use the fiber fields (`f0`, `s0`) from the *rotated* geometry `geo_rot`.

material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
material = pulse.HolzapfelOgden(f0=geo_rot.f0, s0=geo_rot.s0, **material_params)

Ta = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
# eta=0.3 adds 30% transverse active stress
active_model = pulse.ActiveStress(geo_rot.f0, activation=Ta, eta=0.3)
comp_model = pulse.Incompressible()

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# ## 4. Boundary Conditions
#
# ### Neumann BCs (Cavity Pressures)
# We apply pressure loads to the LV and RV endocardial surfaces.

traction_lv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
neumann_lv = pulse.NeumannBC(traction=traction_lv, marker=geometry.markers["LV"][0])

traction_rv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
neumann_rv = pulse.NeumannBC(traction=traction_rv, marker=geometry.markers["RV"][0])

neumann = [neumann_lv, neumann_rv]

# ### Robin BCs (Spring Support)
# Elastic springs on the Epicardium and Base to mimic the pericardium and surrounding tissue.

robin_epi = pulse.RobinBC(
    value=pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e3)),
        "Pa / m",
    ),
    marker=geometry.markers["EPI"][0],
)

robin_base = pulse.RobinBC(
    value=pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e6)),
        "Pa / m",
    ),
    marker=geometry.markers["BASE"][0],
)

# ### Dirichlet BC (Sliding Base)
# Because we rotated the mesh, the basal plane is now perpendicular to the X-axis.
# We constrain the displacement in the x-direction ($u_x = 0$) on the Base.
#
# * **Why?** This prevents the heart from flying away in the X-direction (Rigid Body Motion).
# * **Effect**: Since the normal is X, fixing $u_x$ allows the base to expand and contract freely
#   in the Y and Z directions (in-plane), which is a common approximation for the AV-plane motion constraint.

def dirichlet_bc(V: dolfinx.fem.FunctionSpace):
    # Find facets for the BASE marker
    facets = geometry.facet_tags.find(geometry.markers["BASE"][0])
    # Locate degrees of freedom for the x-component (sub(0)) on these facets
    dofs = dolfinx.fem.locate_dofs_topological(V.sub(0), 2, facets)
    # Constrain u_x to 0.0
    return [dolfinx.fem.dirichletbc(0.0, dofs, V.sub(0))]

bcs = pulse.BoundaryConditions(
    neumann=neumann,
    dirichlet=(dirichlet_bc,),
    robin=(robin_epi, robin_base),
)

# ## 5. Solving the Problem
#
# We set up the `StaticProblem`. We do not use the automatic `base_bc` parameter because we
# explicitly provided our own Dirichlet condition in `bcs`.

problem = pulse.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs,
)

# ### Phase 1: Passive Inflation
# We ramp up the pressure in both ventricles.

vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, outdir / "displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)

pressures = [0.5, 1.0, 1.5] # kPa
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
    p.add_mesh(warped, show_edges=True, color="firebrick", label="Inflated")
    p.add_legend()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot(outdir / "pressure.png")

# ### Phase 2: Active Contraction
# We hold the pressure constant and ramp up the active tension $T_a$.

active_tensions = [3.0, 5.0, 10.0] # kPa
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
    p.add_mesh(warped, show_edges=True, color="red", label="Contracted")
    p.add_legend()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot(outdir / "active.png")
