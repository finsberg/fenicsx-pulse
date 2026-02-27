# # Left Ventricular Ellipsoid with Custom Boundary Conditions
#
# This demo builds upon the [Left Ventricular Ellipsoid Simulation](../geometries/lv_ellipsoid.py).
#
# While the previous example utilized the convenience parameter `base_bc=pulse.BaseBC.fixed`
# to fully clamp the base, this example demonstrates how to:
# 1.  Apply **Robin Boundary Conditions** (springs) to the epicardium and base to mimic
#     surrounding tissue support.
# 2.  Manually define **Dirichlet Boundary Conditions** to constrain specific degrees of freedom.
#
# The geometry generation, constitutive modeling, and active stress implementation remain
# identical to the base example.
#
# ---

# ## Imports

from pathlib import Path
from mpi4py import MPI
import dolfinx
from dolfinx import log
import cardiac_geometries
import cardiac_geometries.geometry
import pulse

# ## Geometry and Materials
#
# We generate the same truncated ellipsoid geometry and define the material model
# (Holzapfel-Ogden with Active Stress) as in the [base example](../geometries/lv_ellipsoid.py).

# 1. Generate Mesh

outdir = Path("lv_ellipsoid_custom_bcs")
outdir.mkdir(parents=True, exist_ok=True)
geodir = outdir / "geometry"

if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        create_fibers=True,
        fiber_space="P_2",
    )

# 2. Load Geometry

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)
geometry = pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})

# 3. Define Constitutive Model

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

# ## Custom Boundary Conditions
#
# Instead of using the preset `base_bc` parameter, we will explicitly define our boundary conditions.
#
# ### 1. Neumann BC (Cavity Pressure)
# Standard pressure load on the endocardium.

traction = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
neumann = pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])

# ### 2. Robin BCs (Elastic Support)
# We apply a spring-like penalty on the Epicardium and the Base to represent the
# resistance provided by the pericardium and surrounding tissue.
#
# $$
# \mathbf{P}\mathbf{N} + k \mathbf{u} = 0
# $$

robin_epi = pulse.RobinBC(
    value=pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e3)),
        "Pa / m",
    ),
    marker=geometry.markers["EPI"][0],
)

robin_base = pulse.RobinBC(
    value=pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e3)),
        "Pa / m",
    ),
    marker=geometry.markers["BASE"][0],
)

# ### 3. Manual Dirichlet BC
# We manually define a Dirichlet condition. In this specific case, we constrain the
# displacement in the x-direction ($u_x = 0$) on the Base. Ideally, the Base plane is
# perpendicular to the X-axis in this geometry, so this prevents the base from moving
# longitudinally while allowing expansion/sliding in the Y-Z plane (resisted only by the Robin spring).

def dirichlet_bc(V: dolfinx.fem.FunctionSpace):
    # Find facets for the BASE marker
    facets = geo.ffun.find(geo.markers["BASE"][0])
    # Locate degrees of freedom for the x-component (sub(0)) on these facets
    dofs = dolfinx.fem.locate_dofs_topological(V.sub(0), 2, facets)
    # Return the Dirichlet BC object
    return [dolfinx.fem.dirichletbc(0.0, dofs, V.sub(0))]

# We collect all conditions into the `BoundaryConditions` container.
bcs = pulse.BoundaryConditions(
    neumann=(neumann,),
    dirichlet=(dirichlet_bc,),
    robin=(robin_epi, robin_base),
)

# ## Solving the Problem
#
# We initialize the `StaticProblem`.
# **Note**: We do *not* pass `parameters={"base_bc": ...}` here, as we are fully controlling
# the boundaries via the `bcs` argument.

problem = pulse.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs,
)

# ### Phase 1: Passive Inflation

vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, outdir / "lv_displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)

pressures = [1.0] # kPa
for i, plv in enumerate(pressures, start=1):
    print(f"Solving for pressure: {plv} kPa")
    traction.assign(plv)
    problem.solve()
    vtx.write(float(i))

# #### Visualization

try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
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
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot(outdir / "lv_ellipsoid_pressure.png")

# ### Phase 2: Active Contraction

active_tensions = [3.0] # kPa
for i, ta in enumerate(active_tensions, start=len(pressures) + 1):
    print(f"Solving for active tension: {ta} kPa")
    Ta.assign(ta)
    problem.solve()
    vtx.write(float(i))

vtx.close()

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
    p.add_mesh(warped, show_edges=False, color="red", label="Contracted")

    p.add_legend()
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot(outdir / "lv_ellipsoid_active.png")
