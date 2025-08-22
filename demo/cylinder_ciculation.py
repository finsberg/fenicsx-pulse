# # Cylinder coupled to a 0D circulatory model and a time-varying elastance model

from pathlib import Path
from mpi4py import MPI
import dolfinx
import logging
import circulation.bestel

from dolfinx import log
import ufl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import adios4dolfinx
import pulse
import cardiac_geometries
import cardiac_geometries.geometry

# Next we set up the logging and the MPI communicator

circulation.log.setup_logging(logging.INFO)
comm = MPI.COMM_WORLD

outdir = Path("results_cylinder_d_shaped")
outdir.mkdir(exist_ok=True)

# Next we create the geometry of the cylinder

geodir = Path("cylinder_d_shaped")

r_inner = 0.02
r_outer = 0.03
height = 0.05
inner_flat_face_distance = 0.015
outer_flat_face_distance = 0.025

import shutil

shutil.rmtree(geodir, ignore_errors=True)
if not geodir.exists():
    comm.barrier()
    cardiac_geometries.mesh.cylinder_D_shaped(
        outdir=geodir,
        create_fibers=True,
        # fiber_space="Quadrature_6",
        fiber_space="DG_1",  # Something wrong with the fibers in quadrature spaces
        r_inner=r_inner,
        r_outer=r_outer,
        height=height,
        inner_flat_face_distance=inner_flat_face_distance,
        outer_flat_face_distance=outer_flat_face_distance,
        char_length=0.01,
        comm=comm,
        fiber_angle_epi=-60,
        fiber_angle_endo=60,
    )

# If the folder already exist, then we just load the geometry

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)

# Next we transform the geometry to a `HeartGeometry` object

geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

# Next we create the material object, and we will use the transversely isotropic version of the {py:class}`Holzapfel Ogden model <pulse.holzapfelogden.HolzapfelOgden>`

material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore

# We use an active stress approach with 30% transverse active stress (see {py:meth}`pulse.active_stress.transversely_active_stress`)

Ta = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa")
active_model = pulse.ActiveStress(geo.f0, activation=Ta)

# a compressible material model

comp_model = pulse.compressibility.Compressible2()

# and assembles the `CardiacModel`

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# Next we set up the boundary conditions. We use a Robin boundary condition on the epicardium and the base of the LV

robin_value = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e7)),
    "Pa / m",
)
# Add robing on both outside, top and bottom
# robin = (
#     pulse.RobinBC(value=robin_value, marker=geometry.markers["TOP"][0]),
#     pulse.RobinBC(value=robin_value, marker=geometry.markers["BOTTOM"][0]),
#     pulse.RobinBC(value=robin_value, marker=geometry.markers["OUTSIDE_CURVED"][0]),
#     pulse.RobinBC(value=robin_value, marker=geometry.markers["OUTSIDE_FLAT"][0]),
# )

traction = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa",
)
neumann = (
    pulse.NeumannBC(traction=traction, marker=geometry.markers["INSIDE_CURVED"][0]),
    pulse.NeumannBC(traction=traction, marker=geometry.markers["INSIDE_FLAT"][0]),
)


# Fix top and bottom in z-direction
def dirichlet_bc(
    V: dolfinx.fem.FunctionSpace,
) -> list[dolfinx.fem.bcs.DirichletBC]:
    Vz, _ = V.sub(2).collapse()
    geometry.mesh.topology.create_connectivity(
        geometry.mesh.topology.dim - 1, geometry.mesh.topology.dim,
    )
    facets_top = geometry.facet_tags.find(
        geometry.markers["TOP"][0],
    )  # Specify the marker used on the boundary
    dofs_top = dolfinx.fem.locate_dofs_topological((V.sub(2), Vz), 2, facets_top)
    facets_bottom = geometry.facet_tags.find(
        geometry.markers["BOTTOM"][0],
    )  # Specify the marker used on the boundary
    dofs_bottom = dolfinx.fem.locate_dofs_topological((V.sub(2), Vz), 2, facets_bottom)
    u_fixed = dolfinx.fem.Function(Vz)
    u_fixed.x.array[:] = 0.0
    return [
        dolfinx.fem.dirichletbc(u_fixed, dofs_top, V.sub(2)),
        dolfinx.fem.dirichletbc(u_fixed, dofs_bottom, V.sub(2)),
    ]


robin = (
    # pulse.RobinBC(value=robin_value, marker=geometry.markers["TOP"][0]),
    # pulse.RobinBC(value=robin_value, marker=geometry.markers["BOTTOM"][0]),
    pulse.RobinBC(value=robin_value, marker=geometry.markers["OUTSIDE_CURVED"][0]),
    pulse.RobinBC(value=robin_value, marker=geometry.markers["OUTSIDE_FLAT"][0]),
)


# We also specify the parameters for the problem and say that we want the base to move freely and that the units of the mesh is meters

parameters = {"mesh_unit": "m"}

# Next we set up the problem.
bcs = pulse.BoundaryConditions(robin=robin, neumann=neumann, dirichlet=(dirichlet_bc,))
problem = pulse.problem.StaticProblem(
    model=model, geometry=geometry, bcs=bcs, parameters=parameters,
)


# Now we can solve the problem

log.set_log_level(log.LogLevel.INFO)
problem.solve()

# We also use the time step from the problem to set the time step for the 0D cell model

dt = 0.01
times = np.arange(0.0, 1.0, dt)

# We solve the Bestel model for the pressure and activation which is already implemented in the [`circulation` package](https://computationalphysiology.github.io/circulation/examples/bestel.html)

pressure_model = circulation.bestel.BestelPressure()
res = solve_ivp(
    pressure_model,
    [0.0, 1.0],
    [0.0],
    t_eval=times,
    method="Radau",
)
# Convert the pressure from Pa to kPa
pressure = res.y[0]

activation_model = circulation.bestel.BestelActivation()
res = solve_ivp(
    activation_model,
    [0.0, 1.0],
    [0.0],
    t_eval=times,
    method="Radau",
)
# Convert the pressure from Pa to kPa
activation = res.y[0]
# activation[:] = 0

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
ax[0].plot(times, pressure)
ax[0].set_title("Pressure")
ax[1].plot(times, activation)
ax[1].set_title("Activation")
fig.savefig(outdir / "pressure_activation.png")

# Next we will save the displacement of the LV to a file

vtx = dolfinx.io.VTXWriter(
    geometry.mesh.comm, f"{outdir}/displacement.bp", [problem.u], engine="BP4",
)
vtx.write(0.0)


def region_curved_locator(x):
    # Values are located for negative x
    # Lets also pick the mid region in z-direction
    return np.logical_and(
        np.logical_and(
            np.logical_and(
                np.logical_and(x[0] < -r_inner / 2, np.abs(x[2]) < 3 * height / 4),
                np.abs(x[2]) > height / 4,
            ),
            x[1] < r_inner,
        ),
        x[1] > -r_inner,
    )


def region_flat_locator(x):
    # Values are located for positive x
    # Lets also pick the mid region in z-direction
    return np.logical_and(
        np.logical_and(
            np.logical_and(
                np.logical_and(x[0] > inner_flat_face_distance / 2, np.abs(x[2]) < 3 * height / 4),
                np.abs(x[2]) > height / 4,
            ),
            x[1] < r_inner,
        ),
        x[1] > -r_inner,
    )


# Locate cells for the curved and flat area
cells_curved = dolfinx.mesh.locate_entities(geometry.mesh, 3, region_curved_locator)
cells_flat = dolfinx.mesh.locate_entities(geometry.mesh, 3, region_flat_locator)

curved_marker = 1
flat_marker = 2

marked_values = np.hstack(
    [
        np.full_like(cells_curved, curved_marker),
        np.full_like(cells_flat, flat_marker),
    ],
)
marked_cells = np.hstack([cells_curved, cells_flat])
sorted_cells = np.argsort(marked_cells)

region_tags = dolfinx.mesh.meshtags(
    geometry.mesh,
    geometry.mesh.topology.dim,
    marked_cells,  # [sorted_cells],
    marked_values,  # [sorted_cells],
)

# Save tags for inspection in paraview
with dolfinx.io.XDMFFile(geometry.mesh.comm, f"{outdir}/regions.xdmf", "w") as xdmf:
    xdmf.write_mesh(geometry.mesh)
    xdmf.write_meshtags(region_tags, geometry.mesh.geometry)

dx_regions = ufl.Measure("dx", domain=geometry.mesh, subdomain_data=region_tags)

# Here we also use [`adios4dolfinx`](https://jsdokken.com/adios4dolfinx/README.html) to save the displacement over at different time steps. Currently it is not a straight forward way to save functions and to later load them in a time dependent simulation in FEniCSx. However `adios4dolfinx` allows us to save the function to a file and later load it in a time dependent simulation. We will first need to save the mesh to the same file.

filename = Path("function_checkpoint.bp")
adios4dolfinx.write_mesh(filename, geometry.mesh)

W = dolfinx.fem.functionspace(geometry.mesh, ("DG", 1))
F = ufl.variable(ufl.grad(problem.u) + ufl.Identity(3))
E = 0.5 * (F.T * F - ufl.Identity(3))
fiber_stress = dolfinx.fem.Function(W, name="fiber_stress")
fiber_stress_expr = dolfinx.fem.Expression(
    ufl.inner(material.sigma(F) * geo.f0, geo.f0),
    W.element.interpolation_points(),
)
radial_stress = dolfinx.fem.Function(W, name="radial_stress")
radial_stress_expr = dolfinx.fem.Expression(
    ufl.inner(material.sigma(F) * geo.n0, geo.n0),
    W.element.interpolation_points(),
)
fiber_strain = dolfinx.fem.Function(W, name="fiber_strain")
fiber_strain_expr = dolfinx.fem.Expression(
    ufl.inner(E * geo.f0, geo.f0),
    W.element.interpolation_points(),
)
radial_strain = dolfinx.fem.Function(W, name="radial_strain")
radial_strain_expr = dolfinx.fem.Expression(
    ufl.inner(E * geo.n0, geo.n0),
    W.element.interpolation_points(),
)
vtx_stress_strain = dolfinx.io.VTXWriter(
    geometry.mesh.comm,
    f"{outdir}/stress_strain.bp",
    [fiber_stress, fiber_strain, radial_stress, radial_strain],
    engine="BP4",
)
vtx_stress_strain.write(0.0)

volume_flat = comm.allreduce(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(dolfinx.fem.Constant(geometry.mesh, 1.0) * dx_regions(flat_marker)),
    ),
    op=MPI.SUM,
)
volume_curved = comm.allreduce(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(dolfinx.fem.Constant(geometry.mesh, 1.0) * dx_regions(curved_marker)),
    ),
    op=MPI.SUM,
)

fiber_stress_flat = []
fiber_stress_flat_form = dolfinx.fem.form(fiber_stress * dx_regions(flat_marker))
fiber_stress_curved = []
fiber_stress_curved_form = dolfinx.fem.form(fiber_stress * dx_regions(curved_marker))

fiber_strain_flat = []
fiber_strain_flat_form = dolfinx.fem.form(fiber_strain * dx_regions(flat_marker))
fiber_strain_curved = []
fiber_strain_curved_form = dolfinx.fem.form(fiber_strain * dx_regions(curved_marker))

radial_stress_flat = []
radial_stress_flat_form = dolfinx.fem.form(radial_stress * dx_regions(flat_marker))
radial_stress_curved = []
radial_stress_curved_form = dolfinx.fem.form(radial_stress * dx_regions(curved_marker))

radial_strain_flat = []
radial_strain_flat_form = dolfinx.fem.form(radial_strain * dx_regions(flat_marker))
radial_strain_curved = []
radial_strain_curved_form = dolfinx.fem.form(radial_strain * dx_regions(curved_marker))


for i, (tai, pi, ti) in enumerate(zip(activation, pressure, times)):
    print(f"Solving for time {ti}, activation {tai}, pressure {pi}")
    traction.assign(pi)
    Ta.assign(tai)
    problem.solve()
    fiber_strain.interpolate(fiber_strain_expr)
    fiber_stress.interpolate(fiber_stress_expr)
    radial_strain.interpolate(radial_strain_expr)
    radial_stress.interpolate(radial_stress_expr)

    fiber_stress_flat.append(
        comm.allreduce(dolfinx.fem.assemble_scalar(fiber_stress_flat_form), op=MPI.SUM)
        / volume_flat,
    )
    fiber_stress_curved.append(
        comm.allreduce(dolfinx.fem.assemble_scalar(fiber_stress_curved_form), op=MPI.SUM)
        / volume_curved,
    )

    fiber_strain_flat.append(
        comm.allreduce(dolfinx.fem.assemble_scalar(fiber_strain_flat_form), op=MPI.SUM)
        / volume_flat,
    )
    fiber_strain_curved.append(
        comm.allreduce(dolfinx.fem.assemble_scalar(fiber_strain_curved_form), op=MPI.SUM)
        / volume_curved,
    )

    radial_stress_flat.append(
        comm.allreduce(dolfinx.fem.assemble_scalar(radial_stress_flat_form), op=MPI.SUM)
        / volume_flat,
    )
    radial_stress_curved.append(
        comm.allreduce(dolfinx.fem.assemble_scalar(radial_stress_curved_form), op=MPI.SUM)
        / volume_curved,
    )

    radial_strain_flat.append(
        comm.allreduce(dolfinx.fem.assemble_scalar(radial_strain_flat_form), op=MPI.SUM)
        / volume_flat,
    )
    radial_strain_curved.append(
        comm.allreduce(dolfinx.fem.assemble_scalar(radial_strain_curved_form), op=MPI.SUM)
        / volume_curved,
    )

    vtx.write(ti)
    vtx_stress_strain.write(ti)

if comm.rank == 0:
    fig, ax = plt.subplots(2, 3, figsize=(10, 10), sharex=True)
    ax[0, 0].plot(times, fiber_stress_flat)
    ax[0, 0].set_title("Fiber Stress (flat)")
    ax[0, 1].plot(times, fiber_stress_curved)
    ax[0, 1].set_title("Fiber Stress (curved)")
    ax[0, 2].plot(times, fiber_strain_flat, label="flat region")
    ax[0, 2].plot(times, fiber_strain_curved, label="curved region")

    ax[0, 2].set_title("Fiber Strain")
    ax[1, 0].plot(times, radial_stress_flat)
    ax[1, 0].set_title("Radial Stress (flat)")
    ax[1, 1].plot(times, radial_stress_curved)
    ax[1, 1].set_title("Radial Stress (curved)")
    ax[1, 2].plot(times, radial_strain_flat, label="flat region")
    ax[1, 2].plot(times, radial_strain_curved, label="curved region")
    ax[1, 2].set_title("Radial Strain")
    for a in ax.flat:
        a.grid()

    ax[0, 2].legend()
    ax[1, 2].legend()

    fig.tight_layout()
    fig.savefig(outdir / "stress_strain.png")
