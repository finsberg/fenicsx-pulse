# # LV Ellipsoid
#

from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import fenicsx_pulse
import ufl

import cardiac_geometries
import cardiac_geometries.geometry

geodir = Path("lv_ellipsoid")
if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid(outdir=geodir, create_fibers=True)
geometry = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)
geometry.dx = ufl.Measure("dx", domain=geometry.mesh, metadata={"quadrature_degree": 4})
geometry.ds = ufl.Measure(
    "ds",
    domain=geometry.mesh,
    subdomain_data=geometry.ffun,
    metadata={"quadrature_degree": 4},
)
geometry.facet_normal = ufl.FacetNormal(geometry.mesh)
geometry.facet_tags = geometry.ffun


material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters()
material = fenicsx_pulse.HolzapfelOgden(f0=geometry.f0, s0=geometry.s0, **material_params)  # type: ignore


Ta = dolfinx.fem.Constant(geometry.mesh, PETSc.ScalarType(0.0))
active_model = fenicsx_pulse.ActiveStress(geometry.f0, activation=Ta)


comp_model = fenicsx_pulse.Incompressible()


model = fenicsx_pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)


def dirichlet_bc(
    state_space: dolfinx.fem.FunctionSpace,
) -> list[dolfinx.fem.bcs.DirichletBC]:
    V, _ = state_space.sub(0).collapse()
    facets = geometry.facet_tags.find(
        geometry.markers["BASE"][0],
    )  # Specify the marker used on the boundary
    geometry.mesh.topology.create_connectivity(
        geometry.mesh.topology.dim - 1,
        geometry.mesh.topology.dim,
    )
    dofs = dolfinx.fem.locate_dofs_topological((state_space.sub(0), V), 2, facets)
    u_fixed = dolfinx.fem.Function(V)
    u_fixed.x.array[:] = 0.0
    return [dolfinx.fem.dirichletbc(u_fixed, dofs, state_space.sub(0))]


traction = dolfinx.fem.Constant(geometry.mesh, PETSc.ScalarType(0.0))
neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])


bcs = fenicsx_pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))


problem = fenicsx_pulse.MechanicsProblem(model=model, geometry=geometry, bcs=bcs)


problem.solve()

u = problem.state.sub(0).collapse()
vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, "lv_displacement.bp", [u], engine="BP4")
vtx.write(0.0)

i = 1
for plv in [0.1, 0.5, 1.0]:
    print(f"plv: {plv}")
    traction.value = plv
    problem.solve()
    u = problem.state.sub(0).collapse()
    vtx.write(float(i))
    i += 1


for ta in [0.1, 0.5, 1.0]:
    print(f"ta: {ta}")
    Ta.value = ta
    problem.solve()
    u = problem.state.sub(0).collapse()
    vtx.write(float(i))
    i += 1

vtx.close()
