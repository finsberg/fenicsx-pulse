# # LV Ellipsoid
#
# In this demo we will show how to simulate an idealized left ventricular geometry. For this we will use [`cardiac-geometries`](https://github.com/ComputationalPhysiology/cardiac-geometriesx) to generate an idealized LV ellipsoid.
#
# First we will make the necessary imports.

from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import fenicsx_pulse
import ufl
import cardiac_geometries
import cardiac_geometries.geometry

# Next we will create the geometry and save it in the folder called `lv_ellipsoid`. We also make sure to generate fibers which can be done analytically and use a second order Lagrange space for the fibers

geodir = Path("lv_ellipsoid")
if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid(outdir=geodir, create_fibers=True, fiber_space="P_2")

# If the folder already exist, then we just load the geometry

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)

# In order to use the geometry with `pulse` we need to convert it to a `fenicsx_pulse.Geometry` object. We can do this by using the `from_cardiac_geometries` method. We also specify that we want to use a quadrature degree of 4
#

geometry = fenicsx_pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})

# Next we create the material object, and we will use the transversely isotropic version of the {py:class}`Holzapfel Ogden model <fenicsx_pulse.holzapfelogden.HolzapfelOgden>`

material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters()
material = fenicsx_pulse.HolzapfelOgden(f0=geometry.f0, s0=geometry.s0, **material_params)  # type: ignore

# We use an active stress approach with 30% transverse active stress (see {py:meth}`fenicsx_pulse.active_stress.transversely_active_stress`)

Ta = dolfinx.fem.Constant(geometry.mesh, PETSc.ScalarType(0.0))
active_model = fenicsx_pulse.ActiveStress(geometry.f0, activation=Ta, eta=0.3)

# We use an incompressible model

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


# We apply a traction in endocardium

traction = dolfinx.fem.Constant(geometry.mesh, PETSc.ScalarType(0.0))
neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])

# and finally combine all the boundary conditions

bcs = fenicsx_pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

# and create a Mixed problem

problem = fenicsx_pulse.MechanicsProblemMixed(model=model, geometry=geometry, bcs=bcs)

# Now we can solve the problem

problem.solve()

# And save the displacement to a file that we can view in Paraview

u = problem.state.sub(0).collapse()
vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, "lv_displacement.bp", [u], engine="BP4")
vtx.write(0.0)

i = 1
for plv in [0.1]: #, 0.5, 1.0]:
    print(f"plv: {plv}")
    traction.value = plv
    problem.solve()
    u = problem.state.sub(0).collapse()
    vtx.write(float(i))
    i += 1


for ta in [0.1]: #, 0.5, 1.0]:
    print(f"ta: {ta}")
    Ta.value = ta
    problem.solve()
    u = problem.state.sub(0).collapse()
    vtx.write(float(i))
    i += 1

vtx.close()
