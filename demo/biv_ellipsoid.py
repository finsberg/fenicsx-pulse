# # BiV Ellipsoid
#
# In this example we will simulate an idealized bi-ventricular geometry. For this we will use [`cardiac-geometries`](https://github.com/ComputationalPhysiology/cardiac-geometriesx) to generate an idealized LV ellipsoid.
#
# First lets do some imports

from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import log
import fenicsx_pulse
import ufl
import ldrb
import cardiac_geometries
import cardiac_geometries.geometry

# and lets turn on logging so that we can see more info from `dolfinx`

log.set_log_level(log.LogLevel.INFO)

# Now we create the geometry using  [`cardiac-geometries`](https://github.com/ComputationalPhysiology/cardiac-geometriesx) and save it to a folder called `biv_ellipsoid`. We will also create fiber orientations using the Laplace-Dirichlet Rule based (LDRB) algorithm, using the library [`fenicsx-ldrb`](https://github.com/finsberg/fenicsx-ldrb) package

geodir = Path("biv_ellipsoid")
if not geodir.exists():
    geo = cardiac_geometries.mesh.biv_ellipsoid(outdir=geodir)
    system = ldrb.dolfinx_ldrb(mesh=geo.mesh, ffun=geo.ffun, markers=geo.markers, alpha_endo_lv=60, alpha_epi_lv=-60, beta_endo_lv=0, beta_epi_lv=0, fiber_space="P_2")
    cardiac_geometries.fibers.utils.save_microstructure(mesh=geo.mesh, functions=[system.f0, system.s0, system.n0], outdir=geodir)

# If the folder exist we just load it

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)

# In order to use the geometry with `pulse` we need to convert it to a `fenicsx_pulse.Geometry` object. We can do this by using the `from_cardiac_geometries` method. We also specify that we want to use a quadrature degree of 4
#

geometry = fenicsx_pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})

# Next we create the material object, and we will use the transversely isotropic version of the {py:class}`Neo Hookean model <fenicsx_pulse.neo_hookean.NeoHookean>`

material = fenicsx_pulse.NeoHookean(mu=dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(15.0)))
# and use an active stress approach

Ta = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
active_model = fenicsx_pulse.ActiveStress(geo.f0, activation=Ta)

# Now we will also implement two different versions, one where we use a compressible model and one where we use an incompressible model. To do this we will introduce a flag

incompressible = False

# And in both cases we will use different compressible models, mechanics problems and different ways to get the displacement.

if incompressible:
    comp_model: fenicsx_pulse.Compressibility = fenicsx_pulse.Incompressible()
    def get_u(problem: fenicsx_pulse.BaseMechanicsProblem) -> dolfinx.fem.Function:
        return problem.state.sub(0).collapse()

    Problem: type[fenicsx_pulse.BaseMechanicsProblem] = fenicsx_pulse.MechanicsProblemMixed
else:
    comp_model = fenicsx_pulse.Compressible()
    def get_u(problem: fenicsx_pulse.BaseMechanicsProblem) -> dolfinx.fem.Function:
        return problem.state

    Problem = fenicsx_pulse.MechanicsProblem

# Now we can assemble the `CardiacModel`

model = fenicsx_pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)


# And we will now implement a Dirichlet BC where we fix the base in the $x$-direction.

def dirichlet_bc(
    state_space: dolfinx.fem.FunctionSpace,
) -> list[dolfinx.fem.bcs.DirichletBC]:

    if incompressible:
        Ux = state_space.sub(0).sub(0)
    else:
        Ux = state_space.sub(0)

    V, _ = Ux.collapse()

    facets = geometry.facet_tags.find(
        geometry.markers["BASE"][0],
    )  # Specify the marker used on the boundary
    geometry.mesh.topology.create_connectivity(
        geometry.mesh.topology.dim - 1,
        geometry.mesh.topology.dim,
    )
    dofs = dolfinx.fem.locate_dofs_topological((Ux, V), 2, facets)
    u_fixed = dolfinx.fem.Function(V)
    u_fixed.x.array[:] = 0.0
    return [dolfinx.fem.dirichletbc(u_fixed, dofs, Ux)]


# We will add a pressure on the LV endocarium

lvp = dolfinx.fem.Constant(geometry.mesh, PETSc.ScalarType(0.0))
neumann_lv = fenicsx_pulse.NeumannBC(traction=lvp, marker=geometry.markers["ENDO_LV"][0])

# and on the RV endocardium

rvp = dolfinx.fem.Constant(geometry.mesh, PETSc.ScalarType(0.0))
neumann_rv = fenicsx_pulse.NeumannBC(traction=lvp, marker=geometry.markers["ENDO_RV"][0])

# We will also add a Robin type spring on the epicardial surface to mimic the pericardium.

pericardium = dolfinx.fem.Constant(geometry.mesh, PETSc.ScalarType(1.0))
robin_per = fenicsx_pulse.RobinBC(value=pericardium, marker=geometry.markers["EPI"][0])

# We collect all the boundary conditions

bcs = fenicsx_pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann_lv, neumann_rv), robin=(robin_per,))

# create the problem

problem = Problem(model=model, geometry=geometry, bcs=bcs)

# and solve

problem.solve()

# Now let us inflate the two ventricles and save the displacement

u = get_u(problem)
U = dolfinx.fem.Function(u.function_space)
U.x.array[:] = u.x.array[:]
vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, "biv_displacement.bp", [U], engine="BP4")
vtx.write(0.0)

i = 1
for plv in [0.1]: #, 0.5, 1.0, 2.0]:
    print(f"plv: {plv}")
    lvp.value = plv
    rvp.value = plv * 0.2
    problem.solve()
    u = get_u(problem)
    U.x.array[:] = u.x.array[:]
    vtx.write(float(i))
    i += 1

# and then apply an active tension

for ta in [0.1] : #, 1.0, 5.0, 10.0]:
    print(f"ta: {ta}")
    Ta.value = ta
    problem.solve()
    u = get_u(problem)
    U.x.array[:] = u.x.array[:]
    vtx.write(float(i))
    i += 1

vtx.close()
