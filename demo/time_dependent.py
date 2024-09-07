from pathlib import Path
import fenicsx_pulse.temporalmechanicsproblem
import fenicsx_pulse.viscoelasticity
from mpi4py import MPI
import dolfinx
from dolfinx import log
import numpy as np
import fenicsx_pulse
import cardiac_geometries
import cardiac_geometries.geometry


geodir = Path("lv_ellipsoid")
if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid(outdir=geodir, create_fibers=True, fiber_space="P_2")

# If the folder already exist, then we just load the geometry

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)
# breakpoint()
# geo.mesh.geometry.x[:] /= 100.0


geometry = fenicsx_pulse.Geometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})

# Next we create the material object, and we will use the transversely isotropic version of the {py:class}`Holzapfel Ogden model <fenicsx_pulse.holzapfelogden.HolzapfelOgden>`

material_params = fenicsx_pulse.HolzapfelOgden.orthotropic_parameters()
# material_params["a"] *= 1000.0
# material_params["a_f"] *= 1000.0
# material_params["a_fs"] *= 1000.0
# material_params["a_s"] *= 1000.0

material = fenicsx_pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore

# We use an active stress approach with 30% transverse active stress (see {py:meth}`fenicsx_pulse.active_stress.transversely_active_stress`)

Ta = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
active_model = fenicsx_pulse.ActiveStress(geo.f0, activation=Ta)

# We use an incompressible model

comp_model = fenicsx_pulse.compressibility.Compressible2()

viscoeleastic_model = fenicsx_pulse.viscoelasticity.Viscous()

# and assembles the `CardiacModel`

model = fenicsx_pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
    viscoelasticity=viscoeleastic_model,
)

traction = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])


bcs = fenicsx_pulse.BoundaryConditions(neumann=(neumann,))

# and create a Mixed problem

problem = fenicsx_pulse.temporalmechanicsproblem.Problem(model=model, geometry=geometry, bcs=bcs)
# problem = fenicsx_pulse.MechanicsProblem(model=model, geometry=geometry, bcs=bcs)

# Now we can solve the problem

log.set_log_level(log.LogLevel.INFO)
problem.solve()

dt = problem.parameters["dt"]
times = np.arange(0.0, 0.1, dt)
pressures = np.linspace(0.0, 20.0, len(times))


vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, "displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)

for i, p in enumerate(pressures):
    traction.value = p
    problem.solve()
    vtx.write((i + 1) * dt)
