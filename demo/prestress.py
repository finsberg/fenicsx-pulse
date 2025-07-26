# # Pre-stressing of a left ventricle ellipsoid

# In this demo we will show how to simulate an idealized left ventricular geometry with pre-stressing
# using the approach described in {cite}`barnafi2024reconstructing`.

from pathlib import Path
from mpi4py import MPI
import dolfinx
import logging

import math
import numpy as np
import pulse.prestress
import pulse
import cardiac_geometries
import cardiac_geometries.geometry


comm = MPI.COMM_WORLD
logging.basicConfig(level=logging.INFO)
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)


geodir = Path("lv_ellipsoid-prestress")
if not geodir.exists():
    comm.barrier()
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        create_fibers=True,
        fiber_space="P_2",
        r_short_endo=0.025,
        r_short_epi=0.035,
        r_long_endo=0.09,
        r_long_epi=0.097,
        psize_ref=0.03,
        mu_apex_endo=-math.pi,
        mu_base_endo=-math.acos(5 / 17),
        mu_apex_epi=-math.pi,
        mu_base_epi=-math.acos(5 / 20),
        comm=comm,
        fiber_angle_epi=-60,
        fiber_angle_endo=60,
    )


geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)

geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})


target_pressure = 2000.0
pressure = pulse.Variable(dolfinx.fem.Constant(geo.mesh, 0.0), "Pa")


material = pulse.material_models.Usyk(f0=geo.f0, s0=geo.s0, n0=geo.n0)
comp = pulse.compressibility.Compressible3(kappa=pulse.Variable(5e4, "Pa"))
model = pulse.CardiacModel(
    material=material, compressibility=comp, active=pulse.active_model.Passive(),
)


alpha_epi = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2e5)),
    "Pa / m",
)
robin_epi = pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])
alpha_epi_perp = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2e5 / 10)),
    "Pa / m",
)
robin_epi_perp = pulse.RobinBC(
    value=alpha_epi_perp, marker=geometry.markers["EPI"][0], perpendicular=True,
)

neumann = pulse.NeumannBC(traction=pressure, marker=geometry.markers["ENDO"][0])

bcs = pulse.BoundaryConditions(neumann=(neumann,), robin=(robin_epi, robin_epi_perp))


prestress_problem = pulse.prestress.PrestressProblem(
    geometry=geometry,
    model=model,
    bcs=bcs,
    parameters={"u_space": "P_2"},
)

ramp_steps = 5
for ramp in np.linspace(0.0, 1.0, ramp_steps):
    pressure.assign(target_pressure * ramp)
    prestress_problem.solve()

with dolfinx.io.VTXWriter(
    comm, "prestress_backward.bp", [prestress_problem.u], engine="BP4",
) as vtx:
    vtx.write(0.0)


geometry.deform(prestress_problem.u)

forward_problem = pulse.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs,
    parameters={"u_space": "P_2"},
)

for ramp in np.linspace(0.0, 1.0, ramp_steps):
    pressure.assign(target_pressure * ramp)
    forward_problem.solve()


with dolfinx.io.VTXWriter(comm, "prestress_forward.bp", [forward_problem.u], engine="BP4") as vtx:
    vtx.write(0.0)


# # References
# ```{bibliography}
# :filter: docname in docnames
# ```
