# # LV ellipsoid with time dependent pressure and activation

# In this example we will solve a time dependent mechanics problem for the left ventricle ellipsoid geometry. The pressure and activation will be time dependent.
# We use the Bestel pressure model and the Bestel activation model

from pathlib import Path
import logging
import fenicsx_pulse.problem
import fenicsx_pulse.viscoelasticity
from mpi4py import MPI
import dolfinx
import math
from dolfinx import log
import numpy as np
import circulation.bestel
from scipy.integrate import solve_ivp
import fenicsx_pulse
import cardiac_geometries
import cardiac_geometries.geometry


logging.basicConfig(level=logging.INFO)
comm = MPI.COMM_WORLD

geodir = Path("bench_ellipsoid")
if not geodir.exists():
    comm.barrier()
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        create_fibers=True,
        fiber_space="Quadrature_6",
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


geometry = fenicsx_pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

# Next we create the material object, and we will use the transversely isotropic version of the {py:class}`Holzapfel Ogden model <fenicsx_pulse.holzapfelogden.HolzapfelOgden>`

material_params = fenicsx_pulse.HolzapfelOgden.orthotropic_parameters()
material = fenicsx_pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore
# material = fenicsx_pulse.material_models.Guccione(f0=geo.f0, s0=geo.s0, n0=geo.n0)

Ta = fenicsx_pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa")
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

traction = fenicsx_pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa")
neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])


alpha_epi = fenicsx_pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e8)), "Pa / m",
)
robin_epi_u = fenicsx_pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])
beta_epi = fenicsx_pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(5e3)), "Pa s/ m",
)
robin_epi_v = fenicsx_pulse.RobinBC(value=beta_epi, marker=geometry.markers["EPI"][0], damping=True)

alpha_base = fenicsx_pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e5)), "Pa / m",
)
robin_base_u = fenicsx_pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])
beta_base = fenicsx_pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(5e3)), "Pa s/ m",
)
robin_base_v = fenicsx_pulse.RobinBC(value=beta_base, marker=geometry.markers["BASE"][0], damping=True)


bcs = fenicsx_pulse.BoundaryConditions(robin=(robin_epi_u, robin_epi_v, robin_base_u, robin_base_v), neumann=(neumann,))

problem = fenicsx_pulse.problem.DynamicProblem(model=model, geometry=geometry, bcs=bcs, parameters={"base_bc": fenicsx_pulse.problem.BaseBC.free})


# Now we can solve the problem

log.set_log_level(log.LogLevel.INFO)
problem.solve()

dt = problem.parameters["dt"].to_base_units()
times = np.arange(0.0, 1.0, dt)

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
vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, "displacement_bench.bp", [problem.u], engine="BP4")
vtx.write(0.0)

volume_form = dolfinx.fem.form(geometry.volume_form(u=problem.u) * geometry.ds(geometry.markers["ENDO"][0]))
initial_volume = geo.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(volume_form))
print(f"Initial volume: {initial_volume}")
import matplotlib.pyplot as plt

volumes = []


for i, (tai, pi, ti) in enumerate(zip(activation, pressure, times)):
    print(f"Solving for time {ti}, activation {tai}, pressure {pi}")
    traction.assign(pi)
    Ta.assign(tai)
    problem.solve()
    vtx.write((i + 1) * dt)

    volumes.append(geo.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(volume_form)))

    if geo.mesh.comm.rank == 0:
        fig, ax = plt.subplots(4, 1, figsize=(10, 10))
        ax[0].plot(times[:i + 1], pressure[:i + 1])
        ax[0].set_title("Pressure")
        ax[1].plot(times[:i + 1], activation[:i + 1])
        ax[1].set_title("Activation")
        ax[2].plot(times[:i + 1], volumes)
        ax[2].set_title("Volume")
        ax[3].plot(volumes, pressure[:i + 1])
        fig.savefig("lv_ellipsoid_time_dependent_bestel.png")
        plt.close(fig)
