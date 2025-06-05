# # BiV ellipsoid with time dependent pressure and activation

# This example is very similar to the [LV example](time_dependent_bestel_lv.py), only that we now will use a biventricular geometry, meaning that there are two different pressure boundary conditions, one for the left ventricle and one for the right ventricle. We will also use different parameters for the Bestel pressure model for the left and right ventricle.
# The reader are referred to the [LV example](time_dependent_bestel_lv.py) for more details on the models used.

from pathlib import Path
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import dolfinx
from dolfinx import log
from scipy.integrate import solve_ivp
import circulation.bestel
import cardiac_geometries
import cardiac_geometries.geometry
import pulse


logging.basicConfig(level=logging.INFO)
comm = MPI.COMM_WORLD

outdir = Path("time-dependent-bestel-biv")
outdir.mkdir(parents=True, exist_ok=True)

geodir = outdir / "geometry"
if not geodir.exists():
    comm.barrier()
    cardiac_geometries.mesh.biv_ellipsoid(
        outdir=geodir,
        create_fibers=True,
        fiber_space="Quadrature_6",
        comm=comm,
        char_length=1.0,
        fiber_angle_epi=-60,
        fiber_angle_endo=60,
    )

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)

# In this case we scale the geometry to be in meters

geo.mesh.geometry.x[:] *= 3e-2

# We create the geometry object and print the volumes of the LV and RV cavities

geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})
print(geometry.volume("ENDO_LV") * 1e6, geometry.volume("ENDO_RV") * 1e6)

material_params = pulse.HolzapfelOgden.orthotropic_parameters()
material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore

Ta = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa")
active_model = pulse.ActiveStress(geo.f0, activation=Ta)
comp_model = pulse.compressibility.Compressible2()
viscoeleastic_model = pulse.viscoelasticity.Viscous()
model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
    viscoelasticity=viscoeleastic_model,
)

# One difference with the LV example is that we now have two different pressure boundary conditions, one for the LV and one for the RV

traction_lv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa")
neumann_lv = pulse.NeumannBC(traction=traction_lv, marker=geometry.markers["ENDO_LV"][0])

traction_rv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa")
neumann_rv = pulse.NeumannBC(traction=traction_rv, marker=geometry.markers["ENDO_RV"][0])

# Otherwize we have the same Robin boundary conditions as in the LV example

alpha_epi = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e8)), "Pa / m",
)
robin_epi_u = pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])
beta_epi = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(5e3)), "Pa s/ m",
)
robin_epi_v = pulse.RobinBC(value=beta_epi, marker=geometry.markers["EPI"][0], damping=True)

alpha_base = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e5)), "Pa / m",
)
robin_base_u = pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])
beta_base = pulse.Variable(
    dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(5e3)), "Pa s/ m",
)
robin_base_v = pulse.RobinBC(value=beta_base, marker=geometry.markers["BASE"][0], damping=True)

bcs = pulse.BoundaryConditions(neumann=(neumann_lv, neumann_rv), robin=(robin_epi_u, robin_epi_v, robin_base_u, robin_base_v))
problem = pulse.problem.DynamicProblem(model=model, geometry=geometry, bcs=bcs)


log.set_log_level(log.LogLevel.INFO)
problem.solve()

dt = problem.parameters["dt"].to_base_units()
times = np.arange(0.0, 1.0, dt)

# We now solve the Bestel pressure model using different parameters for the LV and RV

lv_pressure_model = circulation.bestel.BestelPressure(
    parameters=dict(
        t_sys_pre=0.17,
        t_dias_pre=0.484,
        gamma=0.005,
        a_max=5.0,
        a_min=-30.0,
        alpha_pre=5.0,
        alpha_mid=15.0,
        sigma_pre=12000.0,
        sigma_mid=16000.0,
    ),
)
res_lv = solve_ivp(
    lv_pressure_model,
    [0.0, 1.0],
    [0.0],
    t_eval=times,
    method="Radau",
)
lv_pressure = res_lv.y[0]


rv_pressure_model = circulation.bestel.BestelPressure(
    parameters=dict(
        t_sys_pre=0.17,
        t_dias_pre=0.484,
        gamma=0.005,
        a_max=5.0,
        a_min=-30.0,
        alpha_pre=1.0,
        alpha_mid=10.0,
        sigma_pre=3000.0,
        sigma_mid=4000.0,
    ),
)
res_rv = solve_ivp(
    rv_pressure_model,
    [0.0, 1.0],
    [0.0],
    t_eval=times,
    method="Radau",
)
rv_pressure = res_rv.y[0]

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


fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
ax[0].plot(times, lv_pressure, label="LV")
ax[0].plot(times, rv_pressure, label="RV")
ax[0].set_title("Pressure")
ax[0].legend()
ax[1].plot(times, activation)
ax[1].set_title("Activation")
fig.savefig(outdir / "pressure_activation.png")


vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, outdir / "displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)

volume_form = geometry.volume_form(u=problem.u)
lv_volume_form = dolfinx.fem.form(volume_form * geometry.ds(geometry.markers["ENDO_LV"][0]))
rv_volume_form = dolfinx.fem.form(volume_form * geometry.ds(geometry.markers["ENDO_RV"][0]))

lv_volumes = []
rv_volumes = []
for i, (tai, plv, prv, ti) in enumerate(zip(activation, lv_pressure, rv_pressure, times)):
    print(f"Solving for time {ti}, activation {tai}, lv pressure {plv} and rv pressure {prv}")
    traction_lv.assign(plv)
    traction_rv.assign(prv)
    Ta.assign(tai)
    problem.solve()
    vtx.write((i + 1) * dt)

    lv_volumes.append(dolfinx.fem.assemble_scalar(lv_volume_form))
    rv_volumes.append(dolfinx.fem.assemble_scalar(rv_volume_form))

    if geo.mesh.comm.rank == 0:
        fig, ax = plt.subplots(4, 1, figsize=(10, 10))
        ax[0].plot(times[:i + 1], lv_pressure[:i + 1], label="LV")
        ax[0].plot(times[:i + 1], rv_pressure[:i + 1], label="RV")
        ax[0].set_title("Pressure")
        ax[1].plot(times[:i + 1], activation[:i + 1], label="Activation")
        ax[1].set_title("Activation")
        ax[2].plot(times[:i + 1], lv_volumes, label="LV")
        ax[2].plot(times[:i + 1], rv_volumes, label="RV")
        ax[2].set_title("Volume")
        ax[3].plot(lv_volumes, lv_pressure[:i + 1], label="LV")
        ax[3].plot(rv_volumes, rv_pressure[:i + 1], label="RV")

        for a in ax:
            a.legend()
        fig.savefig(outdir / "biv_ellipsoid_time_dependent_bestel.png")
        plt.close(fig)

    if os.getenv("CI") and i > 2:
        # Early stopping for CI
        break

# <video controls loop autoplay muted>
#   <source src="../_static/time_dependent_bestel_biv.mp4" type="video/mp4">
#   <p>Video showing the motion of the BiV.</p>
# </video>
#
