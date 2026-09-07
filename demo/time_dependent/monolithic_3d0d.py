# # Monolithic 3D-0D coupling: an LV in a closed circulation
#
# An LV ellipsoid is coupled to the closed-loop circulation model of Regazzoni
# et al., with the displacement, the cavity pressure and all twelve circuit
# states solved together in one Newton system. The cavity volume and the
# circuit's own `V_LV` are then never allowed to disagree: the constraint that
# ties them is a row of the system, satisfied to solver tolerance, rather than
# something a fixed number of exchange iterations gets close to.
#
# The alternative is a partitioned scheme, where the two are solved in turn and
# the volume and pressure are passed back and forth until they stop moving.
# That converges to the same answer when it converges, but the agreement is
# only ever as good as the exchange budget allows, and the budget has to be
# spent every step.
#
# ## How the coupling is set up
#
# `circulation` ships the Regazzoni model as a `.ode` file, with each chamber's
# pressure closure in its own component. Removing the LV component leaves a
# model that no longer computes `p_LV` but still carries `V_LV` as a state, and
# expects the pressure to be supplied. That is the seam:
#
# * `V_LV` is an unknown of the Newton system, constrained to the deformed
#   cavity volume;
# * `p_LV` is supplied by the cavity pressure the mechanics problem already
#   carries as a Lagrange multiplier.
#
# The `timing` component goes too, since it computes the beat phase with `Mod`,
# which UFL does not have. The phase is a function of time alone, so it is
# supplied from outside and contributes nothing to any derivative.
#
# ## Activation
#
# `Ta` comes from the Bestel model, which is an ODE in time alone with no
# dependence on the mechanics state. Prescribing its solution is therefore
# exactly equivalent to solving it alongside, with no coupling error of any
# kind. That is deliberate: it leaves the 3D-0D coupling as the only
# approximation in the scheme, which is what makes a comparison against the
# partitioned version mean something.
#
# A model whose tension responds to fibre stretch, such as the crossbridge
# model used by the full-ecosystem demo, is a genuinely coupled subsystem and
# would need its own treatment. Its states are per-quadrature-point fields
# rather than global scalars, so it does not fit the machinery used here.
#
# ## Calibration
#
# The circuit's parameters were tuned against its own LV elastance, and this
# mesh is nothing like that elastance. Its cavity carries no pressure until
# about 125 mL, where the circuit's chamber is unstressed at 42 mL, so through
# the whole of diastole the circuit pushes against a ventricle that has not
# begun to resist. Coupled without further ado, it fills to roughly 197 mL.
#
# So before anything is stepped, the cavity pressure is measured over a grid of
# volumes and activations, and the loading is tuned in pure 0D against that
# measurement. Because the mechanics is quasi-static, that measurement is not an
# approximation of the ventricle -- it is the ventricle, sampled. See
# `calibration.py`; `PULSE_CALIBRATE=0` skips it.

import logging
import os
from pathlib import Path

from mpi4py import MPI

# A sibling module in this directory, not a package -- see `calibration.py`.
import calibration as calib
import circulation
import dolfinx
import io4dolfinx
import matplotlib.pyplot as plt
import numpy as np
from circulation import bestel, regazzoni2020
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp

import cardiac_geometries
import cardiac_geometries.geometry
import pulse
from pulse.circulation import ChamberCoupling, GotranxCirculation, mL, mmHg

circulation.log.setup_logging(logging.INFO)
logging.getLogger("scifem").setLevel(logging.WARNING)
logger = logging.getLogger("pulse")
comm = MPI.COMM_WORLD

# `CI` is parsed rather than merely tested, because `os.getenv` returns a string
# and `CI=0` -- explicitly not CI -- would otherwise be truthy.
_ci = os.getenv("CI", "").strip().lower()
IN_CI = _ci not in ("", "0", "false", "no", "off")

# Sampling the mesh costs upwards of a hundred static solves, which is not
# something a docs build should pay for. In CI the demo runs two steps against
# the circuit's published parameters, where the operating point is beside the
# point. An explicit `PULSE_CALIBRATE` still wins.
if IN_CI:
    os.environ.setdefault("PULSE_CALIBRATE", "0")

# Which active-stress convention the ventricle is built with. The two differ by
# a factor of the fibre stretch, and the partitioned demo this is to be compared
# against uses `StabilizedActiveStress`, whose active stress is
# `[Ta + Ka*dlambda] * F f0 (x) f0 / |F f0|` -- the `stretch` convention, after
# Regazzoni & Quarteroni. With `Ta` prescribed there is no force-generation
# solver to stabilize against, `Ka` is zero and the stabilization term vanishes,
# at which point that class reduces exactly to `ActiveStress` with `stretch`. So
# both arms of the comparison can use the plain class, and must use the same
# convention or the comparison measures the convention as much as the coupling.
#
# `stretch` is also the better-founded reading of a `Ta` that comes from a cell
# model: `P f0` is the force on a *reference* cross-section, and a fixed number
# of crossbridges per reference area is a fixed force per reference area, so
# `|P_a f0| = Ta`. `invariant`, the historical default, makes it `Ta*lambda`.
#
# Set to `invariant` to reproduce the earlier runs; the calibration cache is
# keyed on this, so the two do not overwrite each other.
FORMULATION = pulse.ActiveStressFormulation.stretch

BEAT_LENGTH = 1.0  # s
DT = 0.002  # s
NUM_BEATS = 1 if IN_CI else 2

# The mesh, the prestress solve and the calibration are all cached, so re-running
# this costs only the time-stepping.
cachedir = Path("results_monolithic_3d0d")
geodir = Path("lv_ellipsoid-monolithic-3d0d")
outdir = Path("results_monolithic_3d0d")
outdir.mkdir(exist_ok=True)

# ## Geometry
#
# The same idealized LV ellipsoid the other demos use. The shape represents a
# loaded, end-diastolic configuration, so it is unloaded by prestressing before
# anything is stepped.

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
        mu_apex_endo=-np.pi,
        mu_base_endo=-np.arccos(5 / 17),
        mu_apex_epi=-np.pi,
        mu_base_epi=-np.arccos(5 / 20),
        comm=comm,
        fiber_angle_epi=-60,
        fiber_angle_endo=60,
    )

geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)
geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

target_volume = comm.allreduce(geometry.volume("ENDO"), op=MPI.SUM)
logger.info(f"Target (end-diastolic) volume: {target_volume / mL:.2f} mL")


def build_model(f0, s0, Ta, incompressible=False):
    material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material = pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)  # type: ignore[arg-type]
    comp = pulse.Incompressible() if incompressible else pulse.Compressible()
    return pulse.CardiacModel(
        material=material,
        active=pulse.ActiveStress(f0, activation=Ta, formulation=FORMULATION),
        compressibility=comp,
    )


def robin_bcs():
    return (
        pulse.RobinBC(
            value=pulse.Variable(
                dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0e5)),
                "Pa / m",
            ),
            marker=geometry.markers["EPI"][0],
        ),
        pulse.RobinBC(
            value=pulse.Variable(
                dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1.0e5)),
                "Pa / m",
            ),
            marker=geometry.markers["BASE"][0],
        ),
    )


# ## Activation
#
# Solved once, up front, since it does not depend on anything else.

times = np.arange(0.0, BEAT_LENGTH, DT)
activation_trace = solve_ivp(
    bestel.BestelActivation(),
    [0.0, BEAT_LENGTH],
    [0.0],
    t_eval=times,
    method="Radau",
).y[0]
logger.info(f"Peak synthetic Ta: {activation_trace.max() * 1e-3:.2f} kPa")


def activation_at(t: float) -> float:
    """Ta at time t, repeating each beat."""
    return float(np.interp(t % BEAT_LENGTH, times, activation_trace))


# ## A pressure to prestress against
#
# The circuit is run on its own first, with its own LV elastance, purely to get
# an end-diastolic pressure. That pressure defines the unloaded reference
# configuration, and it has to be fixed before the mesh can be measured -- the
# measurement is taken on the unloaded mesh. So this one number stays at the
# circuit's published value and is a modelling input, not a result. The
# calibration below reports the end-diastolic pressure it settles on, which is
# what to compare it against.

if comm.rank == 0 and not (cachedir / "circ_state.npy").exists():
    cachedir.mkdir(exist_ok=True)
    standalone = regazzoni2020.Regazzoni2020(parameters={"HR": 1.0}, add_units=False)
    history = standalone.solve(
        num_beats=10,
        initial_state={"V_LV": target_volume / mL},
        dt=0.001,
    )
    np.save(
        cachedir / "circ_state.npy",
        dict(zip(standalone.state_names(), standalone.state)),
        allow_pickle=True,
    )
    np.save(cachedir / "p_LV_ED.npy", float(history["p_LV"][-1]))
comm.barrier()

circ_state = np.load(cachedir / "circ_state.npy", allow_pickle=True).item()
p_LV_ED = float(np.load(cachedir / "p_LV_ED.npy"))
p_LV_ED_kPa = circulation.units.ureg.Quantity(p_LV_ED, "mmHg").to("kPa").magnitude
logger.info(f"End-diastolic pressure for prestressing: {p_LV_ED_kPa:.3f} kPa")

# ## Prestressing
#
# Recover the unloaded reference configuration, reusing the cached result.

Ta = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa")
pressure_lv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "kPa")

prestress_fname = cachedir / "prestress_lv.bp"
if not prestress_fname.exists():
    logger.info("Prestressing to recover the unloaded reference configuration...")
    prestress_problem = pulse.unloading.PrestressProblem(
        geometry=geometry,
        model=build_model(geo.f0, geo.s0, Ta),
        bcs=pulse.BoundaryConditions(
            robin=robin_bcs(),
            neumann=(pulse.NeumannBC(traction=pressure_lv, marker=geometry.markers["ENDO"][0]),),
        ),
        parameters={"u_space": "P_2", "mesh_unit": "m"},
        targets=[
            pulse.unloading.TargetPressure(traction=pressure_lv, target=p_LV_ED_kPa, name="LV"),
        ],
        ramp_steps=20,
    )
    u_pre = prestress_problem.unload()
    io4dolfinx.write_function_on_input_mesh(prestress_fname, u_pre, time=0.0, name="u_pre")
comm.barrier()

V_disp = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 2, (3,)))
u_pre = dolfinx.fem.Function(V_disp)
io4dolfinx.read_function(prestress_fname, u_pre, time=0.0, name="u_pre")

geometry.deform(u_pre)
f0 = pulse.utils.map_vector_field(f=geo.f0, u=u_pre, normalize=True, name="f0_unloaded")
s0 = pulse.utils.map_vector_field(f=geo.s0, u=u_pre, normalize=True, name="s0_unloaded")

unloaded_volume = comm.allreduce(geometry.volume("ENDO"), op=MPI.SUM)
logger.info(f"Unloaded volume: {unloaded_volume / mL:.2f} mL")

# ## A volume-controlled problem
#
# Used for two things: measuring the mesh for the calibration, and then
# inflating from the unloaded configuration to the operating point. Both want
# the cavity volume prescribed rather than coupled -- the reference
# configuration is far from end-diastole, and getting there is a ramp, not a
# timestep. It carries the same model and the same boundary conditions as the
# coupled problem, so the pressure it reports at a given volume and activation
# is the pressure the coupled problem would report.

model = build_model(f0, s0, Ta, incompressible=True)
bcs = pulse.BoundaryConditions(robin=robin_bcs())

inflation_volume = dolfinx.fem.Constant(
    geometry.mesh,
    dolfinx.default_scalar_type(unloaded_volume),
)
inflation = pulse.problem.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs,
    cavities=[pulse.problem.Cavity(marker="ENDO", volume=inflation_volume)],
    parameters={"mesh_unit": "m"},
)
inflation.solve()

# ## Calibration
#
# Measure the cavity pressure over a grid of volumes and activations, then tune
# contractility, afterload and preload in pure 0D against that measurement.
# Both halves are cached; the expensive half is the grid.

base_parameters = circulation.base.remove_units(
    regazzoni2020.Regazzoni2020.default_parameters() | {"HR": 1.0},
)
Ta_ref = float(activation_trace.max())


def build_surface():
    logger.info("Measuring the cavity pressure over a grid of volumes and activations...")
    return calib.sample_pv_surface(
        inflation,
        inflation_volume,
        Ta,
        # Centered on the unloaded volume and wide on both sides, so the
        # calibration searches inside the sampled box rather than off its edge.
        # How far down it actually gets is up to the mesh: holding a volume
        # well below the unloaded one takes suction, and the sampler finds the
        # limit rather than being told it.
        volumes=np.linspace(0.4 * unloaded_volume, 2.2 * unloaded_volume, 14) / mL,
        # Up to three times the Bestel peak, which is what bounds `Ta_scale`,
        # and spaced quadratically rather than evenly. The chamber responds to
        # the first few kilopascals far more than to the last few -- the
        # measured pressure climbs 66 mmHg over the first 39 kPa and 30 mmHg
        # over the next 40 -- so an even grid puts its points where the surface
        # is already straight and interpolates across the bend.
        activations=3.0 * Ta_ref * np.linspace(0.0, 1.0, 10) ** 2,
        comm=comm,
    )


# The ellipsoid's unloaded cavity holds far more than a person's, so a textbook
# end-diastolic volume would ask it to fill to a pressure no ventricle fills to.
# The target is read off the measured resting curve instead, at the volume where
# filling reaches 8 mmHg. The ejection fraction and the peak pressure are plain
# numbers, since neither presumes a chamber size.
#
# The activation ceiling caps how much of the sampled grid is used. The mesh
# will not hold its most dilated volumes under the strongest tensions, and every
# volume that fails anywhere below the ceiling has to be dropped for the rest to
# form a rectangle -- so a lower ceiling buys back volume range. Two beats of
# headroom over the contractility the calibration settles on is plenty.
calibration = calib.load_or_calibrate(
    cachedir / f"calibration-{FORMULATION.value}.json",
    build_surface=build_surface,
    activation_at=activation_at,
    base_parameters=base_parameters,
    Ta_ref=Ta_ref,
    comm=comm,
    targets=calib.Targets(EF=0.45, p_max=120.0, filling_pressure=8.0),
    max_activation=2.0 * Ta_ref,
    provenance={"geometry": str(geodir), "beat_length": BEAT_LENGTH, "dt": DT},
)

if calibration is not None:
    logger.info("\n" + calibration.summary())
    circ_state = calibration.initial_state
    circ_parameters = calibration.circulation_parameters(base_parameters)
    Ta_scale = calibration.Ta_scale
else:
    circ_parameters = base_parameters
    Ta_scale = 1.0

# ## Inflation to the operating point
#
# Sampling leaves the mesh wherever the last grid point was, which is at full
# activation. Both the volume and the activation are walked back from there
# together: dropping a few hundred kilopascals of tension in one step is not
# something Newton recovers from, however gentle the volume ramp beside it is.

target = circ_state["V_LV"] * mL
start_volume = float(inflation_volume.value)
start_Ta = float(Ta.value.value)
logger.info(
    f"Returning from {start_volume / mL:.2f} mL at {start_Ta * 1e-3:.1f} kPa "
    f"to {target / mL:.2f} mL at rest",
)
for frac in np.linspace(0.0, 1.0, 30)[1:]:
    inflation_volume.value = start_volume + frac * (target - start_volume)
    Ta.assign((1.0 - frac) * start_Ta)
    inflation.solve()

inflated_volume = comm.allreduce(geometry.volume("ENDO", u=inflation.u), op=MPI.SUM)
p_inflated = float(inflation.cavity_pressures[0].x.array[0])
logger.info(f"Inflated to {inflated_volume / mL:.2f} mL at {p_inflated / mmHg:.2f} mmHg")

# ## The coupled problem
#
# Same model, same boundary conditions; the difference is that the cavity
# volume is no longer prescribed. `V_LV` joins the unknowns, and its row is the
# chamber's own differential equation.

circulation_model = GotranxCirculation(
    ode_file=regazzoni2020.ODE_FILE,
    parameters=regazzoni2020.flat_ode_parameters(circ_parameters),
    drop_components=("timing", "LV"),
)
beat_phase = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))

problem = pulse.problem.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs,
    cavities=[pulse.problem.Cavity(marker="ENDO", volume=None)],
    circulation=circulation_model,
    chambers=[ChamberCoupling(marker="ENDO", volume_state="V_LV", pressure_missing="p_LV")],
    circulation_missing={"beat_phase": beat_phase},
    parameters={"mesh_unit": "m"},
)

# Start from the inflated state, and from the circuit state it was inflated to.
problem.u.x.array[:] = inflation.u.x.array
problem.u_old.x.array[:] = inflation.u.x.array
if problem.is_incompressible:
    problem.p.x.array[:] = inflation.p.x.array
    problem.p_old.x.array[:] = inflation.p.x.array
problem.cavity_pressures[0].x.array[:] = p_inflated
problem.cavity_pressures_old[0].x.array[:] = p_inflated

names = list(circulation_model.state_names)
for name, state, state_old in zip(
    names,
    problem.circulation_states,
    problem.circulation_states_old,
):
    value = float(circ_state[name])
    state.x.array[:] = value
    state_old.x.array[:] = value

problem.circulation_dt.value = DT

# ## Stepping
#
# One solve per step, with no inner iteration between the mechanics and the
# circuit: they are the same system.

i_V_LV = names.index("V_LV")
history = {
    "time": [0.0],
    "V_LV": [float(problem.circulation_states[i_V_LV].x.array[0])],
    "p_LV": [p_inflated / mmHg],
    "Ta": [0.0],
    "iterations": [0],
    "constraint": [0.0],
}

max_steps = 2 if IN_CI else int(NUM_BEATS * BEAT_LENGTH / DT)
t = 0.0
for step in range(max_steps):
    t += DT
    problem.circulation_time.value = t
    beat_phase.value = t % BEAT_LENGTH
    Ta.assign(Ta_scale * activation_at(t))

    if not problem.solve():
        raise RuntimeError(f"Monolithic solve failed at t={t:.4f}")

    volume = comm.allreduce(geometry.volume("ENDO", u=problem.u), op=MPI.SUM)
    V_LV = float(problem.circulation_states[i_V_LV].x.array[0]) * mL

    history["time"].append(t)
    history["V_LV"].append(V_LV / mL)
    history["p_LV"].append(float(problem.cavity_pressures[0].x.array[0]) / mmHg)
    history["Ta"].append(float(Ta.value.value))
    history["iterations"].append(int(problem.problem.solver.getIterationNumber()))
    # How far the cavity volume and the chamber state have drifted apart. In a
    # partitioned scheme this is set by the exchange budget; here it should stay
    # at solver tolerance.
    history["constraint"].append(abs(volume - V_LV) / V_LV)

    if step % 50 == 0:
        logger.info(
            f"t={t:.3f}  V={V_LV / mL:8.2f} mL  p={history['p_LV'][-1]:7.2f} mmHg  "
            f"Ta={history['Ta'][-1] * 1e-3:6.2f} kPa  "
            f"constraint={history['constraint'][-1]:.2e}",
        )

logger.info(f"Worst constraint violation over the run: {max(history['constraint']):.3e}")

# ## Does the coupled run agree with the surrogate it was calibrated against?
#
# The mechanics problem is quasi-static, so its cavity pressure is a function
# of volume and activation and nothing else. The loop this run traced out
# therefore has to lie on the sampled surface, and how far off it lies is
# interpolation error. Anything larger means the calibration and the coupled
# run are not solving the same mechanics problem, which would invalidate the
# operating point rather than merely blur it.

if calibration is not None:
    agreement = calibration.check_against(history["V_LV"], history["Ta"], history["p_LV"])
    logger.info(
        f"Surrogate vs coupled run: {agreement['max_mmHg']:.2f} mmHg worst, "
        f"{agreement['rms_mmHg']:.2f} mmHg rms, "
        f"{agreement['outside_mL']:.1f} mL outside the sampled volume range",
    )

if comm.rank == 0:
    np.savez(
        outdir / f"traces_monolithic-{FORMULATION.value}.npz",
        **{k: np.asarray(v) for k, v in history.items()},
    )

    fig = plt.figure(layout="constrained", figsize=(11, 8))
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])

    ax1.plot(history["V_LV"], history["p_LV"], marker=".", markersize=3, linewidth=0.9)
    ax1.set_xlabel("V [mL]")
    ax1.set_ylabel("p [mmHg]")
    ax1.set_title("Pressure-volume loop")

    ax2.plot(history["time"], history["p_LV"])
    ax2.set_ylabel("p [mmHg]")
    ax3.plot(history["time"], np.asarray(history["Ta"]) * 1e-3)
    ax3.set_ylabel("Ta [kPa]")
    ax4.semilogy(history["time"], np.maximum(history["constraint"], 1e-18))
    ax4.set_ylabel("constraint violation")
    ax4.set_xlabel("Time [s]")

    fig.savefig(outdir / f"monolithic_3d0d-{FORMULATION.value}.png", dpi=140)
    plt.close(fig)

# ## What the calibration was up against
#
# The measured chamber next to the one the circuit ships. The gap between the
# two zero-activation curves is the whole problem: at any volume in the working
# range the mesh needs a fraction of the pressure the circuit's elastance would
# ask for, so an uncalibrated loop fills until the two happen to agree, which
# is well past where a ventricle stops.

if comm.rank == 0 and calibration is not None:
    surface = calibration.surface
    pub = calibration.published_chamber
    Vg = surface.volumes

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    for j, Ta_level in enumerate(surface.activations):
        axL.plot(
            Vg,
            surface.pressures[:, j],
            color=plt.cm.viridis(j / max(len(surface.activations) - 1, 1)),
            label=f"{Ta_level * 1e-3:.0f} kPa",
        )
    axL.plot(Vg, pub["EB"] * (Vg - pub["V0"]), "k--", label="published, at rest")
    axL.plot(Vg, (pub["EA"] + pub["EB"]) * (Vg - pub["V0"]), "k:", label="published, active")
    axL.plot(history["V_LV"], history["p_LV"], color="crimson", linewidth=1.2, label="coupled run")
    axL.set_xlabel("V [mL]")
    axL.set_ylabel("p [mmHg]")
    axL.set_ylim(-20, 260)
    axL.set_title("Measured chamber vs the circuit's own")
    axL.legend(fontsize="x-small", ncols=2)

    # Slopes, not a ratio of pressures. The measured resting pressure passes
    # through zero inside the working range, so dividing by it produces a
    # figure that swings through a pole and changes sign, which says nothing
    # about the chamber. Stiffness is finite everywhere and is the quantity
    # `EA` and `EB` are stated in anyway.
    mid = 0.5 * (Vg[1:] + Vg[:-1])
    for j, Ta_level in enumerate(surface.activations):
        axR.plot(
            mid,
            np.diff(surface.pressures[:, j]) / np.diff(Vg),
            color=plt.cm.viridis(j / max(len(surface.activations) - 1, 1)),
            label=f"{Ta_level * 1e-3:.0f} kPa",
        )
    axR.axhline(pub["EB"], color="k", linestyle="--", label="published, at rest")
    axR.axhline(0.0, color="0.6", linewidth=0.8)
    axR.set_xlabel("V [mL]")
    axR.set_ylabel("dp/dV [mmHg/mL]")
    axR.set_title("Chamber stiffness")
    axR.legend(fontsize="x-small", ncols=2)

    fig.savefig(outdir / f"calibration_surface-{FORMULATION.value}.png", dpi=140)
    plt.close(fig)

logger.info("Done.")
