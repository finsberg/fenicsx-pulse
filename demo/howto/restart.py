# # Restarting a simulation
#
# A long run rarely finishes in one go. A cluster job is killed when it hits
# its wall-time limit, or a run has to continue on another machine. To pick it
# up again we have to read back *everything* that one step carries forward to
# the next, and pulse keeps that in two kinds of state:
#
# - Functions, which `problem.restart_functions()` lists;
# - plain Python values, which `problem.restart_metadata()` returns, together
#   with `CycleController.state_dict()` when a cycle controller drives the
#   cavities.
#
# In this demo we run a left ventricle through a few steps of the cardiac
# cycle three times: once without interruption, once stopping halfway to write
# a checkpoint, and once more from that checkpoint in brand-new objects. Then
# we check that the restarted run equals the uninterrupted one **bit for bit**.
# That is the right bar here. A restart does no arithmetic, it only copies
# state, and on one rank the solver is deterministic, so anything short of an
# exact match means some state was left behind. (In parallel the bar has to
# be lower, for a reason that has nothing to do with restarting; see the
# check at the end.)
#
# Along the way we attach a `pulse.telemetry.PerformanceMonitor`, which records
# where a run spends its time.

import json
import logging
import math
import shutil
from pathlib import Path

from mpi4py import MPI

import dolfinx
import io4dolfinx
import numpy as np

import cardiac_geometries
import cardiac_geometries.geometry
import pulse
from pulse import cycle
from pulse.circulation import mL, mmHg
from pulse.telemetry import PerformanceMonitor

# The monitor reports through Python's `logging`, so we show INFO messages,
# from one rank only; every rank would otherwise log the same per-step line.

comm = MPI.COMM_WORLD
logging.basicConfig(level=logging.INFO if comm.rank == 0 else logging.WARNING)
logging.getLogger("scifem").setLevel(logging.WARNING)

# ## Geometry
#
# We use the LV ellipsoid from cardiac-geometries that pulse's tests use, in metres. `lv_ellipsoid` writes the mesh,
# markers and fibres to `geometry.bp` in the output folder, so a rerun of this
# demo reuses that file instead of meshing again.

outdir = Path("results_restart")
geodir = outdir / "geometry"
if not (geodir / "geometry.bp").exists():
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        r_short_endo=0.025,
        r_short_epi=0.035,
        r_long_endo=0.09,
        r_long_epi=0.097,
        psize_ref=0.03,
        mu_apex_endo=-np.pi,
        mu_base_endo=-np.arccos(5 / 17),
        mu_apex_epi=-np.pi,
        mu_base_epi=-np.arccos(5 / 20),
        create_fibers=True,
        fiber_angle_endo=60,
        fiber_angle_epi=-60,
        fiber_space="P_1",
        comm=comm,
    )
geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)
geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 4})

# ## The cycle
#
# A `pulse.cycle.CycleController` steps the cavity through the five phases of
# the cycle: preload, isovolumic contraction, ejection, isovolumic relaxation
# and filling. The parameters are those of pulse's own LV-cycle test, in SI
# units, except that we shorten preload to 4 ms. That way the run switches to
# isovolumic contraction before we interrupt it, so the restart has to carry a
# phase other than the initial one, and the steps after it run under a
# different constraint from the first.

params = {
    "ENDO": cycle.CycleParams(
        t_zero=2e-3,
        preload_pressure=500.0,
        t_end_diastole=4e-3,
        p_end_diastole=1000.0,
        p_fill=500.0,
        period=0.8,
        windkessel=cycle.Windkessel(
            p_init=9000.0,
            compliance=1.5 * mL / mmHg,
            resistance=1.1 * mmHg / mL,
            characteristic_impedance=0.03 * mmHg / mL,
        ),
        filling=cycle.PrescribedInflow(rate=0.046 * mL / 1e-3),
    ),
}
dt = 2e-3
N = 10  # total steps; we checkpoint after N // 2

# The active tension is a twitch that is zero until 9 ms and would peak at
# 60 kPa at 29 ms; over the 20 ms we simulate it climbs to about 52 kPa. It is
# a function of time alone, so it needs no saving: the restarted run
# recomputes it from `t`.


def twitch(t: float) -> float:
    tau = max(t - 0.004 - 0.005, 0.0)
    return (tau / 0.02) * np.exp(1.0 - tau / 0.02)


# `build` creates the material, the model, the problem, the controller and the
# monitor from scratch. That is the point of it: a new process starts with
# nothing else, so the restart below gets no objects from the run that wrote
# the checkpoint. Only the geometry is shared, and the new process would
# read it from `geometry.bp` just as we did above.


def build():
    material = pulse.HolzapfelOgden(
        f0=geo.f0,
        s0=geo.s0,
        **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
    )
    Ta = pulse.Variable(dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
    model = pulse.CardiacModel(
        material=material,
        active=pulse.ActiveStress(geo.f0, activation=Ta),
        compressibility=pulse.Compressible(),
        viscoelasticity=pulse.Viscous(),
    )
    monitor = PerformanceMonitor(comm=comm)
    control = pulse.problem.CavityControl(geo.mesh)
    problem = pulse.problem.DynamicProblem(
        model=model,
        geometry=geometry,
        cavities=[pulse.problem.Cavity(marker="ENDO", control=control)],
        parameters={
            "base_bc": pulse.problem.BaseBC.fixed,
            "mesh_unit": "m",
            "rho": pulse.Variable(1e3, "kg/m^3"),
            "dt": pulse.Variable(dt, "s"),
        },
        monitor=monitor,
    )
    controller = cycle.CycleController(problem, params)
    controller.initialize(t0=0.0)
    return problem, controller, Ta, monitor


# `advance` takes `n` steps from time `t` and returns the new time. The
# problem times its own solves, but only the driver knows where a step ends,
# so we tell the monitor with `advance_step`; with the default
# `log_frequency=1` it then logs one line per step.


def advance(controller, Ta, monitor, t: float, n: int) -> float:
    for _ in range(n):
        t += dt
        Ta.assign(60.0 * twitch(t))
        if not controller.step(t, dt):
            raise RuntimeError(f"step to t={t:.4f} s did not converge")
        monitor.advance_step(t - dt, t)
    return t


# ## The uninterrupted reference run
#
# We take all `N` steps in one go and keep a copy of every restart Function
# and of the controller's state, to compare against at the end.

problem, controller, Ta, monitor = build()
t = advance(controller, Ta, monitor, 0.0, N)
reference = {name: f.x.array.copy() for name, f in problem.restart_functions()}
reference_cycle = controller.state_dict()
if comm.rank == 0:
    print(f"Reference run: t = {t:.3f} s, phase {controller.cycles['ENDO'].phase.name}")

# ## The interrupted run and its checkpoint
#
# Now we start again from scratch, stop after `N // 2` steps, and write a
# checkpoint as a job would just before its wall time runs out.

problem, controller, Ta, monitor = build()
t = advance(controller, Ta, monitor, 0.0, N // 2)
if comm.rank == 0:
    print(f"Interrupted at t = {t:.3f} s, phase {controller.cycles['ENDO'].phase.name}")
    print("Restart functions:", [name for name, _ in problem.restart_functions()])

# The Functions go to an ADIOS2 file with `io4dolfinx`, the plain values to a
# JSON file. Every name `restart_functions` returns starts with `mechanics_`.
# The prefix keeps the mechanics state apart from the electrophysiology
# state, so that a checkpoint of an electromechanics run can hold both,
# fenicsx-beat's and pulse's, in one file without the names clashing. Here
# they are the displacement `u`, the cavity pressure, and an `_old` copy of
# each, then the velocity and acceleration history `v_old` and `a_old` that
# the dynamic problem adds.
#
# We store `t` itself rather than recompute it from the step count, because
# the two are only tied together while `dt` is fixed. A run that halves its
# step to get through a hard phase has no fixed relation between them.

checkpoint = outdir / "restart.bp"
meta_path = outdir / "restart.json"
if comm.rank == 0:
    # io4dolfinx appends, and a stale file's entry would be the one read back.
    shutil.rmtree(checkpoint, ignore_errors=True)
comm.barrier()
for name, f in problem.restart_functions():
    io4dolfinx.write_function_on_input_mesh(checkpoint, f, time=t, name=name)
meta = {
    "t": t,
    "step": N // 2,
    "problem": problem.restart_metadata(),
    "cycle": controller.state_dict(),
}
if comm.rank == 0:
    meta_path.write_text(json.dumps(meta, indent=2))
comm.barrier()

# ## The restart
#
# Next we build brand-new objects, as a new process would, and restore the
# checkpoint into them in this order:
#
# 1. read the Functions, writing into the problem's own Functions in place;
# 2. `problem.load_restart_metadata`;
# 3. `controller.load_state_dict`.
#
# The Functions come first, because everything else describes them.
# `load_restart_metadata` is documented to run after the Functions are in
# place: with a circulation it restores the circuit's step count and picks the
# time-stepping stencil that applies to those states. The controller's state
# records the volume, pressure and phase that the Functions correspond to.
# Here the problem has no circulation, so its metadata is an empty dict, but
# loading it costs nothing and keeps the code the same for problems that
# have one.

problem, controller, Ta, monitor = build()
meta = json.loads(meta_path.read_text())
for name, f in problem.restart_functions():
    io4dolfinx.read_function(checkpoint, f, time=meta["t"], name=name)
    f.x.scatter_forward()
problem.load_restart_metadata(meta["problem"])
controller.load_state_dict(meta["cycle"])
t = advance(controller, Ta, monitor, meta["t"], N - meta["step"])

# ## The check
#
# Finally we compare every restart Function and the controller's whole state
# with the reference run. On one rank we compare them exactly.
#
# On more ranks that bar is out of reach, and not because of the restart.
# MUMPS, pulse's default direct solver, is not bit-reproducible in parallel:
# two *uninterrupted* runs of this demo on two ranks already differ in the
# last few digits, and a restarted run differs from an uninterrupted one by
# the same amount, up to about $10^{-13}$ of each field's largest value. In
# parallel we therefore accept differences at round-off level. A Function may
# differ by at most $10^{-10}$ times its largest value, and each number in the
# controller's state by the same relative amount. The controller check also
# has a tiny absolute floor, for numbers that are zero up to round-off, such
# as the volume change during an isovolumic phase.

exact = comm.size == 1


def max_abs(x) -> float:
    return comm.allreduce(float(np.max(np.abs(x), initial=0.0)), op=MPI.MAX)


def same_function(values, ref) -> bool:
    if exact:
        return np.array_equal(values, ref)
    return max_abs(values - ref) <= 1e-10 * max_abs(ref)


def same_state(a, b) -> bool:
    if exact or not isinstance(a, (dict, float)):
        return a == b
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-15)


for name, f in problem.restart_functions():
    assert same_function(f.x.array, reference[name]), name
assert same_state(controller.state_dict(), reference_cycle)
if comm.rank == 0:
    if exact:
        print("Restarted run matches the uninterrupted one bit for bit.")
    else:
        print(f"Restarted run matches the uninterrupted one to round-off on {comm.size} ranks.")
monitor.display_summary()

# The summary covers the restarted run only, since its monitor was built
# fresh with everything else. It lists the number of steps, the Newton
# iterations in total and at most per solve, the Newton failures, the linear
# iterations accumulated over all Newton iterations, and the wall time spent
# in each timed section: `newton_solve` and `update_fields`, which the problem
# times itself. When a run slows down after a phase change, this is the first
# place to look. The per-step lines above then show which step it was: each
# gives that step's Newton and linear iterations, and the wall time
# accumulated so far.
#
# ## Beyond this demo
#
# **Monolithic `GotranxCirculation` problems.** When the circulation is
# solved in the same Newton system as the mechanics, `restart_functions`
# already includes every circuit state, each with its `_old` and `_prev`
# copies, and `restart_metadata` holds `circulation_steps`, the number of
# steps the circuit has taken. There is no controller to save, so the same
# code works without the `state_dict` lines.
#
# **The command line.** The `pulse` CLI writes the same kind of checkpoint,
# the `mechanics_*` Functions in `restart.bp` and the problem's restart
# metadata in `restart.json`, and `pulse run --restart` continues from it.
# See [the CLI documentation](../../docs/cli.md) for how it checks that the
# physics has not changed since the checkpoint was written.
