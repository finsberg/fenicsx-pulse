# # A complete cardiac cycle on a prestressed biventricular mesh
#
# In this demo we take both ventricles of a UK Biobank atlas mesh through
# whole heartbeats, using the five-phase, Alya-style cycle of `pulse.cycle`.
# Each ventricle has a single pressure unknown, held by a
# `pulse.problem.CavityControl`. A `pulse.cycle.CycleController` decides, step
# by step, which constraint that unknown satisfies: a prescribed pressure while
# the ventricle is loaded up to end diastole, a fixed volume while both valves
# are shut, a three-element Windkessel while the ventricle ejects, and a volume
# growing at a set rate while it fills. Switching between them
# only changes the values of a few constants, so the problem is built once and
# never rebuilt.
#
# The pipeline is:
#
# 1. **Geometry.** We generate the mesh from the atlas, rotate it so that the
#    base normal points along x, and compute fibres with LDRB, much as in the
#    [rotated BiV demo](../boundary_conditions/ukb_bcs.py).
# 2. **Prestress.** The mesh is imaged at end diastole, so it is already
#    loaded. We recover the unloaded reference configuration by solving the
#    inverse elasticity problem, as in [the BiV prestress demo](../prestress/prestress_biv.py).
# 3. **PRELOAD.** The first phase of the cycle ramps each cavity pressure from
#    zero back up to its end-diastolic value, which inflates the unloaded mesh
#    back to the imaged shape.
# 4. **Beats.** From there the controller takes both ventricles through
#    contraction, ejection, relaxation and filling, and on into the next beat.
#
# A run like this one takes a while. [](../howto/restart.py) shows how to write
# a checkpoint of a `CycleController` run and carry on from it later.

import logging
import os
import shutil
from pathlib import Path

from mpi4py import MPI

# A sibling module in this directory, not a package.
import animation
import dolfinx
import io4dolfinx
import ldrb
import matplotlib.pyplot as plt
import numpy as np

import cardiac_geometries
import cardiac_geometries.geometry
import pulse
from pulse import cycle
from pulse.circulation import mL, mmHg


# Setup logging to print only from rank 0
class MPIFilter(logging.Filter):
    def __init__(self, comm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comm = comm

    def filter(self, record):
        return 1 if self.comm.rank == 0 else 0


outdir = Path("results_biv_complete_cycle")
outdir.mkdir(parents=True, exist_ok=True)
geodir = outdir / "geometry"

comm = MPI.COMM_WORLD
logging.basicConfig(level=logging.INFO)
# The filter sits on the handler, so that it also catches the messages of
# `pulse.cycle`, which logs every phase transition.
mpi_filter = MPIFilter(comm)
for handler in logging.getLogger().handlers:
    handler.addFilter(mpi_filter)
logger = logging.getLogger("pulse")
logging.getLogger("scifem").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# We run two beats of 0.8 s with a time step of 2 ms. Under CI, where this
# page is built, the run is cut to two steps. Setting `PULSE_MAX_STEPS` to a
# positive number cuts it to that many steps instead, which is a convenient way
# to check the first part of a beat without running the whole thing.

_ci = os.getenv("CI", "").strip().lower()
IN_CI = _ci not in ("", "0", "false", "no", "off")
PERIOD = 0.8  # s
NUM_BEATS = 2
DT = 2e-3  # s
max_steps = 2 if IN_CI else int(round(NUM_BEATS * PERIOD / DT))
MAX_STEPS = int(os.getenv("PULSE_MAX_STEPS", "0"))
if MAX_STEPS > 0:
    max_steps = MAX_STEPS


# ## Geometry generation and rotation
#
# We generate the BiV geometry from the UK Biobank Atlas, rotate it to align
# the base normal with the x-axis, and generate fiber fields using LDRB. The
# fibers are based on the fiber orientation angles from {cite}`doste2019rule`.
# The fibers used for the mechanics simulation are in a quadrature space to
# avoid interpolation errors. We also store fibers in a DG 1 space as
# additional data, which is useful if we want to compute fiber stress or strain
# at intermediate points later on.

if not (geodir / "geometry.bp").exists():
    logger.info("Generating and processing geometry...")
    mode = -1
    std = 0
    char_length = 10.0

    geo = cardiac_geometries.mesh.ukb(
        outdir=geodir,
        comm=comm,
        mode=mode,
        std=std,
        case="ED",
        char_length_max=char_length,
        char_length_min=char_length,
        clipped=True,
    )

    # Rotate Mesh (Base Normal -> X-axis)
    geo = geo.rotate(target_normal=[1.0, 0.0, 0.0], base_marker="BASE")

    fiber_angles = dict(
        alpha_endo_lv=60,
        alpha_epi_lv=-60,
        alpha_endo_rv=90,
        alpha_epi_rv=-25,
        beta_endo_lv=-20,
        beta_epi_lv=20,
        beta_endo_rv=0,
        beta_epi_rv=20,
    )

    # Generate Fibers (LDRB)
    system = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=cardiac_geometries.mesh.transform_markers(geo.markers, clipped=True),
        **fiber_angles,
        fiber_space="Quadrature_6",
    )

    # Additional Vectors for Analysis in DG 1 Space for computing stress/strain later
    fiber_space = "DG_1"
    system_fibers = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=cardiac_geometries.mesh.transform_markers(geo.markers, clipped=True),
        **fiber_angles,
        fiber_space=fiber_space,
    )

    # Save Everything
    additional_data = {
        "f0_DG_1": system_fibers.f0,
        "s0_DG_1": system_fibers.s0,
        "n0_DG_1": system_fibers.n0,
    }

    if (geodir / "geometry.bp").exists():
        shutil.rmtree(geodir / "geometry.bp")

    cardiac_geometries.geometry.save_geometry(
        path=geodir / "geometry.bp",
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=geo.markers,
        info=geo.info,
        f0=system.f0,
        s0=system.s0,
        n0=system.n0,
        additional_data=additional_data,
    )

comm.barrier()

# We load the generated geometry

geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)

# and scale it from millimetres to metres. `CavityControl` works in SI units
# throughout (cubic metres, pascals), so a controlled cavity needs
# `mesh_unit == "m"`.

scale = 1e-3
geo.mesh.geometry.x[:] *= scale
mesh_unit = "m"

geometry = pulse.HeartGeometry.from_cardiac_geometries(
    geo, metadata={"quadrature_degree": 6},
)

# The sliding-base condition below holds a single displacement component, the
# x component, on the base, so we check that the base normal really is the x
# axis. We keep the normal to stand the mesh upright in the video.

up = animation.base_normal(geometry, "BASE")
if abs(up[0]) < 0.99:
    raise RuntimeError(
        f"the base normal is {up.round(3)}, not the x axis the sliding-base "
        "condition assumes -- the rotation above did not take effect",
    )

# These are the volumes of the imaged, end-diastolic mesh.

lvv_target = comm.allreduce(geometry.volume("LV"), op=MPI.SUM)
rvv_target = comm.allreduce(geometry.volume("RV"), op=MPI.SUM)
logger.info(
    f"ED Volumes: LV={lvv_target / mL:.2f} mL, RV={rvv_target / mL:.2f} mL",
)

# ## The cycle
#
# Each ventricle carries a `pulse.cycle.CavityCycle`, the state machine that
# moves it through five phases. Every phase sets one constraint on the
# cavity's pressure unknown, computed from the state at the start of the step:
#
# | Phase | Constraint |
# | --- | --- |
# | `PRELOAD` | pressure: a linear ramp, in two pieces |
# | `ISOVOLUMIC_CONTRACTION` | volume: $V$ = the end-diastolic volume |
# | `EJECTION` | pressure: an implicit three-element Windkessel, affine in $V$ |
# | `ISOVOLUMIC_RELAXATION` | volume: $V$ = the end-systolic volume |
# | `FILLING` | volume: $V = V_n + \text{rate}\,\Delta t$ |
#
# In the first beat, PRELOAD's ramp goes from 0 to `preload_pressure` at
# `t_zero`. In every beat it then goes on from `preload_pressure` to
# `p_end_diastole` at `t_end_diastole`. $V_n$ is the volume at the start of the
# step and $\Delta t$ its length.
#
# During ejection the outflow is $Q = -(V - V_n)/\Delta t$, and the Windkessel
# advances its compliance (arterial) pressure by backward Euler,
#
# $$
# P_c = \frac{P_{c,n} + \Delta t\, Q / C}{1 + \Delta t / (R_p C)}, \qquad
# P_v = P_c + R_c Q,
# $$
#
# with $C$ the compliance, $R_p$ the peripheral resistance and $R_c$ the
# characteristic impedance. Written out in terms of the still-unknown volume
# $V$, this makes the cavity pressure $P_v$ an affine function of $V$, which
# Newton then enforces together with the mechanics.
#
# The controller moves on from a phase when:
#
# - `PRELOAD` reaches `t_end_diastole` of the current beat;
# - in `ISOVOLUMIC_CONTRACTION`, the cavity pressure exceeds the compliance
#   pressure $P_c$, i.e. the outflow valve opens;
# - in `EJECTION`, more than `min_ejection_duration` (10 ms) after the valve
#   opened, either the outflow stops or the cavity pressure drops below $P_c$;
# - in `ISOVOLUMIC_RELAXATION`, the cavity pressure falls below `p_fill`;
# - `FILLING` reaches `t_zero` of the next beat, which starts the next
#   `PRELOAD`.
#
# Outside ejection the valve is shut, and once a cavity has ejected, its $P_c$
# drains through $R_p$.
#
# For the left ventricle we use the timings and the Windkessel that `pulse`'s
# own tests run the cycle with. Note that everything in `pulse.cycle` is SI. We
# write the circuit-side values in millilitres and mmHg, and `mL` and `mmHg`
# convert them.
#
# The filling rate is our own. Filling at a prescribed rate knows nothing of
# the next beat, whose PRELOAD starts again from `preload_pressure`. A faster
# rate fills the ventricle past the volume it has at that pressure, and the
# pressure then drops as PRELOAD takes over. We set each ventricle's rate so
# that it is back at about that volume when the next beat starts: the left
# ventricle holds 96.2 mL at 500 Pa and fills to 98.9 mL by the end of the
# first beat; the right holds 69.3 mL at 200 Pa and fills to 69.3 mL.

lv_params = cycle.CycleParams(
    t_zero=0.05,
    preload_pressure=500.0,
    t_end_diastole=0.12,
    p_end_diastole=1000.0,
    p_fill=500.0,
    period=PERIOD,
    windkessel=cycle.Windkessel(
        p_init=9000.0,
        compliance=1.5 * mL / mmHg,
        resistance=1.1 * mmHg / mL,
        characteristic_impedance=0.03 * mmHg / mL,
    ),
    filling=cycle.PrescribedInflow(rate=0.040 * mL / 1e-3),
)

# The right ventricle pumps into the pulmonary circulation, which has a much
# lower resistance and a higher compliance than the systemic one, and it fills
# at a lower pressure. We have tuned these values by hand on this mesh; they
# are not taken from a reference. `p_init` is tuned so that the first beat
# opens the pulmonary valve at about the pressure the compliance has drained to
# by the second beat (0.60 against 0.47 kPa), so both beats peak just under
# 40 mmHg, at 38.2 and 38.0 mmHg.

rv_params = cycle.CycleParams(
    t_zero=0.05,
    preload_pressure=200.0,
    t_end_diastole=0.12,
    p_end_diastole=400.0,
    p_fill=200.0,
    period=PERIOD,
    windkessel=cycle.Windkessel(
        p_init=600.0,
        compliance=4.0 * mL / mmHg,
        resistance=0.15 * mmHg / mL,
        characteristic_impedance=0.01 * mmHg / mL,
    ),
    filling=cycle.PrescribedInflow(rate=0.042 * mL / 1e-3),
)
params = {"LV": lv_params, "RV": rv_params}

# ## Activation
#
# We use a simple twitch as the active tension, the same in every element of
# both ventricles. It starts 5 ms after end diastole, peaks 20 ms later at
# `T_MAX`, and then decays with an e-folding time of 20 ms. By the start of the
# next beat it is negligible, so we can repeat it every `PERIOD`. We tuned
# `T_MAX` by hand, for a left ventricular ejection fraction above 35 % at a
# peak pressure below 140 mmHg.

T_MAX = 90.0  # kPa, uniform over both ventricles


def activation(t: float) -> float:
    """The twitch of tests/test_cycle.py, repeated every beat from end diastole."""
    onset = lv_params.t_end_diastole
    tau = max((t % PERIOD) - onset - 0.005, 0.0)
    return T_MAX * (tau / 0.02) * np.exp(1.0 - tau / 0.02)


# [](land_circulation_biv.py) replaces this prescribed shape with a
# crossbridge model, and a single strength with a different one per region.
#
# ## The model
#
# The material is the transversely isotropic Holzapfel-Ogden model, with an
# active stress along the fibres. The epicardium and the base rest on springs,
# and the base may slide in its own plane but not leave it.
#
# We will solve the dynamic problem, and we damp it with a viscous term.
# Undamped, the wall rings after the switch into the volume constraint of
# isovolumic relaxation: the cavity pressure saw-tooths from step to step, and
# Newton needs many iterations, or diverges. With `Viscous`, the pressure falls
# smoothly through relaxation. The viscous term acts on the strain rate, which
# only the dynamic problem has, so it plays no part in the static prestressing
# solve below.


def setup_problem(geometry, f0, s0, material_params):
    material = pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)
    Ta = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)),
        "kPa",
    )
    active_model = pulse.ActiveStress(f0, activation=Ta)

    model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=pulse.Compressible(),
        viscoelasticity=pulse.Viscous(),
    )

    alpha_epi = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e5)),
        "Pa / m",
    )
    robin_epi = pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])

    alpha_base = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e6)),
        "Pa / m",
    )
    robin_base = pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])

    robin = [robin_epi, robin_base]

    # Dirichlet BC: Sliding Base (ux=0)
    def dirichlet_bc(V: dolfinx.fem.FunctionSpace):
        facets = geometry.facet_tags.find(geometry.markers["BASE"][0])
        dofs = dolfinx.fem.locate_dofs_topological(V.sub(0), 2, facets)
        return [dolfinx.fem.dirichletbc(0.0, dofs, V.sub(0))]

    return model, robin, dirichlet_bc, Ta


material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
model, robin, dirichlet_bc, Ta = setup_problem(
    geometry=geometry,
    f0=geo.f0,
    s0=geo.s0,
    material_params=material_params,
)

# ## Prestressing (inverse elasticity)
#
# The imaged mesh is the shape of the heart at end diastole, under the
# end-diastolic pressures. We prestress to the same pressures that PRELOAD
# ramps up to, `p_end_diastole` of each ventricle. `TargetPressure` takes the
# value in the unit of its traction, kilopascals here.

p_LV_ED = lv_params.p_end_diastole / 1e3  # Pa -> kPa
p_RV_ED = rv_params.p_end_diastole / 1e3  # Pa -> kPa

# Since we want to apply pressures on both ventricles, we create two Neumann BCs.

pressure_lv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "kPa")
pressure_rv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "kPa")
neumann_lv = pulse.NeumannBC(traction=pressure_lv, marker=geometry.markers["LV"][0])
neumann_rv = pulse.NeumannBC(traction=pressure_rv, marker=geometry.markers["RV"][0])

bcs_prestress = pulse.BoundaryConditions(
    robin=robin,
    dirichlet=(dirichlet_bc,),
    neumann=(neumann_lv, neumann_rv),
)

# We store the prestressed displacement in a file to avoid recomputing it. The
# file name carries the two target pressures, so changing `p_end_diastole` of
# either ventricle prestresses again rather than reusing a reference unloaded
# from different pressures.

prestress_fname = outdir / (
    f"prestress_biv_inverse_LV{lv_params.p_end_diastole:g}Pa"
    f"_RV{rv_params.p_end_diastole:g}Pa.bp"
)
if not prestress_fname.exists():
    logger.info(
        f"Start prestressing... Targets: p_LV={p_LV_ED:.2f} kPa, p_RV={p_RV_ED:.2f} kPa",
    )
    prestress_problem = pulse.unloading.PrestressProblem(
        geometry=geometry,
        model=model,
        bcs=bcs_prestress,
        parameters={"u_space": "P_2", "mesh_unit": mesh_unit},
        targets=[
            pulse.unloading.TargetPressure(
                traction=pressure_lv, target=p_LV_ED, name="LV",
            ),
            pulse.unloading.TargetPressure(
                traction=pressure_rv, target=p_RV_ED, name="RV",
            ),
        ],
        ramp_steps=20,
    )

    u_pre = prestress_problem.unload()
    io4dolfinx.write_function_on_input_mesh(
        prestress_fname, u_pre, time=0.0, name="u_pre",
    )
    with dolfinx.io.VTXWriter(
        comm,
        outdir / "prestress_biv_backward.bp",
        [u_pre],
        engine="BP4",
    ) as vtx:
        vtx.write(0.0)

# ## The unloaded reference configuration

V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 2, (3,)))
u_pre = dolfinx.fem.Function(V)
io4dolfinx.read_function(prestress_fname, u_pre, time=0.0, name="u_pre")

# We use the prestressed displacement to deform the mesh to the reference configuration.

logger.info("Deforming mesh to Reference Configuration...")
geometry.deform(u_pre)

# We now map the fiber fields to the reference configuration. The solve uses
# the quadrature fibres; the DG 1 fibres are mapped too, for post-processing
# fibre stress or strain.

logger.info("Mapping fibers to Reference Configuration...")
f0_quad = pulse.utils.map_vector_field(
    f=geo.f0, u=u_pre, normalize=True, name="f0_unloaded",
)
s0_quad = pulse.utils.map_vector_field(
    f=geo.s0, u=u_pre, normalize=True, name="s0_unloaded",
)
f0_map = pulse.utils.map_vector_field(
    geo.additional_data["f0_DG_1"],
    u=u_pre,
    normalize=True,
    name="f0",
)

# Calculate unloaded volumes

lvv_unloaded = comm.allreduce(geometry.volume("LV"), op=MPI.SUM)
rvv_unloaded = comm.allreduce(geometry.volume("RV"), op=MPI.SUM)
logger.info(
    f"Unloaded volumes: LV={lvv_unloaded / mL:.2f} mL, RV={rvv_unloaded / mL:.2f} mL",
)
model, robin, dirichlet_bc, Ta = setup_problem(
    geometry=geometry,
    f0=f0_quad,
    s0=s0_quad,
    material_params=material_params,
)
bcs_forward = pulse.BoundaryConditions(robin=robin, dirichlet=(dirichlet_bc,))

# ## The coupled problem
#
# Each cavity gets a `CavityControl` rather than a volume. We put no Neumann
# pressure on the endocardium: the load on the wall comes from the cavity's
# pressure unknown, whatever constraint it satisfies.
#
# The problem starts from rest in the unloaded configuration, at zero cavity
# pressure. `initialize` reads each cavity's volume and pressure there and puts
# both cavities in PRELOAD. That phase then ramps the pressure back up to
# `p_end_diastole`, which is the pressure we unloaded from, so at
# `t_end_diastole` the wall is back at (close to) the imaged end-diastolic
# shape. This is why PRELOAD's end pressure must match the prestress target:
# any other value would start contraction from a different shape. The ramp
# also takes the place of an explicit inflation to end diastole, such as the
# one in [](monolithic_3d0d_biv.py).

problem = pulse.problem.DynamicProblem(
    model=model,
    geometry=geometry,
    bcs=bcs_forward,
    cavities=[
        pulse.problem.Cavity(marker=name, control=pulse.problem.CavityControl(geometry.mesh))
        for name in ("LV", "RV")
    ],
    parameters={
        "mesh_unit": mesh_unit,
        "u_space": "P_2",
        "rho": pulse.Variable(1e3, "kg/m^3"),
        "dt": pulse.Variable(DT, "s"),
    },
)
controller = cycle.CycleController(problem, params)
controller.initialize(t0=0.0)

# ## Stepping
#
# The loop is plain Python. At each step we set the active tension, note the
# phase each cavity is in, and ask the controller for one step. `step` sets
# each cavity's constraint, solves, and only then decides on the next phase,
# so the phase we note before the call is the one the step is solved under. It
# returns `False` if Newton did not converge, even after one retry, and then
# leaves the state as it was before the call.
#
# `pulse.cycle` logs every phase transition. After each step,
# `controller.records` holds each cavity's volume `V`, pressure `P`, compliance
# pressure `P_c` and outflow `Q`, all in SI units.
#
# We also keep the moving geometry every few steps so that `make_animations.py`
# can render it afterwards. Nothing is recorded under CI, where the run is two
# steps rather than two beats, so the video on the page comes from a saved run
# instead.

history: dict[str, list[float]] = {
    k: []
    for k in (
        "time", "V_LV", "V_RV", "p_LV", "p_RV", "Pc_LV", "Pc_RV",
        "Q_LV", "Q_RV", "phase_LV", "phase_RV", "Ta_LV",
    )
}
recorder = animation.FrameRecorder(geometry.mesh, every=5, enabled=not IN_CI, up=up)
vtx = dolfinx.io.VTXWriter(comm, outdir / "displacement.bp", [problem.u], engine="BP4")

logger.info(f"Running {max_steps} steps of {DT * 1e3:.0f} ms...")
t = 0.0
for step in range(1, max_steps + 1):
    t = step * DT
    Ta.assign(activation(t))
    solved_under = {name: controller.cycles[name].phase for name in params}
    if not controller.step(t, DT):
        raise RuntimeError(f"Step to t={t:.4f} s did not converge (phases {solved_under})")
    history["time"].append(t)
    history["Ta_LV"].append(activation(t))
    for name, record in controller.records.items():
        history[f"V_{name}"].append(record.V / mL)
        history[f"p_{name}"].append(record.P / mmHg)
        history[f"Pc_{name}"].append(record.P_c / mmHg)
        history[f"Q_{name}"].append(record.Q / mL)
        history[f"phase_{name}"].append(int(solved_under[name]))
    recorder.record(problem.u, t, step)
    if step % 10 == 0 or step == max_steps:
        vtx.write(t)
vtx.close()

logger.info("Simulation complete.")

# ## Results
#
# We save the traces (volumes in mL, pressures in mmHg, outflows in mL/s and
# the phase each step was solved under) and the recorded frames, and plot them.
# The phase intervals of the left ventricle are shaded behind its traces.

saved = recorder.save(outdir / "frames.npz")
if saved is not None:
    logger.info(f"Saved {len(recorder.times)} frames of the moving geometry to {saved}")

if comm.rank == 0:
    traces = {k: np.asarray(v) for k, v in history.items()}
    np.savez(outdir / "traces.npz", **traces)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    ax_loop, ax_p, ax_v, ax_c = axes.flat
    for name in ("LV", "RV"):
        colour = animation.CHAMBER_COLOURS[name]
        ax_loop.plot(
            traces[f"V_{name}"], traces[f"p_{name}"], color=colour, label=name, linewidth=1.3,
        )
        ax_p.plot(traces["time"], traces[f"p_{name}"], color=colour, label=f"p {name}")
        ax_v.plot(traces["time"], traces[f"V_{name}"], color=colour, label=f"V {name}")
    ax_loop.set_xlabel("V [mL]")
    ax_loop.set_ylabel("p [mmHg]")
    ax_loop.set_title("Pressure-volume loops")
    ax_loop.legend(frameon=False)
    ax_p.set_xlabel("Time [s]")
    ax_p.set_ylabel("p [mmHg]")
    ax_p.set_title("Cavity pressures (LV phases shaded)")
    ax_v.set_xlabel("Time [s]")
    ax_v.set_ylabel("V [mL]")
    ax_v.set_title("Cavity volumes (LV phases shaded)")
    ax_c.plot(
        traces["time"], traces["p_LV"], color=animation.CHAMBER_COLOURS["LV"], label="p LV",
    )
    ax_c.plot(traces["time"], traces["Pc_LV"], color="0.3", linestyle="--", label="$P_c$ LV")
    ax_c.set_xlabel("Time [s]")
    ax_c.set_ylabel("p [mmHg]")
    ax_c.set_title("LV cavity and compliance pressure")
    ax_c.legend(frameon=False)
    for axis in (ax_p, ax_v):
        animation._shade_phases(axis, traces["time"], traces["phase_LV"])
    fig.savefig(outdir / "complete_cycle.png", dpi=140)
    plt.show()

    if not IN_CI:
        # We read the end-diastolic and end-systolic volumes off the phase
        # trace of the last complete beat, rather than taking the largest and
        # smallest volume: filling at a prescribed rate can carry the volume
        # past end diastole before the next beat starts. The first step solved
        # under isovolumic contraction holds V at the end-diastolic volume, and
        # the first one solved under isovolumic relaxation at the end-systolic
        # volume. A run shorter than one beat reports the beat it is in.
        time = traces["time"]
        complete = int(np.floor(time[-1] / PERIOD + 1e-9))
        beat = max(complete - 1, 0)
        in_beat = (time > beat * PERIOD + 1e-9) & (time <= (beat + 1) * PERIOD + 1e-9)
        label = f"beat {beat + 1}" + ("" if complete > 0 else " (incomplete)")
        for name in ("LV", "RV"):
            phase = traces[f"phase_{name}"][in_beat]
            V_beat = traces[f"V_{name}"][in_beat]
            volumes = {}
            for key, value in (
                ("EDV", cycle.Phase.ISOVOLUMIC_CONTRACTION),
                ("ESV", cycle.Phase.ISOVOLUMIC_RELAXATION),
            ):
                hits = np.flatnonzero(phase == int(value))
                if hits.size:
                    volumes[key] = float(V_beat[hits[0]])
            peak = float(traces[f"p_{name}"][in_beat].max())
            if len(volumes) < 2:
                missing = {"EDV", "ESV"} - set(volumes)
                logger.info(
                    f"{name}, {label}: no {' or '.join(sorted(missing))} "
                    f"(the beat never reached that phase); peak {peak:.1f} mmHg",
                )
                continue
            EDV, ESV = volumes["EDV"], volumes["ESV"]
            logger.info(
                f"{name}, {label}: EDV {EDV:.1f} mL, ESV {ESV:.1f} mL, "
                f"EF {100 * (1 - ESV / EDV):.1f}%, peak {peak:.1f} mmHg",
            )

# ## A whole beat
#
# The figure and the video come from a full run kept in `_static/`, rather
# than from the two steps this page takes under CI. To regenerate them, run
#
# ```bash
# python3 complete_cycle.py
# python3 make_animations.py complete_cycle
# ```
#
# ```{figure} ../../_static/pv_loop_complete_cycle.png
# ---
# name: pv_loop_complete_cycle
# ---
# Both ventricles over two beats. In the second beat the left ventricle ejects
# 42 mL (EF 38%) against a peak of 131 mmHg, and the right 35 mL (EF 46%)
# against 38 mmHg. Each loop closes through PRELOAD, which ramps the pressure
# back up to end diastole. The first left ventricular loop ends systole at
# 74.5 mL rather than 69.1 mL, since its Windkessel starts from `p_init`
# rather than from the pressure it has drained to. The run stops at 1.6 s,
# partway through the second beat's filling.
# ```
#
# <video width="720" controls loop autoplay muted>
#   <source src="../../_static/complete_cycle.mp4" type="video/mp4">
#   <p>The biventricular mesh contracting through two beats, coloured by
#   displacement, with both pressure-volume loops drawn alongside.</p>
# </video>
#
# ## Where to go next
#
# Here the active tension is a prescribed twitch and each ventricle ejects
# into its own Windkessel. [](land_circulation_biv.py) drives a biventricular
# ellipsoid with a crossbridge model and a different strength per region,
# coupled to a full closed-loop circulation, and [](monolithic_3d0d_biv.py)
# solves this mesh and a closed-loop circulation together in a single Newton
# system.

# ## References
# ```{bibliography}
# :filter: docname in docnames
# ```
