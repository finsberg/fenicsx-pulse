# # Monolithic 3D-0D coupling: a UK Biobank biventricular mesh in a closed circulation
#
# This is the biventricular counterpart of [](monolithic_3d0d.py). We couple
# both ventricles of a clipped UK Biobank atlas mesh to the closed-loop
# circulation model of Regazzoni et al., solving the displacement, both cavity
# pressures and all twelve circuit states together in one Newton system. As in
# the LV demo, the constraint tying each cavity to its chamber is a row of that
# system, so the two cannot disagree by more than the solver tolerance.
#
# ## Adding the second ventricle
#
# Structurally this costs us almost nothing, since the coupling machinery
# already takes a list of cavities. Dropping the `RV` component from the `.ode`
# alongside `LV` leaves a circuit that expects both `p_LV` and `p_RV`, and each
# of those is supplied by the Lagrange multiplier its own cavity already
# carries. In practice we need two `Cavity` entries, two `ChamberCoupling`
# entries, and one extra row in the block system.
#
# ## Why there is no calibration here
#
# The LV demo spends most of its length measuring the mesh and tuning the
# circuit's loading against that measurement. It has to, because an idealized
# ellipsoid does not behave like a real ventricle: its unloaded cavity holds
# far more than a person's, and coupled to the published circuit it fills to
# nearly 200 mL.
#
# This mesh is a real heart in a real end-diastolic configuration, so we can
# get the operating point directly. We seed the circuit with the mesh's own
# end-diastolic volumes, run it alone until it reaches a limit cycle, and then
# prestress the mesh to the pressures it settles at. The two then agree at end
# diastole by construction, which leaves contractility -- `TA_SCALE` below --
# as the only free parameter.
#
# ## Units
#
# The UKB mesh comes in millimetres, and we scale it to metres on load, as the
# other demos built on it do. The chamber coupling converts between the
# circuit's millilitres and the mesh's cubic metres, and it assumes metres, so
# leaving the mesh in millimetres would couple the circuit to a cavity a
# billion times the intended size.

import logging
import os
import shutil
from pathlib import Path

from mpi4py import MPI

import circulation
import dolfinx
import io4dolfinx
import ldrb
import matplotlib.pyplot as plt
import numpy as np
from circulation import bestel, regazzoni2020
from circulation.units import kPa_to_mmHg, mmHg_to_kPa
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp

# A sibling module in this directory, not a package.
import animation
import cardiac_geometries
import cardiac_geometries.geometry
import pulse
from pulse.circulation import ChamberCoupling, GotranxCirculation, mL, mmHg

circulation.log.setup_logging(logging.INFO)
logging.getLogger("scifem").setLevel(logging.WARNING)
logger = logging.getLogger("pulse")
comm = MPI.COMM_WORLD

_ci = os.getenv("CI", "").strip().lower()
IN_CI = _ci not in ("", "0", "false", "no", "off")

# Inertia works the same way here as in the LV demo. Quasi-static is the
# default, and switching it on also enables the viscous term and the damping
# Robin conditions, without which the cavity pressures ring against their own
# constraints. As there, `PULSE_DYNAMIC=1` sets the flag without editing the
# file.
DYNAMIC = os.getenv("PULSE_DYNAMIC", "0").strip().lower() in ("1", "true", "yes", "on")
ARM = "dynamic" if DYNAMIC else "quasistatic"

# This is the contractility: the Bestel trace sets the shape of the twitch and
# this sets its size. With end diastole pinned by the prestressing, it is what
# ends up deciding the ejection fraction, and we picked the value by running
# the quasi-static arm and reading that ejection fraction off the loop.
TA_SCALE = 1.0

BEAT_LENGTH = 1.0  # s
DT = 0.002  # s
# Two beats are enough to settle the left ventricle: its end-diastolic and
# end-systolic volumes move by half a percent between the first and the second.
# The right ventricle is still drifting by about seven percent, since the
# pulmonary compartment it fills through is more compliant and takes longer to
# settle. Increase this if the right side matters for what you are doing.
NUM_BEATS = 1 if IN_CI else 2

CHAR_LENGTH = 10.0  # mm; the atlas is smooth, so a coarse mesh suffices

outdir = Path("results_monolithic_3d0d_biv")
geodir = Path("ukb-monolithic-3d0d")
outdir.mkdir(exist_ok=True)

# ## Geometry
#
# We use the mean shape of the atlas (`mode=-1, std=0`) at end diastole,
# clipped at the valve plane so that the mesh has a single `BASE` surface
# rather than four valve annuli, and rotated so that the base normal points
# along x. The fibres come from LDRB, with separate angles for the two
# ventricles.

if not (geodir / "geometry.bp").exists():
    logger.info("Generating the UKB mesh...")
    geo = cardiac_geometries.mesh.ukb(
        outdir=geodir,
        comm=comm,
        mode=-1,
        std=0,
        case="ED",
        char_length_max=CHAR_LENGTH,
        char_length_min=CHAR_LENGTH,
        clipped=True,
    )
    geo = geo.rotate(target_normal=[1.0, 0.0, 0.0], base_marker="BASE")

    system = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=cardiac_geometries.mesh.transform_markers(geo.markers, clipped=True),
        alpha_endo_lv=60,
        alpha_epi_lv=-60,
        alpha_endo_rv=90,
        alpha_epi_rv=-25,
        beta_endo_lv=-20,
        beta_epi_lv=20,
        beta_endo_rv=0,
        beta_epi_rv=20,
        fiber_space="Quadrature_6",
    )
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
    )
comm.barrier()

geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)

# We rotate here, after loading, rather than relying on the rotation in the
# generation step above. That step does rotate the mesh, but the folder also
# holds the `.msh` it was built from, and that unrotated mesh is what
# `from_folder` gives back. Rotating at this point makes the orientation a
# property of what we actually solve on. It matters because the sliding-base
# condition below constrains a single displacement component: on an unrotated
# mesh it would hold the base in a plane that cuts through the ventricle at an
# angle instead of in the base plane itself.
geo = geo.rotate(target_normal=[1.0, 0.0, 0.0], base_marker="BASE")
geo.mesh.geometry.x[:] *= 1e-3  # mm -> m
geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

up = animation.base_normal(geometry, "BASE")
if abs(up[0]) < 0.99:
    raise RuntimeError(
        f"the base normal is {up.round(3)}, not the x axis the sliding-base "
        "condition assumes -- the rotation above did not take effect",
    )

EDV = {
    chamber: comm.allreduce(geometry.volume(chamber), op=MPI.SUM) for chamber in ("LV", "RV")
}
logger.info(f"Mesh end-diastolic volumes: LV {EDV['LV'] / mL:.1f} mL, RV {EDV['RV'] / mL:.1f} mL")


def build_model(f0, s0, Ta):
    material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material = pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)  # type: ignore[arg-type]
    # This does nothing without a strain rate, so the static solves below are
    # unaffected.
    viscoelasticity = (
        pulse.viscoelasticity.Viscous() if DYNAMIC else pulse.viscoelasticity.NoneViscoElasticity()
    )
    return pulse.CardiacModel(
        material=material,
        active=pulse.ActiveStress(f0, activation=Ta, formulation=pulse.ActiveStressFormulation.stretch),
        compressibility=pulse.Compressible(),
        viscoelasticity=viscoelasticity,
    )


def robin_bcs():
    def spring(marker, value, damping=False):
        return pulse.RobinBC(
            value=pulse.Variable(
                dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(value)),
                "Pa s/ m" if damping else "Pa / m",
            ),
            marker=geometry.markers[marker][0],
            damping=damping,
        )

    bcs = [spring("EPI", 1.0e5), spring("BASE", 1.0e6)]
    if DYNAMIC:
        bcs += [spring("EPI", 5.0e3, damping=True), spring("BASE", 5.0e3, damping=True)]
    return tuple(bcs)


def sliding_base(V: dolfinx.fem.FunctionSpace):
    """Hold the base in its own plane but let it slide within it.

    The mesh is rotated so the base normal is x, which is what makes this one
    component rather than a projection.
    """
    facets = geometry.facet_tags.find(geometry.markers["BASE"][0])
    dofs = dolfinx.fem.locate_dofs_topological(V.sub(0), 2, facets)
    return [dolfinx.fem.dirichletbc(0.0, dofs, V.sub(0))]


# ## Activation
#
# We solve the Bestel twitch once up front, since it depends on time alone and
# prescribing it therefore introduces no coupling error.

times = np.arange(0.0, BEAT_LENGTH, DT)
activation = solve_ivp(
    bestel.BestelActivation(),
    [0.0, BEAT_LENGTH],
    [0.0],
    t_eval=times,
    method="Radau",
).y[0]
logger.info(f"Peak activation: {TA_SCALE * activation.max() * 1e-3:.1f} kPa")


def activation_at(t: float) -> float:
    return TA_SCALE * float(np.interp(t % BEAT_LENGTH, times, activation))


# ## The operating point
#
# We seed the circuit with this mesh's own end-diastolic volumes and run it by
# itself until it reaches a limit cycle. Then we prestress the mesh to the
# end-diastolic pressures it arrives at, so that afterwards the two agree on
# both volumes and both pressures at end diastole. This takes the place of the
# calibration that the LV demo needs a separate module for, and it only works
# because the geometry is a real one.

state_file = outdir / "circ_state.npy"
if comm.rank == 0 and not state_file.exists():
    standalone = regazzoni2020.Regazzoni2020(parameters={"HR": 1.0 / BEAT_LENGTH}, add_units=False)
    history_0d = standalone.solve(
        num_beats=10,
        initial_state={"V_LV": EDV["LV"] / mL, "V_RV": EDV["RV"] / mL},
        dt=0.001,
    )
    np.save(
        state_file,
        {
            "state": dict(zip(standalone.state_names(), standalone.state)),
            "p_LV_ED": float(history_0d["p_LV"][-1]),
            "p_RV_ED": float(history_0d["p_RV"][-1]),
        },
        allow_pickle=True,
    )
comm.barrier()

cached = np.load(state_file, allow_pickle=True).item()
circ_state = cached["state"]
p_ED = {"LV": mmHg_to_kPa(cached["p_LV_ED"]), "RV": mmHg_to_kPa(cached["p_RV_ED"])}
logger.info(f"End-diastolic pressures from the circuit: "
            f"LV {p_ED['LV']:.2f} kPa, RV {p_ED['RV']:.2f} kPa")

# ## Prestressing
#
# We recover the unloaded configuration by unloading both cavities together.

Ta = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "Pa")
traction = {
    chamber: pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "kPa")
    for chamber in ("LV", "RV")
}

prestress_fname = outdir / "prestress_biv.bp"
if not prestress_fname.exists():
    logger.info("Prestressing to recover the unloaded reference configuration...")
    prestress_problem = pulse.unloading.PrestressProblem(
        geometry=geometry,
        model=build_model(geo.f0, geo.s0, Ta),
        bcs=pulse.BoundaryConditions(
            robin=robin_bcs(),
            dirichlet=(sliding_base,),
            neumann=tuple(
                pulse.NeumannBC(traction=traction[c], marker=geometry.markers[c][0])
                for c in ("LV", "RV")
            ),
        ),
        parameters={"u_space": "P_2", "mesh_unit": "m"},
        targets=[
            pulse.unloading.TargetPressure(traction=traction[c], target=p_ED[c], name=c)
            for c in ("LV", "RV")
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

unloaded = {c: comm.allreduce(geometry.volume(c), op=MPI.SUM) for c in ("LV", "RV")}
logger.info(f"Unloaded volumes: LV {unloaded['LV'] / mL:.1f} mL, RV {unloaded['RV'] / mL:.1f} mL")

# ## Inflation to end diastole
#
# Here we go from the unloaded configuration back to the volumes the circuit
# was seeded with. We prescribe those volumes rather than coupling them, since
# this is a ramp and not part of the time stepping.

model = build_model(f0, s0, Ta)
bcs = pulse.BoundaryConditions(robin=robin_bcs(), dirichlet=(sliding_base,))

inflation_volume = {
    c: dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(unloaded[c]))
    for c in ("LV", "RV")
}
inflation = pulse.problem.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs,
    cavities=[
        pulse.problem.Cavity(marker=c, volume=inflation_volume[c]) for c in ("LV", "RV")
    ],
    parameters={"mesh_unit": "m"},
)
inflation.solve()

for fraction in np.linspace(0.0, 1.0, 20)[1:]:
    for c in ("LV", "RV"):
        inflation_volume[c].value = unloaded[c] + fraction * (EDV[c] - unloaded[c])
    if not inflation.solve():
        raise RuntimeError(f"inflation failed at {fraction:.2f} of the way to end diastole")

p_inflated = [float(p.x.array[0]) for p in inflation.cavity_pressures]
logger.info(
    f"Inflated to LV {comm.allreduce(geometry.volume('LV', u=inflation.u), op=MPI.SUM) / mL:.1f} mL "
    f"at {p_inflated[0] / mmHg:.1f} mmHg, "
    f"RV {comm.allreduce(geometry.volume('RV', u=inflation.u), op=MPI.SUM) / mL:.1f} mL "
    f"at {p_inflated[1] / mmHg:.1f} mmHg",
)

# ## The coupled problem
#
# Both chamber closures come out of the circuit, and the two cavity pressures
# take their place.

circulation_model = GotranxCirculation(
    ode_file=regazzoni2020.ODE_FILE,
    parameters=regazzoni2020.flat_ode_parameters(
        circulation.base.remove_units(regazzoni2020.Regazzoni2020.default_parameters())
        | {"HR": 1.0 / BEAT_LENGTH},
    ),
    drop_components=("timing", "LV", "RV"),
)
beat_phase = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))

coupled_parameters = {"mesh_unit": "m", "circulation_scheme": "backward_euler"}
if DYNAMIC:
    coupled_parameters |= {
        "rho": pulse.Variable(1e3, "kg/m^3"),
        "dt": pulse.Variable(DT, "s"),
    }

Problem = pulse.problem.DynamicProblem if DYNAMIC else pulse.problem.StaticProblem
problem = Problem(
    model=model,
    geometry=geometry,
    bcs=bcs,
    cavities=[pulse.problem.Cavity(marker=c, volume=None) for c in ("LV", "RV")],
    circulation=circulation_model,
    chambers=[
        ChamberCoupling(marker="LV", volume_state="V_LV", pressure_missing="p_LV"),
        ChamberCoupling(marker="RV", volume_state="V_RV", pressure_missing="p_RV"),
    ],
    circulation_missing={"beat_phase": beat_phase},
    parameters=coupled_parameters,
)

# We start from the inflated configuration and from the circuit state that
# matches it.
problem.u.x.array[:] = inflation.u.x.array
problem.u_old.x.array[:] = inflation.u.x.array
for i, pressure in enumerate(p_inflated):
    problem.cavity_pressures[i].x.array[:] = pressure
    problem.cavity_pressures_old[i].x.array[:] = pressure
if DYNAMIC:
    # The inflation gives us a configuration but no motion, so we start from
    # rest here as well.
    problem.v_old.x.array[:] = 0.0
    problem.a_old.x.array[:] = 0.0

names = list(circulation_model.state_names)
for name, state, state_old in zip(
    names,
    problem.circulation_states,
    problem.circulation_states_old,
):
    # We take the two chamber volumes from the mesh rather than the circuit.
    value = EDV[name[2:]] / mL if name in ("V_LV", "V_RV") else float(circ_state[name])
    state.x.array[:] = value
    state_old.x.array[:] = value

problem.circulation_dt.value = DT

# ## Stepping

index = {name: i for i, name in enumerate(names)}
history: dict[str, list[float]] = {
    key: [] for key in
    (
        "time", "V_LV", "V_RV", "p_LV", "p_RV", "Ta",
        "iterations", "constraint",
    )
}

# We keep the moving geometry every few steps so that `make_animations.py` can
# render it afterwards. Nothing is recorded under CI, where the run is two
# steps rather than a whole beat, so the video on the page comes from a saved
# run instead.
recorder = animation.FrameRecorder(geometry.mesh, every=5, enabled=not IN_CI, up=up)

max_steps = 2 if IN_CI else int(NUM_BEATS * BEAT_LENGTH / DT)
t = 0.0
for step in range(max_steps):
    t += DT
    problem.circulation_time.value = t
    beat_phase.value = t % BEAT_LENGTH
    Ta.assign(activation_at(t))

    if not problem.solve():
        raise RuntimeError(f"Monolithic solve failed at t={t:.4f}")

    worst = 0.0
    for i, chamber in enumerate(("LV", "RV")):
        volume = comm.allreduce(geometry.volume(chamber, u=problem.u), op=MPI.SUM)
        state = float(problem.circulation_states[index[f"V_{chamber}"]].x.array[0]) * mL
        history[f"V_{chamber}"].append(state / mL)
        history[f"p_{chamber}"].append(float(problem.cavity_pressures[i].x.array[0]) / mmHg)
        worst = max(worst, abs(volume - state) / state)

    history["time"].append(t)
    history["Ta"].append(float(Ta.value.value))
    history["iterations"].append(int(problem.problem.solver.getIterationNumber()))
    history["constraint"].append(worst)
    recorder.record(problem.u, t, step)

    if step % 50 == 0:
        logger.info(
            f"t={t:.3f}  LV {history['V_LV'][-1]:6.1f} mL {history['p_LV'][-1]:7.1f} mmHg   "
            f"RV {history['V_RV'][-1]:6.1f} mL {history['p_RV'][-1]:6.1f} mmHg   "
            f"Ta={history['Ta'][-1] * 1e-3:5.1f} kPa  constraint={worst:.1e}",
        )

logger.info(f"Worst constraint violation over the run: {max(history['constraint']):.3e}")

saved = recorder.save(outdir / f"frames-{ARM}.npz")
if saved is not None:
    logger.info(f"Saved {len(recorder.times)} frames of the moving geometry to {saved}")

for chamber in ("LV", "RV"):
    V = np.asarray(history[f"V_{chamber}"])
    p = np.asarray(history[f"p_{chamber}"])
    if V.size > 10:
        EDV_run, ESV_run = float(V.max()), float(V.min())
        logger.info(
            f"{chamber}: EDV {EDV_run:.1f} mL, ESV {ESV_run:.1f} mL, "
            f"SV {EDV_run - ESV_run:.1f} mL, EF {100 * (1 - ESV_run / EDV_run):.1f}%, "
            f"peak {p.max():.1f} mmHg",
        )

if comm.rank == 0:
    np.savez(
        outdir / f"traces_biv-{ARM}.npz",
        **{k: np.asarray(v) for k, v in history.items()},
    )

    fig = plt.figure(layout="constrained", figsize=(11, 8))
    gs = GridSpec(3, 2, figure=fig)
    ax_loop = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[1, 1])
    ax_ta = fig.add_subplot(gs[2, 1])

    for chamber, colour in (("LV", "crimson"), ("RV", "steelblue")):
        ax_loop.plot(
            history[f"V_{chamber}"], history[f"p_{chamber}"], color=colour,
            label=chamber, linewidth=1.1,
        )
        ax_p.plot(history["time"], history[f"p_{chamber}"], color=colour, label=chamber)
        ax_v.plot(history["time"], history[f"V_{chamber}"], color=colour, label=chamber)
    ax_loop.set_xlabel("V [mL]")
    ax_loop.set_ylabel("p [mmHg]")
    ax_loop.set_title(f"Pressure-volume loops ({ARM})")
    ax_loop.legend()
    ax_p.set_ylabel("p [mmHg]")
    ax_p.legend(fontsize="x-small")
    ax_v.set_ylabel("V [mL]")
    ax_ta.plot(history["time"], np.asarray(history["Ta"]) * 1e-3, color="0.3")
    ax_ta.set_ylabel("Ta [kPa]")
    ax_ta.set_xlabel("Time [s]")

    fig.savefig(outdir / f"monolithic_3d0d_biv-{ARM}.png", dpi=140)
    plt.close(fig)

logger.info("Done.")

# ## A whole beat
#
# As in the LV demo, the figure and video come from a full run kept in
# `_static/` rather than from the two steps this page takes under CI:
#
# ```bash
# python3 monolithic_3d0d_biv.py
# python3 make_animations.py monolithic_3d0d_biv
# ```
#
# ```{figure} ../../_static/pv_loop_monolithic_3d0d_biv.png
# ---
# name: pv_loop_monolithic_3d0d_biv
# ---
# Both ventricles over two beats. The left ejects 70 mL against a peak of about
# 100 mmHg, and the right nearly as much against a quarter of that. The left
# loop closes on itself, while the right is still drifting, for the reason
# given at `NUM_BEATS`.
# ```
#
# <video width="720" controls loop autoplay muted>
#   <source src="../../_static/monolithic_3d0d_biv.mp4" type="video/mp4">
#   <p>The biventricular mesh contracting through two beats, coloured by
#   displacement, with both pressure-volume loops drawn alongside.</p>
# </video>
