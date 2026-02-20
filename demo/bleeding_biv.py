# # Bleeding BiV to a 0D circulatory model and a 0D cell model

# This example is similar to the [BiV example](time_dependent_land_circ_biv.py). However, in this example we also simulate a bleeding of the BiV by draining the


from pathlib import Path
from typing import Literal
import json

from mpi4py import MPI
import dolfinx
import logging
import os
from functools import lru_cache

import circulation
from dolfinx import log
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import gotranx
import io4dolfinx
import cardiac_geometries
import cardiac_geometries.geometry
import pulse


# Run first Zenker to get the correct heart rate for normal conditions
def mmHg_to_kPa(x):
    return x * 0.133322


def custom_json(obj):
    if isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj)


def run_zenker(outdir: Path):
    zenker_file = outdir / "zenker.json"
    if not zenker_file.is_file():
        start_withdrawal = 300
        zenker = circulation.zenker.Zenker(
            parameters={
                "start_withdrawal": start_withdrawal,
                "end_withdrawal": 400,
                "flow_withdrawal": -20,
                "flow_infusion": 1,
                "Vv0_min": 1000,
                "Vv0_max": 2000,
            },
        )
        dt_eval = 0.1
        history = zenker.solve(T=700.0, dt=1e-3, dt_eval=dt_eval)
        start_plot = 0
        time = history["time"][start_plot:]

        fig, ax = plt.subplots(7, 2, sharex=True, figsize=(10, 14))
        ax[0, 0].plot(time, history["Vv"][start_plot:])
        ax[0, 0].set_ylabel("Vv [mL]")
        ax[1, 0].plot(time, history["Va"][start_plot:])
        ax[1, 0].set_ylabel("Va [mL]")
        ax[2, 0].plot(time, history["V_ED"][start_plot:])
        ax[2, 0].set_ylabel("V_ED [mL]")
        ax[3, 0].plot(time, history["V_ES"][start_plot:])
        ax[3, 0].set_ylabel("V_ES [mL]")
        SV = np.subtract(history["V_ED"], history["V_ES"])
        ax[4, 0].plot(time, SV[start_plot:], label="SV")
        ax[4, 0].set_ylabel("Stroke volume [mL]")
        ax[0, 1].plot(time, history["fHR"][start_plot:])
        ax[0, 1].set_ylabel("fHR [Hz]")
        CO = SV * history["fHR"]
        ax[1, 1].plot(time, CO[start_plot:], label="CO")
        ax[1, 1].set_ylabel("Cardiac output [mL/s]")
        ax[2, 1].plot(time, history["Pa"][start_plot:])
        ax[2, 1].set_ylabel("Pa [mmHg]")
        ax[3, 1].plot(time, history["S"][start_plot:])
        ax[3, 1].set_ylabel("S")
        ax[4, 1].plot(time, history["Pcvp"][start_plot:])
        ax[4, 1].set_ylabel("Pcvp [mmHg]")
        ax[4, 0].set_xlabel("Time [s]")
        ax[4, 1].set_xlabel("Time [s]")
        ax[5, 0].plot(time, history["R_TPR"][start_plot:])
        ax[5, 0].set_ylabel("R_TPR [mmHg/mL/s]")
        ax[5, 1].plot(time, history["C_PRSW"][start_plot:])
        ax[5, 1].set_ylabel("C_PRSW [mL/mmHg]")
        ax[5, 0].set_xlabel("Time [s]")
        ax[5, 1].set_xlabel("Time [s]")
        ax[6, 0].plot(time, history["Vv0"][start_plot:])
        ax[6, 0].set_ylabel("Vv0 [mL]")
        ax[6, 1].plot(time, history["TotalVolume"][start_plot:])
        ax[6, 1].set_ylabel("Total volume [mL]")
        ax[6, 0].set_xlabel("Time [s]")
        ax[6, 1].set_xlabel("Time [s]")
        fig.savefig(outdir / "zenker.png")

        print("Before withdrawal")
        before_index = int(start_withdrawal / dt_eval)
        HR_before = np.round(history["fHR"][before_index], decimals=1)
        print("HR: ", HR_before)
        print("R_TPR: ", history["R_TPR"][before_index])
        print("C_PRSW: ", history["C_PRSW"][before_index])
        print("Pa: ", history["Pa"][before_index])
        print("Pa kPa: ", mmHg_to_kPa(history["Pa"][before_index]))
        print("Pcvp: ", history["Pcvp"][before_index])
        print("Pcvp kPa: ", mmHg_to_kPa(history["Pcvp"][before_index]))
        print("fHR: ", history["fHR"][before_index])
        print("Total volume: ", history["TotalVolume"][before_index])

        print("\nAfter withdrawal")
        after_index = len(history["time"]) - 1
        HR_after = np.round(history["fHR"][after_index], decimals=1)
        print("HR: ", HR_after)
        print("R_TPR: ", history["R_TPR"][after_index])
        print("C_PRSW: ", history["C_PRSW"][after_index])
        print("Pa: ", history["Pa"][after_index])
        print("Pa kPa: ", mmHg_to_kPa(history["Pa"][after_index]))
        print("Pcvp: ", history["Pcvp"][after_index])
        print("Pcvp kPa: ", mmHg_to_kPa(history["Pcvp"][after_index]))
        print("fHR: ", history["fHR"][after_index])
        print("Volume end: ", history["TotalVolume"][after_index])

        print("\nChanges")

        R_TPR_factor = history["R_TPR"][after_index] / history["R_TPR"][before_index]
        C_PRSW_factor = history["C_PRSW"][after_index] / history["C_PRSW"][before_index]
        print(
            "Volume change: ",
            history["TotalVolume"][after_index] - history["TotalVolume"][before_index],
        )
        print("RTPR factor change: ", R_TPR_factor)
        print("C_PRSW factor change: ", C_PRSW_factor)

        history["HR_before"] = HR_before
        history["R_TPR_factor"] = R_TPR_factor
        history["C_PRSW_factor"] = C_PRSW_factor
        history["HR_after"] = HR_after
        history["before_index"] = before_index
        history["after_index"] = after_index

        Path(zenker_file).write_text(json.dumps(history, indent=4, default=custom_json))

    return json.loads(zenker_file.read_text())


def run_TorOrdLand(
    comm,
    outdir: Path,
    HR: float,
    Ta_factor: float = 1.0,
    dt: float = 0.001,
    label: Literal["before", "after"] = "before",
):
    BCL = 1 / HR
    T = round(BCL * 1000.0)
    print(f"Running TorOrdLand with HR {HR} and BCL {BCL:.3f} ms")
    times = np.arange(0.0, BCL, dt)

    if not Path("TorOrdLand.py").is_file():
        ode = gotranx.load_ode("TorOrdLand.ode")
        ode = ode.remove_singularities()
        code = gotranx.cli.gotran2py.get_code(
            ode,
            scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
            shape=gotranx.codegen.base.Shape.single,
        )
        Path("TorOrdLand.py").write_text(code)
    comm.barrier()
    import TorOrdLand

    TorOrdLand_model = TorOrdLand.__dict__

    Ta_index = TorOrdLand_model["monitor_index"]("Ta")
    y = TorOrdLand_model["init_state_values"]()
    # Get initial parameter values
    p = TorOrdLand_model["init_parameter_values"](i_Stim_Period=T)
    import numba

    fgr = numba.njit(TorOrdLand_model["generalized_rush_larsen"])
    mon = numba.njit(TorOrdLand_model["monitor_values"])
    V_index = TorOrdLand_model["state_index"]("v")
    Ca_index = TorOrdLand_model["state_index"]("cai")

    # Time in milliseconds
    dt_cell = 0.1

    state_file = outdir / f"state_{label}.npy"
    Ta_file = outdir / f"Ta_{label}.npy"
    if not Ta_file.is_file():

        @numba.jit(nopython=True)
        def solve_beat(times, states, dt, p, V_index, Ca_index, Vs, Cais, Tas):
            for i, ti in enumerate(times):
                states[:] = fgr(states, ti, dt, p)
                Vs[i] = states[V_index]
                Cais[i] = states[Ca_index]
                monitor = mon(ti, states, p)
                Tas[i] = monitor[Ta_index]

        nbeats = 10 if os.environ.get("CI") else 200
        times = np.arange(0, T, dt_cell)
        all_times = np.arange(0, T * nbeats, dt_cell)
        Vs = np.zeros(len(times) * nbeats)
        Cais = np.zeros(len(times) * nbeats)
        Tas = np.zeros(len(times) * nbeats)
        for beat in range(nbeats):
            print(f"Solving beat {beat}")
            V_tmp = Vs[beat * len(times) : (beat + 1) * len(times)]
            Cai_tmp = Cais[beat * len(times) : (beat + 1) * len(times)]
            Ta_tmp = Tas[beat * len(times) : (beat + 1) * len(times)]
            solve_beat(times, y, dt_cell, p, V_index, Ca_index, V_tmp, Cai_tmp, Ta_tmp)

        fig, ax = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(10, 10))
        ax[0, 0].plot(all_times, Vs)
        ax[1, 0].plot(all_times, Cais)
        ax[2, 0].plot(all_times, Tas)
        ax[0, 1].plot(times, Vs[-len(times) :])
        ax[1, 1].plot(times, Cais[-len(times) :])
        ax[2, 1].plot(times, Tas[-len(times) :])
        ax[0, 0].set_ylabel("V")
        ax[1, 0].set_ylabel("Cai")
        ax[2, 0].set_ylabel("Ta")
        ax[2, 0].set_xlabel("Time [ms]")
        ax[2, 1].set_xlabel("Time [ms]")

        fig.savefig(outdir / f"Ta_ORdLand_{label}.png")

        np.save(Ta_file, Tas[-len(times) :])
        np.save(state_file, y)

    all_Ta = np.load(Ta_file)
    ts = np.linspace(0, BCL, len(all_Ta))

    def get_activation(t: float):
        return np.interp(t % BCL, ts, all_Ta) * 10.0 * Ta_factor

    return get_activation


def run_3D_model(
    comm,
    geo,
    get_activation,
    outdir: Path,
    label: Literal["before", "after"] = "before",
    num_beats=5,
    dt: float = 0.001,
    R_TPR_factor: float = 1.0,
    C_PRSW_factor: float = 1.0,
    HR: float = 1.0,
    p_AR_SYS: float = 80.0,
    p_AR_PUL: float = 35.0,
    p_VEN_SYS: float = 30.0,
    p_VEN_PUL: float = 24.0,
    mesh_unit: str = "m",
    volume2ml: float = 1000.0,
):
    geometry = pulse.HeartGeometry.from_cardiac_geometries(
        geo,
        metadata={"quadrature_degree": 6},
    )

    material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    # material_params = pulse.HolzapfelOgden.orthotropic_parameters()
    material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore

    # We use an active stress approach with 30% transverse active stress (see {py:meth}`pulse.active_stress.transversely_active_stress`)

    Ta = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)),
        "kPa",
    )
    active_model = pulse.ActiveStress(geo.f0, activation=Ta)

    # We use an incompressible model

    comp_model = pulse.compressibility.Compressible2()

    # and assembles the `CardiacModel`

    model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
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
        value=alpha_epi_perp,
        marker=geometry.markers["EPI"][0],
        perpendicular=True,
    )
    alpha_base = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e6)),
        "Pa / m",
    )
    robin_base = pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])
    robin = [robin_epi, robin_epi_perp, robin_base]

    lvv_initial = comm.allreduce(geometry.volume("LV"), op=MPI.SUM)
    lv_volume = dolfinx.fem.Constant(
        geometry.mesh, dolfinx.default_scalar_type(lvv_initial),
    )
    lv_cavity = pulse.problem.Cavity(marker="LV", volume=lv_volume)

    rvv_initial = comm.allreduce(geometry.volume("RV"), op=MPI.SUM)
    rv_volume = dolfinx.fem.Constant(
        geometry.mesh, dolfinx.default_scalar_type(rvv_initial),
    )
    rv_cavity = pulse.problem.Cavity(marker="RV", volume=rv_volume)

    print("Initial volumes", lvv_initial * volume2ml, rvv_initial * volume2ml)

    cavities = [lv_cavity, rv_cavity]
    parameters = {"base_bc": pulse.problem.BaseBC.free, "mesh_unit": mesh_unit}
    bcs = pulse.BoundaryConditions(robin=robin)
    problem = pulse.problem.StaticProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        cavities=cavities,
        parameters=parameters,
    )

    problem.solve()

    vtx = dolfinx.io.VTXWriter(
        geometry.mesh.comm,
        f"{outdir}/displacement_{label}.bp",
        [problem.u],
        engine="BP4",
    )
    vtx.write(0.0)

    filename = outdir / Path(f"function_checkpoint_{label}.bp")
    io4dolfinx.write_mesh(filename, geometry.mesh)
    output_file = outdir / f"output_{label}.json"

    Ta_history = []

    def callback(model, i: int, t: float, save=True):
        Ta_history.append(get_activation(t))

        if save:
            io4dolfinx.write_function(filename, problem.u, time=t, name="displacement")
            vtx.write(t)
            out = {k: v[: i + 1] for k, v in model.history.items()}
            out["Ta"] = Ta_history
            if comm.rank == 0:
                output_file.write_text(json.dumps(out, indent=4, default=custom_json))

                fig = plt.figure(layout="constrained", figsize=(12, 8))
                gs = GridSpec(3, 4, figure=fig)
                ax1 = fig.add_subplot(gs[:, 0])
                ax2 = fig.add_subplot(gs[:, 1])
                ax3 = fig.add_subplot(gs[0, 2])
                ax4 = fig.add_subplot(gs[1, 2])
                ax5 = fig.add_subplot(gs[0, 3])
                ax6 = fig.add_subplot(gs[1, 3])
                ax7 = fig.add_subplot(gs[2, 2:])

                ax1.plot(model.history["V_LV"][: i + 1], model.history["p_LV"][: i + 1])
                ax1.set_xlabel("LVV [mL]")
                ax1.set_ylabel("LVP [mmHg]")

                ax2.plot(model.history["V_RV"][: i + 1], model.history["p_RV"][: i + 1])
                ax2.set_xlabel("RVV [mL]")
                ax2.set_ylabel("RVP [mmHg]")

                ax3.plot(model.history["time"][: i + 1], model.history["p_LV"][: i + 1])
                ax3.set_ylabel("LVP [mmHg]")
                ax4.plot(model.history["time"][: i + 1], model.history["V_LV"][: i + 1])
                ax4.set_ylabel("LVV [mL]")

                ax5.plot(model.history["time"][: i + 1], model.history["p_RV"][: i + 1])
                ax5.set_ylabel("RVP [mmHg]")
                ax6.plot(model.history["time"][: i + 1], model.history["V_RV"][: i + 1])
                ax6.set_ylabel("RVV [mL]")

                ax7.plot(model.history["time"][: i + 1], Ta_history[: i + 1])
                ax7.set_ylabel("Ta [kPa]")

                for axi in [ax3, ax4, ax5, ax6, ax7]:
                    axi.set_xlabel("Time [s]")

                fig.savefig(outdir / f"pv_loop_incremental_{label}.png")
                plt.close(fig)
            comm.barrier()

    def p_BiV_func(V_LV, V_RV, t):
        print("Calculating pressure at time", t)
        value = get_activation(t)
        print(
            f"Time{t} with activation: {value} and volumes: {V_LV} mL (LV) {V_RV} mL (RV)",
        )
        old_Ta = Ta.value.value
        dTa = value - old_Ta

        new_value_LV = V_LV * (1.0 / volume2ml)
        new_value_RV = V_RV * (1.0 / volume2ml)

        old_value_LV = lv_volume.value.copy()
        old_value_RV = rv_volume.value.copy()

        dLV = new_value_LV - old_value_LV
        dRV = new_value_RV - old_value_RV

        if abs(dLV) > 1e-12 or abs(dRV) > 1e-12 or abs(dTa) > 1e-12:
            # Only solve if there is a change in the volumedLB
            lv_volume.value = new_value_LV
            rv_volume.value = new_value_RV
            # First step can be a bit trick since Ta is non-zero, so we need to ramp it up
            if np.isclose(t, 0.0):
                for next_value in np.linspace(old_Ta, value, 10):
                    Ta.assign(next_value)
                    problem.solve()
            else:
                Ta.assign(value)
                problem.solve()

        lv_pendo_mmHg = circulation.units.kPa_to_mmHg(
            problem.cavity_pressures[0].x.array[0] * 1e-3,
        )
        rv_pendo_mmHg = circulation.units.kPa_to_mmHg(
            problem.cavity_pressures[1].x.array[0] * 1e-3,
        )
        print(f"Compute pressures: {lv_pendo_mmHg} mmHg (LV) {rv_pendo_mmHg} mmHg (RV)")
        return lv_pendo_mmHg, rv_pendo_mmHg

    mL = circulation.units.ureg("mL")
    mmHg = circulation.units.ureg("mmHg")
    s = circulation.units.ureg("s")
    add_units = False
    lvv_init = (
        geo.mesh.comm.allreduce(geometry.volume("LV", u=problem.u), op=MPI.SUM)
        * 1e6
        * 1.0
    )
    rvv_init = (
        geo.mesh.comm.allreduce(geometry.volume("RV", u=problem.u), op=MPI.SUM)
        * 1e6
        * 1.0
    )
    print(f"Initial volume (LV): {lvv_init} mL and (RV): {rvv_init} mL")
    init_state = {
        "V_LV": lvv_initial * 1e6 * mL,
        "V_RV": rvv_initial * 1e6 * mL,
        "p_AR_PUL": p_AR_PUL * mmHg,
        "p_AR_SYS": p_AR_SYS * mmHg,
        "p_VEN_PUL": p_VEN_PUL * mmHg,
        "p_VEN_SYS": p_VEN_SYS * mmHg,
    }

    regazzoni_parmeters = circulation.regazzoni2020.Regazzoni2020.default_parameters()
    regazzoni_parmeters["HR"] = HR
    regazzoni_parmeters["circulation"]["SYS"]["R_AR"] *= R_TPR_factor
    regazzoni_parmeters["circulation"]["SYS"]["R_VEN"] *= R_TPR_factor
    for chamber in ["LA", "RA"]:
        regazzoni_parmeters["chambers"][chamber]["EA"] *= C_PRSW_factor
        regazzoni_parmeters["chambers"][chamber]["EB"] *= C_PRSW_factor

    # Adjust time constant based on HR
    regazzoni_parmeters["chambers"]["LA"]["TC"] = 0.17 / HR * s
    regazzoni_parmeters["chambers"]["LA"]["TR"] = 0.17 / HR * s
    regazzoni_parmeters["chambers"]["LA"]["tC"] = 0.8 / HR * s
    regazzoni_parmeters["chambers"]["RA"]["TC"] = 0.17 / HR * s
    regazzoni_parmeters["chambers"]["RA"]["TR"] = 0.17 / HR * s
    regazzoni_parmeters["chambers"]["RA"]["tC"] = 0.8 / HR * s

    regazzoni = circulation.regazzoni2020.Regazzoni2020(
        add_units=add_units,
        callback=callback,
        p_BiV=p_BiV_func,
        verbose=True,
        comm=comm,
        outdir=outdir,
        initial_state=init_state,
        parameters=regazzoni_parmeters,
    )
    # Set end time for early stopping if running in CI
    end_time = 2 * dt if os.getenv("CI") else None
    regazzoni.solve(
        num_beats=num_beats, initial_state=init_state, dt=dt, T=end_time,
    )  # , checkpoint=RR)
    regazzoni.print_info()


circulation.log.setup_logging(logging.INFO)
log.set_log_level(log.LogLevel.INFO)
logger = logging.getLogger("pulse")
comm = MPI.COMM_WORLD

geodir = Path("ukb")
if not geodir.exists():
    comm.barrier()
    cardiac_geometries.mesh.ukb(
        outdir=geodir,
        comm=comm,
        mode=-1,
        case="ED",
        char_length_max=10.0,
        char_length_min=10.0,
        fiber_angle_endo=60,
        fiber_angle_epi=-60,
        fiber_space="DG_0",
        clipped=True,
    )

outdir = Path("bleeding_biv")
outdir.mkdir(exist_ok=True)

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)
scale_mesh = True

if scale_mesh:
    geo.mesh.geometry.x[:] *= 1e-3
    volume2ml = 1e6
    mesh_unit = "m"
else:
    volume2ml = 1e-3
    mesh_unit = "mm"


dt = 0.001
zenker_history = run_zenker(outdir=outdir)
HR_before = zenker_history["HR_before"]
R_TPR_factor = zenker_history["R_TPR_factor"]
C_PRSW_factor = zenker_history["C_PRSW_factor"]
HR_after = zenker_history["HR_after"]
Pa_before = zenker_history["Pa"][zenker_history["before_index"]]
Pcvp_before = zenker_history["Pcvp"][zenker_history["before_index"]]

Pa_after = zenker_history["Pa"][zenker_history["after_index"]]
Pcvp_after = zenker_history["Pcvp"][zenker_history["after_index"]]
print(f"Pa: {Pa_before} mmHg, Pcvp: {Pcvp_before} mmHg")
print(f"Pa: {Pa_after} mmHg, Pcvp: {Pcvp_after} mmHg")
print(f"HR before: {HR_before}, HR after: {HR_after}")

get_activation_before = run_TorOrdLand(
    comm,
    outdir,
    HR_before,
    Ta_factor=1,
    label="before",
    dt=dt,
)
get_activation_after = run_TorOrdLand(
    comm,
    outdir,
    HR_after,
    Ta_factor=C_PRSW_factor,
    label="after",
    dt=dt,
)

run_3D_model(
    comm=comm,
    geo=geo,
    get_activation=get_activation_before,
    outdir=outdir,
    label="before",
    num_beats=5,
    dt=dt,
    HR=HR_before,
    p_AR_SYS=Pa_before,
    p_AR_PUL=Pa_before * 0.4375,
    p_VEN_SYS=Pcvp_before,
    p_VEN_PUL=Pcvp_before * 0.8,
    mesh_unit=mesh_unit,
    volume2ml=volume2ml,
)

run_3D_model(
    comm=comm,
    geo=geo,
    get_activation=get_activation_after,
    outdir=outdir,
    label="after",
    num_beats=5,
    dt=dt,
    HR=HR_after,
    R_TPR_factor=R_TPR_factor,
    C_PRSW_factor=C_PRSW_factor,
    p_AR_SYS=Pa_after,
    p_AR_PUL=Pa_after * 0.4375,
    p_VEN_SYS=Pcvp_after,
    p_VEN_PUL=Pcvp_after * 0.8,
    mesh_unit=mesh_unit,
    volume2ml=volume2ml,
)

# Below we plot the pressure volume loop, volumes, pressures and flows for 50 beats

# ```{figure} ../_static/bleeding_biv_pv_loop_normal.png
# ---
# name: bleeding_biv_pv_loop_normal
# ---
# Pressure volume loop for the normal BiV.
# ```
#
# ```{figure} ../_static/bleeding_biv_pv_loop_bleeding.png
# ---
# name: bleeding_biv_pv_loop_bleeding
# ---
# Pressure volume loop for the bleeding BiV.
# ```
#

#

# # References
# ```{bibliography}
# :filter: docname in docnames
# ```
