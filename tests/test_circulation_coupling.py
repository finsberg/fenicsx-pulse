"""The 0D circulation states solved alongside the displacement.

Two things have to hold for this coupling to mean anything.

The circuit rows the problem assembles must be the circuit's own equations, not
something scaled or mistranslated on the way in. A real-space test function is
constant, so integrating against it multiplies by the mesh volume, and the
circuit is written in milliliters and millimeters of mercury while the mechanics
is in SI. Any of that going wrong produces a residual that still looks
plausible, so the rows are checked against the same model generated as plain
numpy from the same `.ode` file.

And Newton has to see the coupling. The cavity constraint ties the deformed
cavity volume to a circuit state, and the circuit rows depend on the cavity
pressure; if either cross term is missing from the Jacobian, the solver is
doing a partitioned iteration wearing a monolithic coat. Those blocks are
checked against finite differences.
"""

from __future__ import annotations

import math
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import pulse
from pulse.circulation import ChamberCoupling, GotranxCirculation, mL, mmHg

gotranx = pytest.importorskip("gotranx", reason="gotranx is needed to read .ode files")
circulation = pytest.importorskip("circulation", reason="the .ode file ships with circulation")
cardiac_geometries = pytest.importorskip("cardiac_geometries")


DROP = ("timing", "LV")


@pytest.fixture(scope="module")
def geo(tmp_path_factory):
    geodir = tmp_path_factory.mktemp("lv_circulation")
    comm = MPI.COMM_WORLD
    if not (geodir / "mesh.xdmf").exists():
        comm.barrier()
        cardiac_geometries.mesh.lv_ellipsoid(
            outdir=geodir,
            create_fibers=True,
            fiber_space="Quadrature_6",
            r_short_endo=0.025,
            r_short_epi=0.035,
            r_long_endo=0.09,
            r_long_epi=0.097,
            psize_ref=0.05,
            mu_apex_endo=-math.pi,
            mu_base_endo=-math.acos(5 / 17),
            mu_apex_epi=-math.pi,
            mu_base_epi=-math.acos(5 / 20),
            comm=comm,
            fiber_angle_epi=-60,
            fiber_angle_endo=60,
        )
    return cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)


@pytest.fixture(scope="module")
def flat_parameters():
    from circulation import regazzoni2020

    return regazzoni2020.flat_ode_parameters()


@pytest.fixture(scope="module")
def ode_file():
    from circulation import regazzoni2020

    return Path(regazzoni2020.ODE_FILE)


@pytest.fixture(scope="module")
def numpy_reference(ode_file, flat_parameters):
    """The same split model generated as numpy, to check the UFL rows against."""
    import gotranx.cli.gotran2py
    from gotranx.codegen.python import Format

    ode = gotranx.load_ode(ode_file)
    for name in DROP:
        ode = ode - ode.get_component(name)
    code = gotranx.cli.gotran2py.get_code(ode, format=Format.none)
    namespace: dict = {}
    exec(code, namespace)

    known = {}
    for name, value in flat_parameters.items():
        try:
            namespace["parameter_index"](name)
        except KeyError:
            continue
        known[name] = value
    namespace["_parameters"] = namespace["init_parameter_values"](**known)
    return namespace


@pytest.fixture(scope="module")
def problem(geo, ode_file, flat_parameters):
    geometry = pulse.HeartGeometry.from_cardiac_geometries(
        geo,
        metadata={"quadrature_degree": 4},
    )
    material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore[arg-type]
    Ta = pulse.Variable(dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
    model = pulse.CardiacModel(
        material=material,
        active=pulse.ActiveStress(geo.f0, activation=Ta),
        compressibility=pulse.Compressible(),
    )

    model_0d = GotranxCirculation(
        ode_file=ode_file,
        parameters=flat_parameters,
        drop_components=DROP,
    )
    beat_phase = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0))

    return pulse.problem.StaticProblem(
        model=model,
        geometry=geometry,
        cavities=[pulse.problem.Cavity(marker="ENDO", volume=None)],
        circulation=model_0d,
        chambers=[
            ChamberCoupling(marker="ENDO", volume_state="V_LV", pressure_missing="p_LV"),
        ],
        circulation_missing={"beat_phase": beat_phase},
        parameters={"base_bc": pulse.problem.BaseBC.fixed, "mesh_unit": "m"},
    )


def test_block_system_has_one_row_per_unknown(problem):
    """Every circuit state must add exactly one unknown and one equation."""
    n_circuit = len(problem.circulation.state_names)
    assert n_circuit == 12
    # displacement + one cavity pressure + one row per circuit state
    assert problem.num_states == 1 + 1 + n_circuit

    for attribute in ("states", "old_states", "test_functions", "trial_functions"):
        assert len(getattr(problem, attribute)) == problem.num_states

    R = problem.R
    assert len(R) == problem.num_states
    K = problem.K(R)
    assert all(len(row) == problem.num_states for row in K)


def test_cavity_volume_is_the_chamber_state(problem):
    """The constraint must point at the circuit's volume state, not a constant."""
    index = list(problem.circulation.state_names).index("V_LV")
    V_LV = problem.circulation_states[index]
    # The cavity volume is that state converted from milliliters to cubic metres,
    # so it depends on the state and cannot be a bare constant any more.
    assert V_LV in ufl.algorithms.extract_coefficients(problem.cavities[0].volume)


def test_circuit_rows_equal_the_ode_residual(problem, numpy_reference):
    """Each assembled circuit row must be that state's backward-Euler residual.

    This is what catches the mesh-volume factor a real-space test function
    introduces, a wrong measure, or a missed unit conversion: all of them leave
    a residual that is smooth, finite and wrong.
    """
    rng = np.random.default_rng(0)
    names = list(problem.circulation.state_names)
    offset = problem.num_states - len(names)

    y = np.asarray(problem.circulation.initial_states, dtype=float)
    y = y * rng.uniform(0.85, 1.15, size=y.size)
    y_old = np.asarray(problem.circulation.initial_states, dtype=float)

    dt, t, phase, p_cav_pa = 0.002, 0.31, 0.11, 1500.0

    for state, value in zip(problem.circulation_states, y):
        state.x.array[:] = value
    for state, value in zip(problem.circulation_states_old, y_old):
        state.x.array[:] = value
    problem.circulation_dt.value = dt
    problem.circulation_time.value = t
    problem.circulation_missing["beat_phase"].value = phase
    problem.cavity_pressures[0].x.array[:] = p_cav_pa

    # Reference: the same model as numpy, with the pressure converted the way
    # the coupling is supposed to convert it.
    missing = np.zeros(2)
    missing[numpy_reference["missing_index"]("beat_phase")] = phase
    missing[numpy_reference["missing_index"]("p_LV")] = p_cav_pa / mmHg
    f = np.asarray(
        numpy_reference["rhs"](t, y, numpy_reference["_parameters"], missing),
    )
    expected = (y - y_old) / dt - f

    R = problem.R
    for i, name in enumerate(names):
        assembled = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(R[offset + i]),
        )
        got = problem.geometry.mesh.comm.allreduce(assembled, op=MPI.SUM)
        scale = max(abs(expected[i]), 1.0)
        assert abs(got - expected[i]) / scale < 1e-9, (
            f"row for {name}: assembled {got:.6g}, expected {expected[i]:.6g}"
        )


def test_newton_sees_both_coupling_blocks(problem):
    """The cross terms between the cavity pressure and the circuit must exist.

    A Jacobian missing them would still converge, to the wrong thing: it would
    be a partitioned iteration in disguise. Both directions are checked, since
    the coupling is not symmetric -- the cavity row depends on the volume state,
    and the volume state's row depends on the cavity pressure.
    """
    names = list(problem.circulation.state_names)
    offset = problem.num_states - len(names)
    i_V_LV = offset + names.index("V_LV")
    i_p_cav = 1  # order is (u, cavity pressures, ..., circulation states)

    R = problem.R
    K = problem.K(R)

    def block(i, j):
        value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(K[i][j]))
        return problem.geometry.mesh.comm.allreduce(value, op=MPI.SUM)

    # cavity constraint row differentiated by the chamber volume state
    assert abs(block(i_p_cav, i_V_LV)) > 0.0
    # the chamber's own circuit row differentiated by the cavity pressure
    assert abs(block(i_V_LV, i_p_cav)) > 0.0


def test_coupling_block_matches_finite_differences(problem, numpy_reference):
    """The analytic coupling block must agree with a finite difference of it."""
    names = list(problem.circulation.state_names)
    offset = problem.num_states - len(names)
    i_p_cav = 1

    problem.circulation_dt.value = 0.002
    problem.circulation_time.value = 0.31
    problem.circulation_missing["beat_phase"].value = 0.11

    y = np.asarray(problem.circulation.initial_states, dtype=float)
    for state, value in zip(problem.circulation_states, y):
        state.x.array[:] = value
    for state, value in zip(problem.circulation_states_old, y):
        state.x.array[:] = value

    R = problem.R
    K = problem.K(R)
    comm = problem.geometry.mesh.comm

    def assemble(form):
        return comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)), op=MPI.SUM)

    p0 = 1500.0
    h = 1.0  # Pa; the rows are smooth in pressure, so this is a safe step

    for name in ("V_LV", "V_LA", "p_AR_SYS"):
        row = offset + names.index(name)

        problem.cavity_pressures[0].x.array[:] = p0
        analytic = assemble(K[row][i_p_cav])

        problem.cavity_pressures[0].x.array[:] = p0 + h
        plus = assemble(R[row])
        problem.cavity_pressures[0].x.array[:] = p0 - h
        minus = assemble(R[row])
        numeric = (plus - minus) / (2 * h)

        scale = max(abs(numeric), 1e-8)
        assert abs(analytic - numeric) / scale < 1e-6, (
            f"coupling block for {name}: analytic {analytic:.6g}, finite difference {numeric:.6g}"
        )


def test_circuit_states_advance_and_roll_back(problem):
    """Stepping must carry the circuit forward, and a failed step must undo it.

    `update_old_states` and `reset_states` list each kind of unknown by hand
    rather than walking the state lists, so a new kind of unknown has to be
    added to both. Miss the first and every step solves against the initial
    state; miss the second and a retried step keeps whatever the failed attempt
    left behind.
    """
    states = problem.circulation_states
    old = problem.circulation_states_old

    start = np.array([float(s.x.array[0]) for s in states])
    moved = start * 1.05
    for state, value in zip(states, moved):
        state.x.array[:] = value

    problem.update_old_states()
    assert np.allclose([float(s.x.array[0]) for s in old], moved)

    for state in states:
        state.x.array[:] = 0.0
    problem.reset_states()
    assert np.allclose([float(s.x.array[0]) for s in states], moved)


def test_monolithic_step_converges_and_holds_the_constraint(problem):
    """A few coupled steps must solve, and the constraint must hold at each one.

    The point of solving the circuit and the displacement together is that the
    cavity volume and the chamber's volume state are never allowed to disagree.
    Here that is checked directly: after each step the deformed cavity volume
    must equal the circuit's V_LV, to solver tolerance rather than to whatever a
    fixed number of exchange iterations happens to reach.

    The circuit is started from the mesh's own cavity volume rather than the
    published initial condition, so the constraint holds at t=0 without an
    inflation ramp. That moves the circuit off its own limit cycle, which is
    fine for a convergence check and is not how a real run should be
    initialized.
    """
    comm = problem.geometry.mesh.comm
    names = list(problem.circulation.state_names)
    i_V_LV = names.index("V_LV")

    problem.u.x.array[:] = 0.0
    problem.u_old.x.array[:] = 0.0
    problem.cavity_pressures[0].x.array[:] = 0.0
    problem.cavity_pressures_old[0].x.array[:] = 0.0

    y = np.asarray(problem.circulation.initial_states, dtype=float)
    reference_volume = comm.allreduce(problem.geometry.volume("ENDO"), op=MPI.SUM)
    y[i_V_LV] = reference_volume / mL
    for state, value in zip(problem.circulation_states, y):
        state.x.array[:] = value
    for state, value in zip(problem.circulation_states_old, y):
        state.x.array[:] = value

    dt = 0.001
    problem.circulation_dt.value = dt
    RR = 0.8

    for step in range(3):
        t = (step + 1) * dt
        problem.circulation_time.value = t
        problem.circulation_missing["beat_phase"].value = t % RR
        # A small activation ramp, so the step is not trivially the rest state.
        problem.model.active.activation.assign(2.0 * step)

        assert problem.solve(), f"monolithic solve failed at step {step}"

        volume = comm.allreduce(problem.geometry.volume("ENDO", u=problem.u), op=MPI.SUM)
        V_LV = float(problem.circulation_states[i_V_LV].x.array[0]) * mL
        assert abs(volume - V_LV) / V_LV < 1e-8, (
            f"step {step}: cavity volume {volume:.6e} m^3 does not match "
            f"the circuit state {V_LV:.6e} m^3"
        )


def test_incompressible_constraint_stays_on_its_own_row(geo, ode_file, flat_parameters):
    """The incompressibility row must not move when circuit states are added.

    The block order is (u, cavity pressures, rigid body, p, circulation), so
    `p` is the last row only when there is no circuit. Code that counted back
    from the end put the incompressibility constraint on the last circuit
    state's row instead, mixing test functions from two different spaces.
    """
    geometry = pulse.HeartGeometry.from_cardiac_geometries(
        geo,
        metadata={"quadrature_degree": 4},
    )
    material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    Ta = pulse.Variable(dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0)), "kPa")
    model = pulse.CardiacModel(
        material=pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params),  # type: ignore[arg-type]
        active=pulse.ActiveStress(geo.f0, activation=Ta),
        compressibility=pulse.Incompressible(),
    )

    problem = pulse.problem.StaticProblem(
        model=model,
        geometry=geometry,
        cavities=[pulse.problem.Cavity(marker="ENDO", volume=None)],
        circulation=GotranxCirculation(
            ode_file=ode_file,
            parameters=flat_parameters,
            drop_components=DROP,
        ),
        chambers=[
            ChamberCoupling(marker="ENDO", volume_state="V_LV", pressure_missing="p_LV"),
        ],
        circulation_missing={
            "beat_phase": dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0)),
        },
        parameters={"mesh_unit": "m"},
    )

    # u, cavity pressure, p, then the circuit
    assert problem.num_states == 3 + len(problem.circulation.state_names)
    assert problem.incompressibility_index == 2
    assert problem.states[problem.incompressibility_index] is problem.p

    # Every row must carry exactly one test function, from its own space.
    for i, form in enumerate(problem.R):
        arguments = ufl.algorithms.analysis.extract_arguments(form)
        assert len({id(a) for a in arguments}) <= 1, f"row {i} mixes test functions"
