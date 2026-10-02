"""Wire the builders together, run the (quasi-)static or dynamic time loop, write results and
restarts.

Output folder layout (``conf.output.folder``)::

    config.resolved.toml   the fully resolved configuration of the (latest) run
    run.json               run metadata (versions, ranks, status: running/finished/failed)
    results.bp             io4dolfinx: ``u`` (+ ``p`` if incompressible) every output.save_every
    loads.csv              t [s], every load [Pa], volume_<marker> [m^3] for cavity markers
    restart.bp             io4dolfinx: the ``mechanics_*`` state functions
    restart.json           {"mechanics": {t, step, physics_hash, functions}} of the latest
                           checkpoint
    output.log             log file (``output_all_cpus.log`` too when running on >1 rank)

io4dolfinx *appends* a function written at an already existing timestamp, and ``read_function``
returns the *first* match. So a restarted run never writes a ``results.bp`` timestamp ``<=`` the
last one already there, and readers deduplicate with ``np.unique`` (:func:`read_result_times`).
"""

import csv
import datetime
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from mpi4py import MPI

import dolfinx
import io4dolfinx
import numpy as np

import pulse

from ..telemetry import NullMonitor, PerformanceMonitor
from ..units import mesh_factor
from .bcs import build_bcs
from .config import CAVITY_MARKERS, Config, ConfigError, si
from .geometry import CLIGeometry, build_geometry, check_markers
from .loads import LoadSet, build_loads, make_load_variables, require_optional_packages
from .log import add_logfile_handler, remove_logfile_handlers
from .model import build_model
from .overrides import dump_config, physics_hash

logger = logging.getLogger(__name__)

RESULTS = "results.bp"
RESTART = "restart.bp"
RESTART_META = "restart.json"
RUN_META = "run.json"
LOADS = "loads.csv"
PERFORMANCE = "performance.json"
NAMESPACE = "mechanics"

# Everything pulse writes into an output folder. --overwrite deletes exactly these and nothing
# else, since the output folder may be the config's own directory.
_ARTIFACT_NAMES = (
    RESULTS,
    RESTART,
    RESTART_META,
    RUN_META,
    LOADS,
    PERFORMANCE,
    "config.resolved.toml",
    "output.log",
    "output_all_cpus.log",
    "post",
)


class SolverFailure(RuntimeError):
    """The simulation failed at runtime (exit code 2)."""


def _on_rank0(comm, error_type: type[Exception], fn: Callable[[], Any]) -> None:
    """Run ``fn`` on rank 0 only, and make any failure raise on *every* rank.

    Rank-0-only filesystem work followed by a barrier/collective otherwise deadlocks when rank 0
    raises. A :class:`ConfigError` stays a ConfigError; anything else is raised as
    ``error_type`` with the same message (chained to the original on rank 0).
    """
    original: Exception | None = None
    error: tuple[bool, str] | None = None
    if comm.rank == 0:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - re-raised on every rank below
            original = e
            error = (isinstance(e, ConfigError), str(e) or repr(e))
    error = comm.bcast(error, root=0)
    if error is None:
        return
    is_config, message = error
    exc: Exception = ConfigError(message) if is_config else error_type(message)
    raise exc from original


def _write_json(path: Path, data: dict, comm) -> None:
    def write() -> None:
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)

    _on_rank0(comm, OSError, write)


def _remove_artifacts(folder: Path) -> None:
    paths = [folder / name for name in _ARTIFACT_NAMES]
    for p in paths:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        elif p.exists() or p.is_symlink():
            p.unlink()


def read_result_times(path: Path, comm, name: str = "u") -> np.ndarray:
    """Sorted, unique timestamps of ``name`` in the io4dolfinx file ``path``."""
    return np.unique(io4dolfinx.read_timestamps(filename=path, comm=comm, function_name=name))


def _output_decision(conf: Config, restart: bool, overwrite: bool) -> tuple[str, str]:
    """Decide (on one rank, from the filesystem) what :func:`prepare_output` must do.

    Returns ``("error", message)``, ``("restart", "")``, ``("wipe", "")`` or ``("create", "")``.
    """
    folder = conf.output.folder
    no_checkpoint = (
        f"no restart checkpoint exists yet in {folder} (no {RESTART_META}; the run stopped "
        "before its first checkpoint); use --overwrite to start over"
    )
    if restart:
        if not (folder / RESTART_META).is_file():
            return "error", f"Cannot restart: {no_checkpoint}"
        meta = json.loads((folder / RESTART_META).read_text()).get(NAMESPACE, {})
        try:
            current = physics_hash(conf)
        except ConfigError as e:
            return "error", str(e)
        if meta.get("physics_hash") != current:
            return "error", (
                "Cannot restart: the physics settings differ from the original run "
                "(only time.end_time/num_steps, [output], [postprocess] and [solver] may change). "
                f"Compare with {folder / 'config.resolved.toml'}"
            )
        return "restart", ""
    if (folder / RESULTS).exists() or (folder / RESTART_META).exists():
        if not overwrite:
            if not (folder / RESTART_META).exists():
                return (
                    "error",
                    f"Output folder {folder} already contains results, but {no_checkpoint}",
                )
            return "error", (
                f"Output folder {folder} already contains results. Use --overwrite to replace "
                "them or --restart to continue the run."
            )
        return "wipe", ""
    return "create", ""


def decide_output(conf: Config, restart: bool, overwrite: bool, comm) -> str:
    """Decide what to do with ``conf.output.folder`` without touching it.

    The decision is made on rank 0 only and broadcast, so that every rank raises the same
    :class:`ConfigError` together (never one rank raising while the others wait in a barrier).
    Returns ``"restart"``, ``"wipe"`` or ``"create"`` for :func:`apply_output`.
    """
    decision = None
    if comm.rank == 0:
        try:
            decision = _output_decision(conf, restart, overwrite)
        except Exception as e:  # e.g. a corrupt restart.json: must not raise on rank 0 alone
            decision = ("error", f"Cannot prepare output folder {conf.output.folder}: {e!r}")
    action, message = comm.bcast(decision, root=0)
    if action == "error":
        raise ConfigError(message)
    return action


def apply_output(conf: Config, action: str, comm) -> None:
    """Carry out :func:`decide_output`'s ``action``: create the folder, and for ``"wipe"``
    delete only pulse's own artifacts (see ``_ARTIFACT_NAMES``), never other files."""
    if action == "restart":
        return
    folder = conf.output.folder

    def prepare() -> None:
        if action == "wipe":
            _remove_artifacts(folder)
        folder.mkdir(parents=True, exist_ok=True)

    try:
        _on_rank0(comm, ConfigError, prepare)
    except ConfigError as e:
        raise ConfigError(f"Cannot prepare output folder {folder}: {e}") from e


def prepare_output(conf: Config, restart: bool, overwrite: bool, comm) -> None:
    """Validate/prepare ``conf.output.folder`` for a fresh run, an overwrite or a restart."""
    apply_output(conf, decide_output(conf, restart, overwrite, comm), comm)


def _write_csv_header(path: Path, fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(fields)


def _append_row(path: Path, fields: list[str], row: dict[str, float]) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([repr(float(row[k])) for k in fields])


def _truncate_csv(path: Path, t_max: float, fields: list[str]) -> float:
    """Keep loads.csv rows with t <= t_max (rows written after the restart checkpoint go).

    A missing, empty or header-less file is rewritten with just the header. Returns the largest
    ``t`` kept (``-inf`` if no data row is left).
    """
    rows: list[list[str]] = []
    if path.is_file():
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
    if not rows or rows[0] != fields:
        _write_csv_header(path, fields)
        return -np.inf
    kept = [r for r in rows[1:] if r and float(r[0]) <= t_max]
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w", newline="") as f:
        csv.writer(f).writerows([rows[0], *kept])
    os.replace(tmp, path)
    return max((float(r[0]) for r in kept), default=-np.inf)


def required_markers(conf: Config) -> dict[str, list[str]]:
    """Facet markers each config section needs, for a check against the mesh at build time."""
    out = {
        "load": [load.marker for load in conf.load if load.marker is not None],
        "bcs.robin": [r.marker for r in conf.bcs.robin],
        "bcs.dirichlet": [d.marker for d in conf.bcs.dirichlet],
    }
    if conf.bcs.base_bc == "fixed":
        out["bcs.base_marker"] = [conf.bcs.base_marker]
    return out


def check_config_markers(conf: Config, geo: CLIGeometry) -> None:
    """Check every marker the config names against the mesh (ConfigError on a mismatch).

    Load and boundary-condition markers must exist and be facet markers (dim = tdim - 1);
    ``postprocess.vertex_tags`` must name vertex markers (dim 0). Run before anything is wiped.
    """
    facet_dim = geo.mesh.topology.dim - 1
    for what, names in required_markers(conf).items():
        check_markers(geo, names, what)
        for name in names:
            dim = geo.markers[name][1]
            if dim != facet_dim:
                raise ConfigError(
                    f"{what}: marker {name!r} has dimension {dim}, but must be a facet marker "
                    f"(dimension {facet_dim})",
                )
    for tag, name in conf.postprocess.vertex_tags.items():
        what = f"postprocess.vertex_tags.{tag}"
        check_markers(geo, [name], what)
        dim = geo.markers[name][1]
        if dim != 0 or geo.vfun is None:
            raise ConfigError(
                f"{what}: marker {name!r} is not a vertex marker (dimension {dim}, needs 0 and "
                "vertex tags in the geometry)",
            )


def build_problem(
    conf: Config,
    geo: CLIGeometry,
    model: Any,
    bcs: Any,
    monitor: Any = None,
) -> tuple[Any, Any]:
    """Return ``(problem, dt_constant)``; ``dt_constant`` is None for a static problem."""
    defaults = pulse.StaticProblem.default_parameters()
    parameters: dict[str, Any] = {
        "u_space": conf.problem.u_space,
        "p_space": conf.problem.p_space,
        "rigid_body_constraint": conf.problem.rigid_body_constraint,
        "mesh_unit": conf.geometry.unit,
        "base_bc": pulse.BaseBC(conf.bcs.base_bc),
        "base_marker": conf.bcs.base_marker,
        "petsc_options": {**defaults["petsc_options"], **conf.solver.petsc_options},
    }
    kwargs = dict(
        model=model,
        geometry=geo.geometry,
        bcs=bcs,
        monitor=monitor if monitor is not None else NullMonitor(),
    )
    if conf.problem.type == "static":
        return pulse.StaticProblem(parameters=parameters, **kwargs), None
    dt_constant = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(conf.time.dt_s()))
    parameters.update(
        {
            "dt": pulse.Variable(dt_constant, "s"),
            "rho": pulse.Variable(si(conf.problem.rho), "kg/m**3"),
            "alpha_m": conf.problem.alpha_m,
            "alpha_f": conf.problem.alpha_f,
        },
    )
    return pulse.DynamicProblem(parameters=parameters, **kwargs), dt_constant


@dataclass
class MechanicsSimulation:
    """The step API: build once, then ``step(dt)`` / ``save()`` / ``checkpoint()`` / ``restore()``.

    ``run()`` is only a loop around this; a coupled driver (simcardemsx) calls it directly.
    """

    conf: Config
    geo: CLIGeometry
    problem: Any
    loads: LoadSet
    t: float
    step_index: int = 0
    dt_constant: Any = None
    monitor: Any = field(default_factory=NullMonitor)
    _last_saved: float = field(default=-np.inf, repr=False)
    _last_row: float = field(default=-np.inf, repr=False)
    _checkpoints: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    @property
    def folder(self) -> Path:
        return self.conf.output.folder

    @property
    def comm(self):
        return self.geo.mesh.comm

    def _tol(self) -> float:
        return 1e-9 * self.conf.time.dt_s()

    def cavity_markers(self) -> list[str]:
        return [m for m in self.loads.pressure_markers if m in CAVITY_MARKERS]

    def csv_fields(self) -> list[str]:
        return ["t", *self.loads.names, *(f"volume_{m}" for m in self.cavity_markers())]

    def volumes(self) -> dict[str, float]:
        """Cavity volumes in m^3 (summed over ranks: HeartGeometry.volume is rank-local)."""
        factor = mesh_factor(self.conf.geometry.unit) ** 3
        out = {}
        for marker in self.cavity_markers():
            local = self.geo.geometry.volume(marker, u=self.problem.u)
            out[f"volume_{marker}"] = self.comm.allreduce(local, op=MPI.SUM) * factor
        return out

    def record(self, t: float) -> dict[str, float]:
        with self.monitor.track_time("volumes"):
            volumes = self.volumes()
        return {"t": t, **self.loads.values(t), **volumes}

    def start(self) -> None:
        """Fresh run: create the output folder, write loads.csv's header and set the loads to the
        start time."""
        fields = self.csv_fields()
        folder = self.folder

        def write() -> None:
            folder.mkdir(parents=True, exist_ok=True)
            _write_csv_header(folder / LOADS, fields)

        _on_rank0(self.comm, OSError, write)
        self.loads.update(self.t)

    def step(self, dt: float) -> None:
        """Advance ``t -> t + dt``, halving the step on Newton failure (``solver.max_halvings``).

        Raises :class:`SolverFailure` when the deepest halving fails; the problem (states, old
        states, loads and the ``dt`` Constant) is then back at ``t``, as it was before this call,
        even when some halves had already converged, and ``t``/``step_index`` are unchanged. Any
        other exception raised mid-step (e.g. a PETSc error) rolls back the same way and
        propagates unchanged.
        """
        t0 = self.t
        with self.monitor.track_time("step"):
            snapshot = [(f, f.x.array.copy()) for f in self._state_functions()]
            try:
                self._advance(t0, dt, 0)
            except Exception as e:
                for f, values in snapshot:
                    f.x.array[:] = values
                self.loads.update(t0)
                if isinstance(e, SolverFailure):
                    raise SolverFailure(f"Step to t={t0 + dt:.6g} s failed: {e}") from e
                raise
            finally:
                if self.dt_constant is not None:
                    self.dt_constant.value = dt
        self.t = t0 + dt
        self.step_index += 1
        self.monitor.advance_step(t0, self.t)

    def _state_functions(self) -> list[Any]:
        """Every Function a (partly) converged step changes: states, old states, and for a
        dynamic problem the velocity/acceleration history."""
        functions = [*self.problem.states, *self.problem.old_states]
        functions += [getattr(self.problem, name, None) for name in ("v_old", "a_old")]
        unique = {id(f): f for f in functions if f is not None}
        return list(unique.values())

    def _advance(self, t: float, dt: float, level: int) -> None:
        if self.dt_constant is not None:
            self.dt_constant.value = dt
        with self.monitor.track_time("loads"):
            self.loads.update(t + dt)
        if self.problem.solve():
            return
        self.problem.reset_states()
        self.loads.update(t)
        max_level = self.conf.solver.max_halvings
        if level >= max_level:
            raise SolverFailure(
                f"Newton did not converge at t={t + dt:.6g} s (step {dt:.3g} s after {level} "
                f"halving(s); solver.max_halvings = {max_level})",
            )
        self.monitor.count("halvings")
        logger.warning(
            f"Newton did not converge at t={t + dt:.6g} s; halving the step to {dt / 2:.3g} s "
            f"({level + 1}/{max_level})",
        )
        self._advance(t, dt / 2, level + 1)
        self._advance(t + dt / 2, dt / 2, level + 1)

    def save(self) -> None:
        """Write ``u`` (+ ``p``) to results.bp and a loads.csv row, each only if new."""
        with self.monitor.track_time("save"):
            self._save()

    def _save(self) -> None:
        t = self.t
        tol = self._tol()
        if t > self._last_saved + tol:
            path = self.folder / RESULTS
            io4dolfinx.write_function_on_input_mesh(path, self.problem.u, time=t, name="u")
            if self.problem.is_incompressible:
                io4dolfinx.write_function_on_input_mesh(path, self.problem.p, time=t, name="p")
            self._last_saved = t
        if t > self._last_row + tol:
            row = self.record(t)
            fields = self.csv_fields()
            _on_rank0(self.comm, OSError, lambda: _append_row(self.folder / LOADS, fields, row))
            self._last_row = t

    def restart_functions(self) -> list[tuple[str, dolfinx.fem.Function]]:
        """The state a restart needs, under ``mechanics_*`` names (composable with beat's)."""
        p = self.problem
        out = [("mechanics_u", p.u), ("mechanics_u_old", p.u_old)]
        if p.is_incompressible:
            out += [("mechanics_p", p.p), ("mechanics_p_old", p.p_old)]
        if isinstance(p, pulse.DynamicProblem):
            out += [("mechanics_v_old", p.v_old), ("mechanics_a_old", p.a_old)]
        return out

    def checkpoint(self) -> None:
        """Write the state at ``t`` to restart.bp and point restart.json at it.

        restart.bp already holding ``t`` (a run killed between the two writes, restarted from an
        earlier checkpoint) only updates restart.json: io4dolfinx would append a duplicate and
        read the first one anyway, which is the same state.
        """
        with self.monitor.track_time("checkpoint"):
            self._checkpoint()

    def _checkpoint(self) -> None:
        t = self.t
        functions = self.restart_functions()
        exists = bool(np.any(np.abs(self._checkpoints - t) < self._tol()))
        if not exists:
            for name, f in functions:
                io4dolfinx.write_function_on_input_mesh(
                    self.folder / RESTART,
                    f,
                    time=t,
                    name=name,
                )
        meta = {
            NAMESPACE: {
                "t": t,
                "step": self.step_index,
                "physics_hash": physics_hash(self.conf),
                "functions": [name for name, _ in functions],
            },
        }
        # Written last (and atomically): restart.json only ever names a complete checkpoint.
        _write_json(self.folder / RESTART_META, meta, self.comm)
        self._checkpoints = np.append(self._checkpoints, t)

    def _stored_times(self, name: str) -> np.ndarray:
        return np.asarray(
            io4dolfinx.read_timestamps(
                filename=self.folder / RESTART,
                comm=self.comm,
                function_name=name,
            ),
            dtype=float,
        )

    def restore(self) -> None:
        """Load the latest checkpoint (restart.json) into the problem and set ``t``/``step``."""
        folder = self.folder
        meta = json.loads((folder / RESTART_META).read_text())[NAMESPACE]
        functions = self.restart_functions()
        names = [name for name, _ in functions]
        if meta["functions"] != names:
            raise ConfigError(
                f"Cannot restart: the checkpoint holds {meta['functions']}, this problem "
                f"needs {names}",
            )
        t = float(meta["t"])
        tol = self._tol()
        stored = self._stored_times(names[0])
        if stored.size == 0 or np.min(np.abs(stored - t)) >= tol:
            raise ConfigError(
                f"Cannot restart: {folder / RESTART} has no checkpoint at t={t} s named by "
                f"{folder / RESTART_META} (stored: {sorted(set(stored.tolist()))})",
            )
        t_file = float(stored[np.argmin(np.abs(stored - t))])
        for name, f in functions:
            io4dolfinx.read_function(folder / RESTART, f, time=t_file, name=name)
            f.x.scatter_forward()
        # Only checkpoints whose every function is present may be reused instead of rewritten.
        complete = stored
        for name in names[1:]:
            times = self._stored_times(name)
            complete = complete[[bool(np.any(np.abs(times - c) < tol)) for c in complete]]
        self._checkpoints = complete
        self.t = t
        self.step_index = int(meta["step"])
        if (folder / RESULTS).exists():
            self._last_saved = float(read_result_times(folder / RESULTS, self.comm).max())
        fields = self.csv_fields()
        last_row = [-np.inf]

        def truncate() -> None:
            last_row[0] = _truncate_csv(folder / LOADS, t + tol, fields)

        _on_rank0(self.comm, OSError, truncate)
        # The latest row actually kept, not t: a checkpoint off the old save grid has no row yet.
        self._last_row = float(self.comm.bcast(last_row[0], root=0))
        self.loads.update(t)
        if self.dt_constant is not None:
            self.dt_constant.value = self.conf.time.dt_s()
        logger.info(f"Restarting from t={t} s (step {self.step_index})")


def build_simulation(
    conf: Config,
    comm=MPI.COMM_WORLD,
    *,
    geometry: CLIGeometry | None = None,
    active_model: Any = None,
    monitor: Any = None,
) -> MechanicsSimulation:
    """Build geometry, model, BCs, loads and problem. ``geometry``/``active_model``/``monitor``
    may be injected (simcardemsx); an injected active model replaces ``[active]``."""
    monitor = monitor if monitor is not None else NullMonitor()
    require_optional_packages(conf.load)
    geo = geometry if geometry is not None else build_geometry(conf.geometry, comm)
    check_config_markers(conf, geo)
    variables = make_load_variables(conf.load, geo.mesh)
    activation = variables.get("activation")
    if activation is not None:
        if active_model is not None:
            raise ConfigError(
                "An activation load cannot drive an injected active model; drop the "
                "[[load]] with target = 'activation' and set Ta from the driver instead",
            )
        if conf.active.type == "passive":
            raise ConfigError(
                "An activation load needs [active] type = 'active_stress' (it is 'passive')",
            )
    model = build_model(conf, geo, activation=activation, active_model=active_model)
    pressures = {load.marker: variables[load.name] for load in conf.load if load.marker}
    bcs = build_bcs(conf.bcs, geo, pressures)
    problem, dt_constant = build_problem(conf, geo, model, bcs, monitor=monitor)
    loads = build_loads(conf.load, variables, conf.time)
    return MechanicsSimulation(
        conf=conf,
        geo=geo,
        problem=problem,
        loads=loads,
        t=conf.time.start_s(),
        dt_constant=dt_constant,
        monitor=monitor,
    )


def run(
    conf: Config,
    *,
    restart: bool = False,
    overwrite: bool = False,
    comm=MPI.COMM_WORLD,
) -> Path:
    """Run the simulation described by ``conf``; return the output folder.

    Raises :class:`ConfigError` (exit code 1) for configuration problems and
    :class:`SolverFailure` (exit code 2) for anything failing at runtime. Everything is built
    -- i.e. the config fully validated against the mesh -- before ``--overwrite`` deletes
    anything.
    """
    action = decide_output(conf, restart=restart, overwrite=overwrite, comm=comm)
    folder = conf.output.folder
    monitor = (
        PerformanceMonitor(log_frequency=conf.output.log_every, comm=comm)
        if conf.output.performance
        else None
    )
    try:
        sim = build_simulation(conf, comm, monitor=monitor)
    except (ConfigError, SolverFailure):
        raise
    except Exception as e:
        raise SolverFailure(f"Setting up the simulation failed: {e!r}") from e
    apply_output(conf, action, comm)
    add_logfile_handler(folder, comm=comm)
    try:
        return _run(sim, comm, restart)
    finally:
        remove_logfile_handlers()


def _run(sim: MechanicsSimulation, comm, restart: bool) -> Path:
    conf = sim.conf
    folder = conf.output.folder
    _on_rank0(comm, OSError, lambda: dump_config(conf, folder / "config.resolved.toml"))
    record: dict[str, Any] = {
        "pulse": pulse.__version__,
        "dolfinx": dolfinx.__version__,
        "n_ranks": comm.size,
        "start": datetime.datetime.now().isoformat(),
        "restart": restart,
        "status": "running",
    }
    _write_json(folder / RUN_META, record, comm)
    try:
        _time_loop(sim, restart)
    except Exception as e:
        record["status"] = "failed"
        record["error"] = str(e) if isinstance(e, ConfigError) else repr(e)
        _write_json(folder / RUN_META, record, comm)
        if isinstance(e, (ConfigError, SolverFailure)):
            raise
        raise SolverFailure(str(e)) from e
    finally:
        # Also on failure: the timings up to a solver failure are what one wants to look at.
        if isinstance(sim.monitor, PerformanceMonitor):
            sim.monitor.display_summary()
            _on_rank0(comm, OSError, lambda: sim.monitor.save_summary(folder / PERFORMANCE))
    record["status"] = "finished"
    record["end"] = datetime.datetime.now().isoformat()
    _write_json(folder / RUN_META, record, comm)
    logger.info(f"Simulation finished. Results in {folder / RESULTS}")
    return folder


def _time_loop(sim: MechanicsSimulation, restart: bool) -> None:
    conf = sim.conf
    dt = conf.time.dt_s()
    n_steps = conf.time.n_steps()
    save_stride = conf.output.save_stride(dt)
    ckpt_stride = conf.output.checkpoint_stride(dt)
    if restart:
        sim.restore()
        if sim.step_index >= n_steps:
            logger.info("Nothing to do: the restart point is at or beyond the end time")
            return
    else:
        sim.start()
    step0 = sim.step_index
    tic = perf_counter()
    while sim.step_index < n_steps:
        if sim.step_index % save_stride == 0:
            sim.save()
        sim.step(dt)
        if ckpt_stride and sim.step_index % ckpt_stride == 0 and sim.step_index < n_steps:
            sim.checkpoint()
        if sim.step_index % conf.output.log_every == 0:
            rate = (sim.step_index - step0) / (perf_counter() - tic)
            logger.info(
                f"t={sim.t:.6g} s  step {sim.step_index}/{n_steps}  {rate:.2f} steps/s  "
                f"ETA {(n_steps - sim.step_index) / rate:.0f} s",
            )
    sim.save()
    sim.checkpoint()
