"""``[prestress]``: recover the unloaded reference configuration of an imaged (loaded) mesh.

The imaged mesh is loaded by the cavity pressures it was imaged at. `PrestressProblem` solves
the inverse elasticity problem for the displacement ``u_pre`` from the unloaded to the imaged
configuration; the mesh is then deformed by it (moved to the unloaded configuration) and the
fibres are mapped along. Results are cached in ``prestress.cache_folder/<hash16>/``, keyed on
everything that changes them, installed by an atomic rename like the geometry cache, and never
deleted by ``--overwrite``.
"""

import dataclasses
import gc
import hashlib
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import io4dolfinx
import numpy as np

import pulse

from ..circulation import mL
from ..units import mesh_factor
from .bcs import _dirichlet, build_bcs
from .config import Config, ConfigError, DirichletConfig
from .geometry import CLIGeometry
from .model import build_compressibility, build_material

logger = logging.getLogger(__name__)

U_PRE = "u_pre.bp"
META = "meta.json"


@dataclass
class PrestressResult:
    u_pre: dolfinx.fem.Function  # unloaded -> imaged, on the imaged mesh
    targets: dict[str, float]  # marker -> Pa
    imaged_volumes: dict[str, float]  # marker -> m^3
    hash: str


def prestress_targets(conf: Config) -> dict[str, float]:
    """Target cavity pressures (Pa): each cycle cavity's p_end_diastole, else the targets."""
    assert conf.prestress is not None
    if conf.circulation.type == "cycle":
        return {
            c.marker: float(c.p_end_diastole.to("Pa").magnitude) for c in conf.circulation.cavity
        }
    return {t.marker: float(t.pressure.to("Pa").magnitude) for t in conf.prestress.target}


def prestress_hash(conf: Config) -> str:
    """Hash of everything that changes the unloaded configuration."""
    assert conf.prestress is not None
    geometry = conf.geometry.model_dump(mode="json")
    if geometry.get("type") != "folder":
        geometry.pop("folder", None)
    data = {
        "geometry": geometry,
        **conf.model_dump(mode="json", include={"material", "compressibility", "bcs"}),
        "targets": prestress_targets(conf),
        "ramp_steps": conf.prestress.ramp_steps,
        "spaces": [conf.problem.u_space, conf.problem.p_space],
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def cavity_volume(conf: Config, geo: CLIGeometry, marker: str, u: Any = None) -> float:
    """Volume (m^3) of the cavity bounded by ``marker``, summed over ranks."""
    local = geo.geometry.volume(marker, u=u)
    return geo.mesh.comm.allreduce(local, op=MPI.SUM) * mesh_factor(conf.geometry.unit) ** 3


def _displacement_space(conf: Config, geo: CLIGeometry) -> Any:
    family, degree = conf.problem.u_space.split("_")
    return dolfinx.fem.functionspace(geo.mesh, (family, int(degree), (geo.mesh.geometry.dim,)))


def _unload(conf: Config, geo: CLIGeometry, targets: dict[str, float]) -> dolfinx.fem.Function:
    assert conf.prestress is not None
    from .runner import SolverFailure

    model = pulse.CardiacModel(
        material=build_material(conf.material, geo),
        active=pulse.Passive(),  # Ta = 0 at the imaged state, as in every template
        compressibility=build_compressibility(conf.compressibility),
        viscoelasticity=pulse.viscoelasticity.NoneViscoElasticity(),
    )
    pressures = {
        marker: pulse.Variable(
            dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0)),
            "Pa",
        )
        for marker in targets
    }
    bcs = build_bcs(conf.bcs, geo, pressures)
    if conf.bcs.base_bc == "fixed":  # PrestressProblem ignores parameters["base_bc"]
        clamp = _dirichlet(geo, DirichletConfig(marker=conf.bcs.base_marker))
        bcs = bcs._replace(dirichlet=[*bcs.dirichlet, clamp])
    defaults = pulse.unloading.PrestressProblem.default_parameters()
    problem = pulse.unloading.PrestressProblem(
        geometry=geo.geometry,
        model=model,
        bcs=bcs,
        parameters={
            "u_space": conf.problem.u_space,
            "p_space": conf.problem.p_space,
            "mesh_unit": conf.geometry.unit,
            "petsc_options": {**defaults["petsc_options"], **conf.solver.petsc_options},
        },
        targets=[
            pulse.unloading.TargetPressure(traction=pressures[m], target=p, name=m)
            for m, p in targets.items()
        ],
        ramp_steps=conf.prestress.ramp_steps,
    )
    try:
        return problem.unload()
    except Exception as e:  # scifem's Newton solver raises when it gives up
        raise SolverFailure(
            f"Prestressing to {targets} Pa failed: {e}. Try more prestress.ramp_steps.",
        ) from e


def _cache_is_valid(folder: Path, h: str) -> bool:
    """Never raises (it runs on rank 0 only, ahead of a broadcast)."""
    meta = folder / META
    try:
        if not meta.is_file() or not (folder / U_PRE).exists():
            return False
    except OSError:
        return False
    try:
        data = json.loads(meta.read_text())
    except (OSError, ValueError):  # unreadable, non-UTF8 or not JSON: recompute and replace
        return False
    return isinstance(data, dict) and data.get("hash") == h


def _install(folder: Path, u_pre: dolfinx.fem.Function, meta: dict[str, Any], comm) -> None:
    from .runner import _on_rank0

    name = f".tmp-{folder.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}" if comm.rank == 0 else None
    tmp = folder.with_name(comm.bcast(name, root=0))
    _on_rank0(comm, OSError, lambda: tmp.mkdir(parents=True, exist_ok=True))
    io4dolfinx.write_function_on_input_mesh(tmp / U_PRE, u_pre, time=0.0, name="u_pre")

    def install() -> None:
        (tmp / META).write_text(json.dumps(meta, indent=2))
        if folder.exists():  # a stale entry for this hash (e.g. half-written by a killed job)
            shutil.rmtree(folder)
        os.rename(tmp, folder)

    try:
        _on_rank0(comm, OSError, install)
    finally:
        if comm.rank == 0:
            shutil.rmtree(tmp, ignore_errors=True)


def build_prestress(
    conf: Config,
    geo: CLIGeometry,
    comm=MPI.COMM_WORLD,
    *,
    warn_if_missing: bool = False,
) -> PrestressResult:
    """The unloaded configuration of ``geo`` (still the imaged mesh): cached, or solved now."""
    assert conf.prestress is not None
    for name in ("f0", "s0", "n0"):
        f = getattr(geo, name)
        if f is not None and not isinstance(f, dolfinx.fem.Function):
            raise ConfigError(
                f"[prestress] needs fibre fields stored as Functions; geometry {name} is a "
                f"{type(f).__name__}",
            )
    targets = prestress_targets(conf)
    h = prestress_hash(conf)
    folder = conf.prestress.cache_folder / h[:16]
    imaged = {marker: cavity_volume(conf, geo, marker) for marker in targets}
    u_pre = dolfinx.fem.Function(_displacement_space(conf, geo), name="u_pre")
    cached = comm.bcast(_cache_is_valid(folder, h) if comm.rank == 0 else None, root=0)
    if cached:
        logger.info(f"Reusing the unloaded reference configuration cached in {folder}")
        io4dolfinx.read_function(folder / U_PRE, u_pre, time=0.0, name="u_pre")
        u_pre.x.scatter_forward()
    else:
        message = f"No cached unloaded configuration in {folder}; prestressing to {targets} Pa"
        if warn_if_missing:
            logger.warning(f"{message} (recomputing: the run being continued used one)")
        else:
            logger.info(message)
        u_pre.interpolate(_unload(conf, geo, targets))
        meta = {
            "hash": h,
            "targets_Pa": targets,
            "imaged_volumes_m3": imaged,
            "pulse": pulse.__version__,
        }
        _install(folder, u_pre, meta, comm)
    return PrestressResult(u_pre=u_pre, targets=targets, imaged_volumes=imaged, hash=h)


def apply_prestress(conf: Config, geo: CLIGeometry, result: PrestressResult) -> CLIGeometry:
    """Move ``geo``'s mesh to the unloaded configuration and map its fibres along.

    The mesh is deformed in place, so an injected geometry object must not be passed to
    ``build_simulation`` twice.
    """
    geo.geometry.deform(result.u_pre)
    fibres: dict[str, Any] = {}
    for name in ("f0", "s0", "n0"):
        f = getattr(geo, name)
        if f is None:
            continue
        assert isinstance(f, dolfinx.fem.Function)  # checked in build_prestress
        fibres[name] = pulse.utils.map_vector_field(
            f=f,
            u=result.u_pre,
            normalize=True,
            name=f"{name}_unloaded",
        )
    for marker, imaged in result.imaged_volumes.items():
        unloaded = cavity_volume(conf, geo, marker)
        logger.info(f"{marker}: imaged {imaged / mL:.2f} mL, unloaded {unloaded / mL:.2f} mL")
    return dataclasses.replace(geo, **fibres)


@dataclass
class InflationState:
    """The re-inflated state, to be copied into the run's problem before it starts."""

    u: np.ndarray
    p: np.ndarray | None
    cavity_pressures: dict[str, float]  # marker -> Pa

    def apply(self, problem: Any) -> None:
        problem.u.x.array[:] = self.u
        problem.u_old.x.array[:] = self.u
        if self.p is not None:
            problem.p.x.array[:] = self.p
            problem.p_old.x.array[:] = self.p
        for cavity, pressure, pressure_old in zip(
            problem.cavities,
            problem.cavity_pressures,
            problem.cavity_pressures_old,
        ):
            if cavity.marker in self.cavity_pressures:
                pressure.x.array[:] = self.cavity_pressures[cavity.marker]
                pressure_old.x.array[:] = self.cavity_pressures[cavity.marker]
        # v_old and a_old of a dynamic problem stay 0: the run starts from rest.


def inflate(
    conf: Config,
    geo: CLIGeometry,
    model: Any,
    bcs: Any,
    result: PrestressResult,
    markers: list[str],
    monitor: Any = None,
) -> InflationState:
    """Ramp each cavity's volume from unloaded back to imaged, in ``inflate_steps`` static solves.

    Must run *before* the run's problem is built: `CardiacModel` registers ``u``/``p`` with the
    problem built last, and that must be the run's.
    """
    from .runner import SolverFailure

    state, failed_at = _ramp_to_imaged(conf, geo, model, bcs, result, markers, monitor)
    if state is None:
        raise SolverFailure(
            f"Re-inflation failed at {failed_at:.2f} of the way to the imaged volumes; try "
            "more prestress.inflate_steps",
        )
    return state


def _ramp_to_imaged(
    conf: Config,
    geo: CLIGeometry,
    model: Any,
    bcs: Any,
    result: PrestressResult,
    markers: list[str],
    monitor: Any,
) -> tuple[InflationState | None, float]:
    """Run the ramp; return ``(state, 1.0)`` or ``(None, failed_fraction)``. The problem is
    freed on every rank before returning, so no exception frame ever holds it."""
    from .runner import problem_parameters

    assert conf.prestress is not None
    unloaded = {m: cavity_volume(conf, geo, m) for m in markers}
    volumes = {
        m: dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(unloaded[m])) for m in markers
    }
    problem: Any = pulse.StaticProblem(
        model=model,
        geometry=geo.geometry,
        bcs=bcs,
        cavities=[pulse.problem.Cavity(marker=m, volume=v) for m, v in volumes.items()],
        parameters=problem_parameters(conf),
        monitor=monitor if monitor is not None else pulse.telemetry.NullMonitor(),
    )
    state: InflationState | None = None
    failed_at = 1.0
    n = conf.prestress.inflate_steps
    for k in range(1, n + 1):
        fraction = k / n
        for m in markers:
            volumes[m].value[...] = unloaded[m] + fraction * (
                result.imaged_volumes[m] - unloaded[m]
            )
        if not problem.solve():
            failed_at = fraction
            break
    else:
        state = InflationState(
            u=problem.u.x.array.copy(),
            p=problem.p.x.array.copy() if problem.is_incompressible else None,
            cavity_pressures={
                m: float(p.x.array[0]) for m, p in zip(markers, problem.cavity_pressures)
            },
        )
        for m in markers:
            logger.info(
                f"{m}: re-inflated to {result.imaged_volumes[m] / mL:.2f} mL at "
                f"{state.cavity_pressures[m] / 1e3:.3f} kPa",
            )
    # PETSc's destructors are collective: free the problem on every rank at the same point.
    problem = None
    gc.collect()
    geo.mesh.comm.barrier()
    return state, failed_at
