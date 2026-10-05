"""Postprocess a `pulse run` output folder into ``<output.folder>/post/``.

VTX of ``u``, derived fibre fields (DG1), point traces and plots of loads.csv. The simulation is
rebuilt from the config (so the model, fibres and active tension are available) and filled from
``results.bp``; ``post`` refuses results written with different physics settings.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Callable

from mpi4py import MPI

import dolfinx
import io4dolfinx
import numpy as np
import ufl

from .config import Config, ConfigError
from .overrides import load_config, physics_hash
from .runner import (
    LOADS,
    NAMESPACE,
    RESTART_META,
    RESULTS,
    MechanicsSimulation,
    _on_rank0,
    build_simulation,
    read_result_times,
)

logger = logging.getLogger(__name__)


def _check_matches_run(conf: Config, comm) -> None:
    """Refuse a config whose physics differ from the run that wrote ``results.bp``.

    ``pulse post`` rebuilds the geometry, model and loads from the config, so e.g. an edited
    ``geometry.nx`` would otherwise crash reading ``results.bp`` or, on a same-topology mesh
    (e.g. an edited ``material.mu``), silently produce wrong derived fields. Only what
    ``physics_hash`` excludes (``[output]``, ``[postprocess]``, ``[solver]``, the run length) may
    differ. The recorded hash is taken from ``restart.json`` or, if the run was killed before its
    first checkpoint, recomputed from ``config.resolved.toml``; if neither exists the results are
    refused as unverifiable.
    Checked on rank 0 and broadcast, so every rank raises together.
    """
    folder = conf.output.folder
    resolved = folder / "config.resolved.toml"

    def check() -> None:
        meta = folder / RESTART_META
        if meta.is_file():
            recorded = json.loads(meta.read_text())[NAMESPACE]["physics_hash"]
        elif resolved.is_file():
            recorded = physics_hash(load_config(resolved, environ={}))
        else:
            raise ConfigError(
                f"Cannot verify that {folder / RESULTS} was written with this config: neither "
                f"{meta} nor {resolved} exists",
            )
        if recorded != physics_hash(conf):
            raise ConfigError(
                f"The config's physics settings differ from the run that wrote "
                f"{folder / RESULTS} (only [output], [postprocess], [solver] and "
                f"time.end_time may change for pulse post). Compare with {resolved}",
            )

    _on_rank0(comm, ConfigError, check)


def _rank0_guarded(comm, ok: bool, step_name: str, fn: Callable[[], None]) -> bool:
    """Run ``fn`` on rank 0 only, if no earlier step already failed, and make the resulting
    "did visualization succeed so far" flag identical on every rank.

    This is the crux of keeping the plots MPI-safe: rank 0's matplotlib plotting (a broken
    backend, disk errors, etc.) must never raise past this point while the other ranks carry
    on into a later collective call that rank 0 then never reaches - that would deadlock
    every other rank waiting on rank 0 forever. Catching the exception here and broadcasting
    ``ok`` (always rank 0's value, via ``root=0``) means every rank agrees, after this call,
    on whether to keep going - so any subsequent collective call is either entered by every
    rank or skipped by every rank together.
    """
    if ok and comm.rank == 0:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - any rendering failure is recoverable here
            ok = False
            logger.warning(f"Visualization failed at {step_name!r}, skipping it: {e!r}")
    return comm.bcast(ok, root=0)


def _interpolation_points(V):
    points = V.element.interpolation_points
    return points() if callable(points) else points


def _read_step(sim: MechanicsSimulation, path: Path, t: float) -> None:
    io4dolfinx.read_function(path, sim.problem.u, time=t, name="u")
    sim.problem.u.x.scatter_forward()
    if sim.problem.is_incompressible:
        io4dolfinx.read_function(path, sim.problem.p, time=t, name="p")
        sim.problem.p.x.scatter_forward()
    sim.loads.update(float(t))


def _write_vtx(sim: MechanicsSimulation, path: Path, times, post: Path) -> None:
    out = post / "displacement.bp"
    with dolfinx.io.VTXWriter(sim.comm, out, [sim.problem.u], engine="BP4") as vtx:
        for t in times:
            _read_step(sim, path, t)
            vtx.write(float(t))
    logger.info(f"VTX output for ParaView written to {out}")


def _write_fields(sim: MechanicsSimulation, path: Path, times, post: Path) -> None:
    names = sim.conf.postprocess.fields
    if sim.geo.f0 is None:
        raise ConfigError("postprocess.fields needs a fibre field (geometry.fibers)")
    mesh = sim.geo.mesh
    V = dolfinx.fem.functionspace(mesh, ("DG", 1))
    u = sim.problem.u
    F = ufl.variable(ufl.grad(u) + ufl.Identity(3))  # model.sigma differentiates w.r.t. F
    f0 = sim.geo.f0
    exprs = {}
    if "fiber_stress" in names:
        f = F * f0 / ufl.sqrt(ufl.inner(F * f0, F * f0))
        sigma = sim.problem.model.sigma(F)
        exprs["fiber_stress"] = ufl.inner(sigma * f, f)
    if "fiber_strain" in names:
        E = 0.5 * (F.T * F - ufl.Identity(3))
        exprs["fiber_strain"] = ufl.inner(E * f0, f0)
    points = _interpolation_points(V)
    compiled = {n: dolfinx.fem.Expression(e, points) for n, e in exprs.items()}
    functions = {n: dolfinx.fem.Function(V, name=n) for n in exprs}
    out = post / "fields.bp"
    with dolfinx.io.VTXWriter(mesh.comm, out, list(functions.values()), engine="BP4") as vtx:
        for t in times:
            _read_step(sim, path, t)
            for n, expr in compiled.items():
                functions[n].interpolate(expr)
            vtx.write(float(t))
    logger.info(f"Derived fields {sorted(exprs)} written to {out}")


def _probe_points(sim: MechanicsSimulation) -> dict[str, np.ndarray]:
    """Reference coordinates of every [postprocess] point and vertex tag (same on all ranks)."""
    conf = sim.conf.postprocess
    points = {name: np.asarray(xyz, dtype=float) for name, xyz in conf.points.items()}
    mesh = sim.geo.mesh
    for name, marker in conf.vertex_tags.items():
        if sim.geo.vfun is None or marker not in sim.geo.markers:
            raise ConfigError(
                f"postprocess.vertex_tags.{name}: no vertex marker {marker!r}; markers: "
                f"{sorted(sim.geo.markers)}",
            )
        vertices = sim.geo.vfun.find(sim.geo.markers[marker][0])
        local = np.zeros((0, 3))
        if len(vertices):
            dofs = dolfinx.mesh.entities_to_geometry(mesh, 0, vertices)
            local = mesh.geometry.x[dofs.reshape(-1)]
        gathered = np.vstack(mesh.comm.allgather(local))
        if len(gathered) == 0:
            raise ConfigError(f"postprocess.vertex_tags.{name}: marker {marker!r} tags no vertex")
        points[name] = gathered[0]
    return points


def _evaluate(u: dolfinx.fem.Function, points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mesh = u.function_space.mesh
    names = list(points)
    xyz = np.array([points[n] for n in names], dtype=mesh.geometry.x.dtype).reshape(-1, 3)
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, xyz)
    colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, xyz)
    values = np.zeros((len(names), 3))
    found = np.zeros(len(names))
    for i in range(len(names)):
        cells = colliding.links(i)
        if len(cells):
            values[i] = u.eval(xyz[i], cells[:1])
            found[i] = 1.0
    values = mesh.comm.allreduce(values, op=MPI.SUM)
    found = mesh.comm.allreduce(found, op=MPI.SUM)
    missing = [n for n, c in zip(names, found) if c == 0]
    if missing:
        raise ConfigError(f"postprocess points outside the mesh: {missing}")
    return {n: values[i] / found[i] for i, n in enumerate(names)}


def _write_points(sim: MechanicsSimulation, path: Path, times, post: Path) -> None:
    points = _probe_points(sim)
    _evaluate(sim.problem.u, points)  # fail before reading every step if a point is outside
    rows = []
    for t in times:
        _read_step(sim, path, t)
        values = _evaluate(sim.problem.u, points)
        row = {"t": float(t)}
        for name, (ux, uy, uz) in values.items():
            row.update({f"{name}_ux": ux, f"{name}_uy": uy, f"{name}_uz": uz})
        rows.append(row)

    def write() -> None:
        with open(post / "points.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    _on_rank0(sim.comm, OSError, write)


def _plots(folder: Path, post: Path) -> None:
    """loads.png (loads in kPa, volumes in mL vs t) and one PV loop per cavity marker."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(folder / LOADS) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    t = np.array([float(r["t"]) for r in rows])
    columns = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "t"}
    volumes = {k: v for k, v in columns.items() if k.startswith("volume_")}
    loads = {k: v for k, v in columns.items() if not k.startswith("volume_")}
    fig, axes = plt.subplots(
        1,
        2 if volumes else 1,
        figsize=(10 if volumes else 5, 4),
        squeeze=False,
    )
    for name, values in loads.items():
        axes[0, 0].plot(t, values / 1e3, label=name)
    axes[0, 0].set(xlabel="t [s]", ylabel="load [kPa]")
    axes[0, 0].legend()
    for name, values in volumes.items():
        axes[0, 1].plot(t, values * 1e6, label=name)
    if volumes:
        axes[0, 1].set(xlabel="t [s]", ylabel="volume [mL]")
        axes[0, 1].legend()
    fig.tight_layout()
    fig.savefig(post / "loads.png")
    plt.close(fig)
    for name, values in volumes.items():
        marker = name[len("volume_") :]
        pressure = loads.get(f"pressure_{marker}")
        if pressure is None:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(values * 1e6, pressure / 1e3)
        ax.set(xlabel="volume [mL]", ylabel="pressure [kPa]", title=f"PV loop {marker}")
        fig.tight_layout()
        fig.savefig(post / f"pv_loop_{marker}.png")
        plt.close(fig)


def run_post(conf: Config, comm=MPI.COMM_WORLD) -> Path:
    folder = conf.output.folder
    path = folder / RESULTS
    if not path.exists():
        raise ConfigError(f"No results found at {path}. Run `pulse run <config>` first.")
    _check_matches_run(conf, comm)
    sim = build_simulation(conf, comm)
    times = read_result_times(path, comm)
    post = folder / "post"
    _on_rank0(comm, OSError, lambda: post.mkdir(parents=True, exist_ok=True))
    if conf.postprocess.vtx:
        _write_vtx(sim, path, times, post)
    if conf.postprocess.fields:
        _write_fields(sim, path, times, post)
    if conf.postprocess.points or conf.postprocess.vertex_tags:
        _write_points(sim, path, times, post)
    if conf.postprocess.plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            logger.warning("matplotlib is not installed; skipping plots")
        else:
            _rank0_guarded(comm, True, "plots", lambda: _plots(folder, post))
    logger.info(f"Postprocessing written to {post}")
    return post
