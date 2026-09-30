"""Build the mesh, markers and fibres from ``[geometry]``.

Generated cardiac-geometriesx meshes are cached in ``geometry.folder/<hash16>/`` (the same
layout as beat), installed by an atomic rename; nothing else in ``geometry.folder`` is touched.
"""

import hashlib
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt

import pulse

from .config import (
    GENERATED_GEOMETRY_TYPES,
    GENERATORS,
    BoxGeometry,
    ConfigError,
    FibersConfig,
    GeometryConfig,
)

logger = logging.getLogger(__name__)

META_FILE = "pulse_geometry.json"


@dataclass
class CLIGeometry:
    geometry: pulse.HeartGeometry
    f0: Any = None
    s0: Any = None
    n0: Any = None
    cfun: dolfinx.mesh.MeshTags | None = None
    vfun: dolfinx.mesh.MeshTags | None = None

    @property
    def mesh(self) -> dolfinx.mesh.Mesh:
        return self.geometry.mesh

    @property
    def markers(self) -> dict[str, tuple[int, int]]:
        return self.geometry.markers


def get_marker(geo: CLIGeometry, name: str) -> tuple[int, int]:
    if name not in geo.markers:
        raise ConfigError(f"Marker {name!r} not found in geometry markers {sorted(geo.markers)}")
    return geo.markers[name]


def check_markers(geo: CLIGeometry, names: Iterable[str], what: str) -> None:
    missing = sorted(set(names) - set(geo.markers))
    if missing:
        raise ConfigError(
            f"{what}: marker(s) {missing} not in the geometry; available: {sorted(geo.markers)}",
        )


def _geometry_hash(conf: GeometryConfig) -> str:
    # unit/scale/quadrature_degree don't change the generated mesh files; folder is the cache
    # location itself. fibers stays in: it drives create_fibers.
    exclude = {"folder", "unit", "scale", "quadrature_degree"}
    blob = json.dumps(conf.model_dump(mode="json", exclude=exclude), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _needs_regeneration(meta: Path, current_hash: str) -> tuple[bool, str | None]:
    """Whether the cached geometry at ``meta`` is stale.

    Called on rank 0 only, before the result is broadcast to the other ranks: must never raise,
    or the other ranks would deadlock waiting on the broadcast. A missing, unreadable or corrupt
    metadata file (e.g. left behind by a job killed mid-write) just means "regenerate".
    """
    if not meta.is_file():
        return True, None
    try:
        cached_hash = json.loads(meta.read_text()).get("hash")
    except (OSError, ValueError, AttributeError) as e:
        # ValueError also catches json.JSONDecodeError and UnicodeDecodeError.
        return True, f"Ignoring unreadable/corrupt {meta} ({e}); regenerating geometry"
    return cached_hash != current_hash, None


def cache_folder(conf: GeometryConfig) -> Path:
    """The folder a generated geometry is cached in: ``geometry.folder/<hash16>/``.

    Keyed by the geometry hash so that different geometry parameters (e.g. an array-job sweep
    over ``geometry.dx``) never share, overwrite or delete each other's mesh, and so that
    ``geometry.folder`` itself -- possibly the config's own directory -- is never deleted.
    """
    return Path(conf.folder) / _geometry_hash(conf)[:16]


def _install_generated(tmp: Path, target: Path, geometry_type: str, h: str) -> None:
    """Move the freshly generated ``tmp`` folder into place as ``target`` (rank 0 only).

    ``target`` is a hash-keyed subfolder pulse itself owns, so a stale/corrupt one may be
    replaced. If a concurrent job already installed a complete entry for the same hash, it is
    reused and ``tmp`` discarded. The metadata file is written into ``tmp`` *before* the
    (atomic) rename, so ``target`` only ever appears complete.
    """
    (tmp / META_FILE).write_text(json.dumps({"type": geometry_type, "hash": h}, indent=2))
    for _ in range(3):
        if not _needs_regeneration(target / META_FILE, h)[0]:
            shutil.rmtree(tmp, ignore_errors=True)  # another job finished first: reuse its
            return
        if target.exists():
            # Stale or partial entry pulse created: move it aside first (atomic), then delete.
            trash = target.with_name(f".trash-{target.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}")
            try:
                os.rename(target, trash)
            except FileNotFoundError:
                pass  # another job removed/replaced it meanwhile; re-check
            else:
                shutil.rmtree(trash, ignore_errors=True)
        try:
            os.rename(tmp, target)
            return
        except OSError:
            continue  # another job installed ``target`` in between; re-check
    raise OSError(f"Could not install the generated geometry into {target}")


def ensure_generated(conf: GeometryConfig, comm: MPI.Intracomm) -> Path:
    """Generate a cardiac-geometriesx mesh into :func:`cache_folder` unless it is up to date."""
    import cardiac_geometries as cg

    from .runner import _on_rank0

    h = _geometry_hash(conf)
    target = cache_folder(conf)
    meta = target / META_FILE
    need = True
    if comm.rank == 0:
        need, warning = _needs_regeneration(meta, h)
        if warning:
            logger.warning(warning)
    need = comm.bcast(need, root=0)
    if not need:
        logger.info(f"Reusing cached {conf.type} geometry in {target}")
        return target

    logger.info(f"Generating {conf.type} geometry in {target}")
    name = f".tmp-{target.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}" if comm.rank == 0 else None
    tmp: Path = target.with_name(comm.bcast(name, root=0))
    raw = tmp.with_name(tmp.name + "-raw")
    _on_rank0(comm, OSError, lambda: target.parent.mkdir(parents=True, exist_ok=True))
    generator = getattr(cg.mesh, GENERATORS[conf.type])
    kwargs = conf.generator_kwargs()
    kwargs["create_fibers"] = conf.fibers.type == "from_geometry"
    rotate = getattr(conf, "rotate_base_normal", None)
    try:
        if rotate is None:
            generator(outdir=tmp, comm=comm, **kwargs)
        else:
            g = generator(outdir=raw, comm=comm, **kwargs)
            g.rotate(target_normal=list(rotate), base_marker="BASE").save_folder(folder=tmp)
            comm.barrier()
            if comm.rank == 0:
                shutil.rmtree(raw, ignore_errors=True)
    except BaseException as e:
        # No collective here (ranks may fail independently): best-effort cleanup only.
        if comm.rank == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(raw, ignore_errors=True)
        if isinstance(e, ImportError):
            # e.g. BiV/UKB fibres need fenicsx-ldrb, UKB meshes need ukb-atlas.
            raise ConfigError(
                f"Generating a {conf.type!r} geometry needs an optional package: {e}",
            ) from e
        raise
    _on_rank0(comm, OSError, lambda: _install_generated(tmp, target, conf.type, h))
    return target


def _axis_fibers(mesh: dolfinx.mesh.Mesh, fibers: FibersConfig) -> tuple[Any, Any, Any]:
    if fibers.type == "none":
        return None, None, None
    if fibers.type != "axis":
        raise ConfigError("geometry.type = 'box' supports fibers.type = 'axis' or 'none'")
    axis = "xyz".index(fibers.direction)
    vectors = []
    for k in range(3):
        vec = np.zeros(3, dtype=dolfinx.default_scalar_type)
        vec[(axis + k) % 3] = 1.0
        vectors.append(dolfinx.fem.Constant(mesh, vec))
    return vectors[0], vectors[1], vectors[2]


def _coord_locator(axis: int, coord: float) -> Any:
    def locator(x: npt.NDArray[np.float64]) -> Any:
        return np.isclose(x[axis], coord)

    return locator


def _box(conf: BoxGeometry, comm: MPI.Intracomm) -> CLIGeometry:
    cell = getattr(dolfinx.mesh.CellType, conf.cell_type)
    lengths = np.array([conf.lx, conf.ly, conf.lz]) * conf.scale
    mesh = dolfinx.mesh.create_box(
        comm,
        [[0.0, 0.0, 0.0], lengths.tolist()],
        [conf.nx, conf.ny, conf.nz],
        cell,
    )
    boundaries: list[pulse.Marker] = []
    for axis, letter in enumerate("XYZ"):
        for side, coord in ((0, 0.0), (1, float(lengths[axis]))):
            boundaries.append(
                pulse.Marker(
                    name=f"{letter}{side}",
                    marker=2 * axis + side + 1,
                    dim=2,
                    locator=_coord_locator(axis, coord),
                ),
            )
    geometry = pulse.HeartGeometry(
        mesh=mesh,
        boundaries=boundaries,
        metadata={"quadrature_degree": conf.quadrature_degree},
    )
    f0, s0, n0 = _axis_fibers(mesh, conf.fibers)
    return CLIGeometry(geometry=geometry, f0=f0, s0=s0, n0=n0)


def _from_folder(folder: Path, conf: GeometryConfig, comm: MPI.Intracomm) -> CLIGeometry:
    import cardiac_geometries as cg

    if not Path(folder).is_dir():
        raise ConfigError(f"Geometry folder {folder} does not exist")
    g = cg.geometry.Geometry.from_folder(comm=comm, folder=folder)
    if conf.scale != 1.0:
        g.mesh.geometry.x[:] *= conf.scale
    geometry = pulse.HeartGeometry.from_cardiac_geometries(
        g,
        metadata={"quadrature_degree": conf.quadrature_degree},
    )
    f0 = s0 = n0 = None
    if conf.fibers.type == "from_geometry":
        if g.f0 is None:
            raise ConfigError(
                "geometry.fibers.type = 'from_geometry' but the geometry has no fibre field; "
                "use fibers.type = 'none' or a mesh with fibres",
            )
        f0, s0, n0 = g.f0, g.s0, g.n0
    elif conf.fibers.type == "axis":
        f0, s0, n0 = _axis_fibers(g.mesh, conf.fibers)
    return CLIGeometry(
        geometry=geometry,
        f0=f0,
        s0=s0,
        n0=n0,
        cfun=g.cfun,
        vfun=getattr(g, "vfun", None),
    )


def build_geometry(conf: GeometryConfig, comm: MPI.Intracomm = MPI.COMM_WORLD) -> CLIGeometry:
    if conf.type == "box":
        return _box(conf, comm)
    if conf.type == "folder":
        return _from_folder(conf.folder, conf, comm)
    if conf.type in GENERATED_GEOMETRY_TYPES:
        return _from_folder(ensure_generated(conf, comm), conf, comm)
    raise ConfigError(f"Unsupported geometry type {conf.type!r}")  # pragma: no cover
