"""Surfaces fixed in space, for a :class:`~pulse.boundary_conditions.PericardiumBC` to hold a
boundary against.

A surface gives the signed distance to it, positive outside, in one of two ways:

- as a UFL expression of the point (:class:`PericardialSurface`), like the
  :class:`EllipsoidSurface`;
- as numbers, point by point (:class:`SampledSurface`), like the :class:`MeshSurface` through
  the marked facets of a mesh.

A form cannot evaluate a sampled surface where the boundary has moved to. A problem therefore
wraps it in a :class:`TrackedSurface`, which the solver updates before every evaluation of the
residual and the Jacobian.
"""

import logging
import math
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import dolfinx
import numpy as np
import numpy.typing as npt
import ufl
from packaging.version import Version

logger = logging.getLogger(__name__)

_dolfinx_version = Version(dolfinx.__version__)


class HasInfo(typing.Protocol):
    """A geometry that knows how its mesh was made, like those of cardiac-geometriesx."""

    @property
    def info(self) -> Mapping[str, typing.Any]: ...


class HasFacetTags(typing.Protocol):
    """A mesh with tags on its facets and names for their values, like a
    :class:`pulse.Geometry`."""

    mesh: dolfinx.mesh.Mesh

    # Read-only, so that a `dict` and a `MeshTags` that cannot be `None` satisfy them too
    @property
    def markers(self) -> Mapping[str, tuple[int, int]]: ...

    @property
    def facet_tags(self) -> dolfinx.mesh.MeshTags | None: ...


class PericardialSurface(typing.Protocol):
    """A surface fixed in space that a :class:`~pulse.boundary_conditions.PericardiumBC`
    holds a boundary against."""

    def signed_distance(self, x: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """The signed distance from the point ``x`` to the surface, positive outside, in
        mesh units. It must be smooth near the surface, and its gradient there of unit
        length and normal to the surface."""
        ...


class SampledSurface(typing.Protocol):
    """A surface fixed in space whose signed distance is computed point by point."""

    def evaluate(
        self,
        x: npt.NDArray[np.float64],
        hessian: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
        """The signed distance from each of the points ``x`` (one per row) to the surface,
        positive outside, in mesh units, its gradient, and with ``hessian`` its Hessian
        (otherwise ``None``). It must be twice continuously differentiable near the
        surface."""
        ...


@dataclass(slots=True)
class EllipsoidSurface:
    r"""An ellipsoid of revolution with semi-axis ``r_long`` along ``axis`` and ``r_short``
    across it, centred at ``center``. Lengths are in mesh units.

    The signed distance is the first-order one of the level set
    :math:`\phi = \sqrt{(s / r_\mathrm{long})^2 + \rho^2 / r_\mathrm{short}^2} - 1`, with
    :math:`s` and :math:`\rho` the axial and radial coordinates: :math:`d = \phi /
    |\nabla \phi|`. It is exact on the surface and for a sphere, and its error grows with
    the square of the distance over the radius of curvature. It is symmetric about the axis,
    so a rotation about the axis leaves it unchanged everywhere.
    """

    r_long: float
    r_short: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self):
        if self.r_long <= 0 or self.r_short <= 0:
            raise ValueError(f"Radii must be positive, got {self.r_long}, {self.r_short}")
        length = math.sqrt(sum(a * a for a in self.axis))
        if length == 0:
            raise ValueError("axis must be nonzero")
        self.axis = typing.cast(
            tuple[float, float, float],
            tuple(float(a) / length for a in self.axis),
        )

    @classmethod
    def from_cardiac_geometries(
        cls,
        geo: HasInfo | Mapping[str, typing.Any],
        surface: str = "epi",
    ) -> "EllipsoidSurface":
        """The epicardial (``surface="epi"``) or endocardial (``"endo"``) ellipsoid of a
        cardiac-geometriesx ``lv_ellipsoid``, from its ``info`` (or the info dict itself).
        Its long axis is x and its centre the origin. If the mesh was scaled after it was
        made, scale the radii likewise."""
        info = geo if isinstance(geo, Mapping) else geo.info
        mesh_type = info.get("mesh_type")
        if mesh_type != "lv_ellipsoid":
            raise ValueError(
                f"Can only build an EllipsoidSurface from an 'lv_ellipsoid', got {mesh_type!r}",
            )
        if surface not in ("epi", "endo"):
            raise ValueError(f"surface must be 'epi' or 'endo', got {surface!r}")
        return cls(r_long=info[f"r_long_{surface}"], r_short=info[f"r_short_{surface}"])

    def signed_distance(self, x: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        a = ufl.as_vector(self.axis)
        y = x - ufl.as_vector(self.center)
        s = ufl.dot(y, a)
        y_radial = y - s * a
        psi = (s / self.r_long) ** 2 + ufl.dot(y_radial, y_radial) / self.r_short**2
        grad_psi = 2 * s / self.r_long**2 * a + 2 * y_radial / self.r_short**2
        # phi = sqrt(psi) - 1 and grad(phi) = grad(psi) / (2 sqrt(psi))
        return 2 * (psi - ufl.sqrt(psi)) / ufl.sqrt(ufl.dot(grad_psi, grad_psi))


# Points evaluated at a time by a MeshSurface, to bound the memory of the work arrays
_CHUNK = 4096
# The sum M of the weighted normals is as long as the sum W of the weights on a flat surface.
# Instead of by its length it is divided by sqrt((|M|^2 + (_CANCEL W)^2) / (1 + _CANCEL^2)),
# which is the same there, so that the mean normal fades out where the normals around a point
# cancel (far inside a closed surface) instead of turning on the spot
_CANCEL = 0.05


@dataclass(slots=True)
class _Samples:
    """Points with a normal and a weight each, binned in cubes of side ``radius`` to find the
    ones within ``radius`` of a point."""

    points: npt.NDArray[np.float64]
    normals: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    radius: float
    origin: npt.NDArray[np.float64] = field(init=False)
    shape: npt.NDArray[np.int64] = field(init=False)
    order: npt.NDArray[np.int64] = field(init=False)
    keys: npt.NDArray[np.int64] = field(init=False)

    def __post_init__(self):
        self.origin = self.points.min(axis=0) - self.radius
        cubes = self._cubes(self.points)
        self.shape = cubes.max(axis=0) + 2
        keys = self._key(cubes)
        self.order = np.argsort(keys, kind="stable")
        self.keys = keys[self.order]

    def _cubes(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        return np.floor((x - self.origin) / self.radius).astype(np.int64)

    def _key(self, cubes: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        return (cubes[:, 0] * self.shape[1] + cubes[:, 1]) * self.shape[2] + cubes[:, 2]

    def pairs(self, x: npt.NDArray[np.float64]):
        """Every (point of ``x``, sample) pair within ``radius`` of each other, sorted by
        point: the indices of the two, and the vector and distance between them."""
        home = self._cubes(np.clip(x, self.origin - self.radius, None))
        points, samples = [], []
        for shift in np.ndindex(3, 3, 3):
            cubes = home + np.array(shift) - 1
            valid = np.flatnonzero(np.all((cubes >= 0) & (cubes < self.shape), axis=1))
            keys = self._key(cubes[valid])
            start = np.searchsorted(self.keys, keys, side="left")
            count = np.searchsorted(self.keys, keys, side="right") - start
            within = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
            points.append(np.repeat(valid, count))
            samples.append(self.order[np.repeat(start, count) + within])
        point = np.concatenate(points)
        sample = np.concatenate(samples)
        q = x[point] - self.points[sample]
        distance = np.linalg.norm(q, axis=1)
        keep = np.flatnonzero(distance < self.radius)
        keep = keep[np.argsort(point[keep], kind="stable")]
        return point[keep], sample[keep], q[keep], distance[keep]


def _beyond_the_edge(
    corners: npt.NDArray[np.float64],
    normals: npt.NDArray[np.float64],
    extension: float,
    vertices: npt.NDArray[np.int64] | None = None,
):
    """Samples that continue a surface of triangles past its edge, for ``extension``.

    ``corners`` holds the three vertices of each triangle and ``normals`` its normal, of any
    length. An edge of only one triangle is on the edge of the surface. From each, a strip
    as wide as the edge runs outwards in the plane of its triangle, at right angles to the
    edge, with samples a width apart. Returns their points, normals (as long as their areas)
    and areas.

    ``vertices``, one row of three ids per triangle, says which corners are the same vertex.
    Without it, corners are the same if their coordinates are. That fails on a mesh whose
    copies of a node on different processes differ by round-off: the edges between the
    processes then look like edges of the surface, and strips run across it.
    """
    if extension <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    if vertices is None:
        _, vertex = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
    else:
        _, vertex = np.unique(np.asarray(vertices).reshape(-1), return_inverse=True)
    vertex = vertex.reshape(-1, 3)
    triangle = np.repeat(np.arange(len(corners)), 3)
    local = np.tile(np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]), (len(corners), 1))
    ends = np.sort(vertex[triangle[:, None], local[:, :2]], axis=1)
    _, edge, count = np.unique(ends, axis=0, return_inverse=True, return_counts=True)
    on_edge = count[edge.reshape(-1)] == 1
    triangle, local = triangle[on_edge], local[on_edge]

    a, b, c = (corners[triangle, local[:, i]] for i in range(3))
    along = b - a
    width = np.linalg.norm(along, axis=1)
    along /= width[:, None]
    middle = 0.5 * (a + b)
    outwards = middle - c
    outwards -= np.einsum("ij,ij->i", outwards, along)[:, None] * along
    outwards /= np.linalg.norm(outwards, axis=1)[:, None]
    unit = normals[triangle] / np.linalg.norm(normals[triangle], axis=1)[:, None]

    steps = np.ceil(extension / width).astype(np.int64)
    strip = np.repeat(np.arange(len(triangle)), steps)
    step = np.arange(steps.sum()) - np.repeat(np.cumsum(steps) - steps, steps)
    points = middle[strip] + ((step + 0.5) * width[strip])[:, None] * outwards[strip]
    areas = width[strip] ** 2
    return points, unit[strip] * areas[:, None], areas


def _symmetric(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]):
    """u v^T + v u^T, row by row."""
    uv = u[:, :, None] * v[:, None, :]
    return uv + uv.transpose(0, 2, 1)


@dataclass
class MeshSurface:
    r"""A smooth surface through the facets of a mesh, as they were when it was made.

    Each facet contributes its centroid :math:`\mathbf{p}_i`, unit normal
    :math:`\mathbf{n}_i` and area :math:`A_i`. With the weights
    :math:`w_i(\mathbf{x}) = A_i\, \varphi(|\mathbf{x} - \mathbf{p}_i| / \rho)`, where
    :math:`\varphi(r) = (1 - r)^4 (4 r + 1)` and :math:`\rho` is ``radius``, and
    :math:`\langle a \rangle = \sum_i w_i a_i / \sum_i w_i`, the signed distance is

    .. math::
        d(\mathbf{x}) = \frac{1}{2} \left[ \langle \mathbf{n} \cdot (\mathbf{x} - \mathbf{p})
        \rangle + \bar{\mathbf{n}} \cdot (\mathbf{x} - \langle \mathbf{p} \rangle) \right],
        \qquad \bar{\mathbf{n}} = \frac{\langle \mathbf{n} \rangle}{|\langle \mathbf{n}
        \rangle|}.

    The first term is the mean distance to the planes of the facets nearby, which lies
    below a convex surface. The second is the distance to the plane through their centroid,
    which lies above it by as much, so on a sphere their mean has no error. :math:`d` is
    twice continuously differentiable, also across the edges between the facets. It differs
    from the true distance by a few percent of the facet size on a well-resolved surface,
    and features smaller than ``radius`` are smoothed out.

    The surface reaches as far as ``radius`` from the centroids. A point further than that
    from all of them gets the distance to the plane of the nearest facet instead, and a
    warning, once. Choose a ``radius`` larger than the gap that the springs are meant to
    allow.

    ``points``, ``normals`` (pointing outwards) and ``areas`` have one row per facet, and
    lengths are in mesh units. Use :meth:`from_geometry` or :meth:`from_facets` to read
    them from a mesh. The last ``rim`` rows are not facets, but continue the surface past
    its edge; see :meth:`from_facets`.
    """

    points: npt.NDArray[np.float64]
    normals: npt.NDArray[np.float64]
    areas: npt.NDArray[np.float64]
    radius: float
    rim: int = 0
    _samples: _Samples = field(init=False, repr=False)
    _warned: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        self.points = np.array(self.points, dtype=np.float64).reshape(-1, 3)
        self.normals = np.array(self.normals, dtype=np.float64).reshape(-1, 3)
        self.areas = np.array(self.areas, dtype=np.float64).reshape(-1)
        if not len(self.points) == len(self.normals) == len(self.areas):
            raise ValueError("points, normals and areas must have one row per facet")
        if len(self.points) == 0:
            raise ValueError("A MeshSurface needs at least one facet")
        if not self.radius > 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        self.radius = float(self.radius)
        lengths = np.linalg.norm(self.normals, axis=1)
        if np.any(lengths == 0) or np.any(self.areas <= 0):
            raise ValueError("Every facet needs a nonzero normal and a positive area")
        self.normals = self.normals / lengths[:, None]

        self._samples = _Samples(self.points, self.normals, self.areas, self.radius)

    def __repr__(self) -> str:
        facets = len(self.points) - self.rim
        beyond = f" and {self.rim} beyond their edge" if self.rim else ""
        return f"MeshSurface({facets} facets{beyond}, radius={self.radius:.4g})"

    @classmethod
    def from_facets(
        cls,
        mesh: dolfinx.mesh.Mesh,
        facets: npt.NDArray[np.int32],
        radius: float | None = None,
        extension: float | None = None,
    ) -> "MeshSurface":
        """The surface through the boundary ``facets`` (local indices) of ``mesh``, a mesh
        of tetrahedra, where it is now. The normals point out of the mesh.

        ``radius`` defaults to twice the size of the facets (the edge length of an
        equilateral triangle with their mean area), and to no less than a tenth of the
        size of the surface (the square root of its area), about a centimetre for a heart.
        The first gives enough facets to average over, the second room for the boundary
        to move away from the surface on a fine mesh.

        Where the facets end, at the base of a ventricle for example, the surface goes on
        for ``extension`` (by default twice the radius) in the plane of the last facet, so
        that a boundary that slides past the edge stays on a surface. The samples of that
        strip come after those of the facets; ``rim`` counts them.

        The surface is the same on every process.
        """
        tdim = mesh.topology.dim
        fdim = tdim - 1
        if mesh.topology.cell_type != dolfinx.mesh.CellType.tetrahedron:
            raise NotImplementedError("A MeshSurface needs a mesh of tetrahedra")
        mesh.topology.create_connectivity(fdim, tdim)
        mesh.topology.create_connectivity(tdim, fdim)
        facets = np.unique(np.asarray(facets, dtype=np.int32))
        facets = facets[facets < mesh.topology.index_map(fdim).size_local]
        X = mesh.geometry.x
        # The vertices come first among the nodes of a facet
        nodes = dolfinx.mesh.entities_to_geometry(mesh, fdim, facets)[:, :3]
        corners = X[nodes]
        # The same on every process that holds the node, unlike its coordinates, which can
        # differ by round-off there (see `_beyond_the_edge`)
        vertices = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)[nodes]
        normals = 0.5 * np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        facet_to_cell = mesh.topology.connectivity(fdim, tdim)
        cells = facet_to_cell.array[facet_to_cell.offsets[facets]]
        cell_nodes = dolfinx.mesh.entities_to_geometry(mesh, tdim, cells)
        inwards = X[cell_nodes].mean(axis=1) - corners.mean(axis=1)
        normals[np.einsum("ij,ij->i", normals, inwards) > 0] *= -1
        # Every process gets all the facets, in the same order
        corners = np.concatenate(mesh.comm.allgather(corners))
        normals = np.concatenate(mesh.comm.allgather(normals))
        vertices = np.concatenate(mesh.comm.allgather(vertices))
        if len(corners) == 0:
            raise ValueError("No facets to make a MeshSurface from")
        points = corners.mean(axis=1)
        areas = np.linalg.norm(normals, axis=1)
        if radius is None:
            facet_size = math.sqrt(4 * areas.mean() / math.sqrt(3))
            radius = max(2 * facet_size, 0.1 * math.sqrt(areas.sum()))
        if extension is None:
            extension = 2 * radius
        rim_points, rim_normals, rim_areas = _beyond_the_edge(corners, normals, extension, vertices)
        return cls(
            points=np.concatenate((points, rim_points)),
            normals=np.concatenate((normals, rim_normals)),
            areas=np.concatenate((areas, rim_areas)),
            radius=radius,
            rim=len(rim_points),
        )

    @classmethod
    def from_geometry(
        cls,
        geometry: HasFacetTags,
        marker: str | int | Sequence[str | int],
        radius: float | None = None,
        extension: float | None = None,
    ) -> "MeshSurface":
        """The surface through the facets of ``geometry`` (a :class:`~pulse.Geometry`) with
        the given marker or markers, by name or by value, for example ``"EPI"``. See
        :meth:`from_facets`."""
        if geometry.facet_tags is None:
            raise ValueError("The geometry has no facet tags")
        markers = [marker] if isinstance(marker, (str, int, np.integer)) else list(marker)
        values = [geometry.markers[m][0] if isinstance(m, str) else int(m) for m in markers]
        facets = np.concatenate([geometry.facet_tags.find(value) for value in values])
        return cls.from_facets(geometry.mesh, facets, radius=radius, extension=extension)

    def evaluate(
        self,
        x: npt.NDArray[np.float64],
        hessian: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
        """The signed distance from each of the points ``x`` to the surface, its gradient,
        and with ``hessian`` its Hessian (otherwise ``None``)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
        d = np.zeros(len(x))
        grad = np.zeros((len(x), 3))
        hess = np.zeros((len(x), 3, 3)) if hessian else None
        for start in range(0, len(x), _CHUNK):
            chunk = slice(start, start + _CHUNK)
            d[chunk], grad[chunk], H = self._evaluate(x[chunk], hessian)
            if hess is not None:
                hess[chunk] = H
        return d, grad, hess

    def _evaluate(self, x: npt.NDArray[np.float64], hessian: bool):
        samples = self._samples
        rho = samples.radius
        eye = np.eye(3)
        point, sample, q, distance = samples.pairs(x)
        found, first = np.unique(point, return_index=True)

        def total(values):
            """The sum of the values of the pairs of each point."""
            sums = np.zeros((len(x), *values.shape[1:]))
            if len(found) > 0:
                sums[found] = np.add.reduceat(values, first, axis=0)
            return sums

        r = distance / rho
        s = 1 - r
        A = samples.weights[sample]
        n = samples.normals[sample]
        # The distance to the plane of the sample
        a = np.einsum("pi,pi->p", n, q)
        # The weight A phi(r), with gradient c q and Hessian c I + e q q^T
        w = A * s**4 * (4 * r + 1)
        c = -20 * A * s**3 / rho**2
        gw = c[:, None] * q

        # Sums over the samples near each point, and their gradients
        W, gW = total(w), total(gw)
        S, gS = total(w * a), total(a[:, None] * gw + w[:, None] * n)
        M, gM = total(w[:, None] * n), total(n[:, :, None] * gw[:, None, :])
        # q depends on the point as well as w does
        C = total(w[:, None] * q)
        gC = total(q[:, :, None] * gw[:, None, :]) + W[:, None, None] * eye

        covered = W > 0
        W1 = np.where(covered, W, 1.0)
        cancel = _CANCEL**2
        m1 = np.sqrt((np.einsum("pi,pi->p", M, M) + cancel * W1**2) / (1 + cancel))
        D = m1 * W1
        # The mean distance to the planes of the samples, and its gradient
        f1 = S / W1
        g1 = (gS - f1[:, None] * gW) / W1[:, None]
        # The distance to the plane through their centroid, f2 = M.C / (m W) with m the
        # length of M, but for _CANCEL
        MC = np.einsum("pi,pi->p", M, C)
        gMC = np.einsum("pij,pi->pj", gM, C) + np.einsum("pi,pij->pj", M, gC)
        gm = (np.einsum("pi,pij->pj", M, gM) + cancel * W[:, None] * gW) / (1 + cancel)
        gm /= m1[:, None]
        gD = gm * W1[:, None] + m1[:, None] * gW
        f2 = MC / D
        g2 = (gMC - f2[:, None] * gD) / D[:, None]

        d = 0.5 * (f1 + f2)
        grad = 0.5 * (g1 + g2)
        H = None
        if hessian:
            e = np.divide(60 * A * s**2, rho**4 * r, out=np.zeros_like(r), where=r > 0)
            hw = c[:, None, None] * eye + e[:, None, None] * q[:, :, None] * q[:, None, :]
            hW = total(hw)
            hS = total(a[:, None, None] * hw + _symmetric(gw, n))
            hM = total(n[:, :, None, None] * hw[:, None, :, :])
            hC = total(q[:, :, None, None] * hw[:, None, :, :])
            hC += np.einsum("ik,pj->pijk", eye, gW) + np.einsum("ij,pk->pijk", eye, gW)
            h1 = (hS - _symmetric(g1, gW) - f1[:, None, None] * hW) / W1[:, None, None]
            gMgC = np.einsum("pij,pik->pjk", gM, gC)
            hMC = (
                np.einsum("pijk,pi->pjk", hM, C)
                + gMgC
                + gMgC.transpose(0, 2, 1)
                + np.einsum("pi,pijk->pjk", M, hC)
            )
            hm = (
                np.einsum("pij,pik->pjk", gM, gM)
                + np.einsum("pi,pijk->pjk", M, hM)
                + cancel * (gW[:, :, None] * gW[:, None, :] + W[:, None, None] * hW)
            ) / (1 + cancel)
            hm = (hm - gm[:, :, None] * gm[:, None, :]) / m1[:, None, None]
            hD = hm * W1[:, None, None] + _symmetric(gm, gW) + m1[:, None, None] * hW
            h2 = (hMC - _symmetric(g2, gD) - f2[:, None, None] * hD) / D[:, None, None]
            H = 0.5 * (h1 + h2)

        far = np.flatnonzero(~covered)
        if len(far) > 0:
            if not self._warned:
                self._warned = True
                logger.warning(
                    f"{len(far)} points are further than its radius from {self}; there the "
                    "surface is the plane of the nearest facet. Give it a larger radius if "
                    "the boundary is meant to move this far",
                )
            # A few at a time: every point is compared with every sample
            for block in np.array_split(far, len(far) // 256 + 1):
                gaps = x[block, None, :] - samples.points[None, :, :]
                nearest = np.argmin(np.einsum("psi,psi->ps", gaps, gaps), axis=1)
                grad[block] = samples.normals[nearest]
                d[block] = np.einsum("pi,pi->p", grad[block], x[block] - samples.points[nearest])
                if H is not None:
                    H[block] = 0.0
        return d, grad, H


# The nodes of a degree 2 Lagrange element on a triangle, as a quadrature rule: the rule that
# is exact for functions linear on each of the four triangles between the nodes
_NODAL_POINTS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]],
)
_NODAL_WEIGHTS = np.array([1 / 24, 1 / 24, 1 / 24, 1 / 8, 1 / 8, 1 / 8])


@dataclass
class _TrackedPoint:
    """The quadratic expansion of the distance about where a point of the boundary is."""

    expression: dolfinx.fem.Expression
    moves: bool
    x: dolfinx.fem.Function
    d: dolfinx.fem.Function
    n: dolfinx.fem.Function
    H: dolfinx.fem.Function


class TrackedSurface:
    r"""A :class:`SampledSurface` as the forms of a problem see it, on the boundary ``facets``
    of ``mesh``, a mesh of tetrahedra.

    The distance to a sampled surface is not a UFL expression. For every point
    :math:`\mathbf{x}` it is asked the distance of (say :math:`\mathbf{X} + \mathbf{u}`),
    this keeps the distance :math:`d_k`, its gradient :math:`\mathbf{n}_k` and its Hessian
    :math:`\mathbf{H}_k` at the position :math:`\mathbf{x}_k` the point had at the last
    :meth:`update`, and gives the expansion

    .. math::
        d(\mathbf{x}) = d_k + \mathbf{n}_k \cdot (\mathbf{x} - \mathbf{x}_k) + \frac{1}{2}
        (\mathbf{x} - \mathbf{x}_k) \cdot \mathbf{H}_k (\mathbf{x} - \mathbf{x}_k).

    Its value and first two derivatives are those of the surface at :math:`\mathbf{x}_k`.
    So a residual and a Jacobian assembled right after an :meth:`update` are those of the
    surface itself, to which Newton's method then converges as usual. A problem updates
    before each of them.

    The four are kept at the nodes of a degree 2 Lagrange space, and :attr:`ds` integrates
    with these nodes as quadrature points, which is where the expansion is exact. Integrate
    everything that holds a distance from here over :attr:`ds`.
    """

    def __init__(
        self,
        surface: SampledSurface,
        mesh: dolfinx.mesh.Mesh,
        facets: npt.NDArray[np.int32],
        ds: ufl.Measure,
    ):
        if mesh.topology.cell_type != dolfinx.mesh.CellType.tetrahedron:
            raise NotImplementedError("A sampled surface needs a mesh of tetrahedra")
        self.surface = surface
        self.mesh = mesh
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        self._scalar = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
        self._vector = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))
        self._tensor = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3, 3)))
        self._cells = dolfinx.mesh.compute_incident_entities(mesh.topology, facets, tdim - 1, tdim)
        self._nodes = dolfinx.fem.locate_dofs_topological(self._scalar, tdim - 1, facets)
        #: The measure over the boundary of the mesh with the nodes as quadrature points
        self.ds = ds.reconstruct(
            metadata={
                "quadrature_rule": "custom",
                "quadrature_points": _NODAL_POINTS,
                "quadrature_weights": _NODAL_WEIGHTS,
                "quadrature_degree": 2,
            },
        )
        self._points: list[tuple[ufl.core.expr.Expr, _TrackedPoint]] = []

    def _point(self, x: ufl.core.expr.Expr) -> _TrackedPoint:
        while isinstance(x, ufl.classes.Variable):
            x = x.ufl_operands[0]
        for known, point in self._points:
            if known == x:
                return point
        if _dolfinx_version >= Version("0.10"):
            nodes = self._vector.element.interpolation_points
        else:
            nodes = self._vector.element.interpolation_points()  # type: ignore[operator]
        point = _TrackedPoint(
            expression=dolfinx.fem.Expression(x, nodes),
            moves=len(ufl.algorithms.extract_coefficients(x)) > 0,
            x=dolfinx.fem.Function(self._vector),
            d=dolfinx.fem.Function(self._scalar),
            n=dolfinx.fem.Function(self._vector),
            H=dolfinx.fem.Function(self._tensor),
        )
        self._points.append((x, point))
        self._update(point, hessian=True)
        return point

    def _update(self, point: _TrackedPoint, hessian: bool) -> None:
        point.x.interpolate(point.expression, self._cells)
        x = point.x.x.array.reshape(-1, 3)[self._nodes]
        d, n, H = self.surface.evaluate(x, hessian=hessian)
        point.d.x.array[self._nodes] = d
        point.n.x.array.reshape(-1, 3)[self._nodes] = n
        if H is not None:
            point.H.x.array.reshape(-1, 9)[self._nodes] = H.reshape(-1, 9)

    def signed_distance(self, x: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """The expansion of the distance about where the point ``x`` was at the last
        :meth:`update`. It is exact at the nodes on the facets, at that time."""
        point = self._point(x)
        dx = x - point.x
        return point.d + ufl.dot(point.n, dx) + 0.5 * ufl.dot(dx, point.H * dx)

    def update(self, hessian: bool = True, fixed: bool = False) -> None:
        """Expand about where the points are now. Without ``hessian`` the Hessians stay as
        they were, which a residual does not see. ``fixed`` also updates the points that do
        not depend on any function, for a mesh that was moved."""
        for _, point in self._points:
            if point.moves or fixed:
                self._update(point, hessian=hessian)
