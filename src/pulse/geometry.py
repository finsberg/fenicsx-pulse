import logging
import typing
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt
import ufl

from . import exceptions

logger = logging.getLogger(__name__)


class Marker(NamedTuple):
    name: str
    marker: int
    dim: int
    locator: typing.Callable[[npt.NDArray[np.float64]], bool]


class CardiacGeometriesObject(typing.Protocol):
    mesh: dolfinx.mesh.Mesh
    markers: dict[str, tuple[int, int]]

    # Read-only, so that both `ffun: MeshTags` and the `ffun: MeshTags | None` of
    # `cardiac_geometries.geometry.Geometry` satisfy it; it becomes the optional `facet_tags`.
    @property
    def ffun(self) -> dolfinx.mesh.MeshTags | None: ...


#: Quadrature degree of ``dx`` and ``ds`` when ``metadata`` names none, as the CLI's
#: ``geometry.quadrature_degree``. Without one, UFL estimates the degree from each integrand,
#: which for the exponentials, logarithms and inverses of cardiac models is very high: on a
#: 4x4x4 cube a 4-iteration Newton solve took 98 s instead of 0.15 s. Pass
#: ``metadata={"quadrature_degree": -1}`` to get that estimate anyway.
DEFAULT_QUADRATURE_DEGREE = 4

#: A cavity surface whose net (vector) area is below this fraction of its area has no opening.
CLOSED_CAVITY_TOLERANCE = 1e-10


def _coordinate_space(
    mesh: dolfinx.mesh.Mesh,
) -> tuple[dolfinx.fem.FunctionSpace, npt.NDArray[np.int32]]:
    """The space of the coordinate element of ``mesh``, whose dofs are its nodes, and the dof
    that each node is, by row of ``mesh.geometry.x``."""
    geometry = mesh.geometry
    # dolfinx 0.10 deprecated `dofmap` for the list of mixed-topology meshes
    dofmap = geometry.dofmaps[0] if hasattr(geometry, "dofmaps") else geometry.dofmap
    domain = mesh.ufl_domain()
    assert domain is not None
    V = dolfinx.fem.functionspace(mesh, domain.ufl_coordinate_element())
    dof = np.empty(len(geometry.x), dtype=np.int32)
    dof[dofmap.reshape(-1)] = V.dofmap.list.reshape(-1)
    return V, dof


def nodal_values(
    u: dolfinx.fem.Function,
    mesh: dolfinx.mesh.Mesh | None = None,
) -> npt.NDArray[np.floating]:
    """``u`` at the nodes of ``mesh``, by default its own, one row per row of
    ``mesh.geometry.x``.

    ``u`` may also live on another mesh that covers ``mesh``, such as a copy of it, whether
    partitioned the same way or not. Every process that holds a node gets the value its owner
    computes: evaluated on each process, in different cells, the copies of a node shared
    between processes can differ by round-off.
    """
    mesh = u.function_space.mesh if mesh is None else mesh
    V, dof = _coordinate_space(mesh)
    w = dolfinx.fem.Function(V)
    if u.function_space.mesh is mesh:
        w.interpolate(u)
    else:
        cells = np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32)
        w.interpolate_nonmatching(
            u,
            cells,
            dolfinx.fem.create_interpolation_data(V, u.function_space, cells),
        )
    w.x.scatter_forward()
    return w.x.array.reshape(-1, mesh.geometry.dim)[dof]


def move_nodes(mesh: dolfinx.mesh.Mesh, x: npt.NDArray[np.floating]) -> None:
    """Move the nodes of ``mesh`` to ``x``, one row per row of ``mesh.geometry.x``.

    Every process that holds a node puts it where its owner does, so all copies of a node end
    up at the same coordinates, bit for bit, even if they started apart. A mesh whose copies
    differ breaks anything that matches points across processes by their coordinates.
    """
    V, dof = _coordinate_space(mesh)
    gdim = mesh.geometry.dim
    w = dolfinx.fem.Function(V)
    values = w.x.array.reshape(-1, gdim)
    values[dof] = x
    w.x.scatter_forward()
    mesh.geometry.x[:, :gdim] = values[dof]


@dataclass(slots=True, kw_only=True)
class Geometry:
    mesh: dolfinx.mesh.Mesh
    boundaries: typing.Sequence[Marker] = ()
    # Form metadata of dx and ds; quadrature_degree defaults to DEFAULT_QUADRATURE_DEGREE
    metadata: dict[str, typing.Any] = field(default_factory=dict)
    _facet_indices: npt.NDArray[np.int32] = field(init=False, repr=False)
    _facet_markers: npt.NDArray[np.int32] = field(init=False, repr=False)
    _sorted_facets: npt.NDArray[np.int32] = field(init=False, repr=False)
    facet_tags: dolfinx.mesh.MeshTags | None = field(default=None, repr=False)
    markers: dict[str, tuple[int, int]] = field(default_factory=dict)
    dx: ufl.Measure = field(init=False, repr=False)
    ds: ufl.Measure = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.metadata = {"quadrature_degree": DEFAULT_QUADRATURE_DEGREE, **self.metadata}
        # Check if facet_tags are empty. If so, create them
        if self.facet_tags is None:
            facet_indices, facet_markers = [], []
            # TODO: Handle when dim is not 2
            for _, marker, dim, locator in self.boundaries:
                facets = dolfinx.mesh.locate_entities(self.mesh, dim, locator)
                facet_indices.append(facets)
                facet_markers.append(np.full_like(facets, marker))

            hstack = lambda x: np.array(x) if len(x) == 0 else np.hstack(x).astype(np.int32)
            self._facet_indices = hstack(facet_indices)
            self._facet_markers = hstack(facet_markers)
            self._sorted_facets = np.argsort(self._facet_indices).astype(np.int32)
            entities: npt.NDArray[np.int32] = (
                np.array([], dtype=np.int32)
                if len(self._sorted_facets) == 0
                else self._facet_indices[self._sorted_facets]
            )
            values: npt.NDArray[np.int32] = (
                np.array([], dtype=np.int32)
                if len(self._sorted_facets) == 0
                else self._facet_markers[self._sorted_facets]
            )
            self.facet_tags = dolfinx.mesh.meshtags(
                self.mesh,
                self.facet_dimension,
                entities,
                values,
            )
        if not self.markers:
            self.markers = dict((x[0], (x[1], x[2])) for x in self.boundaries)
        self.mesh.topology.create_connectivity(self.facet_dimension, self.dim)
        self._set_measures()
        logger.debug("Created Geometry with %d boundaries", len(self.markers))
        logger.debug("Markers: %s", ", ".join(self.markers.keys()))
        logger.debug("Metadata: %s", self.metadata)

    def _set_measures(self) -> None:
        self.dx = ufl.Measure("dx", domain=self.mesh, metadata=self.metadata)
        self.ds = ufl.Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=self.facet_tags,
            metadata=self.metadata,
        )

    def deform(self, u: dolfinx.fem.Function | None) -> None:
        """Deform the geometry by a displacement field.
        Note that this will modify the mesh geometry in place.

        Parameters
        ----------
        u : dolfinx.fem.Function
            Displacement field to deform the geometry with. It may live on a copy of the
            mesh.
        """
        if u is None:
            return
        if not isinstance(u, dolfinx.fem.Function):
            raise TypeError("Displacement field must be a dolfinx.fem.Function")

        # Every copy of a node moves to the same coordinates (see `move_nodes`)
        gdim = self.mesh.geometry.dim
        move_nodes(self.mesh, self.mesh.geometry.x[:, :gdim] + nodal_values(u, self.mesh))

    @classmethod
    def from_cardiac_geometries(
        cls,
        geo: CardiacGeometriesObject,
        metadata: dict[str, typing.Any] | None = None,
    ):
        metadata = metadata or {}
        return cls(mesh=geo.mesh, metadata=metadata, facet_tags=geo.ffun, markers=geo.markers)

    @property
    def facet_dimension(self) -> int:
        return self.mesh.topology.dim - 1

    @property
    def dim(self) -> int:
        return self.mesh.topology.dim

    def surface_area(self, marker: str) -> float:
        marker_id = self.markers[marker][0]
        return self.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.as_ufl(1.0) * self.ds(marker_id))),
            op=MPI.SUM,
        )

    def dump_mesh_tags(self, fname: str) -> None:
        assert self.facet_tags is not None
        if self.facet_tags.values.size == 0:
            raise exceptions.MeshTagNotFoundError
        self.mesh.topology.create_connectivity(self.facet_dimension, self.dim)

        with dolfinx.io.XDMFFile(
            self.mesh.comm,
            Path(fname).with_suffix(".xdmf"),
            "w",
        ) as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_meshtags(self.facet_tags, x=self.mesh.geometry)

    @property
    def facet_normal(self) -> ufl.FacetNormal:
        return ufl.FacetNormal(self.mesh)

    @property
    def X(self) -> ufl.SpatialCoordinate:
        return ufl.SpatialCoordinate(self.mesh)

    @property
    def comm(self) -> MPI.Comm:
        return self.mesh.comm


# Note slots doesn't work due to https://github.com/python/cpython/issues/90562
@dataclass(kw_only=True)
class HeartGeometry(Geometry):
    # Compiled forms of the surface integrals `_opening` needs, per cavity. The forms and not
    # their values: the mesh may move between two calls (`deform`, the fixed-point unloader).
    _opening_forms: dict[int | tuple[int, ...], list[dolfinx.fem.Form]] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )

    def base_center_form(
        self,
        base: str = "BASE",
        u: dolfinx.fem.Function | None = None,
    ) -> list[dolfinx.fem.forms.Form]:
        """Return the normal of the base

        Parameters
        ----------
        base : str, optional
            Marker for the base, by default "BASE"
        u : dolfinx.fem.Function | None, optional
            Displacement field, by default None

        Returns
        -------
        npt.NDArray[np.float64]
            Normal of the base
        """
        if u is None:
            b_vec = [(self.X[i]) * self.ds(self.markers[base][0]) for i in range(3)]
        else:
            b_vec = [(self.X[i] + u[i]) * self.ds(self.markers[base][0]) for i in range(3)]

        return dolfinx.fem.form(b_vec)

    def base_center(
        self,
        base: str = "BASE",
        u: dolfinx.fem.Function | None = None,
        dtype=np.float64,
    ) -> npt.NDArray[np.float64]:
        """Return the normal of the base

        Parameters
        ----------
        base : str, optional
            Marker for the base, by default "BASE"
        u : dolfinx.fem.Function | None, optional
            Displacement field, by default None

        Returns
        -------
        npt.NDArray[np.float64]
            Normal of the base
        """
        forms = self.base_center_form(base=base, u=u)
        base_area = self.surface_area(base)
        return np.array(
            [dolfinx.fem.assemble_scalar(bi) / base_area for bi in forms],
            dtype=dtype,
        )

    def _opening(
        self,
        marker_id: int | tuple[int, ...],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        r"""Unit normal and centre of the opening of the cavity bounded by ``marker_id``.

        Both come from the cavity's own surface :math:`\Gamma`, whose normal
        :math:`\mathbf{N}` points into the cavity. A closed surface has no net area, so
        :math:`\int_\Gamma \mathbf{N}\, dS = A \mathbf{a}` is the vector area of the lid that
        would close the cavity, and by the divergence theorem
        :math:`\int_\Gamma \mathbf{X} (\mathbf{N} \cdot \mathbf{a})\, dS` is, in the plane
        of the lid, :math:`A` times the lid's centroid. Returns ``None`` when the surface has
        no opening.
        """
        forms = self._opening_forms.get(marker_id)
        if forms is None:
            X, N, ds = self.X, self.facet_normal, self.ds(marker_id)
            integrands = [ufl.as_ufl(1.0), *(N[j] for j in range(3))]
            integrands += [X[i] * N[j] for i in range(3) for j in range(3)]
            forms = self._opening_forms[marker_id] = dolfinx.fem.form([f * ds for f in integrands])
        local = np.array([dolfinx.fem.assemble_scalar(form) for form in forms]).real
        values = self.mesh.comm.allreduce(local, op=MPI.SUM)
        area, vector_area, moment = values[0], values[1:4], values[4:].reshape(3, 3)

        opening_area = np.linalg.norm(vector_area)
        if opening_area <= CLOSED_CAVITY_TOLERANCE * area:
            return None
        normal = vector_area / opening_area
        center = moment @ normal / opening_area
        center -= np.dot(center, normal) * normal
        return normal, center

    def volume_form(
        self,
        u: dolfinx.fem.Function | None = None,
        marker: str | typing.Sequence[str] | None = None,
    ) -> ufl.core.expr.Expr:
        r"""Return the integrand whose integral over the surface ``marker`` is the volume of
        that cavity.

        A ventricle's cavity is open at the base, so its volume is that of the cavity closed
        by a flat lid. With :math:`\mathbf{a}` the unit normal of the lid and
        :math:`\mathbf{b}` its centre, the field
        :math:`\boldsymbol{\xi} = (\mathbf{I} - \mathbf{a} \otimes \mathbf{a})
        (\mathbf{x} - \mathbf{b})` has divergence 2 and no flux through the lid, so

        .. math::
            V = -\frac{1}{2} \int_\Gamma \boldsymbol{\xi} \cdot \mathbf{n}\, da ,
            \qquad \mathbf{n}\, da = J \mathbf{F}^{-T} \mathbf{N}\, dA ,

        with :math:`\mathbf{n}` pointing into the cavity. :math:`\mathbf{a}` and
        :math:`\mathbf{b}` are those of the mesh as it is when this is called (see
        :meth:`_opening`) and do not follow ``u``. The volume then does not depend on where
        the cavity is: it is the same wherever the origin is and however the mesh is turned,
        a rigid translation by ``u`` leaves it unchanged, and so does a rigid rotation about
        the centre of the opening. The cavity pressure is conjugate to this volume, so it
        exerts no net force on the wall, as if the lid carried its share.

        It is the volume closed by a flat lid while the opening stays normal to
        :math:`\mathbf{a}`. An opening of area :math:`A` that tilts by :math:`\theta` while
        its centre moves a distance :math:`d` in its own plane adds an error of at most
        :math:`\frac{1}{2} A d \sin\theta`.

        A surface with no opening gets :math:`-\frac{1}{3} \mathbf{x} \cdot \mathbf{n}`,
        which is exact for it.

        Parameters
        ----------
        u : dolfinx.fem.Function | None, optional
            Optional displacement field, by default None
        marker : str | typing.Sequence[str] | None, optional
            Marker (or markers) of the surface of the cavity. Without it the integrand is
            :math:`-\frac{1}{3} \mathbf{x} \cdot \mathbf{n}` for any surface, which measures
            an open cavity from the origin of the coordinates (deprecated).

        Returns
        -------
        ufl.core.expr.Expr
            The integrand, to be multiplied by ``ds`` of the cavity's marker

        Raises
        ------
        exceptions.MarkerNotFoundError
            If the marker is not found in the geometry
        """
        if u is None:
            x, n = self.X, self.facet_normal
        else:
            F = ufl.Identity(3) + ufl.grad(u)
            x, n = self.X + u, ufl.det(F) * ufl.inv(F).T * self.facet_normal

        if marker is None:
            warnings.warn(
                "volume_form without a marker measures the cavity from the origin of the "
                "coordinates, so the volume changes when the cavity moves. Pass the marker "
                "of the cavity's surface.",
                DeprecationWarning,
                stacklevel=2,
            )
            opening = None
        else:
            opening = self._opening(self.get_marker_ids(marker))
        if opening is None:
            return (-1 / 3) * ufl.dot(x, n)

        a, b = (
            dolfinx.fem.Constant(self.mesh, value.astype(dolfinx.default_scalar_type))
            for value in opening
        )
        xi = x - b
        return (-1 / 2) * ufl.dot(xi - ufl.dot(xi, a) * a, n)

    def volume(
        self,
        marker: str | typing.Sequence[str],
        u: dolfinx.fem.Function | None = None,
    ) -> float:
        """Return the volume of the cavity for a given marker, closed by a flat lid over its
        opening; see :meth:`volume_form`.

        Parameters
        ----------
        marker : str
            Marker for the surface of the cavity
        u : dolfinx.fem.Function | None, optional
            Optional displacement field, by default None

        Returns
        -------
        float
            Volume of the cavity

        Raises
        ------
        exceptions.MarkerNotFoundError
            If the marker is not found in the geometry
        """

        marker_id = self.get_marker_ids(marker)
        form = self.volume_form(u=u, marker=marker)
        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(form * self.ds(marker_id))).real

    def get_marker_ids(
        self,
        marker: str | typing.Sequence[str],
    ) -> int | tuple[int, ...]:
        if isinstance(marker, str):
            if marker not in self.markers:
                raise exceptions.MarkerNotFoundError(marker)
            return self.markers[marker][0]
        else:
            ids = []
            for m in marker:
                if m not in self.markers:
                    raise exceptions.MarkerNotFoundError(m)
                ids.append(self.markers[m][0])
            return tuple(ids)
