import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt
import ufl

from . import exceptions


class Marker(NamedTuple):
    name: str
    marker: int
    dim: int
    locator: typing.Callable[[npt.NDArray[np.float64]], bool]


class CardiacGeometriesObject(typing.Protocol):
    mesh: dolfinx.mesh.Mesh
    ffun: dolfinx.mesh.MeshTags
    markers: dict[str, tuple[int, int]]


@dataclass(slots=True, kw_only=True)
class Geometry:
    mesh: dolfinx.mesh.Mesh
    boundaries: typing.Sequence[Marker] = ()
    metadata: dict[str, typing.Any] = field(default_factory=dict)
    _facet_indices: npt.NDArray[np.int32] = field(init=False, repr=False)
    _facet_markers: npt.NDArray[np.int32] = field(init=False, repr=False)
    _sorted_facets: npt.NDArray[np.int32] = field(init=False, repr=False)
    facet_tags: dolfinx.mesh.MeshTags = field(
        default_factory=lambda: dolfinx.mesh.MeshTags([]),
        repr=False,
    )
    markers: dict[str, tuple[int, int]] = field(default_factory=dict)
    dx: ufl.Measure = field(init=False, repr=False)
    ds: ufl.Measure = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Check if facet_tags are empty. If so, create them
        if not hasattr(self.facet_tags, "values"):
            facet_indices, facet_markers = [], []
            # TODO: Handle when dim is not 2
            for _, marker, dim, locator in self.boundaries:
                facets = dolfinx.mesh.locate_entities(self.mesh, dim, locator)
                facet_indices.append(facets)
                facet_markers.append(np.full_like(facets, marker))

            hstack = lambda x: np.array(x) if len(x) == 0 else np.hstack(x).astype(np.int32)
            self._facet_indices = hstack(facet_indices)
            self._facet_markers = hstack(facet_markers)
            self._sorted_facets = np.argsort(self._facet_indices)
            entities = (
                [] if len(self._sorted_facets) == 0 else self._facet_indices[self._sorted_facets]
            )
            values = (
                [] if len(self._sorted_facets) == 0 else self._facet_markers[self._sorted_facets]
            )
            self.facet_tags = dolfinx.mesh.meshtags(
                self.mesh,
                self.facet_dimension,
                entities,
                values,
            )
        if not self.markers:
            self.markers = dict((x[0], (x[1], x[2])) for x in self.boundaries)
        self._set_measures()

    def _set_measures(self) -> None:
        self.dx = ufl.Measure("dx", domain=self.mesh, metadata=self.metadata)
        self.ds = ufl.Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=self.facet_tags,
            metadata=self.metadata,
        )

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
    def comm(self) -> MPI.Intracomm:
        return self.mesh.comm


class BaseData(NamedTuple):
    centroid: npt.NDArray[np.float64]
    vector: npt.NDArray[np.float64]
    normal: npt.NDArray[np.float64]


def compute_base_data(
    mesh: dolfinx.mesh.Mesh,
    facet_tags: dolfinx.mesh.MeshTags,
    marker,
) -> BaseData:
    """Compute the centroid, vector and normal of the base

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh
    facet_tags : dolfinx.mesh.MeshTags
        The facet tags
    marker : _type_
        Marker for the base

    Returns
    -------
    BaseData
        NamedTuple containing the centroid, vector and normal of the base
    """
    base_facets = facet_tags.find(marker)
    base_midpoints = mesh.comm.gather(
        dolfinx.mesh.compute_midpoints(mesh, 2, base_facets),
        root=0,
    )
    base_vector = np.zeros(3)
    base_centroid = np.zeros(3)
    base_normal = np.zeros(3)
    if mesh.comm.rank == 0:
        bm = np.concatenate(base_midpoints)
        base_centroid = bm.mean(axis=0)
        # print("Base centroid", len(base_midpoints))
        base_points_centered = bm - base_centroid
        u, s, vh = np.linalg.svd(base_points_centered)
        base_normal = vh[-1, :]
        # Initialize vector to be used for cross product
        vector_init = np.array([0, 1, 0])

        # If the normal is parallel to the initial vector, change the initial vector
        if np.allclose(np.abs(base_normal), np.abs(vector_init)):
            vector_init = np.array([0, 0, 1])

        # Find two vectors in the plane, orthogonal to the normal
        vector = np.cross(base_normal, vector_init)
        base_vector = np.cross(base_normal, vector)

    base_centroid = mesh.comm.bcast(base_centroid, root=0)
    base_vector = mesh.comm.bcast(base_vector, root=0)
    base_normal = mesh.comm.bcast(base_normal, root=0)
    return BaseData(centroid=base_centroid, vector=base_vector, normal=base_normal)


# Note slots doesn't work due to https://github.com/python/cpython/issues/90562
@dataclass(kw_only=True)
class HeartGeometry(Geometry):
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

    def volume_form(
        self,
        u: dolfinx.fem.Function | None = None,
        b: ufl.Coefficient = ufl.as_vector([0.0, 0.0, 0.0]),
    ) -> dolfinx.fem.forms.Form:
        """Return the form for the volume of the cavity
        for a given marker

        Parameters
        ----------
        u : dolfinx.fem.Function | None, optional
            Optional displacement field, by default None
        base : str, optional
            Marker for the base, by default "BASE"

        Returns
        -------
        dolfinx.fem.forms.Form
            The form for the volume of the cavity

        Raises
        ------
        exceptions.MarkerNotFoundError
            If the marker is not found in the geometry
        """
        X = self.X

        if u is None:
            return (-1 / 3) * ufl.dot(X - b, self.facet_normal)
        else:
            F = ufl.Identity(3) + ufl.grad(u)
            J = ufl.det(F)
            return (-1 / 3) * J * ufl.dot(X + u - b, ufl.inv(F).T * self.facet_normal)

    def volume(
        self,
        marker: str,
        u: dolfinx.fem.Function | None = None,
        base: str = "BASE",
    ) -> float:
        """Return the volume of the cavity for a given marker

        Parameters
        ----------
        marker : str
            Marker for the surface of the cavity
        u : dolfinx.fem.Function | None, optional
            Optional displacement field, by default None
        base : str, optional
            Marker for the base, by default "BASE"

        Returns
        -------
        float
            Volume of the cavity

        Raises
        ------
        exceptions.MarkerNotFoundError
            If the marker is not found in the geometry
        """
        if marker not in self.markers:
            raise exceptions.MarkerNotFoundError(marker)
        marker_id = self.markers[marker][0]

        if marker not in self.markers:
            raise exceptions.MarkerNotFoundError(marker)
        marker_id = self.markers[marker][0]
        b = ufl.as_vector(self.base_center(base=base, u=u))

        form = self.volume_form(u=u, b=b)
        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(form * self.ds(marker_id)))
