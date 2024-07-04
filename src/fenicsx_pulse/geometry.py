import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

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


@dataclass(slots=True)
class Geometry:
    mesh: dolfinx.mesh.Mesh
    boundaries: typing.Sequence[Marker] = ()
    metadata: dict[str, typing.Any] = field(default_factory=dict)
    _facet_indices: npt.NDArray[np.int32] = field(init=False, repr=False)
    _facet_markers: npt.NDArray[np.int32] = field(init=False, repr=False)
    _sorted_facets: npt.NDArray[np.int32] = field(init=False, repr=False)
    facet_tags: dolfinx.mesh.MeshTags = field(init=False, repr=False)
    markers: dict[str, tuple[int, int]] = field(init=False)
    dx: ufl.Measure = field(init=False, repr=False)
    ds: ufl.Measure = field(init=False, repr=False)

    def __post_init__(self) -> None:
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
        entities = [] if len(self._sorted_facets) == 0 else self._facet_indices[self._sorted_facets]
        values = [] if len(self._sorted_facets) == 0 else self._facet_markers[self._sorted_facets]
        self.facet_tags = dolfinx.mesh.meshtags(
            self.mesh,
            self.facet_dimension,
            entities,
            values,
        )
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
        obj = cls(mesh=geo.mesh, metadata=metadata)
        obj.facet_tags = geo.ffun
        obj.markers = geo.markers
        obj._set_measures()
        return obj

    @property
    def facet_dimension(self) -> int:
        return self.mesh.topology.dim - 1

    @property
    def dim(self) -> int:
        return self.mesh.topology.dim

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
