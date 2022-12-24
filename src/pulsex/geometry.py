import typing
from dataclasses import dataclass
from dataclasses import field
from typing import NamedTuple

import dolfinx
import numpy as np
import ufl


class Marker(NamedTuple):
    marker: int
    dim: int
    locator: typing.Callable[[np.ndarray[np.float64]], bool]


@dataclass(slots=True)
class Geometry:
    mesh: dolfinx.mesh.Mesh
    boundaries: typing.Sequence[Marker] = ()
    metadata: dict[str, typing.Any] = field(default_factory=dict)
    _facet_indices: np.ndarray[np.int32] = field(init=False, repr=False)
    _facet_markers: np.ndarray[np.int32] = field(init=False, repr=False)
    _sorted_facets: np.ndarray[np.int32] = field(init=False, repr=False)
    facet_tags: dolfinx.mesh.MeshTagsMetaClass = field(init=False, repr=False)
    dx: ufl.Measure = field(init=False, repr=False)
    ds: ufl.Measure = field(init=False, repr=False)

    def __post_init__(self) -> None:
        facet_indices, facet_markers = [], []

        # TODO: Handle when dim is not 2
        for (marker, dim, locator) in self.boundaries:
            facets = dolfinx.mesh.locate_entities(self.mesh, dim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))
        self._facet_indices = np.hstack(facet_indices).astype(np.int32)
        self._facet_markers = np.hstack(facet_markers).astype(np.int32)
        self._sorted_facets = np.argsort(self._facet_indices)

        self.facet_tags = dolfinx.mesh.meshtags(
            self.mesh,
            self.facet_dimension,
            self._facet_indices[self._sorted_facets],
            self._facet_markers[self._sorted_facets],
        )
        self.dx = ufl.Measure("dx", domain=self.mesh, metadata=self.metadata)
        self.ds = ufl.Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=self.facet_tags,
            metadata=self.metadata,
        )

    @property
    def facet_dimension(self) -> int:
        return self.mesh.topology.dim - 1

    @property
    def dim(self) -> int:
        return self.mesh.topology.dim

    def dump_mesh_tags(self, fname: str) -> None:
        self.mesh.topology.create_connectivity(self.facet_dimension, self.dim)
        with dolfinx.io.XDMFFile(self.mesh.comm, fname, "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_meshtags(self.facet_tags)

    @property
    def markers(self) -> tuple[int, ...]:
        return tuple(x[0] for x in self.boundaries)

    @property
    def facet_normal(self) -> ufl.FacetNormal:
        return ufl.FacetNormal(self.mesh)
