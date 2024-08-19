import numpy as np
import pytest
import ufl

import cardiac_geometries
import fenicsx_pulse


def test_geometry_empty_initialization(mesh):
    geo = fenicsx_pulse.Geometry(mesh=mesh)
    assert geo.facet_tags.values.size == 0
    assert geo.facet_tags.indices.size == 0
    assert geo._facet_indices.size == 0
    assert geo._facet_markers.size == 0
    assert geo._sorted_facets.size == 0
    assert geo.facet_dimension == 2
    assert geo.dim == 3
    assert geo.markers == {}
    assert geo.dx == ufl.Measure("dx", domain=mesh)
    assert geo.ds == ufl.Measure("ds", domain=mesh, subdomain_data=geo.facet_tags)


def test_geometry_with_boundary_and_metadata(mesh):
    boundaries = [
        ("left", 1, 2, lambda x: np.isclose(x[0], 0)),
        ("right", 2, 2, lambda x: np.isclose(x[0], 1)),
    ]
    metadata = {"quadrature_degree": 4}
    geo = fenicsx_pulse.Geometry(
        mesh=mesh,
        boundaries=boundaries,
        metadata=metadata,
    )
    assert set(geo.facet_tags.values) == {1, 2}
    assert geo._facet_indices.size == 36
    assert geo._facet_markers.size == 36
    assert geo._sorted_facets.size == 36
    assert geo.markers == {"left": (1, 2), "right": (2, 2)}

    assert geo.dx == ufl.Measure("dx", domain=mesh, metadata=metadata)
    assert geo.ds == ufl.Measure(
        "ds",
        domain=mesh,
        subdomain_data=geo.facet_tags,
        metadata=metadata,
    )


def test_dump_mesh_tags(tmp_path, mesh):
    geo = fenicsx_pulse.Geometry(
        mesh=mesh,
        boundaries=[("marker", 1, 2, lambda x: np.isclose(x[0], 0))],
    )
    path = tmp_path.with_suffix(".xdmf")
    assert not path.is_file()
    geo.dump_mesh_tags(path)
    assert path.is_file()
    assert path.with_suffix(".h5").is_file()


def test_dump_mesh_tags_raises_MeshTagNotFoundError(tmp_path, mesh):
    geo = fenicsx_pulse.Geometry(mesh=mesh)
    with pytest.raises(fenicsx_pulse.exceptions.MeshTagNotFoundError):
        geo.dump_mesh_tags(tmp_path)


def test_geometry_from_cardiac_geometries(tmp_path):
    geo1 = cardiac_geometries.mesh.lv_ellipsoid(outdir=tmp_path)
    geo2 = fenicsx_pulse.Geometry.from_cardiac_geometries(geo1)

    assert geo1.mesh is geo2.mesh
    assert geo1.markers is geo2.markers
    assert geo1.ffun is geo2.facet_tags
