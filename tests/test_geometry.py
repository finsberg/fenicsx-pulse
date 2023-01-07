import fenicsx_pulse
import numpy as np
import pytest
import ufl


def test_geometry_empty_initialization(mesh):
    geo = fenicsx_pulse.Geometry(mesh=mesh)
    assert geo.facet_tags.values.size == 0
    assert geo.facet_tags.indices.size == 0
    assert geo._facet_indices.size == 0
    assert geo._facet_markers.size == 0
    assert geo._sorted_facets.size == 0
    assert geo.facet_dimension == 2
    assert geo.dim == 3
    assert geo.markers == ()
    assert geo.dx == ufl.Measure("dx", domain=mesh)
    assert geo.ds == ufl.Measure("ds", domain=mesh, subdomain_data=geo.facet_tags)


def test_geometry_with_boundary_and_metadata(mesh):
    boundaries = [
        (1, 2, lambda x: np.isclose(x[0], 0)),
        (2, 2, lambda x: np.isclose(x[0], 1)),
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
    assert geo.markers == (1, 2)

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
        boundaries=[(1, 2, lambda x: np.isclose(x[0], 0))],
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
