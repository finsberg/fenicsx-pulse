import math

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import cardiac_geometries
import pulse


def test_geometry_empty_initialization(mesh):
    geo = pulse.Geometry(mesh=mesh)
    assert geo.facet_tags.values.size == 0
    assert geo.facet_tags.indices.size == 0
    assert geo._facet_indices.size == 0
    assert geo._facet_markers.size == 0
    assert geo._sorted_facets.size == 0
    assert geo.facet_dimension == 2
    assert geo.dim == 3
    assert geo.markers == {}
    md = {"quadrature_degree": pulse.geometry.DEFAULT_QUADRATURE_DEGREE}
    assert geo.dx == ufl.Measure("dx", domain=mesh, metadata=md)
    assert geo.ds == ufl.Measure("ds", domain=mesh, subdomain_data=geo.facet_tags, metadata=md)


@pytest.mark.parametrize(
    "metadata, degree",
    [
        ({}, pulse.geometry.DEFAULT_QUADRATURE_DEGREE),
        ({"quadrature_rule": "default"}, pulse.geometry.DEFAULT_QUADRATURE_DEGREE),
        ({"quadrature_degree": 2}, 2),
        ({"quadrature_degree": -1}, -1),  # -1: let UFL estimate the degree
    ],
)
def test_geometry_default_quadrature_degree(mesh, metadata, degree):
    geo = pulse.Geometry(mesh=mesh, metadata=dict(metadata))
    assert geo.metadata == {**metadata, "quadrature_degree": degree}
    assert geo.dx.metadata()["quadrature_degree"] == degree
    assert geo.ds.metadata()["quadrature_degree"] == degree


def test_geometry_with_boundary_and_metadata(mesh):
    boundaries = [
        ("left", 1, 2, lambda x: np.isclose(x[0], 0)),
        ("right", 2, 2, lambda x: np.isclose(x[0], 1)),
    ]
    metadata = {"quadrature_degree": 4}
    geo = pulse.Geometry(
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
    geo = pulse.Geometry(
        mesh=mesh,
        boundaries=[("marker", 1, 2, lambda x: np.isclose(x[0], 0))],
    )
    path = tmp_path.with_suffix(".xdmf")
    assert not path.is_file()
    geo.dump_mesh_tags(path)
    assert path.is_file()
    assert path.with_suffix(".h5").is_file()


def test_dump_mesh_tags_raises_MeshTagNotFoundError(tmp_path, mesh):
    geo = pulse.Geometry(mesh=mesh)
    with pytest.raises(pulse.exceptions.MeshTagNotFoundError):
        geo.dump_mesh_tags(tmp_path)


def test_geometry_from_cardiac_geometries(tmp_path):
    geo1 = cardiac_geometries.mesh.lv_ellipsoid(outdir=tmp_path)
    geo2 = pulse.Geometry.from_cardiac_geometries(geo1)

    assert geo1.mesh is geo2.mesh
    assert geo1.markers is geo2.markers
    assert geo1.ffun is geo2.facet_tags


def rotate_geo(geo, theta):
    rotate = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
    )
    translate = geo.mesh.geometry.x.mean(axis=0)
    geo.mesh.geometry.x[:] = geo.mesh.geometry.x - translate
    geo.mesh.geometry.x[:] = geo.mesh.geometry.x @ rotate.T
    geo.mesh.geometry.x[:] = geo.mesh.geometry.x + translate
    return geo


#: An LV ellipsoid with its long axis along x. The endocardium ends on the circle
#: x = r_long_endo * cos(mu_base_endo) = 5 around the axis, so the cavity's opening is a
#: plane that does not pass through the origin.
LV_ELLIPSOID = dict(
    r_short_endo=6.0,
    r_short_epi=10.0,
    r_long_endo=17.0,
    r_long_epi=20.0,
    psize_ref=3,
    mu_apex_endo=-math.pi,
    mu_base_endo=-math.acos(5 / 17),
    mu_apex_epi=-math.pi,
    mu_base_epi=-math.acos(5 / 20),
)
LV_OPENING_CENTER = (5.0, 0.0, 0.0)


def test_HeartGeometry_lv(tmp_path):
    geo1 = cardiac_geometries.mesh.lv_ellipsoid(outdir=tmp_path, **LV_ELLIPSOID)
    geo2 = pulse.HeartGeometry.from_cardiac_geometries(geo1)

    endo_volume = 1772.9570488626932
    assert np.isclose(geo2.volume("ENDO"), endo_volume)

    # Now we rotate the geometry
    rotate_geo(geo2, np.pi)

    # But volume should be the same
    assert np.isclose(geo2.volume("ENDO"), endo_volume, atol=1e-7)


def test_HeartGeometry_biv(tmp_path):
    geo1 = cardiac_geometries.mesh.biv_ellipsoid(
        outdir=tmp_path,
    )
    geo2 = pulse.HeartGeometry.from_cardiac_geometries(geo1)

    endo_lv_volume = 32.549231806345155
    assert np.isclose(geo2.volume(["LV"]), endo_lv_volume, rtol=0.05)
    endo_rv_volume = 18.204026065014258
    assert np.isclose(geo2.volume(["RV"]), endo_rv_volume, rtol=0.05)


# --- The cavity volume does not depend on where the cavity is ----------------------------------


@pytest.fixture(scope="module")
def lv(tmp_path_factory):
    geo = cardiac_geometries.mesh.lv_ellipsoid(
        outdir=tmp_path_factory.mktemp("lv"),
        **LV_ELLIPSOID,
    )
    return pulse.HeartGeometry.from_cardiac_geometries(geo)


def _integral(geometry, integrand, marker="ENDO"):
    form = dolfinx.fem.form(integrand * geometry.ds(geometry.markers[marker][0]))
    return geometry.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)


def _volume(geometry, u=None, marker="ENDO"):
    return geometry.mesh.comm.allreduce(geometry.volume(marker, u=u), op=MPI.SUM)


def _rotation(axis, angle):
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _affine(geometry, gradient=np.zeros((3, 3)), center=(0, 0, 0), translation=(0, 0, 0)):
    """The displacement ``gradient (X - center) + translation`` as a function."""
    V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 1, (3,)))
    u = dolfinx.fem.Function(V)
    c, t = np.reshape(center, (3, 1)), np.reshape(translation, (3, 1))
    u.interpolate(lambda x: np.asarray(gradient) @ (x - c) + t)
    return u


def test_cavity_volume_is_the_cavity_closed_by_a_flat_lid(lv):
    # (X_0 - 5, 0, 0) is another field of divergence 1 with no flux through the plane x = 5
    X, N = lv.X, lv.facet_normal
    closed = _integral(lv, -(X[0] - LV_OPENING_CENTER[0]) * N[0])

    assert np.isclose(_volume(lv), closed, rtol=1e-12)

    with pytest.warns(DeprecationWarning):
        from_origin = _integral(lv, lv.volume_form())
    # Measured from the origin, the cone between the origin and the opening is missing
    opening_area = _integral(lv, N[0])
    assert np.isclose(from_origin, closed - LV_OPENING_CENTER[0] * opening_area / 3, rtol=1e-12)


def test_cavity_volume_does_not_depend_on_where_the_mesh_is(lv):
    volume = _volume(lv)
    x = lv.mesh.geometry.x
    reference = x.copy()
    try:
        x[:] = reference @ _rotation((0.3, -0.5, 0.8), 1.1).T + np.array([30.0, -10.0, 20.0])
        assert np.isclose(_volume(lv), volume, rtol=1e-12)
        with pytest.warns(DeprecationWarning):
            from_origin = _integral(lv, lv.volume_form())
        assert not np.isclose(from_origin, volume, rtol=0.1)
    finally:
        x[:] = reference


@pytest.mark.parametrize("translation", [(-1.5, 0.0, 0.0), (0.0, 2.0, 0.0), (1.0, -2.0, 3.0)])
def test_cavity_volume_is_unchanged_by_a_rigid_translation(lv, translation):
    u = _affine(lv, translation=translation)

    assert np.isclose(_volume(lv, u), _volume(lv), rtol=1e-12)

    # From the origin, moving the opening by d along its normal changes the volume by -d A / 3
    with pytest.warns(DeprecationWarning):
        from_origin = [_integral(lv, lv.volume_form(v)) for v in (None, u)]
    opening_area = _integral(lv, lv.facet_normal[0])
    assert np.isclose(
        from_origin[1] - from_origin[0],
        -translation[0] * opening_area / 3,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "axis, angle",
    [((1.0, 0.0, 0.0), 0.7), ((0.0, 1.0, 0.0), 0.3), ((0.3, -0.5, 0.8), 1.1)],
)
def test_cavity_volume_is_unchanged_by_a_rotation_about_its_opening(lv, axis, angle):
    u = _affine(lv, gradient=_rotation(axis, angle) - np.eye(3), center=LV_OPENING_CENTER)

    assert np.isclose(_volume(lv, u), _volume(lv), rtol=1e-10)


@pytest.mark.parametrize("widening", [0.0, 0.1])
def test_cavity_pressure_exerts_no_net_force_or_moment(lv, widening):
    """The pressure is conjugate to the volume, so the derivative of the volume along a
    rigid motion is the force (or moment) a unit pressure exerts on the wall. Checked at
    rest and with the cavity widened in the plane of its opening.
    """
    grad_u = widening * np.diag([0.0, 1.0, 1.0])
    u = _affine(lv, gradient=grad_u, center=LV_OPENING_CENTER)
    ds = lv.ds(lv.markers["ENDO"][0])
    volume = lv.volume_form(u, marker="ENDO") * ds
    with pytest.warns(DeprecationWarning):
        from_origin = lv.volume_form(u) * ds
    area = lv.surface_area("ENDO")

    def derivative(form, direction):
        return lv.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.derivative(form, u, direction))),
            op=MPI.SUM,
        )

    for e in np.eye(3):
        assert abs(derivative(volume, _affine(lv, translation=e))) < 1e-10 * area
        # an infinitesimal rotation about e of the deformed cavity, through its opening
        spin = np.cross(e, np.eye(3)).T @ (np.eye(3) + grad_u)
        rotation = _affine(lv, gradient=spin, center=LV_OPENING_CENTER)
        assert abs(derivative(volume, rotation)) < 1e-10 * area * LV_OPENING_CENTER[0]

    # From the origin, a third of the pressure on the lid is left pushing along the axis
    opening_area = (1 + widening) ** 2 * _integral(lv, lv.facet_normal[0])
    along_axis = derivative(from_origin, _affine(lv, translation=(1.0, 0.0, 0.0)))
    assert np.isclose(along_axis, -opening_area / 3, rtol=1e-10)


def test_closed_surface_has_no_opening(mesh):
    everywhere = pulse.Marker("ALL", 1, 2, lambda x: np.full(x.shape[1], True))
    geo = pulse.HeartGeometry(mesh=mesh, boundaries=[everywhere])

    assert geo._opening(1) is None
    # The normal points out of the cube, so it is the cavity around it that is measured
    assert np.isclose(_volume(geo, marker="ALL"), -1.0)


def test_flat_surface_encloses_no_volume(mesh):
    face = pulse.Marker("FACE", 1, 2, lambda x: np.isclose(x[0], 1.0))
    geo = pulse.HeartGeometry(mesh=mesh, boundaries=[face])

    assert np.isclose(_volume(geo, marker="FACE"), 0.0, atol=1e-14)


def test_deform_moves_every_copy_of_a_node_alike():
    """A node shared between processes has a copy on each. `deform` must move them all to
    the same coordinates, bit for bit, and each by the displacement at the node."""
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD,
        4,
        4,
        4,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    )
    geometry = pulse.Geometry(mesh=mesh)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))
    u = dolfinx.fem.Function(V)

    def displacement(x):
        return 0.1 * np.vstack((np.sin(3 * x[1]), x[0] * x[2], np.cos(2 * x[0]) * x[1]))

    u.interpolate(displacement)
    x0 = mesh.geometry.x.copy()
    geometry.deform(u)
    # The vertices are dofs of u, so u is exact there
    np.testing.assert_allclose(mesh.geometry.x, x0 + displacement(x0.T).T, rtol=0, atol=1e-14)

    ids = np.concatenate(mesh.comm.allgather(np.asarray(mesh.geometry.input_global_indices)))
    x = np.concatenate(mesh.comm.allgather(mesh.geometry.x))
    order = np.argsort(ids, kind="stable")
    ids, x = ids[order], x[order]
    first = np.searchsorted(ids, ids)
    np.testing.assert_array_equal(x, x[first])
