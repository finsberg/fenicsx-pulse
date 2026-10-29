"""MeshSurface: a smooth surface through marked facets, for a PericardiumBC.

The surface itself is numbers: its distance must be the distance to the surface the facets
were sampled from, with the gradient and the Hessian of that distance, and they must be the
derivatives of each other at any distance from it.

In a problem it is seen through a TrackedSurface, a quadratic expansion about where the
boundary is, integrated over the nodes. Right after an update the forms must be those of the
surface itself, and a solve must update before every evaluation. Both are checked exactly
against the EllipsoidSurface, by feeding its distance through the same machinery as numbers.
With the MeshSurface of the epicardium a solve must then give what the ellipsoid gives, up
to how well the facets approximate it.
"""

import logging
import math

from mpi4py import MPI

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import pytest
import scifem
import ufl

import pulse

cardiac_geometries = pytest.importorskip("cardiac_geometries")

K = 1e6  # Pa/m
R_LONG, R_SHORT = 0.097, 0.035


def _sphere(n, radius=1.0, centre=(0.0, 0.0, 0.0)):
    """n points spread evenly over a sphere, their normals and the area each stands for."""
    i = np.arange(n) + 0.5
    polar = np.arccos(1 - 2 * i / n)
    azimuth = np.pi * (1 + 5**0.5) * i
    normals = np.column_stack(
        (np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)),
    )
    areas = np.full(n, 4 * np.pi * radius**2 / n)
    return np.asarray(centre) + radius * normals, normals, areas


def _directions(n, seed=0):
    v = np.random.default_rng(seed).standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def _finite_differences(surface, x, eps=1e-6):
    gradient = np.zeros((len(x), 3))
    hessian = np.zeros((len(x), 3, 3))
    for j in range(3):
        step = np.zeros(3)
        step[j] = eps
        d_plus, g_plus, _ = surface.evaluate(x + step, hessian=False)
        d_minus, g_minus, _ = surface.evaluate(x - step, hessian=False)
        gradient[:, j] = (d_plus - d_minus) / (2 * eps)
        hessian[:, :, j] = (g_plus - g_minus) / (2 * eps)
    return gradient, hessian


def test_a_plane_is_exact():
    """Every facet has the same plane, so the distance is the height above it."""
    xy = np.stack(np.meshgrid(np.arange(10.0), np.arange(10.0)), axis=-1).reshape(-1, 2)
    points = np.column_stack((xy, np.zeros(len(xy))))
    normals = np.tile([0.0, 0.0, 1.0], (len(xy), 1))
    surface = pulse.MeshSurface(points, normals, np.ones(len(xy)), radius=2.5)

    rng = np.random.default_rng(1)
    x = np.column_stack((rng.uniform(0, 9, (50, 2)), rng.uniform(-1, 1, 50)))
    d, gradient, hessian = surface.evaluate(x)
    np.testing.assert_allclose(d, x[:, 2], atol=1e-12)
    np.testing.assert_allclose(gradient, np.tile([0.0, 0.0, 1.0], (50, 1)), atol=1e-12)
    np.testing.assert_allclose(hessian, 0.0, atol=1e-11)


def test_a_sphere_gives_its_distance_normal_and_curvature():
    radius = 2.0
    centre = np.array([0.3, -0.2, 0.1])
    points, normals, areas = _sphere(4000, radius, centre)
    # Three times the spacing of the points, as from_facets takes it
    surface = pulse.MeshSurface(points, normals, areas, radius=0.35)

    offsets = np.linspace(-0.05, 0.05, 200)
    radial = _directions(200)
    x = centre + (radius + offsets)[:, None] * radial
    d, gradient, hessian = surface.evaluate(x)

    np.testing.assert_allclose(d, offsets, atol=2e-4 * radius)
    np.testing.assert_allclose(gradient, radial, atol=2e-2)
    # The curvature of the sphere through the point, in the tangent plane
    expected = (np.eye(3) - radial[:, :, None] * radial[:, None, :]) / (radius + offsets)[
        :,
        None,
        None,
    ]
    np.testing.assert_allclose(hessian, expected, atol=0.05 / radius)


def test_gradient_and_hessian_match_finite_differences():
    """On an ellipsoid, whose curvature varies, and off the surface on either side."""
    unit, _, _ = _sphere(3000)
    semi_axes = np.array([2.0, 1.0, 0.7])
    points = unit * semi_axes
    normals = unit / semi_axes
    surface = pulse.MeshSurface(points, normals, np.full(3000, 0.005), radius=0.25)

    rng = np.random.default_rng(2)
    x = points[rng.integers(0, 3000, 100)] + 0.03 * rng.standard_normal((100, 3))
    d, gradient, hessian = surface.evaluate(x)
    fd_gradient, fd_hessian = _finite_differences(surface, x)

    np.testing.assert_allclose(gradient, fd_gradient, atol=1e-8)
    np.testing.assert_allclose(hessian, fd_hessian, atol=1e-6 * np.abs(hessian).max())
    np.testing.assert_allclose(hessian, hessian.transpose(0, 2, 1), atol=1e-10)
    assert surface.evaluate(x, hessian=False)[2] is None
    # A point on a centroid is no special case
    d0, g0, _ = surface.evaluate(points[:5])
    assert np.all(np.isfinite(d0)) and np.all(np.isfinite(g0))


def test_distance_reaches_as_far_as_the_radius():
    """Up to most of the radius off the surface the distance is still the distance, and
    its derivatives are still its derivatives."""
    points, normals, areas = _sphere(4000, 2.0)
    surface = pulse.MeshSurface(points, normals, areas, radius=0.35)

    radial = _directions(100, seed=6)
    for height in (-0.25, -0.1, 0.1, 0.25):
        x = (2.0 + height) * radial
        d, gradient, hessian = surface.evaluate(x)
        np.testing.assert_allclose(d, height, atol=5e-3)
        fd_gradient, fd_hessian = _finite_differences(surface, x)
        np.testing.assert_allclose(gradient, fd_gradient, atol=1e-7)
        np.testing.assert_allclose(hessian, fd_hessian, atol=1e-5 * max(1.0, np.abs(hessian).max()))


def test_out_of_reach_is_the_plane_of_the_nearest_facet(caplog):
    points, normals, areas = _sphere(500)
    surface = pulse.MeshSurface(points, normals, areas, radius=0.3)
    x = np.array([[0.0, 0.0, 3.0], [0.1, 0.2, -0.4], [0.0, 0.0, 1.01]])
    with caplog.at_level(logging.WARNING, logger="pulse.surface"):
        d, gradient, hessian = surface.evaluate(x)
    assert "2 points are further than its radius" in caplog.text

    for i in (0, 1):
        nearest = np.argmin(np.linalg.norm(points - x[i], axis=1))
        np.testing.assert_allclose(gradient[i], normals[nearest])
        np.testing.assert_allclose(d[i], normals[nearest] @ (x[i] - points[nearest]))
        np.testing.assert_allclose(hessian[i], 0.0)
    assert d[2] == pytest.approx(0.01, abs=2e-3)

    # It warns once
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pulse.surface"):
        surface.evaluate(x)
    assert caplog.text == ""


def test_invalid_surfaces_raise():
    points, normals, areas = _sphere(10)
    with pytest.raises(ValueError, match="radius"):
        pulse.MeshSurface(points, normals, areas, radius=0.0)
    with pytest.raises(ValueError, match="one row per facet"):
        pulse.MeshSurface(points, normals[:5], areas, radius=1.0)
    with pytest.raises(ValueError, match="at least one facet"):
        pulse.MeshSurface(points[:0], normals[:0], areas[:0], radius=1.0)
    with pytest.raises(ValueError, match="positive area"):
        pulse.MeshSurface(points, normals, 0 * areas, radius=1.0)


@pytest.fixture(scope="module")
def lv_folder(tmp_path_factory):
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path_factory.mktemp("lv_mesh_surface"), root=0)
    if comm.rank == 0:
        cardiac_geometries.mesh.lv_ellipsoid(
            outdir=folder,
            r_short_endo=0.025,
            r_short_epi=R_SHORT,
            r_long_endo=0.09,
            r_long_epi=R_LONG,
            psize_ref=0.008,
            mu_apex_endo=-math.pi,
            mu_base_endo=-math.acos(5 / 17),
            mu_apex_epi=-math.pi,
            mu_base_epi=-math.acos(5 / 20),
            comm=MPI.COMM_SELF,
        )
    comm.barrier()
    return folder


def _load(folder):
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=MPI.COMM_WORLD, folder=folder)
    return geo, pulse.HeartGeometry.from_cardiac_geometries(geo)


@pytest.fixture(scope="module")
def lv(lv_folder):
    geo, geometry = _load(lv_folder)
    return geo, geometry, pulse.MeshSurface.from_geometry(geometry, "EPI")


def _model(incompressible=False):
    return pulse.CardiacModel(
        material=pulse.NeoHookean(mu=pulse.Variable(10.0, "kPa")),
        active=pulse.Passive(),
        compressibility=pulse.Incompressible() if incompressible else pulse.Compressible(),
    )


def _constant(mesh, value, unit):
    return pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(value)), unit)


def _pericardium(geometry, surface, stiffness=K, **kwargs):
    return pulse.PericardiumBC(
        stiffness=_constant(geometry.mesh, stiffness, "Pa / m"),
        marker=geometry.markers["EPI"][0],
        surface=surface,
        **kwargs,
    )


def _static(geometry, incompressible=False, **bcs):
    return pulse.StaticProblem(
        model=_model(incompressible),
        geometry=geometry,
        bcs=pulse.BoundaryConditions(**bcs),
        parameters={"base_bc": pulse.BaseBC.fixed},
    )


def _assemble(form):
    vec = dolfinx.fem.assemble_vector(dolfinx.fem.form(form))
    vec.scatter_reverse(dolfinx.la.InsertMode.add)
    return vec.array[: vec.index_map.size_local * vec.block_size]


def _integrate(form):
    return MPI.COMM_WORLD.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)),
        op=MPI.SUM,
    )


def _dot(a, b):
    return MPI.COMM_WORLD.allreduce(float(np.dot(a, b)), op=MPI.SUM)


def _norm(array):
    return math.sqrt(_dot(array, array))


def _rotate(u, degrees, shift=1.0):
    """Rigid rotation about the long (x) axis, plus a translation ``shift`` * (0, 3, 1) mm."""
    th = np.radians(degrees)

    def f(x):
        return np.vstack(
            (
                0.0 * x[0],
                np.cos(th) * x[1] - np.sin(th) * x[2] - x[1] + shift * 3e-3,
                np.sin(th) * x[1] + np.cos(th) * x[2] - x[2] + shift * 1e-3,
            ),
        )

    u.interpolate(f)
    u.x.scatter_forward()


def _ellipsoid_distance(x):
    """The first-order distance to the epicardial ellipsoid, as the EllipsoidSurface's, and
    the gradient of its level set. Analytic in x, so that it takes complex steps."""
    psi = (x[:, 0] / R_LONG) ** 2 + (x[:, 1] ** 2 + x[:, 2] ** 2) / R_SHORT**2
    grad = 2 * np.column_stack((x[:, 0] / R_LONG**2, x[:, 1] / R_SHORT**2, x[:, 2] / R_SHORT**2))
    return 2 * (psi - np.sqrt(psi)) / np.sqrt((grad * grad).sum(axis=1)), grad


class _SampledEllipsoid:
    """The EllipsoidSurface as a sampled surface: the same distance, as numbers."""

    @staticmethod
    def _gradient(x):
        gradient = np.zeros((len(x), 3))
        for j in range(3):
            step = np.zeros(3, dtype=complex)
            step[j] = 1e-30j
            gradient[:, j] = _ellipsoid_distance(x + step)[0].imag / 1e-30
        return gradient

    def evaluate(self, x, hessian=True):
        d = _ellipsoid_distance(x)[0]
        H = None
        if hessian:
            H = np.zeros((len(x), 3, 3))
            for j in range(3):
                step = np.zeros(3)
                step[j] = 1e-7
                H[:, :, j] = (self._gradient(x + step) - self._gradient(x - step)) / 2e-7
            H = 0.5 * (H + H.transpose(0, 2, 1))
        return d, self._gradient(x), H


def _on_the_nodes(problem, geo, tracked):
    """Have ``problem`` integrate its pericardium, the EllipsoidSurface, over the nodes, as a
    tracked surface is."""
    exact = pulse.EllipsoidSurface.from_cardiac_geometries(geo)
    assert problem._pericardial_surfaces[0][0] == exact
    problem._pericardial_surfaces = [(exact, tracked.ds)]
    problem._init_forms()


def test_from_geometry_reads_the_marked_facets(lv):
    geo, geometry, surface = lv
    comm = geometry.mesh.comm
    epi = geometry.markers["EPI"][0]
    facets = geometry.facet_tags.find(epi)
    owned = facets < geometry.mesh.topology.index_map(2).size_local
    n = comm.allreduce(int(owned.sum()), op=MPI.SUM)

    assert len(surface.points) == n + surface.rim
    assert surface.areas[:n].sum() == pytest.approx(_integrate(1.0 * geometry.ds(epi)), rel=1e-12)
    # The normals point out of the mesh, along those of the ellipsoid
    _, outwards = _ellipsoid_distance(surface.points[:n])
    assert np.all(np.einsum("ij,ij->i", surface.normals[:n], outwards) > 0)
    # The centroids of the facets lie just inside the ellipsoid
    d, _ = _ellipsoid_distance(surface.points[:n])
    assert np.all(d < 1e-12) and np.all(d > -1e-3)
    # Every process has the same surface
    assert len(set(comm.allgather(surface.points.tobytes()))) == 1
    assert repr(surface) == (
        f"MeshSurface({n} facets and {surface.rim} beyond their edge, radius={surface.radius:.4g})"
    )

    by_value = pulse.MeshSurface.from_geometry(geometry, epi)
    np.testing.assert_array_equal(by_value.points, surface.points)
    # A tenth of the size of the surface, which is more than two facets here
    assert surface.radius == pytest.approx(0.1 * math.sqrt(surface.areas[:n].sum()), rel=1e-12)
    assert surface.radius > 2 * math.sqrt(4 * surface.areas[:n].mean() / math.sqrt(3))
    # The epicardium and the endocardium together still end at the base, on two edges
    both = pulse.MeshSurface.from_geometry(geometry, ["EPI", "ENDO"], radius=0.01)
    endo = geometry.facet_tags.find(geometry.markers["ENDO"][0])
    endo = endo[endo < geometry.mesh.topology.index_map(2).size_local]
    assert len(both.points) - both.rim == n + comm.allreduce(len(endo), op=MPI.SUM)
    assert both.radius == 0.01
    assert both.rim > 0


def test_surface_goes_on_beyond_its_edge(lv):
    """The epicardium ends at the base, at x = 5/20 of the long radius. Past it the surface
    continues in the planes of the last facets, for twice the radius by default."""
    _, geometry, surface = lv
    n = len(surface.points) - surface.rim
    rim_x = 5 / 20 * R_LONG
    assert surface.rim > 0
    beyond = surface.points[n:]
    assert np.all(beyond[:, 0] > rim_x)
    assert beyond[:, 0].max() == pytest.approx(rim_x + 2 * surface.radius, rel=0.2)
    np.testing.assert_allclose(np.linalg.norm(surface.normals[n:], axis=1), 1.0)

    cut = pulse.MeshSurface.from_geometry(geometry, "EPI", extension=0.0)
    assert cut.rim == 0
    np.testing.assert_array_equal(cut.points, surface.points[:n])
    assert repr(cut) == f"MeshSurface({n} facets, radius={cut.radius:.4g})"

    # A point on the rim that slides up along the surface stays on it. Without the
    # continuation it runs out of facets, and what is left tilts away
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    mu = -np.arccos(5 / 20)
    on_rim = np.column_stack(
        (
            np.full(40, rim_x),
            R_SHORT * np.sin(mu) * np.cos(theta),
            R_SHORT * np.sin(mu) * np.sin(theta),
        ),
    )
    tangent = np.column_stack(
        (
            np.full(40, -R_LONG * np.sin(mu)),
            R_SHORT * np.cos(mu) * np.cos(theta),
            R_SHORT * np.cos(mu) * np.sin(theta),
        ),
    )
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    slid = on_rim + surface.radius * tangent
    gap = surface.evaluate(slid)[0] - surface.evaluate(on_rim)[0]
    assert np.abs(gap).max() < 0.05 * surface.radius
    assert np.all(surface.evaluate(slid)[1] @ np.array([1.0, 0.0, 0.0]) < 0.3)


def test_strips_beyond_the_edge_of_a_square():
    """Two triangles make a unit square. Its four sides are edges of one triangle each, the
    diagonal of two."""
    corners = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ],
    )
    normals = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    points, strip_normals, areas = pulse.surface._beyond_the_edge(corners, normals, 2.5)

    assert len(points) == 4 * 3  # three samples a unit apart cover 2.5 from each side
    np.testing.assert_allclose(areas, 1.0)
    np.testing.assert_allclose(strip_normals, np.tile([0.0, 0.0, 1.0], (12, 1)))
    expected = [(0.5, -d) for d in (0.5, 1.5, 2.5)] + [(1 + d, 0.5) for d in (0.5, 1.5, 2.5)]
    expected += [(0.5, 1 + d) for d in (0.5, 1.5, 2.5)] + [(-d, 0.5) for d in (0.5, 1.5, 2.5)]
    assert {tuple(np.round(q, 12)) for q in points[:, :2]} == set(expected)
    np.testing.assert_allclose(points[:, 2], 0.0)

    assert len(pulse.surface._beyond_the_edge(corners, normals, 0.0)[0]) == 0
    # A closed surface has no edge
    tetrahedron = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = tetrahedron[[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]]
    assert len(pulse.surface._beyond_the_edge(faces, np.ones((4, 3)), 1.0)[0]) == 0


def test_corners_are_the_same_vertex_by_id_not_by_coordinates():
    """Copies of a node on different processes can differ by round-off, after
    `Geometry.deform` for example. Matched by coordinates, the diagonal of the square is then
    two edges of one triangle each, and strips run across the square. Matched by vertex id,
    it is one edge of two, as it should be."""
    corners = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ],
    )
    # The second triangle's copy of (1, 1, 0) is one ulp off
    corners[1, 1, 0] = np.nextafter(1.0, 2.0)
    vertices = np.array([[0, 1, 2], [0, 2, 3]])
    normals = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])

    by_coordinates = pulse.surface._beyond_the_edge(corners, normals, 2.5)[0]
    by_vertex = pulse.surface._beyond_the_edge(corners, normals, 2.5, vertices)[0]
    assert len(by_coordinates) > 4 * 3
    assert len(by_vertex) == 4 * 3
    outside = ((by_vertex[:, :2] < 0) | (by_vertex[:, :2] > 1)).any(axis=1)
    assert outside.all()


def test_a_surface_through_a_deformed_mesh_has_only_its_own_edge(lv):
    """The epicardium of the LV ellipsoid has one edge, at the base. Deformed, on any number
    of processes, it still has: the strips beyond it number as many as when the same
    surface is made in one piece. That is not so when the copies of the nodes on the
    process boundaries move apart by round-off and their corners are matched by
    coordinates."""
    _, geometry, _ = lv
    mesh = geometry.mesh
    x0 = mesh.geometry.x.copy()
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))
    u = dolfinx.fem.Function(V)
    u.interpolate(
        lambda x: (
            1e-3 * np.vstack((np.sin(30 * x[1]), np.cos(20 * x[2]) * x[0] * 10, np.sin(40 * x[0])))
        ),
    )
    try:
        geometry.deform(u)
        surface = pulse.MeshSurface.from_geometry(geometry, "EPI", radius=0.01)
        # The same facets, in one piece on every process: the corners of the triangles
        # that meet at an edge then come from the same node, so coordinates match
        facets = geometry.facet_tags.find(geometry.markers["EPI"][0])
        facets = facets[facets < mesh.topology.index_map(2).size_local]
        nodes = dolfinx.mesh.entities_to_geometry(mesh, 2, facets)[:, :3]
        ids = np.concatenate(
            mesh.comm.allgather(np.asarray(mesh.geometry.input_global_indices)[nodes]),
        )
        points = np.concatenate(mesh.comm.allgather(mesh.geometry.x[nodes]))
        # Every id gets the coordinates of its first copy
        _, first = np.unique(ids.reshape(-1), return_index=True)
        canonical = dict(zip(ids.reshape(-1)[first], points.reshape(-1, 3)[first]))
        one_piece = np.array([[canonical[i] for i in row] for row in ids])
        normals = np.cross(one_piece[:, 1] - one_piece[:, 0], one_piece[:, 2] - one_piece[:, 0])
        expected = pulse.surface._beyond_the_edge(one_piece, normals, 2 * surface.radius)[0]
        assert surface.rim == len(expected)
    finally:
        mesh.geometry.x[:] = x0


def test_mesh_surface_is_the_ellipsoid_it_was_meshed_from(lv):
    """On and near the ellipsoid, away from the apex, where the facets are too coarse for its
    curvature, and from the rim at the base."""
    _, _, surface = lv
    rng = np.random.default_rng(3)
    mu = rng.uniform(-0.8 * np.pi, -np.arccos(5 / 20) - 0.3, 500)
    theta = rng.uniform(0, 2 * np.pi, 500)
    on = np.column_stack(
        (
            R_LONG * np.cos(mu),
            R_SHORT * np.sin(mu) * np.cos(theta),
            R_SHORT * np.sin(mu) * np.sin(theta),
        ),
    )
    _, normals = _ellipsoid_distance(on)
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    d, gradient, _ = surface.evaluate(on)
    # The facets are chords of the ellipsoid, so the surface through them lies inside it
    assert 0 < d.mean() < 3e-4
    assert d.std() < 5e-5
    length = np.linalg.norm(gradient, axis=1)
    np.testing.assert_allclose(length, 1.0, atol=2e-2)
    cosine = np.einsum("ij,ij->i", gradient, normals) / length
    assert np.degrees(np.arccos(np.clip(cosine, -1, 1)).max()) < 2.0
    for offset in (-2e-3, 2e-3):
        moved = surface.evaluate(on + offset * normals)[0] - d
        np.testing.assert_allclose(moved, offset, rtol=3e-2)


def test_a_sampled_surface_needs_its_tracked_surface(lv):
    _, geometry, surface = lv
    bc = _pericardium(geometry, surface)
    X = ufl.SpatialCoordinate(geometry.mesh)
    with pytest.raises(TypeError, match="TrackedSurface"):
        bc.traction(X, X)


def test_a_surface_around_another_boundary_warns(lv, caplog):
    _, geometry, surface = lv
    with caplog.at_level(logging.WARNING, logger="pulse.boundary_conditions"):
        _static(geometry, pericardium=[_pericardium(geometry, surface)])
    assert "does not match" not in caplog.text
    endo = pulse.MeshSurface.from_geometry(geometry, "ENDO")
    with caplog.at_level(logging.WARNING, logger="pulse.boundary_conditions"):
        _static(geometry, pericardium=[_pericardium(geometry, endo)])
    assert "does not match" in caplog.text


def test_rigid_rotation_about_the_long_axis_costs_little(lv):
    """The rotation slides the epicardium along itself. The facets are not a surface of
    revolution, so the springs feel a little, far less than a RobinBC along either normal."""
    _, geometry, surface = lv
    problem = _static(geometry, pericardium=[_pericardium(geometry, surface)])
    _rotate(problem.u, 20.0, shift=0.0)
    (tracked,) = problem._tracked_surfaces
    tracked.update()
    residual = _norm(_assemble(problem._pericardium_form(problem.u)[0]))

    for normal in ("current", "reference"):
        robin = pulse.RobinBC(
            value=_constant(geometry.mesh, K, "Pa / m"),
            marker=geometry.markers["EPI"][0],
            normal=normal,
        )
        robin_problem = _static(geometry, robin=[robin])
        robin_problem.u.x.array[:] = problem.u.x.array
        robin_residual = _norm(_assemble(robin_problem._robin_form(robin_problem.u)[0]))
        assert residual < 2e-2 * robin_residual


@pytest.mark.parametrize("unilateral", [False, True])
def test_springs_are_derivative_of_energy_and_jacobian_of_springs(lv, unilateral):
    """By central differences along w, with the surface updated at every point."""
    _, geometry, surface = lv
    bc = _pericardium(geometry, surface, unilateral=unilateral)
    problem = _static(geometry, pericardium=[bc])
    (tracked,) = problem._tracked_surfaces
    u = problem.u
    _rotate(u, 15.0)
    u0 = u.x.array.copy()
    w = dolfinx.fem.Function(problem.u_space)
    w.interpolate(lambda x: np.vstack((x[1] * x[2], x[0] * x[2], x[0] * x[1])) * 1e-1)
    w.x.scatter_forward()
    owned = slice(0, problem.u_space.dofmap.index_map.size_local * 3)

    X = ufl.SpatialCoordinate(geometry.mesh)
    g = tracked.signed_distance(X + u) - tracked.signed_distance(X)
    if unilateral:
        g = ufl.max_value(g, 0.0)
    energy = 0.5 * K * g**2 * tracked.ds(bc.marker)
    residual = problem._pericardium_form(u)[0]

    def at(eps, form):
        u.x.array[:] = u0 + eps * w.x.array
        tracked.update()
        return _integrate(form) if form is energy else _assemble(form)

    eps = 1e-3
    springs = at(0.0, residual)
    A = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(ufl.derivative(residual, u, problem.du)),
    )
    A.assemble()
    assert A.norm() > 0
    assert A.isSymmetric(1e-10 * A.norm())
    Aw = A.createVecLeft()
    A.mult(w.x.petsc_vec, Aw)

    dE = (at(eps, energy) - at(-eps, energy)) / (2 * eps)
    assert abs(dE) > 0
    assert _dot(springs, w.x.array[owned]) == pytest.approx(dE, rel=1e-6)

    fd = (at(eps, residual) - at(-eps, residual)) / (2 * eps)
    np.testing.assert_allclose(Aw.array, fd, rtol=0, atol=1e-5 * _norm(fd))
    at(0.0, residual)


def test_tracked_surface_gives_the_forms_of_the_surface_itself(lv):
    """After an update, the springs and their Jacobian are those of the EllipsoidSurface
    integrated over the same nodes."""
    geo, geometry, _ = lv
    sampled = _static(geometry, pericardium=[_pericardium(geometry, _SampledEllipsoid())])
    (tracked,) = sampled._tracked_surfaces
    exact = _static(
        geometry,
        pericardium=[_pericardium(geometry, pulse.EllipsoidSurface.from_cardiac_geometries(geo))],
    )
    _on_the_nodes(exact, geo, tracked)

    matrices = []
    residuals = []
    for problem in (sampled, exact):
        _rotate(problem.u, 15.0)
        tracked.update()
        residual = problem._pericardium_form(problem.u)[0]
        residuals.append(_assemble(residual))
        A = dolfinx.fem.petsc.assemble_matrix(
            dolfinx.fem.form(ufl.derivative(residual, problem.u, problem.du)),
        )
        A.assemble()
        matrices.append(A)

    assert _norm(residuals[1]) > 0
    np.testing.assert_allclose(residuals[0], residuals[1], rtol=0, atol=1e-11 * _norm(residuals[1]))
    difference = matrices[0].copy()
    difference.axpy(-1.0, matrices[1])
    assert difference.norm() < 1e-6 * matrices[1].norm()

    # Away from where it was updated the expansion is no longer the surface
    _rotate(sampled.u, 16.0)
    moved = _assemble(sampled._pericardium_form(sampled.u)[0])
    assert _norm(moved - residuals[0]) > 1e-3 * _norm(residuals[0])


def _inflate(geometry, surface, pressure=1.5e3, steps=3, prepare=None, incompressible=False):
    traction = _constant(geometry.mesh, 0.0, "Pa")
    problem = _static(
        geometry,
        incompressible=incompressible,
        neumann=[pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])],
        pericardium=[_pericardium(geometry, surface)],
    )
    if prepare is not None:
        prepare(problem)
    iterations = []
    for p in np.linspace(0.0, pressure, steps + 1)[1:]:
        traction.assign(p)
        assert problem.solve()
        iterations.append(problem.problem.solver.getIterationNumber())
    return problem, iterations


@pytest.mark.parametrize("incompressible", [False, True])
def test_solve_updates_the_surface_at_every_evaluation(lv, incompressible):
    """Newton with a line search converges to the solution of the EllipsoidSurface integrated
    over the same nodes, in as many iterations. It would not if any residual or Jacobian were
    assembled with the surface expanded about another point. The incompressible problem has
    a pressure as well, so that the solver's vector holds two blocks."""
    geo, geometry, _ = lv
    sampled, iterations = _inflate(geometry, _SampledEllipsoid(), incompressible=incompressible)
    (tracked,) = sampled._tracked_surfaces
    exact, exact_iterations = _inflate(
        geometry,
        pulse.EllipsoidSurface.from_cardiac_geometries(geo),
        prepare=lambda problem: _on_the_nodes(problem, geo, tracked),
        incompressible=incompressible,
    )

    assert iterations == exact_iterations
    difference = _norm(sampled.u.x.array - exact.u.x.array)
    assert difference < 1e-7 * _norm(exact.u.x.array)


def test_solve_with_the_mesh_surface_matches_the_ellipsoid(lv):
    """Inflate against the mesh's own epicardium, and against the ellipsoid it approximates."""
    geo, geometry, surface = lv
    problem, iterations = _inflate(geometry, surface)
    exact, exact_iterations = _inflate(
        geometry,
        pulse.EllipsoidSurface.from_cardiac_geometries(geo),
    )

    difference = _norm(problem.u.x.array - exact.u.x.array)
    assert difference < 3e-2 * _norm(exact.u.x.array)
    assert max(iterations) <= max(exact_iterations) + 1

    # The last evaluation of the solver was at the solution it returned
    (tracked,) = problem._tracked_surfaces
    V = tracked._vector
    position = dolfinx.fem.Function(V)
    X = ufl.SpatialCoordinate(geometry.mesh)
    position.interpolate(dolfinx.fem.Expression(X + problem.u, V.element.interpolation_points))
    nodes = tracked._nodes
    x = position.x.array.reshape(-1, 3)[nodes]
    moving = [point for _, point in tracked._points if point.moves]
    assert len(moving) == 1
    np.testing.assert_allclose(moving[0].x.x.array.reshape(-1, 3)[nodes], x, atol=1e-14)
    np.testing.assert_allclose(moving[0].d.x.array[nodes], surface.evaluate(x)[0], atol=1e-14)


def test_dynamic_problem_tracks_the_surface(lv):
    """Two steps of a DynamicProblem, whose forms hold the displacement at a time between the
    steps, with a dashpot. As for the static problem, against the EllipsoidSurface on the
    same nodes."""
    geo, geometry, _ = lv
    solutions = []
    tracked = None
    for surface in (_SampledEllipsoid(), pulse.EllipsoidSurface.from_cardiac_geometries(geo)):
        traction = _constant(geometry.mesh, 1.0e3, "Pa")
        bc = _pericardium(geometry, surface, damping=_constant(geometry.mesh, 5e3, "Pa s / m"))
        problem = pulse.DynamicProblem(
            model=_model(),
            geometry=geometry,
            bcs=pulse.BoundaryConditions(
                neumann=[pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])],
                pericardium=[bc],
            ),
            parameters={"base_bc": pulse.BaseBC.fixed, "dt": pulse.Variable(1e-3, "s")},
        )
        if tracked is None:
            (tracked,) = problem._tracked_surfaces
        else:
            _on_the_nodes(problem, geo, tracked)
        for _ in range(2):
            assert problem.solve()
        solutions.append(problem.u.x.array.copy())
    assert _norm(solutions[1]) > 0
    assert _norm(solutions[0] - solutions[1]) < 1e-7 * _norm(solutions[1])


def _clamp_base(geometry):
    def clamp(V):
        facets = geometry.facet_tags.find(geometry.markers["BASE"][0])
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
        return [dolfinx.fem.dirichletbc(dolfinx.fem.Function(V), dofs)]

    return clamp


PRESSURE = 1.5e3  # Pa


def test_prestress_recovers_reference_with_a_mesh_surface(lv_folder):
    """Inflate the LV against its own epicardium, then unload the result against the same
    surface. The inverse problem recovers the original mesh when it has the same springs, and
    misses by far more without them."""
    _, geometry = _load(lv_folder)
    surface = pulse.MeshSurface.from_geometry(geometry, "EPI")
    endo = geometry.markers["ENDO"][0]
    pressure = _constant(geometry.mesh, 0.0, "Pa")
    problem = pulse.StaticProblem(
        model=_model(),
        geometry=geometry,
        bcs=pulse.BoundaryConditions(
            neumann=[pulse.NeumannBC(traction=pressure, marker=endo)],
            dirichlet=[_clamp_base(geometry)],
            pericardium=[_pericardium(geometry, surface)],
        ),
    )
    for p in np.linspace(0.0, PRESSURE, 4)[1:]:
        pressure.assign(p)
        assert problem.solve()
    X = geometry.mesh.geometry.x.copy()
    comm = geometry.mesh.comm
    scale = comm.allreduce(np.abs(problem.u.x.array).max(), op=MPI.MAX)

    error = {}
    for with_pericardium in (True, False):
        _, loaded = _load(lv_folder)
        loaded.deform(problem.u)
        traction = _constant(loaded.mesh, 0.0, "Pa")
        prestress = pulse.unloading.PrestressProblem(
            geometry=loaded,
            model=_model(),
            bcs=pulse.BoundaryConditions(
                neumann=[pulse.NeumannBC(traction=traction, marker=endo)],
                dirichlet=[_clamp_base(loaded)],
                pericardium=[_pericardium(loaded, surface)] if with_pericardium else [],
            ),
            targets=[pulse.unloading.TargetPressure(traction=traction, target=PRESSURE)],
            ramp_steps=4,
        )
        u_inverse = prestress.unload()
        x = loaded.mesh.geometry.x
        X_recovered = x + scifem.evaluate_function(u_inverse, x, broadcast=False)
        local = np.abs(X_recovered - X).max() if X.size else 0.0
        error[with_pericardium] = comm.allreduce(local, op=MPI.MAX) / scale

    assert error[True] < 1.5e-2
    assert error[True] < 0.25 * error[False]


def test_prestress_problem_tracks_the_surface(lv):
    """The inverse problem moves the reference point, not the current one. Against the
    EllipsoidSurface on the same nodes, as for the static problem."""
    geo, geometry, _ = lv
    solutions = []
    tracked = None
    for surface in (_SampledEllipsoid(), pulse.EllipsoidSurface.from_cardiac_geometries(geo)):
        traction = _constant(geometry.mesh, 0.0, "Pa")
        prestress = pulse.unloading.PrestressProblem(
            geometry=geometry,
            model=_model(),
            bcs=pulse.BoundaryConditions(
                neumann=[pulse.NeumannBC(traction=traction, marker=geometry.markers["ENDO"][0])],
                dirichlet=[_clamp_base(geometry)],
                pericardium=[_pericardium(geometry, surface)],
            ),
            targets=[pulse.unloading.TargetPressure(traction=traction, target=PRESSURE)],
            ramp_steps=3,
        )
        if tracked is None:
            ((tracked, _),) = prestress._pericardial_surfaces
        else:
            _on_the_nodes(prestress, geo, tracked)
        solutions.append(prestress.unload().x.array.copy())
    assert _norm(solutions[1]) > 0
    assert _norm(solutions[0] - solutions[1]) < 1e-7 * _norm(solutions[1])


def test_a_mesh_of_hexahedra_is_not_supported():
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD,
        2,
        2,
        2,
        cell_type=dolfinx.mesh.CellType.hexahedron,
    )
    mesh.topology.create_connectivity(2, 3)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    with pytest.raises(NotImplementedError, match="tetrahedra"):
        pulse.MeshSurface.from_facets(mesh, facets)
