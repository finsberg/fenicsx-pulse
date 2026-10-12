"""PericardiumBC: springs on the true gap to a fixed surface, which let the boundary slide.

A rigid rotation of an LV ellipsoid about its long axis slides the epicardium along itself, so
the springs must feel nothing, where a RobinBC registers a gap. The springs must be the
derivative of the energy 1/2 k g^2 dA, with a symmetric Jacobian that Newton sees, and a normal
displacement delta must give the traction k delta. The dashpot must oppose the normal velocity
and ignore sliding. The inverse (prestress) problem must put the springs at rest in the same
configuration as the forward problem, so unloading a forward solution recovers the mesh.
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


@pytest.fixture(scope="module")
def lv_folder(tmp_path_factory):
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path_factory.mktemp("lv_pericardium"), root=0)
    if comm.rank == 0:
        cardiac_geometries.mesh.lv_ellipsoid(
            outdir=folder,
            r_short_endo=0.025,
            r_short_epi=0.035,
            r_long_endo=0.09,
            r_long_epi=0.097,
            psize_ref=0.02,
            mu_apex_endo=-math.pi,
            mu_base_endo=-math.acos(5 / 17),
            mu_apex_epi=-math.pi,
            mu_base_epi=-math.acos(5 / 20),
            comm=MPI.COMM_SELF,
        )
    comm.barrier()
    return folder


def _load(folder, scale=1.0):
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=MPI.COMM_WORLD, folder=folder)
    geo.mesh.geometry.x[:] *= scale
    return geo, pulse.HeartGeometry.from_cardiac_geometries(geo)


@pytest.fixture(scope="module")
def lv(lv_folder):
    return _load(lv_folder)


def _model():
    return pulse.CardiacModel(
        material=pulse.NeoHookean(mu=pulse.Variable(10.0, "kPa")),
        active=pulse.Passive(),
        compressibility=pulse.Compressible(),
    )


def _constant(mesh, value, unit):
    return pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(value)), unit)


def _pericardium(geo, geometry, stiffness=K, scale=1.0, **kwargs):
    surface = pulse.EllipsoidSurface.from_cardiac_geometries(geo)
    surface = pulse.EllipsoidSurface(r_long=surface.r_long * scale, r_short=surface.r_short * scale)
    return pulse.PericardiumBC(
        stiffness=_constant(geometry.mesh, stiffness, "Pa / m"),
        marker=geometry.markers["EPI"][0],
        surface=surface,
        **kwargs,
    )


def _static(geometry, **bcs):
    return pulse.StaticProblem(
        model=_model(),
        geometry=geometry,
        bcs=pulse.BoundaryConditions(**bcs),
        parameters={"base_bc": pulse.BaseBC.fixed},
    )


def _assemble(form):
    vec = dolfinx.fem.assemble_vector(dolfinx.fem.form(form))
    vec.scatter_reverse(dolfinx.la.InsertMode.add)
    return vec.array[: vec.index_map.size_local * vec.block_size]


def _norm(array):
    return math.sqrt(MPI.COMM_WORLD.allreduce(float(np.dot(array, array)), op=MPI.SUM))


def _rotate(u, degrees, scale=1.0):
    """Rigid rotation about the long (x) axis, plus a translation ``scale`` * (0, 3, 1) mm."""
    th = np.radians(degrees)

    def f(x):
        return np.vstack(
            (
                0.0 * x[0],
                np.cos(th) * x[1] - np.sin(th) * x[2] - x[1] + scale * 3e-3,
                np.sin(th) * x[1] + np.cos(th) * x[2] - x[2] + scale * 1e-3,
            ),
        )

    u.interpolate(f)


def _ellipsoid_normal(surface, X):
    Xv = ufl.variable(X)
    n = ufl.diff(surface.signed_distance(Xv), Xv)
    return n / ufl.sqrt(ufl.dot(n, n))


def test_boundary_conditions_default_has_no_pericardium():
    assert pulse.BoundaryConditions().pericardium == ()


def test_ellipsoid_surface_from_cardiac_geometries(lv):
    geo, geometry = lv
    surface = pulse.EllipsoidSurface.from_cardiac_geometries(geo)
    assert (surface.r_long, surface.r_short) == (0.097, 0.035)
    assert pulse.EllipsoidSurface.from_cardiac_geometries(geo.info, "endo").r_short == 0.025
    with pytest.raises(ValueError, match="lv_ellipsoid"):
        pulse.EllipsoidSurface.from_cardiac_geometries({"mesh_type": "biv_ellipsoid"})

    # The epicardial facets lie on the surface, up to the facets' chord error
    X = ufl.SpatialCoordinate(geometry.mesh)
    ds = geometry.ds(geometry.markers["EPI"][0])
    area = geometry.mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ds)),
        op=MPI.SUM,
    )
    d2 = geometry.mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(surface.signed_distance(X) ** 2 * ds)),
        op=MPI.SUM,
    )
    assert math.sqrt(d2 / area) < 1e-3


def test_a_surface_in_the_wrong_unit_warns(lv, caplog):
    geo, geometry = lv
    with caplog.at_level(logging.WARNING, logger="pulse.boundary_conditions"):
        _static(geometry, pericardium=[_pericardium(geo, geometry)])
    assert "does not match" not in caplog.text
    with caplog.at_level(logging.WARNING, logger="pulse.boundary_conditions"):
        _static(geometry, pericardium=[_pericardium(geo, geometry, scale=1e3)])
    assert "does not match" in caplog.text


def test_rigid_rotation_about_the_long_axis_costs_nothing(lv):
    """The rotation slides the epicardium along itself. Both RobinBC normals see a gap."""
    geo, geometry = lv
    problem = _static(geometry, pericardium=[_pericardium(geo, geometry)])
    _rotate(problem.u, 20.0, scale=0.0)
    residual = _assemble(problem._pericardium_form(problem.u)[0])

    for normal in ("current", "reference"):
        robin = pulse.RobinBC(
            value=_constant(geometry.mesh, K, "Pa / m"),
            marker=geometry.markers["EPI"][0],
            normal=normal,
        )
        robin_problem = _static(geometry, robin=[robin])
        robin_problem.u.x.array[:] = problem.u.x.array
        robin_residual = _norm(_assemble(robin_problem._robin_form(robin_problem.u)[0]))
        assert _norm(residual) < 1e-10 * robin_residual


@pytest.mark.parametrize("unilateral", [False, True])
def test_springs_are_derivative_of_energy(lv, unilateral):
    geo, geometry = lv
    bc = _pericardium(geo, geometry, unilateral=unilateral)
    problem = _static(geometry, pericardium=[bc])
    _rotate(problem.u, 15.0)

    X = ufl.SpatialCoordinate(geometry.mesh)
    g = bc.surface.signed_distance(X + problem.u) - bc.surface.signed_distance(X)
    if unilateral:
        g = ufl.max_value(g, 0.0)
    energy = 0.5 * K * g**2 * geometry.ds(bc.marker)
    expected = _assemble(ufl.derivative(energy, problem.u, problem.u_test))
    actual = _assemble(problem._pericardium_form(problem.u)[0])

    assert _norm(expected) > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10 * _norm(expected))


@pytest.mark.parametrize("unilateral", [False, True])
def test_normal_displacement_gives_k_delta(lv, unilateral):
    """Moving each epicardial point delta along the surface normal gives the traction k delta
    along it, to O(delta / radius). Unilateral springs only resist outward displacement."""
    geo, geometry = lv
    bc = _pericardium(geo, geometry, unilateral=unilateral)
    problem = _static(geometry, pericardium=[bc])
    X = ufl.SpatialCoordinate(geometry.mesh)
    n = _ellipsoid_normal(bc.surface, X)
    ds = geometry.ds(bc.marker)

    delta = 1e-5
    expected = _assemble(K * delta * ufl.dot(n, problem.u_test) * ds)
    outward = _assemble(problem._pericardium_form(delta * n)[0])
    inward = _assemble(problem._pericardium_form(-delta * n)[0])

    # About 1% on this coarse mesh, whose facets lie up to a millimetre off the surface
    assert _norm(outward - expected) < 3e-2 * _norm(expected)
    if unilateral:
        assert _norm(inward) == 0
    else:
        assert _norm(inward + expected) < 3e-2 * _norm(expected)


def test_mesh_unit_scales_like_robin(lv_folder):
    """The same problem in millimetres: each residual entry is a force times a length, which
    is 1e6 larger in N mm than in N m, since the traction is the same."""
    residuals = []
    for scale, unit in [(1.0, "m"), (1e3, "mm")]:
        geo, geometry = _load(lv_folder, scale)
        problem = pulse.StaticProblem(
            model=_model(),
            geometry=geometry,
            bcs=pulse.BoundaryConditions(pericardium=[_pericardium(geo, geometry, scale=scale)]),
            parameters={"mesh_unit": unit},
        )
        _rotate(problem.u, 15.0, scale=scale)
        residuals.append(_norm(_assemble(problem._pericardium_form(problem.u)[0])))
    assert residuals[1] == pytest.approx(1e6 * residuals[0], rel=1e-8)


@pytest.mark.parametrize("unilateral", [False, True])
def test_jacobian_is_symmetric_and_matches_finite_differences(lv, unilateral):
    geo, geometry = lv
    problem = _static(geometry, pericardium=[_pericardium(geo, geometry, unilateral=unilateral)])
    _rotate(problem.u, 15.0)
    residual = problem._pericardium_form(problem.u)[0]
    jacobian = ufl.derivative(residual, problem.u, problem.du)

    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(jacobian))
    A.assemble()
    assert A.norm() > 0
    assert A.isSymmetric(1e-10 * A.norm())

    # Directional derivative along w, by central differences
    w = dolfinx.fem.Function(problem.u_space)
    w.interpolate(lambda x: np.vstack((x[1] * x[2], x[0] * x[2], x[0] * x[1])) * 1e-1)
    Aw = A.createVecLeft()
    A.mult(w.x.petsc_vec, Aw)
    u0 = problem.u.x.array.copy()
    eps = 1e-7
    problem.u.x.array[:] = u0 + eps * w.x.array
    plus = _assemble(residual)
    problem.u.x.array[:] = u0 - eps * w.x.array
    minus = _assemble(residual)
    problem.u.x.array[:] = u0
    fd = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(Aw.array, fd, rtol=0, atol=1e-5 * _norm(fd))


@pytest.mark.parametrize("unilateral", [False, True])
def test_dashpot_opposes_normal_velocity_and_ignores_sliding(lv, unilateral):
    geo, geometry = lv
    c = 5e3
    bc = _pericardium(
        geo,
        geometry,
        stiffness=0.0,
        damping=_constant(geometry.mesh, c, "Pa s / m"),
        unilateral=unilateral,
    )
    problem = pulse.DynamicProblem(
        model=_model(),
        geometry=geometry,
        bcs=pulse.BoundaryConditions(pericardium=[bc]),
        parameters={"base_bc": pulse.BaseBC.fixed, "dt": pulse.Variable(1e-3, "s")},
    )
    X = ufl.SpatialCoordinate(geometry.mesh)
    n = _ellipsoid_normal(bc.surface, X)
    ds = geometry.ds(bc.marker)
    # Displaced slightly outward, so that unilateral dashpots are engaged
    u = 1e-6 * n

    sliding = ufl.cross(ufl.as_vector((1.0, 0.0, 0.0)), X)  # rotation about the long axis
    assert _norm(_assemble(problem._pericardium_form(u, sliding)[0])) < 1e-10

    expected = _assemble(c * ufl.dot(n, problem.u_test) * ds)
    actual = _assemble(problem._pericardium_form(u, n)[0])
    assert _norm(actual - expected) < 3e-2 * _norm(expected)

    # The dashpot dissipates: its force does positive work against any velocity
    v = ufl.as_vector((0.0, 1.0, 0.5))
    power = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.replace(problem._pericardium_form(u, v)[0], {problem.u_test: v})),
    )
    assert MPI.COMM_WORLD.allreduce(power, op=MPI.SUM) > 0

    # A static problem has no velocity, so no dashpot
    static = _static(geometry, pericardium=[bc])
    assert _norm(_assemble(static._pericardium_form(u)[0])) < 1e-10


def _clamp_base(geometry):
    def clamp(V):
        facets = geometry.facet_tags.find(geometry.markers["BASE"][0])
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
        return [dolfinx.fem.dirichletbc(dolfinx.fem.Function(V), dofs)]

    return clamp


PRESSURE = 1.5e3  # Pa


def test_prestress_recovers_reference_with_pericardium(lv_folder):
    """Inflate the LV against pericardial springs, then unload the result. The inverse problem
    recovers the original mesh to discretisation error (about 0.9% here) when it has the same
    springs. It misses by 3% if they act on the loaded area instead of the unloaded one, and by
    150% without them."""
    geo, geometry = _load(lv_folder)
    bc = _pericardium(geo, geometry)
    pressure = _constant(geometry.mesh, 0.0, "Pa")
    endo = geometry.markers["ENDO"][0]
    problem = pulse.StaticProblem(
        model=_model(),
        geometry=geometry,
        bcs=pulse.BoundaryConditions(
            neumann=[pulse.NeumannBC(traction=pressure, marker=endo)],
            dirichlet=[_clamp_base(geometry)],
            pericardium=[bc],
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
                pericardium=[bc] if with_pericardium else [],
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
