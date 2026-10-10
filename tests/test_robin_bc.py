"""RobinBC springs and dashpots: the reference/current normal choice, and the same choice in
the dynamic (dashpot) and inverse (prestress) problems.

The reference-normal spring must be the derivative of the energy 1/2 k (Q u . u) dA, so its
Jacobian is symmetric. The dashpot must honour ``perpendicular`` and ``normal`` exactly as the
spring does. The prestress problem must pull its spring back to the same configuration the
forward problem uses: unloading a forward solution recovers the reference geometry only when
both use the same normal.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import pulse

K = 5e4  # Pa/m


def _geometry(n=3):
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    boundaries = [
        pulse.Marker(name="X0", marker=1, dim=2, locator=lambda x: np.isclose(x[0], 0.0)),
        pulse.Marker(name="X1", marker=2, dim=2, locator=lambda x: np.isclose(x[0], 1.0)),
        pulse.Marker(name="Z1", marker=3, dim=2, locator=lambda x: np.isclose(x[2], 1.0)),
    ]
    return pulse.Geometry(mesh=mesh, boundaries=boundaries)


def _model():
    return pulse.CardiacModel(
        material=pulse.NeoHookean(mu=pulse.Variable(10.0, "kPa")),
        active=pulse.Passive(),
        compressibility=pulse.Compressible(),
    )


def _robin(mesh, value, **kwargs):
    unit = "Pa s / m" if kwargs.get("damping") else "Pa / m"
    k = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(value)), unit)
    return pulse.RobinBC(value=k, marker=2, **kwargs)


def _assemble(form):
    vec = dolfinx.fem.assemble_vector(dolfinx.fem.form(form))
    vec.scatter_reverse(dolfinx.la.InsertMode.add)
    return vec.array[: vec.index_map.size_local * vec.block_size]


def _rotate(u, degrees):
    """Rigid rotation about the z-axis through (0.5, 0.5, 0)."""
    th = np.radians(degrees)

    def f(x):
        y0, y1 = x[0] - 0.5, x[1] - 0.5
        return np.vstack(
            (
                np.cos(th) * y0 - np.sin(th) * y1 - y0,
                np.sin(th) * y0 + np.cos(th) * y1 - y1,
                0.0 * x[2],
            ),
        )

    u.interpolate(f)


def test_default_normal_is_what_pulse_did_before():
    """Unset, springs act along the current normal and dashpots along the reference one."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    assert _robin(mesh, K).normal == pulse.RobinNormal.current
    assert _robin(mesh, K, damping=True).normal == pulse.RobinNormal.reference
    assert _robin(mesh, K, normal="reference").normal == pulse.RobinNormal.reference
    assert _robin(mesh, K, damping=True, normal="current").normal == pulse.RobinNormal.current


@pytest.mark.parametrize("perpendicular", [False, True])
def test_reference_spring_is_derivative_of_energy(perpendicular):
    geometry = _geometry()
    robin = _robin(geometry.mesh, K, perpendicular=perpendicular, normal="reference")
    problem = pulse.StaticProblem(
        model=_model(),
        geometry=geometry,
        bcs=pulse.BoundaryConditions(robin=(robin,)),
    )
    _rotate(problem.u, 20.0)

    N = geometry.facet_normal
    nn = ufl.outer(N, N)
    Q = ufl.Identity(3) - nn if perpendicular else nn
    energy = 0.5 * K * ufl.dot(Q * problem.u, problem.u) * geometry.ds(2)
    expected = _assemble(ufl.derivative(energy, problem.u, problem.u_test))
    actual = _assemble(problem._robin_form(problem.u)[0])

    assert np.linalg.norm(expected) > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12 * np.linalg.norm(expected))


def test_reference_and_current_agree_on_translation_and_differ_on_rotation():
    geometry = _geometry()
    residuals = {}
    for normal in ("reference", "current"):
        problem = pulse.StaticProblem(
            model=_model(),
            geometry=geometry,
            bcs=pulse.BoundaryConditions(robin=(_robin(geometry.mesh, K, normal=normal),)),
        )
        problem.u.interpolate(lambda x: np.vstack((0.1 + 0 * x[0], 0.05 + 0 * x[0], 0 * x[0])))
        translated = _assemble(problem._robin_form(problem.u)[0])
        _rotate(problem.u, 20.0)
        rotated = _assemble(problem._robin_form(problem.u)[0])
        residuals[normal] = (translated, rotated)

    (t_ref, r_ref), (t_cur, r_cur) = residuals["reference"], residuals["current"]
    np.testing.assert_allclose(t_cur, t_ref, rtol=1e-12, atol=1e-12 * np.linalg.norm(t_ref))
    assert np.linalg.norm(r_cur - r_ref) > 0.1 * np.linalg.norm(r_ref)


@pytest.mark.parametrize("normal", ["reference", "current"])
@pytest.mark.parametrize("perpendicular", [False, True])
def test_dashpot_honours_perpendicular(perpendicular, normal):
    """With u = 0 both normals coincide: a normal dashpot sees only the normal velocity, a
    perpendicular one only the tangential velocity (the X1 face has normal e_x)."""
    geometry = _geometry()
    c = 5e3
    robin = _robin(geometry.mesh, c, damping=True, perpendicular=perpendicular, normal=normal)
    problem = pulse.DynamicProblem(
        model=_model(),
        geometry=geometry,
        bcs=pulse.BoundaryConditions(robin=(robin,)),
        parameters={"dt": pulse.Variable(1e-3, "s")},
    )
    u = dolfinx.fem.Function(problem.u_space)
    v = dolfinx.fem.Function(problem.u_space)
    normal_velocity = np.array([1.0, 0.0, 0.0])
    tangential_velocity = np.array([0.0, 1.0, 0.0])

    out = {}
    for name, vel in [("normal", normal_velocity), ("tangential", tangential_velocity)]:
        v.interpolate(lambda x, vel=vel: np.tile(vel[:, None], (1, x.shape[1])))
        expected = _assemble(c * ufl.dot(v, problem.u_test) * geometry.ds(2))
        out[name] = (_assemble(problem._robin_form(u, v)[0]), expected)

    seen, ignored = ("tangential", "normal") if perpendicular else ("normal", "tangential")
    actual, expected = out[seen]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12 * np.linalg.norm(expected))
    assert np.linalg.norm(out[ignored][0]) < 1e-12 * np.linalg.norm(expected)


def _bending_bcs(geometry, normal, traction):
    def clamp(V):
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, geometry.facet_tags.find(1))
        return [dolfinx.fem.dirichletbc(dolfinx.fem.Function(V), dofs)]

    return pulse.BoundaryConditions(
        neumann=(pulse.NeumannBC(traction=traction, marker=3),),
        dirichlet=(clamp,),
        robin=(_robin(geometry.mesh, K, normal=normal),),
    )


def _forward(normal, pressure):
    geometry = _geometry(n=4)
    traction = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)),
        "Pa",
    )
    problem = pulse.StaticProblem(
        model=_model(),
        geometry=geometry,
        bcs=_bending_bcs(geometry, normal, traction),
    )
    for p in np.linspace(0.0, pressure, 6)[1:]:
        traction.assign(p)
        assert problem.solve()
    return geometry, problem.u


def _unload(u_forward, normal, pressure):
    loaded = _geometry(n=4)
    loaded.deform(u_forward)
    traction = pulse.Variable(
        dolfinx.fem.Constant(loaded.mesh, dolfinx.default_scalar_type(0.0)),
        "Pa",
    )
    prestress = pulse.unloading.PrestressProblem(
        geometry=loaded,
        model=_model(),
        bcs=_bending_bcs(loaded, normal, traction),
        targets=[pulse.unloading.TargetPressure(traction=traction, target=pressure)],
        ramp_steps=6,
    )
    u_inverse = prestress.unload()
    import scifem

    x = loaded.mesh.geometry.x
    return x + scifem.evaluate_function(u_inverse, x, broadcast=False)


@pytest.mark.parametrize("normal", ["reference", "current"])
def test_prestress_recovers_reference_with_the_same_normal(normal):
    """Bend a clamped cube so the sprung face rotates, then unload the result. The inverse
    problem recovers the original mesh to discretisation error (about 0.5% here) when its spring
    uses the forward problem's normal, and misses by about 4% with the other one. So an inverse
    spring pulled back to the wrong configuration fails this test."""
    pressure = 1.5e3
    other = {"reference": "current", "current": "reference"}[normal]
    reference, u_forward = _forward(normal, pressure)
    comm = reference.mesh.comm
    X = reference.mesh.geometry.x.copy()
    scale = comm.allreduce(np.abs(u_forward.x.array).max(), op=MPI.MAX)

    error = {}
    for inverse_normal in (normal, other):
        X_recovered = _unload(u_forward, inverse_normal, pressure)
        local = np.abs(X_recovered - X).max() if X.size else 0.0
        error[inverse_normal] = comm.allreduce(local, op=MPI.MAX) / scale

    assert error[normal] < 1e-2
    assert error[normal] < 0.25 * error[other]
