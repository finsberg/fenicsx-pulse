"""The axisymmetric geometry, on the meridian of a hollow cylinder of radii A < B and length L:
the measures integrate over the solid of revolution, the deformation gradient is that of the
revolved deformation, and a pressure load is the derivative of the cavity volume."""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import pulse

A, B, L = 1.0, 2.0, 3.0
ENDO, EPI, BOTTOM, TOP = 1, 2, 3, 4


def cylinder_geometry(n: int = 8, **kwargs) -> pulse.AxisymmetricHeartGeometry:
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, A]), np.array([L, B])],
        [3 * n, n],
        dolfinx.mesh.CellType.triangle,
    )
    boundaries = [
        pulse.Marker("ENDO", ENDO, 1, lambda x: np.isclose(x[1], A)),
        pulse.Marker("EPI", EPI, 1, lambda x: np.isclose(x[1], B)),
        pulse.Marker("BOTTOM", BOTTOM, 1, lambda x: np.isclose(x[0], 0.0)),
        pulse.Marker("TOP", TOP, 1, lambda x: np.isclose(x[0], L)),
    ]
    return pulse.AxisymmetricHeartGeometry(
        mesh=mesh,
        boundaries=boundaries,
        axis_marker=None,
        **kwargs,
    )


def assemble(form) -> float:
    return MPI.COMM_WORLD.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)),
        op=MPI.SUM,
    )


def displacement(geo, expr) -> dolfinx.fem.Function:
    V = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 2, geo.displacement_shape))
    u = dolfinx.fem.Function(V)
    u.interpolate(expr)
    return u


def test_requires_2D_mesh():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    with pytest.raises(ValueError, match="geometric dimension 2"):
        pulse.AxisymmetricHeartGeometry(mesh=mesh)


def test_measures_integrate_over_solid_of_revolution():
    geo = cylinder_geometry()
    assert geo.axisymmetric
    assert geo.displacement_shape == (3,)
    assert np.isclose(assemble(1.0 * geo.dx), np.pi * (B**2 - A**2) * L)
    assert np.isclose(geo.surface_area("ENDO"), 2 * np.pi * A * L)
    # A tuple of subdomains is weighted once
    assert np.isclose(assemble(1.0 * geo.ds((ENDO, EPI))), 2 * np.pi * (A + B) * L)
    assert np.isclose(geo.volume("ENDO"), np.pi * A**2 * L)
    assert np.allclose(geo.base_center("BOTTOM"), [0.0, 0.0, 0.0])
    assert np.allclose(geo.base_center("TOP"), [L, 0.0, 0.0])


def test_rigid_motions_do_not_strain():
    geo = cylinder_geometry()
    # Translation along the axis and a rigid twist
    u = displacement(geo, lambda x: (0.3 + 0 * x[0], 0 * x[0], 0.7 + 0 * x[0]))
    F = geo.deformation_gradient(u)
    assert np.isclose(assemble(ufl.inner(F - ufl.Identity(3), F - ufl.Identity(3)) * geo.dx), 0)
    assert np.isclose(geo.volume("ENDO", u), np.pi * A**2 * L)


@pytest.mark.parametrize("eps, delta, kappa", [(0.1, 0.0, 0.0), (0.1, 0.2, 0.0), (0, 0, 0.3)])
def test_deformation_gradient(eps, delta, kappa):
    """Radial expansion r = (1 + eps) R, axial stretch z = (1 + delta) Z and a twist
    phi = kappa Z that grows along the axis."""
    geo = cylinder_geometry()
    u = displacement(geo, lambda x: (delta * x[0], eps * x[1], kappa * x[0]))
    r = (1 + eps) * geo.X[1]
    F_exact = ufl.as_matrix(
        [
            [1 + delta, 0, 0],
            [0, 1 + eps, 0],
            [r * kappa, 0, 1 + eps],
        ],
    )
    F = geo.deformation_gradient(u)
    assert np.isclose(assemble(ufl.inner(F - F_exact, F - F_exact) * geo.dx), 0.0)
    assert np.isclose(
        geo.volume("ENDO", u),
        np.pi * ((1 + eps) * A) ** 2 * (1 + delta) * L,
    )


def test_neumann_load_is_derivative_of_volume():
    r"""The virtual work of a pressure p on the cavity, :math:`\int p \, n \cdot \delta x \, da`,
    is :math:`-p \, \delta V` away from the rim of the cavity, also with a twist."""
    geo = cylinder_geometry(n=4)
    u = displacement(
        geo,
        lambda x: (0.05 * np.sin(x[0]) * x[1], 0.1 * np.cos(x[0]), 0.2 * x[0] * x[1]),
    )
    v = ufl.TestFunction(u.function_space)
    F = geo.deformation_gradient(u)
    n = ufl.det(F) * ufl.inv(F).T * geo.facet_normal
    ds = geo.ds(ENDO)
    work = ufl.inner(geo.displacement_variation(u, v), n) * ds
    dV = ufl.derivative(geo.volume_form(u, marker="ENDO") * ds, u, v)

    b = dolfinx.fem.assemble_vector(dolfinx.fem.form(work + dV))
    b.scatter_reverse(dolfinx.la.InsertMode.add)

    V = u.function_space
    assert geo.facet_tags is not None
    rim = np.unique(
        np.hstack(
            [
                dolfinx.fem.locate_dofs_topological(V, 1, geo.facet_tags.find(marker))
                for marker in (BOTTOM, TOP)
            ],
        ),
    )
    bs = V.dofmap.bs
    num_owned = V.dofmap.index_map.size_local
    away = np.setdiff1d(np.arange(num_owned), rim)
    values = b.array.reshape(-1, bs)[away]
    # and the load does something there, so the test is not vacuous
    work_vec = dolfinx.fem.assemble_vector(dolfinx.fem.form(work))
    work_vec.scatter_reverse(dolfinx.la.InsertMode.add)
    scale = MPI.COMM_WORLD.allreduce(np.abs(work_vec.array).max(), op=MPI.MAX)
    assert scale > 1e-2
    assert MPI.COMM_WORLD.allreduce(np.abs(values).max(initial=0), op=MPI.MAX) < 1e-10 * scale


def test_inflation_of_incompressible_tube():
    r"""An incompressible neo-Hookean tube in plane strain under an inner pressure p.
    The radii go as :math:`r^2 = R^2 + c`, and with the hoop stretch
    :math:`\lambda = r / R`, :math:`p = \mu \int_{r_A}^{r_B} (\lambda^2 - \lambda^{-2})\, dr / r`.
    """
    geo = cylinder_geometry(n=4)
    mu = 10.0  # kPa
    p = 5.0  # kPa
    model = pulse.CardiacModel(
        material=pulse.NeoHookean(mu=pulse.Variable(mu, "kPa")),
        active=pulse.Passive(),
        compressibility=pulse.Incompressible(),
    )

    def plane_strain(V):
        # No axial displacement and no twist anywhere
        tdim = geo.mesh.topology.dim
        geo.mesh.topology.create_connectivity(tdim, tdim)
        cells = np.arange(geo.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        bcs = []
        for i in (0, 2):
            dofs = dolfinx.fem.locate_dofs_topological(V.sub(i), tdim, cells)
            bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), dofs, V.sub(i)))
        return bcs

    pressure = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0))
    bcs = pulse.BoundaryConditions(
        neumann=(pulse.NeumannBC(traction=pulse.Variable(pressure, "kPa"), marker=ENDO),),
        dirichlet=(plane_strain,),
    )
    problem = pulse.StaticProblem(model=model, geometry=geo, bcs=bcs)
    for value in np.linspace(0, p, 5)[1:]:
        pressure.value = value
        assert problem.solve()

    r_A2 = geo.volume("ENDO", problem.u) / (np.pi * L)
    c = r_A2 - A**2
    assert c > 0.1  # inflated
    r = np.linspace(np.sqrt(A**2 + c), np.sqrt(B**2 + c), 20001)
    lam2 = r**2 / (r**2 - c)
    p_exact = mu * np.trapezoid((lam2 - 1 / lam2) / r, r)
    assert np.isclose(p_exact, p, rtol=1e-4)


@pytest.mark.parametrize(
    "bcs, parameters, match",
    [
        (
            dict(robin=(pulse.RobinBC(value=pulse.Variable(1.0, "Pa / m"), marker=EPI),)),
            {},
            "Robin",
        ),
        ({}, {"rigid_body_constraint": True}, "rigid body"),
    ],
)
def test_unsupported(bcs, parameters, match):
    geo = cylinder_geometry(n=1)
    model = pulse.CardiacModel(
        material=pulse.NeoHookean(),
        active=pulse.Passive(),
        compressibility=pulse.Compressible(),
    )
    with pytest.raises(NotImplementedError, match=match):
        pulse.StaticProblem(
            model=model,
            geometry=geo,
            bcs=pulse.BoundaryConditions(**bcs),
            parameters=parameters,
        )


def test_symmetry_dirichlet_needs_axis_marker():
    geo = cylinder_geometry(n=1)
    geo.axis_marker = "APEX"
    V = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 2, (3,)))
    with pytest.raises(pulse.exceptions.MarkerNotFoundError):
        geo.symmetry_dirichlet(V)
