import math

import dolfinx
import numpy as np
import pytest
import ufl

import pulse

# def test


@pytest.mark.parametrize("isotropy", (pulse.active_stress.ActiveStressModels.transversely,))
@pytest.mark.parametrize(
    "comp_model_cls",
    (pulse.compressibility.Incompressible, pulse.compressibility.Compressible),
)
def test_CardiacModel_HolzapfelOgden(comp_model_cls, isotropy, mesh, u):
    # material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    material = pulse.HolzapfelOgden(
        f0=f0,
        s0=s0,
        a=1.0,
        b=0.0,
        a_f=1.0,
        b_f=0.0,
        a_fs=0.0,
        b_fs=0.0,
    )
    comp_model = comp_model_cls()
    active_model = pulse.ActiveStress(f0, isotropy=isotropy)
    model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )
    comp_model.register(p=1000.0)
    u.interpolate(lambda x: x / 10.0)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    psi = model.strain_energy(C)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))

    # value_mat = dolfinx.fem.assemble_scalar(dolfinx.fem.form(material.strain_energy(C) * ufl.dx))
    # value_comp = dolfinx.fem.assemble_scalar(
    #     dolfinx.fem.form(comp_model.strain_energy(C) * ufl.dx),
    # )
    # value_active = dolfinx.fem.assemble_scalar(
    #     dolfinx.fem.form(active_model.strain_energy(C) * ufl.dx)
    # )

    # F = I + 0.1 I, C = 1.21 I, I4f = 1.21
    # J = det(F) = 1.1 ** 3, J^{-2/3} = 1.1 ** -2

    if isinstance(comp_model, pulse.compressibility.Incompressible):
        # psi = 0.5 * a * (I1 - 3) + 0.5 * a_f * (I4f - 1)**2 + p (J - 1)
        # psi = 0.5 * 1000*1 * (3 * 1.21 - 3) + 0.5 * 1000.0 * 1.0 * (1.21 - 1)**2 +
        # 1000.0 * (1.1 ** 3 - 1) = 668.05
        assert math.isclose(value, 668.05)
    else:
        # J^{-2/3} = 1.1 ** -2
        # psi = 0.5 * a * (J^{-2/3} * I1 - 3) + 0.5 * a_f * (^{-2/3} * I4f - 1)**2
        # + kappa * (J * ln(J) - J + 1)

        # psi = 0.5 * 1000*1 * (1.1 ** -2  * 3 * 1.21 - 3)
        # + 0.5 * 1000.0 * 1.0 * (1.1 ** -2  * 1.21 - 1)**2
        # + 1e6 * (1.1 ** 3 * math.log(1.1 ** 3) - 1.1 ** 3 + 1) = 49573.547958669194
        assert math.isclose(value, 49573.547958669194)


@pytest.mark.parametrize("isotropy", (pulse.active_stress.ActiveStressModels.transversely,))
@pytest.mark.parametrize(
    "comp_model_cls",
    (pulse.compressibility.Incompressible, pulse.compressibility.Compressible),
)
def test_CardiacModel_NeoHookean(comp_model_cls, isotropy, mesh, u):
    material = pulse.NeoHookean(
        mu=dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(15.0)),
    )
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    comp_model = comp_model_cls()
    active_model = pulse.ActiveStress(f0, isotropy=isotropy)
    model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )
    comp_model.register(p=dolfinx.fem.Constant(mesh, 1.0))
    u.interpolate(lambda x: x / 10.0)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    psi = model.strain_energy(C)

    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))

    if isinstance(comp_model, pulse.compressibility.Incompressible):
        assert math.isclose(value, 4725.331000000082)
    else:
        assert math.isclose(value, 49573.54795867355)


@pytest.mark.parametrize("isotropy", (pulse.active_stress.ActiveStressModels.transversely,))
@pytest.mark.parametrize(
    "comp_model_cls",
    (pulse.compressibility.Incompressible, pulse.compressibility.Compressible),
)
def test_CardiacModel_Guccione(comp_model_cls, isotropy, mesh, u):
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    n0 = dolfinx.fem.Constant(mesh, (0.0, 0.0, 1.0))
    material_params = pulse.Guccione.default_parameters()
    material = pulse.Guccione(f0=f0, s0=s0, n0=n0, **material_params)
    active_model = pulse.ActiveStress(f0, isotropy=isotropy)
    comp_model = comp_model_cls()
    model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )
    comp_model.register(p=dolfinx.fem.Constant(mesh, 1.0))
    u.interpolate(lambda x: x / 10.0)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    psi = model.strain_energy(C)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))

    if isinstance(comp_model, pulse.compressibility.Incompressible):
        assert math.isclose(value, 141.78170311802802)
    else:
        assert math.isclose(value, 49573.54795866912)


def _active_models(mesh, u):
    """One instance of every shipped active model, each with a nonzero tension."""
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    Ta = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(50.0)), "kPa")
    Ka = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(500.0)), "kPa")

    frank_starling = pulse.active_stress.FrankStarlingActiveStress(f0, activation=Ta)
    frank_starling.register(u)

    return {
        "passive": pulse.active_model.Passive(),
        "invariant": pulse.ActiveStress(
            f0,
            activation=Ta,
            formulation=pulse.active_stress.ActiveStressFormulation.invariant,
        ),
        "stretch": pulse.ActiveStress(
            f0,
            activation=Ta,
            formulation=pulse.active_stress.ActiveStressFormulation.stretch,
        ),
        "stabilized": pulse.active_stress.StabilizedActiveStress(
            f0=f0,
            activation=Ta,
            active_stiffness=Ka,
            lmbda_prev=dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.05)),
        ),
        "frank_starling": frank_starling,
    }


@pytest.mark.parametrize(
    "active_name",
    ("invariant", "stretch", "stabilized", "frank_starling"),
)
@pytest.mark.parametrize(
    "comp_model_cls",
    (pulse.compressibility.Incompressible, pulse.compressibility.Compressible),
)
def test_active_stress_is_consistent_between_S_P_and_strain_energy(
    comp_model_cls,
    active_name,
    mesh,
    u,
):
    """The active contribution to S, to P and to the total energy must agree.

    Evaluated at a deformation with J != 1, so that the isochoric split is
    actually doing something, and taken as the difference against the same
    model with a `Passive` active component so that only the active term is
    under test.

    An active model evaluated on Cdev in one of the three routes and on C in
    another disagrees by an isotropic term of order Ta/3. That does not vanish
    as J -> 1, and no amount of Newton convergence reveals it, since only S
    enters the residual.
    """
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    material = pulse.HolzapfelOgden(f0=f0, s0=s0, **pulse.HolzapfelOgden.orthotropic_parameters())
    comp_model = comp_model_cls()
    comp_model.register(p=dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1000.0)))

    def cardiac_model(active):
        return pulse.CardiacModel(material=material, active=active, compressibility=comp_model)

    active_model = _active_models(mesh, u)[active_name]
    model = cardiac_model(active_model)
    passive = cardiac_model(pulse.active_model.Passive())

    # A non-isochoric, non-symmetric deformation: J = 1.1 * 0.95 * 1.02 != 1
    u.interpolate(lambda x: np.vstack([0.1 * x[0], -0.05 * x[1], 0.02 * x[2]]))
    F = ufl.variable(pulse.kinematics.DeformationGradient(u))
    C = ufl.variable(F.T * F)

    tensor_space = dolfinx.fem.functionspace(mesh, ("DG", 0, (3, 3)))

    def values(expr):
        f = dolfinx.fem.Function(tensor_space)
        f.interpolate(dolfinx.fem.Expression(expr, tensor_space.element.interpolation_points))
        return f.x.array.copy()

    S = values(model.S(C) - passive.S(C))
    S_from_P = values(ufl.inv(F) * (model.P(F) - passive.P(F)))
    S_from_psi = values(
        2.0 * ufl.diff(model.strain_energy(C) - passive.strain_energy(C), C),
    )

    scale = np.abs(S).max()
    assert scale > 0.0
    assert np.abs(S - S_from_P).max() < 1e-10 * scale
    assert np.abs(S - S_from_psi).max() < 1e-10 * scale
