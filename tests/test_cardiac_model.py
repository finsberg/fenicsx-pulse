import math

import dolfinx
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
