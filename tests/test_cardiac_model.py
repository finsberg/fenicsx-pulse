import math

import dolfinx
import pytest
import ufl

import pulse

# def test


@pytest.mark.parametrize("isotropy", (pulse.active_stress.ActiveStressModels.transversely,))
@pytest.mark.parametrize(
    "comp_model",
    (pulse.compressibility.Incompressible(), pulse.compressibility.Compressible()),
)
def test_CardiacModel_HolzapfelOgden(comp_model, isotropy, mesh, u):
    material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    material = pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)

    active_model = pulse.ActiveStress(f0, isotropy=isotropy)
    model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )
    u.interpolate(lambda x: x / 10.0)
    F = pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F, p=1.0)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))

    if isinstance(comp_model, pulse.compressibility.Incompressible):
        assert math.isclose(value, 53647.14346074856)
    else:
        assert math.isclose(value, 103220.36041941364)


@pytest.mark.parametrize("isotropy", (pulse.active_stress.ActiveStressModels.transversely,))
@pytest.mark.parametrize(
    "comp_model",
    (pulse.compressibility.Incompressible(), pulse.compressibility.Compressible()),
)
def test_CardiacModel_NeoHookean(comp_model, isotropy, mesh, u):
    material = pulse.NeoHookean(
        mu=dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(15.0)),
    )
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    active_model = pulse.ActiveStress(f0, isotropy=isotropy)
    model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )
    u.interpolate(lambda x: x / 10.0)
    F = pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F, p=1.0)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))

    if isinstance(comp_model, pulse.compressibility.Incompressible):
        assert math.isclose(value, 4725.331000000082)
    else:
        assert math.isclose(value, 54298.54795867078)


@pytest.mark.parametrize("isotropy", (pulse.active_stress.ActiveStressModels.transversely,))
@pytest.mark.parametrize(
    "comp_model",
    (pulse.compressibility.Incompressible(), pulse.compressibility.Compressible()),
)
def test_CardiacModel_Guccione(comp_model, isotropy, mesh, u):
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    n0 = dolfinx.fem.Constant(mesh, (0.0, 0.0, 1.0))
    material_params = pulse.Guccione.default_parameters()
    material = pulse.Guccione(f0=f0, s0=s0, n0=n0, **material_params)
    active_model = pulse.ActiveStress(f0, isotropy=isotropy)
    model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )
    u.interpolate(lambda x: x / 10.0)
    F = pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F, p=1.0)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))

    if isinstance(comp_model, pulse.compressibility.Incompressible):
        assert math.isclose(value, 141.78170311802802)
    else:
        assert math.isclose(value, 49714.998661786354)
