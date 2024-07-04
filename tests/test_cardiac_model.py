import math

import dolfinx
import fenicsx_pulse
import pytest
import ufl

# def test


@pytest.mark.parametrize("isotropy", (fenicsx_pulse.active_stress.ActiveStressModels.transversely,))
@pytest.mark.parametrize(
    "comp_model",
    (fenicsx_pulse.compressibility.Incompressible(), fenicsx_pulse.compressibility.Compressible()),
)
def test_CardiacModel_HolzapfelOgden(comp_model, isotropy, mesh, u):
    material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters()
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    material = fenicsx_pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)

    active_model = fenicsx_pulse.ActiveStress(f0, isotropy=isotropy)
    model = fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
        decouple_deviatoric_volumetric=False,
    )
    u.interpolate(lambda x: x / 10.0)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F, p=1.0)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))

    if isinstance(comp_model, fenicsx_pulse.compressibility.Incompressible):
        assert math.isclose(value, 53.977812460748154)
    else:
        assert math.isclose(value, 49.57354795865461)


@pytest.mark.parametrize("isotropy", (fenicsx_pulse.active_stress.ActiveStressModels.transversely,))
@pytest.mark.parametrize(
    "comp_model",
    (fenicsx_pulse.compressibility.Incompressible(), fenicsx_pulse.compressibility.Compressible()),
)
def test_CardiacModel_NeoHookean(comp_model, isotropy, mesh, u):
    material = fenicsx_pulse.NeoHookean(
        mu=dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(15.0)),
    )
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    active_model = fenicsx_pulse.ActiveStress(f0, isotropy=isotropy)
    model = fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
        decouple_deviatoric_volumetric=False,
    )
    u.interpolate(lambda x: x / 10.0)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F, p=1.0)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))

    if isinstance(comp_model, fenicsx_pulse.compressibility.Incompressible):
        assert math.isclose(value, 5.056)
    else:
        assert math.isclose(value, 49.57354795865461)


@pytest.mark.parametrize("isotropy", (fenicsx_pulse.active_stress.ActiveStressModels.transversely,))
@pytest.mark.parametrize(
    "comp_model",
    (fenicsx_pulse.compressibility.Incompressible(), fenicsx_pulse.compressibility.Compressible()),
)
def test_CardiacModel_Guccione(comp_model, isotropy, mesh, u):
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    n0 = dolfinx.fem.Constant(mesh, (0.0, 0.0, 1.0))
    material_params = fenicsx_pulse.Guccione.default_parameters()
    material = fenicsx_pulse.Guccione(f0=f0, s0=s0, n0=n0, **material_params)
    active_model = fenicsx_pulse.ActiveStress(f0, isotropy=isotropy)
    model = fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
        decouple_deviatoric_volumetric=False,
    )
    u.interpolate(lambda x: x / 10.0)
    F = fenicsx_pulse.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F, p=1.0)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))

    if isinstance(comp_model, fenicsx_pulse.compressibility.Incompressible):
        assert math.isclose(value, 0.47245070311801096)
    else:
        assert math.isclose(value, 49.57354795865461)
