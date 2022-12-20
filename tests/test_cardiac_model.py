import math

import dolfinx
import pulsex
import ufl


def test_CardiacModel(mesh, u):

    material_params = pulsex.HolzapfelOgden.transversely_isotropic_parameters()
    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    material = pulsex.HolzapfelOgden(f0=f0, s0=s0, **material_params)

    active_model = pulsex.ActiveStress(f0)
    comp_model = pulsex.Compressible()

    model = pulsex.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )
    u.interpolate(lambda x: x / 10.0)
    F = pulsex.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    assert math.isclose(value, 103.22036041941614)
