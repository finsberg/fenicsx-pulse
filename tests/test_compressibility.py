import math

import dolfinx
import pulsex
import ufl


def test_Incompressible(u, P1):
    p = dolfinx.fem.Function(P1)
    p.x.set(3.14)
    u.interpolate(lambda x: x)
    F = pulsex.kinematics.DeformationGradient(u)
    J = pulsex.kinematics.Jacobian(F)
    comp = pulsex.compressibility.Incompressible(p)
    psi = comp.strain_energy(J)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    assert math.isclose(value, 3.14 * (8 - 1))


def test_Compressible(u):

    u.interpolate(lambda x: x)
    F = pulsex.kinematics.DeformationGradient(u)
    J = pulsex.kinematics.Jacobian(F)
    kappa = 1234
    comp = pulsex.compressibility.Compressible(kappa=kappa)
    psi = comp.strain_energy(J)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    assert math.isclose(value, kappa * (8 * math.log(8) - 8 + 1))
