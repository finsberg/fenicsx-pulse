import math

import dolfinx
import pytest
import ufl

import pulse


def test_Incompressible(u, P1) -> None:
    p = dolfinx.fem.Function(P1)
    p.x.array[:] = 3.14
    u.interpolate(lambda x: x)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    comp = pulse.compressibility.Incompressible()
    comp.register(p)
    psi = comp.strain_energy(C)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    assert math.isclose(value, 3.14 * (8 - 1))


def test_Incompressible_with_missing_pressure_raises_MissingModelAttribute(u) -> None:
    u.interpolate(lambda x: x)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    comp = pulse.compressibility.Incompressible()
    with pytest.raises(pulse.exceptions.MissingModelAttribute):
        comp.strain_energy(C)


def test_Compressible(u) -> None:
    u.interpolate(lambda x: x)
    F = pulse.kinematics.DeformationGradient(u)
    C = F.T * F
    kappa = 1234
    comp = pulse.compressibility.Compressible(kappa=pulse.Variable(kappa, "Pa"))
    psi = comp.strain_energy(C)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi * ufl.dx))
    assert math.isclose(value, kappa * (8 * math.log(8) - 8 + 1))
