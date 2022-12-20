import pulsex
import ufl
import utils


def test_linear_elastic_model(mesh, u):
    E = 2.0
    nu = 0.2
    model = pulsex.LinearElastic(E=E, nu=nu)

    u.interpolate(lambda x: x)
    F = pulsex.kinematics.DeformationGradient(u)
    # F = 2I, e = I, tr(e) = 3
    # sigma = (E / (1 + nu)) * (e + (nu / (1 - 2 * nu)) * tr(e) * I
    # sigma = (E / (1 + nu)) * (1 + (nu / (1 - 2 * nu)) * 3) * I
    sigma = model.sigma(F)
    I = ufl.Identity(3)
    zero = sigma - (E / (1 + nu)) * (1 + (nu / (1 - 2 * nu)) * 3) * I
    assert utils.matrix_is_zero(zero)
