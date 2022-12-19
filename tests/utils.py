import dolfinx
import numpy as np
import ufl


def matrix_is_zero(A: ufl.core.expr.Expr) -> bool:
    n = ufl.domain.find_geometric_dimension(A)
    for i in range(n):
        for j in range(n):
            value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(A[i, j] * ufl.dx))
            is_zero = np.isclose(value, 0)
            if not is_zero:
                return False
    return True


def IsochoricDeformationGradient(u):
    from pulsex import kinematics

    return kinematics.IsochoricDeformationGradient(kinematics.DeformationGradient(u))
