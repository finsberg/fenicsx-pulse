import dolfinx
import numpy as np
import ufl


def matrix_is_zero(A: ufl.core.expr.Expr) -> bool:
    n = ufl.domain.find_geometric_dimension(A)
    for i in range(n):
        for j in range(n):
            value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(A[i, j] * ufl.dx))
            print(i, j, value)
            is_zero = np.isclose(value, 0)
            if not is_zero:
                return False
    return True


def IsochoricDeformationGradient(u) -> ufl.core.expr.Expr:
    from pulse import kinematics

    return kinematics.IsochoricDeformationGradient(kinematics.DeformationGradient(u))


def float2object(
    f: float,
    obj_str: str,
    mesh: dolfinx.mesh.Mesh,
    V: dolfinx.fem.FunctionSpace,
):
    if obj_str == "float":
        return f
    if obj_str == "Constant":
        return dolfinx.fem.Constant(mesh, f)
    if obj_str == "Function":
        v = dolfinx.fem.Function(V)
        v.x.array[:] = f
        return v
    raise ValueError(f"Invalid object string {obj_str!r}")
