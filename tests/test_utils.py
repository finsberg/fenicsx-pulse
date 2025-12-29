import dolfinx
import numpy as np

from pulse.utils import map_vector_field


def test_map_vector_field(mesh):
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,))), name="u")

    # Define a simple displacement field that rotate the mesh 45 degrees around the z-axis
    # but do not scale the mesh
    def displacement(x):
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        x_new = c * x[0] - s * x[1] - x[0]
        y_new = s * x[0] + c * x[1] - x[1]
        return np.array([x_new, y_new, x[2]])

    u.interpolate(displacement)
    f0 = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,))), name="f0")
    f0.interpolate(lambda x: (np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0])))
    f0_mapped = map_vector_field(f0, u=u, normalize=True, name="f0_mapped")

    f_expected = dolfinx.fem.Function(
        dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,))),
        name="f_expected",
    )
    f_expected.interpolate(
        lambda x: (
            np.sqrt(2) / 2 * np.ones_like(x[0]),
            np.sqrt(2) / 2 * np.ones_like(x[0]),
            np.zeros_like(x[0]),
        ),
    )
    assert np.allclose(f0_mapped.x.array, f_expected.x.array)
