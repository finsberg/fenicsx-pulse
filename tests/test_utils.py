from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import fenicsx_pulse


@pytest.mark.parametrize("element", [("P", 1), ("P", 2)])
@pytest.mark.parametrize(
    "celltype",
    [dolfinx.cpp.mesh.CellType.triangle, dolfinx.cpp.mesh.CellType.quadrilateral],
)
def test_vertex_to_dofmap(element, celltype):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, celltype)
    V = dolfinx.fem.functionspace(mesh, element)

    v2d = fenicsx_pulse.utils.vertex_to_dofmap(V)

    X = ufl.SpatialCoordinate(mesh)

    s = dolfinx.fem.Function(V)
    s.interpolate(dolfinx.fem.Expression(X[0] + X[1], V.element.interpolation_points()))

    vert_values = mesh.geometry.x.sum(1)
    assert np.allclose(vert_values, s.x.array[v2d])
