import dolfinx
import pytest
import ufl
from mpi4py import MPI


@pytest.fixture(scope="session")
def mesh():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)


@pytest.fixture(scope="session")
def P2(mesh):
    return dolfinx.fem.FunctionSpace(mesh, ufl.VectorElement("CG", mesh.ufl_cell(), 2))


@pytest.fixture
def u(P2):
    return dolfinx.fem.Function(P2)
