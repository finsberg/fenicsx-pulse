import dolfinx
import pytest
from mpi4py import MPI


@pytest.fixture(scope="session")
def mesh():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)


@pytest.fixture(scope="session")
def P1(mesh):
    return dolfinx.fem.functionspace(mesh, ("Lagrange", 1))


@pytest.fixture(scope="session")
def P2(mesh):
    return dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim,)))


@pytest.fixture
def u(P2):
    return dolfinx.fem.Function(P2)
