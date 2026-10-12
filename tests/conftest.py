from mpi4py import MPI

import dolfinx
import pytest


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


def _assert_one_space_per_block(problem):
    """Every block row/column of `problem` is on its own space object.

    dolfinx's blocked assembly (0.11 nightly on) refuses a form whose rows or
    columns share one space, so two real-valued unknowns need two (cloned)
    spaces. The test/trial functions and old states of a block must be on the
    same space as its state.
    """
    spaces = [f.ufl_function_space()._cpp_object for f in problem.states]
    assert len({id(V) for V in spaces}) == len(spaces)
    for name in ("test_functions", "trial_functions", "old_states"):
        others = [f.ufl_function_space()._cpp_object for f in getattr(problem, name)]
        assert all(a is b for a, b in zip(others, spaces, strict=True)), name


@pytest.fixture
def assert_one_space_per_block():
    return _assert_one_space_per_block
