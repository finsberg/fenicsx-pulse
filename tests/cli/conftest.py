"""Fixtures shared by the CLI tests.

The runner assumes one output folder shared by every rank, but pytest's ``tmp_path`` differs per
rank under ``mpirun``; every rank therefore uses rank 0's.
"""

import gc

from mpi4py import MPI

import pytest


@pytest.fixture
def tmp_path(tmp_path):
    return MPI.COMM_WORLD.bcast(tmp_path, root=0)


@pytest.fixture(autouse=True)
def _collect_garbage():
    """Free the previous test's dolfinx/PETSc objects on every rank at the same point.

    Their ``__del__`` destroys PETSc objects collectively; left to the cyclic garbage collector,
    that happens at a different allocation on each rank under ``mpirun`` and deadlocks.
    """
    gc.collect()
    MPI.COMM_WORLD.barrier()
    yield
