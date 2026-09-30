"""Fixtures shared by the CLI tests.

The runner assumes one output folder shared by every rank, but pytest's ``tmp_path`` differs per
rank under ``mpirun``; every rank therefore uses rank 0's.
"""

from mpi4py import MPI

import pytest


@pytest.fixture
def tmp_path(tmp_path):
    return MPI.COMM_WORLD.bcast(tmp_path, root=0)
