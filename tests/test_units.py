import numpy as np

from fenicsx_pulse import units


def test_Variable():
    value = units.Variable(1.0, unit="kPa / cm")
    assert np.isclose(value.to_base_units(), 1e3 / 1e-2)
