import logging

import fenicsx_pulse
import fenicsx_pulse.cli


def test_version(caplog):
    caplog.set_level(logging.INFO)
    ret = fenicsx_pulse.cli.main(["version"])
    assert ret == 0
    assert caplog.records[0].msg == f"fenicsx-pulse: {fenicsx_pulse.__version__}"
