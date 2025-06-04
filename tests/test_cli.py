import logging

import pulse


def test_version(caplog):
    caplog.set_level(logging.INFO)
    ret = pulse.cli.main(["version"])
    assert ret == 0
    assert caplog.records[0].msg == f"fenicsx-pulse: {pulse.__version__}"
