import csv

import pytest
from cli_helpers import write_cfg

from pulse.cli.config import ConfigError
from pulse.cli.overrides import load_config
from pulse.cli.postprocess import run_post
from pulse.cli.runner import run


def _run(tmp_path, **over):
    post = {"fields": ["fiber_stress", "fiber_strain"], "points": {"tip": [1.0, 0.5, 0.5]}}
    conf = load_config(write_cfg(tmp_path, postprocess=post, **over), environ={})
    run(conf)
    return conf


def test_post_writes_everything(tmp_path):
    conf = _run(
        tmp_path,
        geometry={"type": "box", "nx": 2, "ny": 2, "nz": 2, "fibers": {"type": "axis"}},
    )
    post = run_post(conf)
    assert (post / "displacement.bp").exists()
    assert (post / "fields.bp").exists()
    with open(post / "points.csv") as f:
        rows = list(csv.DictReader(f))
    assert [float(r["t"]) for r in rows] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert set(rows[0]) == {"t", "tip_ux", "tip_uy", "tip_uz"}
    assert float(rows[-1]["tip_ux"]) != 0.0
    pytest.importorskip("matplotlib")
    assert (post / "loads.png").exists()


def test_post_refuses_other_physics(tmp_path):
    conf = _run(tmp_path)
    changed = load_config(
        conf.output.folder.parent / "config.toml",
        sets=['material.mu="16 kPa"'],
        environ={},
    )
    with pytest.raises(ConfigError, match="physics"):
        run_post(changed)


def test_post_without_results(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    with pytest.raises(ConfigError, match="pulse run"):
        run_post(conf)


def test_point_outside_mesh_is_config_error(tmp_path):
    conf = _run(tmp_path)
    outside = load_config(
        conf.output.folder.parent / "config.toml",
        sets=["postprocess.points.far=[5.0, 5.0, 5.0]"],
        environ={},
    )
    with pytest.raises(ConfigError, match="far"):
        run_post(outside)
