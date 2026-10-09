import csv

import pytest
from cli_helpers import write_cfg

from pulse.cli.config import ConfigError
from pulse.cli.overrides import load_config
from pulse.cli.postprocess import _plots, column_groups, run_post
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


def test_post_plots_coupled_columns_by_prefix():
    columns = [
        "t", "activation", "pressure_X1", "volume_X1",
        "volume_LV", "pressure_LV", "phase_LV", "Pc_LV", "Q_LV",
        "circ_V_LA", "circ_p_AR_SYS", "circ_Q_MV",
    ]  # fmt: skip
    groups = column_groups(columns, ["activation", "pressure_X1"])
    assert groups["loads"] == ["activation", "pressure_X1"]  # never phase_/Pc_/Q_/circ_
    assert groups["cavities"] == ["X1", "LV"]
    assert groups["phases"] == ["LV"]
    assert groups["circulation"] == ["V_LA", "p_AR_SYS", "Q_MV"]


@pytest.mark.skip_in_parallel
def test_post_writes_coupled_plots(tmp_path):
    pytest.importorskip("matplotlib")
    folder, post = tmp_path / "out", tmp_path / "out" / "post"
    post.mkdir(parents=True)
    header = "t,activation,volume_LV,pressure_LV,phase_LV,circ_V_LA,circ_p_AR_SYS\n"
    rows = "".join(
        f"{0.001 * i},{100.0 * i},{1e-4 + 1e-6 * i},{1000.0 + 50 * i},{min(i // 2, 4)},"
        f"{50 + i},{80 - i}\n"
        for i in range(10)
    )
    (folder / "loads.csv").write_text(header + rows)
    _plots(folder, post, ["activation"])
    names = {p.name for p in post.iterdir()}
    assert {"loads.png", "cavities.png", "pv_loop_LV.png", "circulation.png"} <= names
