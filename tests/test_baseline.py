"""Baseline regression tests.

tests/golden/ holds the published outputs from before the configuration refactor. The
pipeline run with ModelConfig.baseline() must reproduce them to within TOL (floating-point
noise only).

Run:  python3 -m pytest tests -q        (or: python3 tests/test_baseline.py)
The full-pipeline test takes about a minute (four specifications of double ML).
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GOLD = ROOT / "tests/golden"
TOL = 1e-9

from config import ModelConfig  # noqa: E402
import pipeline  # noqa: E402


def compare_csv(a: Path, b: Path):
    x, y = pd.read_csv(a), pd.read_csv(b)
    assert list(x.columns) == list(y.columns), f"{a.name}: columns differ"
    assert len(x) == len(y), f"{a.name}: row count differs"
    num = x.select_dtypes("number").columns
    for c in x.columns:
        if c in num:
            d = np.nanmax(np.abs(x[c].values - y[c].values)) if len(x) else 0
            assert (np.isnan(x[c].values) == np.isnan(y[c].values)).all(), f"{a.name}:{c} NaN pattern differs"
            assert not d > TOL, f"{a.name}:{c} max diff {d}"
        else:
            assert (x[c].fillna("").astype(str) == y[c].fillna("").astype(str)).all(), f"{a.name}:{c} differs"


def compare_json(gold, new, path="$"):
    """Every value in `gold` must match `new`; extra keys in `new` are allowed."""
    if isinstance(gold, dict):
        for k, v in gold.items():
            assert k in new, f"{path}.{k} missing"
            compare_json(v, new[k], f"{path}.{k}")
    elif isinstance(gold, list):
        assert len(gold) == len(new), f"{path}: length differs"
        for i, (g, n) in enumerate(zip(gold, new)):
            compare_json(g, n, f"{path}[{i}]")
    elif isinstance(gold, (int, float)) and not isinstance(gold, bool):
        assert abs(gold - new) <= TOL, f"{path}: {gold} vs {new}"
    else:
        assert gold == new, f"{path}: {gold!r} vs {new!r}"


def test_full_pipeline_reproduces_published_outputs():
    import chart
    import model
    import robustness
    cfg = ModelConfig.baseline()
    with tempfile.TemporaryDirectory() as tmp:
        out, proc = Path(tmp), Path(tmp) / "processed"
        model.main(cfg, out, proc, verbose=False)
        robustness.main(cfg, out, proc)
        chart.main(out, png=False)
        for f in ["decomposition.csv", "national_effects.csv", "capacity_outlook.csv"]:
            compare_csv(GOLD / f, out / f)
        for f in ["estimation_sample.csv", "features_all.csv"]:
            compare_csv(GOLD / f, proc / f)
        for f in ["model_results.json", "robustness.json", "chart_data.json"]:
            compare_json(json.loads((GOLD / f).read_text()), json.loads((out / f).read_text()), f)


def test_published_estimates_reproduce_decomposition():
    """The app's path: baseline estimates loaded from JSON + mechanical attribution."""
    cfg = ModelConfig.baseline()
    est = pipeline.estimates_from_results_json(GOLD / "model_results.json")
    _, _, res = pipeline.run(cfg, est=est)
    with tempfile.TemporaryDirectory() as tmp:
        res.decomposition.to_csv(Path(tmp) / "d.csv", index=False)
        compare_csv(GOLD / "decomposition.csv", Path(tmp) / "d.csv")


def test_mechanical_assumptions_do_not_change_estimation_key():
    base = ModelConfig.baseline()
    for k, v in {"spreading.fixed_share": 0.3, "inflation.pass_through": 0.8,
                 "inflation.current_year_weight": 1.0, "capacity.dc_share_2025_2027": 0.6,
                 "capacity.dc_share_band": 0.1, "uncertainty.combine": "quadrature"}.items():
        assert base.with_values(**{k: v}).estimation_key() == base.estimation_key(), k
    for k, v in {"capacity.pass_through_vert_int": 0.5, "dc_load.validation_tol": 0.5,
                 "estimation.lag": False, "fuel.heat_rate": 8.0}.items():
        assert base.with_values(**{k: v}).estimation_key() != base.estimation_key(), k


def test_mechanical_invariants():
    """Spreading never changes the net DC effect; the capacity share never changes the
    national bar or the total capacity effect; inflation never changes the national total."""
    base = ModelConfig.baseline()
    est = pipeline.estimates_from_results_json(GOLD / "model_results.json")
    panel = pipeline.load_panel()
    _, _, r0 = pipeline.run(base, panel, est)
    _, _, r1 = pipeline.run(base.with_values(**{"spreading.fixed_share": 0.2}), panel, est)
    _, _, r2 = pipeline.run(base.with_values(**{"capacity.dc_share_2025_2027": 0.7}), panel, est)
    _, _, r3 = pipeline.run(base.with_values(**{"inflation.pass_through": 0.5}), panel, est)
    d0, d1, d2, d3 = (r.decomposition for r in (r0, r1, r2, r3))
    assert np.allclose(d0._dc_net, d1._dc_net, atol=1e-12)
    assert not np.allclose(d0["Data centers: fixed-cost spreading"], d1["Data centers: fixed-cost spreading"])
    cap_total = lambda d: d["Capacity market (non-DC)"] + d._dc_capacity
    assert np.allclose(cap_total(d0), cap_total(d2), atol=1e-12)
    assert np.allclose(pd.Series(r0.national), pd.Series(r2.national), atol=1e-12, equal_nan=True)
    assert np.allclose(pd.Series(r0.national), pd.Series(r3.national), atol=1e-12, equal_nan=True)
    nat_total = lambda d: d["National: inflation"] + d["National: grid costs & interest rates above inflation"]
    assert np.allclose(nat_total(d0), nat_total(d3), atol=1e-12)


def test_config_round_trip():
    cfg = ModelConfig.baseline().with_values(**{"capacity.pass_through_vert_int": 0.35})
    assert ModelConfig.from_json(cfg.to_json()) == cfg


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
