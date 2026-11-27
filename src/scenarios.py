"""Precompute headline estimates for common alternative assumptions.

Each scenario changes assumptions that require re-estimation (one at a time, from the
baseline). The dashboard loads these instantly instead of re-estimating. Mechanical
assumptions never need a scenario: they are applied on top of any estimates.

Writes output/scenarios/index.json and output/scenarios/<estimation_key>.json.
Usage: python3 scenarios.py   (about 12 s per scenario)
"""
import json
from pathlib import Path

from config import ModelConfig
from pipeline import estimate, load_panel, prepare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/scenarios"

SCENARIOS = [
    ("Virginia/WV capacity pass-through 0%", {"capacity.pass_through_vert_int": 0.0}),
    ("Virginia/WV capacity pass-through 50%", {"capacity.pass_through_vert_int": 0.5}),
    ("Virginia/WV capacity pass-through 100%", {"capacity.pass_through_vert_int": 1.0}),
    ("Residential load factor at peak 0.45", {"capacity.residential_peak_load_factor": 0.45}),
    ("Residential load factor at peak 0.65", {"capacity.residential_peak_load_factor": 0.65}),
    ("Illinois not in PJM", {"capacity.illinois_pjm_weight": 0.0}),
    ("All of Illinois in PJM", {"capacity.illinois_pjm_weight": 1.0}),
    ("Virginia uses the PJM-wide capacity price", {"capacity.use_dominion_zone": False}),
    ("Validation tolerance 20%", {"dc_load.validation_tol": 0.2}),
    ("Validation tolerance 50%", {"dc_load.validation_tol": 0.5}),
    ("Slower national data center growth (2025: 210, 2026: 230 TWh)",
     {"dc_load.national_2025_twh": 210.0, "dc_load.national_2026_twh": 230.0}),
    ("Faster national data center growth (2025: 250, 2026: 320 TWh)",
     {"dc_load.national_2025_twh": 250.0, "dc_load.national_2026_twh": 320.0}),
    ("Allocate remaining load by commercial + industrial sales", {"dc_load.remainder_allocation": "commercial_industrial"}),
    ("Allocate remaining load by total sales", {"dc_load.remainder_allocation": "total"}),
    ("No floor on the state-specific measure", {"dc_load.floor_frac": 0.0}),
    ("Floor at 50% of national scaling", {"dc_load.floor_frac": 0.5}),
    ("Trend window 2014–2019", {"dc_load.trend_start": 2014}),
    ("National-scaling load measure", {"estimation.exposure": "national_scaling"}),
    ("No one-year lag", {"estimation.lag": False}),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    base = ModelConfig.baseline()
    index = []
    for label, changes in [("Baseline", {})] + SCENARIOS:
        cfg = base.with_values(**changes)
        key = cfg.estimation_key()
        path = OUT / f"{key}.json"
        if not path.exists():
            est = estimate(prepare(panel, cfg), cfg, label)
            path.write_text(json.dumps(est.to_dict(), default=float))
        index.append({"label": label, "changes": changes, "key": key})
        print(f"{key}  {label}")
    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
