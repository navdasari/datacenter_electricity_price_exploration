"""Run the published model: the headline specification plus three alternatives.

All assumptions live in config.py and all modeling logic in pipeline.py; this script only
chooses the specifications and writes the outputs.

Usage:  python3 model.py [--config scenario.json] [--out DIR] [--processed DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ModelConfig
from pipeline import load_panel, run

ROOT = Path(__file__).resolve().parents[1]

SPECS = {  # name -> estimation overrides
    "state-specific, with lag (headline)": {"estimation.exposure": "state_specific", "estimation.lag": True},
    "state-specific, no lag": {"estimation.exposure": "state_specific", "estimation.lag": False},
    "national scaling, with lag": {"estimation.exposure": "national_scaling", "estimation.lag": True},
    "national scaling, no lag (original)": {"estimation.exposure": "national_scaling", "estimation.lag": False},
}
HEADLINE = "state-specific, with lag (headline)"


def main(cfg: ModelConfig, out_dir: Path, proc_dir: Path, verbose=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    spec_results, head = {}, None
    for name, overrides in SPECS.items():
        c = cfg.with_values(**overrides)
        prep, est, res = run(c, panel)
        spec_results[name] = {"exposure": c.estimation.exposure_col(), "lag": c.estimation.lag, "n_obs": est.n_obs,
                              "dml_theta": est.theta.to_dict(), "dml_cov": est.cov.values.tolist(),
                              "ols_theta": est.ols_theta.to_dict(), "ols_cov": est.ols_cov.values.tolist(),
                              "cumulative_2022_2025": res.cumulative_2022_2025}
        if verbose:
            print(f"\n== {name}: n={est.n_obs}  DML theta {est.theta.round(3).to_dict()}")
            for k, v in res.cumulative_2022_2025.items():
                print(f"   {k:28s} DC net {v['dc_net']:+.2f}  [{v['lo']:+.2f}, {v['hi']:+.2f}]   actual {v['actual']:+.2f}")
        if name == HEADLINE:
            head = (c, prep, est, res)

    c, prep, est, res = head
    res.decomposition.to_csv(out_dir / "decomposition.csv", index=False)
    res.national_frame().to_csv(out_dir / "national_effects.csv")
    res.outlook.to_csv(out_dir / "capacity_outlook.csv", index=False)
    D = list(est.theta.index)
    results = {
        "headline_spec": HEADLINE,
        "dml_theta": est.theta.to_dict(), "dml_cov": est.cov.values.tolist(), "dml_terms": D,
        "dml_se": dict(zip(D, np.sqrt(np.diag(est.cov.values)))),
        "ols_fe_theta": est.ols_theta.to_dict(), "ols_fe_se": est.ols_se.to_dict(),
        "other_betas": est.betas.to_dict(), "other_se": est.betas_se.to_dict(),
        "n_obs": est.n_obs, "n_states": est.n_states, "ytd_months": prep.last,
        "specs": spec_results, "dc_load_validation": prep.validation, "national_dc_test": res.national_test,
        "assumptions": {"FIXED_SHARE": c.spreading.fixed_share, "DC_CAP_SHARE": c.capacity.dc_cap_share(),
                        "CAP_TO_CKWH": c.capacity.cap_to_ckwh(),
                        "PASS_THROUGH_VERT_INT": c.capacity.pass_through_vert_int},
        "config": c.to_dict(), "estimation_key": c.estimation_key(),
    }
    (out_dir / "model_results.json").write_text(json.dumps(results, indent=2, default=float))
    proc_dir.mkdir(parents=True, exist_ok=True)
    prep.est.to_csv(proc_dir / "estimation_sample.csv", index=False)
    pd.concat([prep.ann, prep.ytd.assign(period="ytd")]).to_csv(proc_dir / "features_all.csv", index=False)
    if verbose:
        print("\nnational DC test:", {k: (round(v, 3) if isinstance(v, float) else "...") for k, v in res.national_test.items()})
        print(res.outlook.round(2).to_string(index=False))
        print(pd.DataFrame(prep.validation).T.round(2))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="JSON scenario file (defaults to the baseline)")
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--processed", default=str(ROOT / "data/processed"))
    a = ap.parse_args()
    cfg = ModelConfig.from_json(Path(a.config).read_text()) if a.config else ModelConfig.baseline()
    main(cfg, Path(a.out), Path(a.processed))
