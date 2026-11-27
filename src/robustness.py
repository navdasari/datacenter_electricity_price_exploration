"""Robustness checks for the data center effect.

1. Gradient boosting + SHAP on the two-way-demeaned panel: the slope of the data center
   SHAP values on data center share gives a model-agnostic analogue of theta.
2. Synthetic control for Virginia, Oregon and Arizona (high data center share states):
   donor pool = states with low data center exposure; pre-period 2010-2019.

Settings come from config.RobustnessConfig. Usage: python3 robustness.py [--out DIR] [--processed DIR]
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.optimize import minimize

from config import ModelConfig, RobustnessConfig
from pipeline import X_COLS, within

ROOT = Path(__file__).resolve().parents[1]
PROC, OUTD = ROOT / "data/processed", ROOT / "output"


def shap_check(est):
    D = ["dc_share", "dc_share_cap"]
    cols = D + X_COLS
    W = within(est, ["y_net_cap"] + cols)
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=8, min_child_samples=10,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    m.fit(W[cols], W.y_net_cap)
    sv = shap.TreeExplainer(m).shap_values(W[cols])
    sv = pd.DataFrame(sv, columns=cols)
    cap = est.capmkt.values > 0
    out = {}
    for name, mask in [("non_capacity_market", ~cap), ("capacity_market", cap)]:
        x = W.dc_share.values[mask]
        y = (sv.dc_share + sv.dc_share_cap).values[mask]
        out[name] = float(np.polyfit(x, y, 1)[0])
    imp = sv.abs().mean().sort_values(ascending=False)
    return out, imp.round(4).to_dict()


def synth(panel, treated, donors, pre=range(2010, 2020), post=range(2020, 2026)):
    Y = panel.pivot(index="year", columns="state", values="res_price")
    y1, Y0 = Y[treated], Y[donors]
    pre_idx = [y for y in pre if y in Y.index]

    def loss(w):
        return np.sum((y1.loc[pre_idx].values - Y0.loc[pre_idx].values @ w) ** 2)

    n = len(donors)
    r = minimize(loss, np.ones(n) / n, method="SLSQP", bounds=[(0, 1)] * n,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 500})
    w = r.x
    gap = y1 - Y0.values @ w
    pre_rmspe = np.sqrt(np.mean(gap.loc[pre_idx] ** 2))
    post_idx = [y for y in post if y in Y.index]
    post_rmspe = np.sqrt(np.mean(gap.loc[post_idx] ** 2))
    return gap, pd.Series(w, index=donors), pre_rmspe, post_rmspe


def synth_all(ann, cfg: RobustnessConfig = RobustnessConfig()):
    pre = range(cfg.synth_pre[0], cfg.synth_pre[1] + 1)
    post = range(cfg.synth_post[0], cfg.synth_post[1] + 1)
    p = ann[ann.year.between(cfg.synth_pre[0], cfg.synth_post[1])]
    dc23 = p[p.year == 2023].set_index("state").dc_share
    # Donors: low data center exposure, not in a capacity market (whose 2025 shock would bias the gap)
    capm = p[p.year == 2023].set_index("state").capmkt
    donors = [s for s in dc23.index if dc23[s] < cfg.donor_max_dc_share and capm[s] == 0 and s not in cfg.donor_exclude]
    res = {}
    for t in ["VA", "OR", "AZ"]:
        gap, w, pre_r, post_r = synth(p, t, donors, pre, post)
        # placebo-in-space: rank of post/pre RMSPE ratio
        ratios = []
        for d in donors:
            g2, _, pr2, po2 = synth(p, d, [x for x in donors if x != d], pre, post)
            ratios.append(po2 / pr2)
        ratio = post_r / pre_r
        dshare = dc23.get(t) - p[(p.year == 2019) & (p.state == t)].dc_share.iloc[0]
        dshare25 = p[(p.year == 2025) & (p.state == t)].dc_share.iloc[0] - p[(p.year == 2019) & (p.state == t)].dc_share.iloc[0]
        res[t] = {
            "gap_by_year": gap.loc[2015:2025].round(3).to_dict(),
            "avg_post_gap_cents": float(gap.loc[2020:2025].mean()),
            "gap_2025_cents": float(gap.loc[2025]),
            "dc_share_change_2019_2025_pp": float(dshare25),
            "implied_effect_per_pp": float(gap.loc[2025] / dshare25) if dshare25 else None,
            "pre_rmspe": float(pre_r), "post_pre_ratio": float(ratio),
            "placebo_p_value": float((np.sum(np.array(ratios) >= ratio) + 1) / (len(ratios) + 1)),
            "top_donors": w[w > 0.02].sort_values(ascending=False).round(3).to_dict(),
        }
    return res, donors


def main(cfg: ModelConfig, out_dir: Path, proc_dir: Path):
    est = pd.read_csv(proc_dir / "estimation_sample.csv")
    feats = pd.read_csv(proc_dir / "features_all.csv")
    ann = feats[feats.period.isna()]
    slopes, imp = shap_check(est)
    sc, donors = synth_all(ann, cfg.robustness)
    out = {"shap_dc_slope": slopes, "shap_mean_abs_importance": imp, "synthetic_control": sc, "donor_pool": donors}
    (out_dir / "robustness.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTD))
    ap.add_argument("--processed", default=str(PROC))
    a = ap.parse_args()
    out = main(ModelConfig.baseline(), Path(a.out), Path(a.processed))
    sc = out["synthetic_control"]
    print(json.dumps({k: v for k, v in out.items() if k != "synthetic_control"}, indent=1, default=float))
    for t, r in sc.items():
        print(t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "gap_by_year"})
        print("   gap:", r["gap_by_year"])
