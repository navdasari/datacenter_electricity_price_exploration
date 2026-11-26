"""Estimate factor effects on residential electricity prices and decompose yearly changes.

0. Capacity-market costs (PJM, ISO-NE) are computed mechanically from auction prices
   and removed from the price before estimation; the PJM portion attributed to data
   centers by the market monitor (~45% from 2025/26) is assigned to data centers.
1. Double ML (partially linear model, two-way state + year fixed effects via
   within-transform, LightGBM nuisance models, state-grouped cross-fitting) for the
   local data center effect, allowed to differ for capacity-market states.
2. Two-way fixed-effects regression of (adjusted price - DC effect) on the other factors.
3. Yearly decomposition for the target markets: contribution_k = beta_k * delta x_k.
   The year effects become a "national common factors" bar. The DC net effect is split
   into a mechanical fixed-cost-spreading term (negative) and an implied cost-push term.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold

from build_panel import aggregate, datacenter_load
from markets import MARKETS

ROOT = Path(__file__).resolve().parents[1]
PROC, OUTD = ROOT / "data/processed", ROOT / "output"
OUTD.mkdir(exist_ok=True)

# --- Assumptions (documented in README) ---------------------------------------
FIXED_SHARE = 0.5          # share of residential rate that is fixed cost re-spread over load
DC_CAP_SHARE = {2025: 0.45, 2026: 0.45, 2027: 0.45}  # IMM: ~45% of PJM capacity charges from DC load
CAP_TO_CKWH = 365 / (8760 * 0.55) / 10  # $/MW-day -> cents/kWh (residential load factor at peak ~0.55)
PASS_THROUGH_VERT_INT = 0.2  # Dominion/WV utilities self-supply most capacity: muted retail exposure
EST_YEARS = range(2010, 2026)
# Restructured retail states: gas sets the marginal wholesale price that flows to retail,
# regardless of the in-state generation mix.
RETAIL_CHOICE = {"CT", "DE", "DC", "IL", "ME", "MD", "MA", "NH", "NJ", "NY", "OH", "PA", "RI", "TX"}


GROUPS = {
    "Natural gas / fuel": ["gas_cost", "gas_cost_l1", "gas_marg", "gas_marg_l1"],
    "Hydro conditions": ["hydro_share"],
    "Weather": ["hdd", "cdd"],
    "Lower use per customer (efficiency, rooftop solar)": ["ln_kwh_cust"],
    "Non-DC load growth": ["ln_nondc_load"],
}
X_COLS = [c for g in GROUPS.values() for c in g]


def features(a):
    a = a.copy()
    a["dc_share"] = a.dc_mwh / a.tot_mwh * 100
    a["capmkt"] = ((a.pjm_w + a.isone_w) > 0).astype(float)
    a["dc_share_cap"] = a.dc_share * a.capmkt
    # delivery-year mapping is approximate at calendar-year resolution
    dcf = a.year.map(DC_CAP_SHARE).fillna(0) * (a.pjm_w > 0)
    rho = np.where(a.cap_restructured > 0, 1.0, PASS_THROUGH_VERT_INT)
    a["cap_ckwh"] = a.cap_price * CAP_TO_CKWH * rho
    a["cap_dc"], a["cap_nondc"] = a.cap_ckwh * dcf, a.cap_ckwh * (1 - dcf)
    a["y_net_cap"] = a.res_price - a.cap_ckwh
    # marginal-cost gas exposure in restructured retail states (7.5 MMBtu/MWh heat rate)
    a["gas_marg"] = a.gas_price * 7.5 / 10 * a.state.isin(RETAIL_CHOICE)
    a["ln_kwh_cust"] = np.log(a.res_kwh_per_cust)
    a["ln_nondc_load"] = np.log(a.tot_mwh - a.dc_mwh)
    a["hdd"], a["cdd"] = a.hdd / 1000, a.cdd / 1000
    return a


def within(df, cols, iters=50):
    """Two-way (state + year) within transform by alternating projections."""
    w = df[cols].astype(float).copy()
    for _ in range(iters):
        w = w - w.groupby(df.state.values).transform("mean")
        w = w - w.groupby(df.year.values).transform("mean")
    return w


def cluster_ols(y, X, groups):
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})


def dml(a, n_rep=5):
    D = ["dc_share", "dc_share_cap"]
    W = within(a, ["y_net_cap"] + D + X_COLS)
    Xf = W[X_COLS]
    states = a.state.values
    thetas, ses = [], []
    for rep in range(n_rep):
        rng = np.random.default_rng(rep)
        perm = {s: rng.random() for s in np.unique(states)}
        order = np.array([perm[s] for s in states])
        res = {c: np.zeros(len(a)) for c in ["y_net_cap"] + D}
        for tr, te in GroupKFold(n_splits=5).split(Xf, groups=np.round(order * 1000)):
            for c in res:
                m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=10,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=rep, verbose=-1)
                m.fit(Xf.iloc[tr], W[c].iloc[tr])
                res[c][te] = W[c].iloc[te] - m.predict(Xf.iloc[te])
        f = cluster_ols(res["y_net_cap"], np.column_stack([res[d] for d in D]), states)
        thetas.append(f.params)
        ses.append(f.bse)
    th = np.mean(thetas, axis=0)
    # median-aggregation style SE: include across-split variance
    se = np.sqrt(np.mean(np.array(ses) ** 2 + (np.array(thetas) - th) ** 2, axis=0))
    return dict(zip(D, th)), dict(zip(D, se))


def ols_fe(a, include_dc=True):
    cols = (["dc_share", "dc_share_cap"] if include_dc else []) + X_COLS
    W = within(a, ["y_net_cap"] + cols)
    return cluster_ols(W.y_net_cap, W[cols], a.state.values)


def other_betas(a, theta):
    a = a.copy()
    a["y_adj"] = a.y_net_cap - theta["dc_share"] * a.dc_share - theta["dc_share_cap"] * a.dc_share_cap
    W = within(a, ["y_adj"] + X_COLS)
    return cluster_ols(W.y_adj, W[X_COLS], a.state.values)


def period_frames(m, dc):
    """Return (annual frame incl. 2010-2025, ytd frame for Jan-Sep 2025 & 2026)."""
    ann = aggregate(m)
    ann = ann[ann.n_months == 12].merge(dc, on=["state", "year"])
    last = int(m[m.state == "VA"].query("year==2026").month.max())
    ytd = aggregate(m, range(1, last + 1))
    ytd = ytd[ytd.year.isin([2025, 2026]) & (ytd.n_months == last)].merge(dc, on=["state", "year"])
    ytd["dc_mwh"] *= last / 12
    ann = ann.sort_values(["state", "year"])
    ann, ytd = features(ann), features(ytd)
    ann["gas_cost_l1"] = ann.groupby("state").gas_cost.shift(1)
    ann["gas_marg_l1"] = ann.groupby("state").gas_marg.shift(1)
    # YTD lag: prior year's same-month window (2026 YTD -> 2025 YTD; 2025 YTD -> 2024 YTD)
    y24 = features(aggregate(m, range(1, last + 1)).query("year==2024 and n_months==@last").merge(dc, on=["state", "year"]))
    lagf = pd.concat([y24, ytd[ytd.year == 2025]]).assign(year=lambda d: d.year + 1).set_index(["state", "year"])
    idx = ytd.set_index(["state", "year"]).index
    ytd["gas_cost_l1"] = idx.map(lagf.gas_cost)
    ytd["gas_marg_l1"] = idx.map(lagf.gas_marg)
    return ann, ytd, last


def inflation_component(m, last):
    """Inflation pass-through: prior national avg residential price x CPI inflation
    (average of this year's and last year's rate, as rate cases lag). Pass-through of 1
    is calibrated, not estimated: it matches the 2012-2019 national common trend."""
    cpi = m.groupby(["year", "month"]).cpi.first()
    out = {}
    def window(months):
        c = cpi[cpi.index.get_level_values(1).isin(months)].groupby(level=0).mean()
        g = m[m.month.isin(months)].groupby("year")[["res_rev", "res_mwh"]].sum()
        return np.log(c).diff(), g.res_rev / g.res_mwh * 100
    pi, p = window(range(1, 13))
    for y in range(2012, 2026):
        out[y] = p[y - 1] * (pi[y] + pi[y - 1]) / 2
    pi9, p9 = window(range(1, last + 1))
    out[2026] = p9[2025] * (pi9[2026] + pi9[2025]) / 2
    return out


def decompose(ann, ytd, theta, betas, infl):
    rows = []
    def contrib(prev, cur):
        out = {}
        dx = cur[X_COLS].values.astype(float) - prev[X_COLS].values.astype(float)
        for g, cols in GROUPS.items():
            out[g] = sum(betas[c] * dx[X_COLS.index(c)] for c in cols)
        out["Capacity market (non-DC)"] = cur.cap_nondc - prev.cap_nondc
        th = theta["dc_share"] + theta["dc_share_cap"] * cur.capmkt
        local = th * (cur.dc_share - prev.dc_share)
        cap_dc = cur.cap_dc - prev.cap_dc
        mech = -FIXED_SHARE * prev.res_price * (cur.dc_mwh - prev.dc_mwh) / prev.tot_mwh
        # Local net effect = spreading (-) + local cost push (+). Spreading is the mechanical
        # estimate unless the estimated net is even more negative, in which case push = 0.
        spread = mech if local >= mech else local
        net = local + cap_dc
        out["Data centers: fixed-cost spreading"] = spread
        out["Data centers: cost push"] = (local - spread) + cap_dc
        out["_dc_net"], out["_dc_capacity"], out["_dc_local"] = net, cap_dc, local
        out["_actual"] = cur.res_price - prev.res_price
        return out

    def frames(year):
        if year < 2026:
            return ann[ann.year == year - 1].set_index("state"), ann[ann.year == year].set_index("state")
        return ytd[ytd.year == 2025].set_index("state"), ytd[ytd.year == 2026].set_index("state")

    # National common factor = average (across all states) of the change not explained by
    # state-specific measured factors; equals the change in the year fixed effect.
    national = {}
    for year in range(2011, 2027):
        p, c = frames(year)
        sts = [s for s in c.index if s in p.index]
        resid = [contrib(p.loc[s], c.loc[s]) for s in sts]
        national[year] = np.mean([r["_actual"] - sum(v for k, v in r.items() if not k.startswith("_")) for r in resid])

    for mkt, sts in MARKETS.items():
        for year in range(2020, 2027):
            p, c = frames(year)
            w = p.loc[sts, "res_mwh"] / p.loc[sts, "res_mwh"].sum()
            agg = {"National: inflation": infl[year],
                   "National: grid costs & interest rates above inflation": national[year] - infl[year]}
            for s in sts:
                for k, v in contrib(p.loc[s], c.loc[s]).items():
                    agg[k] = agg.get(k, 0) + w[s] * v
            # actual change for the region from aggregated revenue / sales
            pr = lambda d: d.loc[sts, "res_rev"].sum() / d.loc[sts, "res_mwh"].sum() * 100
            agg["_actual"] = pr(c) - pr(p)
            agg["_price_prev"], agg["_price"] = pr(p), pr(c)
            explained = sum(v for k, v in agg.items() if not k.startswith("_"))
            agg["Other / unexplained (T&D, policy, storms, rate cases)"] = agg["_actual"] - explained
            rows.append({"market": mkt, "year": year, **agg})
    pd.DataFrame({"national_common": national, "inflation": infl}).rename_axis("year").to_csv(OUTD / "national_effects.csv")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    m = pd.read_parquet(PROC / "panel_monthly.parquet")
    ann0 = aggregate(m)
    dc, _ = datacenter_load(ann0[ann0.n_months == 12])
    ann, ytd, last = period_frames(m, dc)
    est = ann[ann.year.isin(EST_YEARS)].dropna(subset=["res_price"] + X_COLS).reset_index(drop=True)
    print(f"estimation sample: {len(est)} state-years, {est.state.nunique()} states; YTD months: 1-{last}")

    theta, se = dml(est)
    fe = ols_fe(est)
    ob = other_betas(est, theta)
    print("DML theta:", {k: round(v, 3) for k, v in theta.items()}, "SE:", {k: round(v, 3) for k, v in se.items()})
    print(fe.summary().tables[1])
    print(ob.summary().tables[1])

    dec = decompose(ann, ytd, theta, ob.params, inflation_component(m, last))
    dec.to_csv(OUTD / "decomposition.csv", index=False)
    results = {
        "dml_theta": theta, "dml_se": se,
        "ols_fe_theta": {k: float(fe.params[k]) for k in ["dc_share", "dc_share_cap"]},
        "ols_fe_se": {k: float(fe.bse[k]) for k in ["dc_share", "dc_share_cap"]},
        "other_betas": ob.params.to_dict(), "other_se": ob.bse.to_dict(),
        "n_obs": len(est), "n_states": int(est.state.nunique()), "ytd_months": last,
        "assumptions": {"FIXED_SHARE": FIXED_SHARE, "DC_CAP_SHARE": DC_CAP_SHARE, "CAP_TO_CKWH": CAP_TO_CKWH},
    }
    (OUTD / "model_results.json").write_text(json.dumps(results, indent=2, default=float))
    est.to_csv(PROC / "estimation_sample.csv", index=False)
    pd.concat([ann, ytd.assign(period="ytd")]).to_csv(PROC / "features_all.csv", index=False)
    pd.set_option("display.width", 250)
    print(dec.round(2).to_string())
