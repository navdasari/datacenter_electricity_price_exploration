"""Estimate factor effects on residential electricity prices and decompose yearly changes.

0. Capacity-market costs (PJM, ISO-NE) are computed mechanically from auction prices
   and removed from the price before estimation. The PJM portion attributed to data
   centers by the market monitor (45% of 2025/26-2027/28, 40% of 2028/29 charges,
   +/-5 points for the uncertainty range) is assigned to data centers.
1. Double ML (partially linear model, two-way state + year fixed effects via
   within-transform, LightGBM nuisance models, state-grouped cross-fitting) for the
   local data center effect: data center share of state load, its one-year lag, and both
   interacted with a capacity-market indicator. Run for four specifications (exposure
   measure x lags); the headline is the state-specific measure with lags.
2. Two-way fixed-effects regression of (adjusted price - DC effect) on the other factors.
3. Yearly decomposition for the target markets: contribution_k = beta_k * delta x_k.
   The year effects become the national bars (inflation + above inflation). The DC net
   effect is split into a mechanical fixed-cost-spreading term (negative) and an implied
   cost-push term, and carries a 95% range from the DML covariance and the capacity band.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold

from build_panel import DC_CAP_SHARE, INP, aggregate, datacenter_load
from markets import MARKETS

ROOT = Path(__file__).resolve().parents[1]
PROC, OUTD = ROOT / "data/processed", ROOT / "output"
OUTD.mkdir(exist_ok=True)

# --- Assumptions (documented in README) ---------------------------------------
FIXED_SHARE = 0.5          # share of residential rate that is fixed cost re-spread over load
CAP_TO_CKWH = 365 / (8760 * 0.55) / 10  # $/MW-day -> cents/kWh (residential load factor at peak ~0.55)
PASS_THROUGH_VERT_INT = 0.2  # Dominion/WV utilities self-supply most capacity: muted retail exposure
EST_YEARS = range(2010, 2026)
# Restructured retail states: gas sets the marginal wholesale price that flows to retail,
# regardless of the in-state generation mix.
RETAIL_CHOICE = {"CT", "DE", "DC", "IL", "ME", "MD", "MA", "NH", "NJ", "NY", "OH", "PA", "RI", "TX"}

SPECS = {  # name -> (exposure column, include one-year lag)
    "state-specific, with lag (headline)": ("dc_mwh", True),
    "state-specific, no lag": ("dc_mwh", False),
    "national scaling, with lag": ("dc_mwh_v1", True),
    "national scaling, no lag (original)": ("dc_mwh_v1", False),
}
HEADLINE = "state-specific, with lag (headline)"

GROUPS = {
    "Natural gas / fuel": ["gas_cost", "gas_cost_l1", "gas_marg", "gas_marg_l1"],
    "Hydro conditions": ["hydro_share"],
    "Weather": ["hdd", "cdd"],
    "Lower use per customer (efficiency, rooftop solar)": ["ln_kwh_cust"],
    "Non-DC load growth": ["ln_nondc_load"],
}
X_COLS = [c for g in GROUPS.values() for c in g]
D_ALL = ["dc_share", "dc_share_cap", "dc_share_l1", "dc_share_cap_l1"]


def d_cols(lag):
    return D_ALL if lag else D_ALL[:2]


def features(a, exposure="dc_mwh"):
    a = a.copy()
    a["dc_mwh"] = a[exposure]
    a["dc_share"] = a.dc_mwh / a.tot_mwh * 100
    a["capmkt"] = ((a.pjm_w + a.isone_w) > 0).astype(float)
    a["dc_share_cap"] = a.dc_share * a.capmkt
    rho = np.where(a.cap_restructured > 0, 1.0, PASS_THROUGH_VERT_INT)
    a["cap_ckwh"] = a.cap_price * CAP_TO_CKWH * rho
    for suf in ["", "_lo", "_hi"]:
        a["cap_dc" + suf] = a["cap_dc_price" + suf] * CAP_TO_CKWH * rho
    a["cap_nondc"] = a.cap_ckwh - a.cap_dc
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


def dml(a, D, n_rep=5):
    """Returns theta (Series), covariance (DataFrame). Covariance = mean clustered
    covariance across cross-fitting repetitions + between-repetition spread."""
    W = within(a, ["y_net_cap"] + D + X_COLS)
    Xf = W[X_COLS]
    states = a.state.values
    thetas, covs = [], []
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
        f = cluster_ols(res["y_net_cap"], pd.DataFrame({d: res[d] for d in D}), states)
        thetas.append(f.params.values)
        covs.append(f.cov_params().values)
    T = np.array(thetas)
    th = T.mean(axis=0)
    dev = T - th
    cov = np.mean(covs, axis=0) + dev.T @ dev / len(T)
    return pd.Series(th, index=D), pd.DataFrame(cov, index=D, columns=D)


def ols_fe(a, D):
    cols = D + X_COLS
    W = within(a, ["y_net_cap"] + cols)
    return cluster_ols(W.y_net_cap, W[cols], a.state.values)


def other_betas(a, theta):
    a = a.copy()
    a["y_adj"] = a.y_net_cap - sum(theta[d] * a[d] for d in theta.index)
    W = within(a, ["y_adj"] + X_COLS)
    return cluster_ols(W.y_adj, W[X_COLS], a.state.values)


def add_lags(ann, ytd, m, dc, last, exposure):
    ann = ann.sort_values(["state", "year"])
    for c in ["gas_cost", "gas_marg", "dc_share"]:
        ann[c + "_l1"] = ann.groupby("state")[c].shift(1)
    # YTD lag: prior year's same-month window (2026 YTD -> 2025 YTD; 2025 YTD -> 2024 YTD)
    y24 = aggregate(m, range(1, last + 1)).query("year==2024 and n_months==@last").merge(dc, on=["state", "year"])
    y24[["dc_mwh", "dc_mwh_v1"]] *= last / 12
    y24 = features(y24, exposure)
    lagf = pd.concat([y24, ytd[ytd.year == 2025]]).assign(year=lambda d: d.year + 1).set_index(["state", "year"])
    idx = ytd.set_index(["state", "year"]).index
    for c in ["gas_cost", "gas_marg", "dc_share"]:
        ytd[c + "_l1"] = idx.map(lagf[c])
    for f in (ann, ytd):
        f["dc_share_cap_l1"] = f.dc_share_l1 * f.capmkt
    return ann, ytd


def period_frames(m, dc, exposure):
    """Return (annual frame 2010-2025, ytd frame for Jan-last 2025 & 2026)."""
    ann = aggregate(m)
    ann = ann[ann.n_months == 12].merge(dc, on=["state", "year"])
    last = int(m[m.state == "VA"].query("year==2026").month.max())
    ytd = aggregate(m, range(1, last + 1))
    ytd = ytd[ytd.year.isin([2025, 2026]) & (ytd.n_months == last)].merge(dc, on=["state", "year"])
    ytd[["dc_mwh", "dc_mwh_v1"]] *= last / 12
    ann, ytd = features(ann, exposure), features(ytd, exposure)
    ann, ytd = add_lags(ann, ytd, m, dc, last, exposure)
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


def frames_for(ann, ytd, year):
    if year < 2026:
        return ann[ann.year == year - 1].set_index("state"), ann[ann.year == year].set_index("state")
    return ytd[ytd.year == 2025].set_index("state"), ytd[ytd.year == 2026].set_index("state")


def dc_gradient(prev, cur, D):
    """Change in each DC regressor; the local DC effect is theta . gradient."""
    return np.array([float(cur[d] - prev[d]) for d in D])


def decompose(ann, ytd, theta, cov, betas, infl):
    D = list(theta.index)
    rows = []

    def contrib(prev, cur):
        out = {}
        dx = cur[X_COLS].values.astype(float) - prev[X_COLS].values.astype(float)
        for g, cols in GROUPS.items():
            out[g] = sum(betas[c] * dx[X_COLS.index(c)] for c in cols)
        out["Capacity market (non-DC)"] = cur.cap_nondc - prev.cap_nondc
        grad = dc_gradient(prev, cur, D)
        local = float(theta.values @ grad)
        cap_dc = cur.cap_dc - prev.cap_dc
        mech = -FIXED_SHARE * prev.res_price * (cur.dc_mwh - prev.dc_mwh) / prev.tot_mwh
        # Local net effect = spreading (-) + local cost push (+). Spreading is the mechanical
        # estimate unless the estimated net is even more negative, in which case push = 0.
        spread = mech if local >= mech else local
        out["Data centers: fixed-cost spreading"] = spread
        out["Data centers: cost push"] = (local - spread) + cap_dc
        out["_dc_net"], out["_dc_capacity"], out["_dc_local"] = local + cap_dc, cap_dc, local
        out["_dc_cap_lo"], out["_dc_cap_hi"] = cur.cap_dc_lo - prev.cap_dc_lo, cur.cap_dc_hi - prev.cap_dc_hi
        for i, gv in enumerate(grad):
            out[f"_g{i}"] = gv
        out["_actual"] = cur.res_price - prev.res_price
        return out

    # National common factor = average (across all states) of the change not explained by
    # state-specific measured factors; equals the change in the year fixed effect.
    national = {}
    for year in range(2011, 2027):
        p, c = frames_for(ann, ytd, year)
        sts = [s for s in c.index if s in p.index]
        resid = [contrib(p.loc[s], c.loc[s]) for s in sts]
        national[year] = np.mean([r["_actual"] - sum(v for k, v in r.items() if not k.startswith("_")) for r in resid])

    for mkt, sts in MARKETS.items():
        for year in range(2020, 2027):
            p, c = frames_for(ann, ytd, year)
            w = p.loc[sts, "res_mwh"] / p.loc[sts, "res_mwh"].sum()
            agg = {"National: inflation": infl[year],
                   "National: grid costs & interest rates above inflation": national[year] - infl[year]}
            for s in sts:
                for k, v in contrib(p.loc[s], c.loc[s]).items():
                    agg[k] = agg.get(k, 0) + w[s] * v
            pr = lambda d: d.loc[sts, "res_rev"].sum() / d.loc[sts, "res_mwh"].sum() * 100
            agg["_actual"] = pr(c) - pr(p)
            agg["_price_prev"], agg["_price"] = pr(p), pr(c)
            explained = sum(v for k, v in agg.items() if not k.startswith("_"))
            agg["Other / unexplained (T&D, policy, storms, rate cases)"] = agg["_actual"] - explained
            rows.append({"market": mkt, "year": year, **agg})
    dec = pd.DataFrame(rows)
    add_ranges(dec, cov)
    return dec, national


def add_ranges(dec, cov):
    """95% range of the DC net effect: sampling uncertainty of theta (local channel) plus
    the +/-5 point band on the market monitor's capacity attribution."""
    G = dec[[f"_g{i}" for i in range(len(cov))]].values
    sd = np.sqrt(np.einsum("ij,jk,ik->i", G, cov.values, G))
    dec["_dc_local_sd"] = sd
    dec["_dc_net_lo"] = dec._dc_local - 1.96 * sd + dec._dc_cap_lo
    dec["_dc_net_hi"] = dec._dc_local + 1.96 * sd + dec._dc_cap_hi


def cumulative(dec, cov, years):
    """Cumulative DC effect and 95% range per market over a span of years."""
    d = dec[dec.year.isin(years)]
    gcols = [f"_g{i}" for i in range(len(cov))]
    out = {}
    for mkt, g in d.groupby("market", sort=False):
        G = g[gcols].sum().values
        sd = float(np.sqrt(G @ cov.values @ G))
        loc, cap = g._dc_local.sum(), g._dc_capacity.sum()
        out[mkt] = {"actual": g._actual.sum(), "dc_net": loc + cap, "dc_local": loc, "dc_capacity": cap,
                    "lo": loc - 1.96 * sd + g._dc_cap_lo.sum(), "hi": loc + 1.96 * sd + g._dc_cap_hi.sum()}
    return out


def capacity_outlook(m):
    """PJM capacity cost in residential rates by calendar year, using auction results
    already set through 2028/29: total and data-center share (with band)."""
    rows = []
    last_full = 2028
    cap = pd.read_csv(INP / "capacity_prices.csv")
    pjm = cap[(cap.market == "PJM") & (cap.zone == "RTO")].set_index("delivery_year_start").price
    dom = cap[(cap.market == "PJM") & (cap.zone == "DOM")].set_index("delivery_year_start").price
    w = m[(m.year == 2025)].groupby("state").res_mwh.sum()
    groups = {"PJM: Virginia (Dominion)": (["VA"], PASS_THROUGH_VERT_INT),
              "PJM: restructured states": (MARKETS["PJM: restructured states"], 1.0)}
    for mkt, (sts, rho) in groups.items():
        for y in range(2023, last_full + 1):
            tot = dcv = lo = hi = 0.0
            for mo in range(1, 13):
                dy = y if mo >= 6 else y - 1
                price = dom.get(dy, pjm.get(dy)) if sts == ["VA"] else pjm.get(dy)
                share = DC_CAP_SHARE.get(dy, 0.0)
                c = price * CAP_TO_CKWH * rho / 12
                tot += c
                dcv += c * share
                lo += c * max(share - 0.05, 0) * (share > 0)
                hi += c * (share + 0.05) * (share > 0)
            rows.append({"market": mkt, "year": y, "capacity_ckwh": tot, "dc_capacity_ckwh": dcv,
                         "dc_lo": lo, "dc_hi": hi})
    return pd.DataFrame(rows)


def national_dc_test(m, dc, national, infl):
    """Does national data center growth explain the 'above inflation' national bar?
    Regress it on the change in the national DC share of US load (14 yearly points)."""
    tot = m[m.month <= 12].groupby("year").tot_mwh.sum()
    nat = dc.groupby("year").dc_mwh_v1.sum()
    share = (nat / tot * 100).loc[2010:2025]
    above = pd.Series({y: national[y] - infl[y] for y in range(2012, 2026)})
    dshare = share.diff().loc[2012:2025]
    r = sm.OLS(above, sm.add_constant(dshare.rename("d_dc_share"))).fit(cov_type="HC1")
    implied = float(r.params["d_dc_share"] * dshare.loc[2022:2025].sum())
    return {"coef_per_pp": float(r.params["d_dc_share"]), "se": float(r.bse["d_dc_share"]),
            "p_value": float(r.pvalues["d_dc_share"]), "r2": float(r.rsquared),
            "implied_2022_2025_cents": implied, "above_inflation_2022_2025_cents": float(above.loc[2022:2025].sum()),
            "dc_share_by_year": share.round(3).to_dict()}


def run_spec(m, dc, exposure, lag):
    ann, ytd, last = period_frames(m, dc, exposure)
    D = d_cols(lag)
    est = ann[ann.year.isin(EST_YEARS)].dropna(subset=["res_price"] + X_COLS + D).reset_index(drop=True)
    theta, cov = dml(est, D)
    fe = ols_fe(est, D)
    return ann, ytd, last, est, theta, cov, fe


if __name__ == "__main__":
    m = pd.read_parquet(PROC / "panel_monthly.parquet")
    ann0 = aggregate(m)
    dc, _, validation = datacenter_load(ann0[ann0.n_months == 12])
    infl = inflation_component(m, int(m[m.state == "VA"].query("year==2026").month.max()))

    spec_results = {}
    for name, (exposure, lag) in SPECS.items():
        ann, ytd, last, est, theta, cov, fe = run_spec(m, dc, exposure, lag)
        ob = other_betas(est, theta)
        dec, national = decompose(ann, ytd, theta, cov, ob.params, infl)
        cum = cumulative(dec, cov, range(2022, 2026))
        D = list(theta.index)
        fe_cov = fe.cov_params().loc[D, D]
        spec_results[name] = {"exposure": exposure, "lag": lag, "n_obs": len(est),
                              "dml_theta": theta.to_dict(), "dml_cov": cov.values.tolist(),
                              "ols_theta": fe.params[D].to_dict(), "ols_cov": fe_cov.values.tolist(),
                              "cumulative_2022_2025": cum}
        print(f"\n== {name}: n={len(est)}  DML theta {theta.round(3).to_dict()}")
        for k, v in cum.items():
            print(f"   {k:28s} DC net {v['dc_net']:+.2f}  [{v['lo']:+.2f}, {v['hi']:+.2f}]   actual {v['actual']:+.2f}")
        if name == HEADLINE:
            head = (ann, ytd, last, est, theta, cov, fe, ob, dec, national)

    ann, ytd, last, est, theta, cov, fe, ob, dec, national = head
    dec.to_csv(OUTD / "decomposition.csv", index=False)
    pd.DataFrame({"national_common": national, "inflation": infl}).rename_axis("year").to_csv(OUTD / "national_effects.csv")
    outlook = capacity_outlook(m)
    outlook.to_csv(OUTD / "capacity_outlook.csv", index=False)
    ntest = national_dc_test(m, dc, national, infl)
    D = list(theta.index)
    results = {
        "headline_spec": HEADLINE,
        "dml_theta": theta.to_dict(), "dml_cov": cov.values.tolist(), "dml_terms": D,
        "dml_se": dict(zip(D, np.sqrt(np.diag(cov.values)))),
        "ols_fe_theta": fe.params[D].to_dict(), "ols_fe_se": fe.bse[D].to_dict(),
        "other_betas": ob.params.to_dict(), "other_se": ob.bse.to_dict(),
        "n_obs": len(est), "n_states": int(est.state.nunique()), "ytd_months": last,
        "specs": spec_results, "dc_load_validation": validation, "national_dc_test": ntest,
        "assumptions": {"FIXED_SHARE": FIXED_SHARE, "DC_CAP_SHARE": DC_CAP_SHARE, "CAP_TO_CKWH": CAP_TO_CKWH,
                        "PASS_THROUGH_VERT_INT": PASS_THROUGH_VERT_INT},
    }
    (OUTD / "model_results.json").write_text(json.dumps(results, indent=2, default=float))
    est.to_csv(PROC / "estimation_sample.csv", index=False)
    pd.concat([ann, ytd.assign(period="ytd")]).to_csv(PROC / "features_all.csv", index=False)
    print("\nnational DC test:", {k: (round(v, 3) if isinstance(v, float) else "...") for k, v in ntest.items()})
    print(outlook.round(2).to_string(index=False))
    print(pd.DataFrame(validation).T.round(2))
