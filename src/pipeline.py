"""The model as three stages, each driven by a ModelConfig.

  prepare(panel, cfg)            capacity costs, data center load, features, lags   (~1 s)
  estimate(prep, cfg)            double ML + fixed-effects regressions              (~12 s)
  attribute(prep, est, cfg)      mechanical splits, national bars, ranges, outlook  (~1 s)

estimate() depends only on the assumptions flagged reestimate=True in config.py, so its
result can be cached by cfg.estimation_key() and reused whenever only mechanical
assumptions change. run(cfg) chains the three stages.

Method summary:
0. Capacity-market costs (PJM, ISO-NE) are computed mechanically from auction prices and
   removed from the price before estimation. The PJM portion attributed to data centers
   by the market monitor is assigned to data centers.
1. Double ML (partially linear model, two-way state + year fixed effects via
   within-transform, LightGBM nuisance models, state-grouped cross-fitting) for the local
   data center effect: data center share of state load, optionally its one-year lag, each
   interacted with a capacity-market indicator.
2. Two-way fixed-effects regression of (adjusted price - DC effect) on the other factors.
3. Yearly decomposition for the target markets: contribution_k = beta_k * delta x_k. The
   year effects become the national bars (inflation + above inflation). The DC net effect
   is split into a mechanical fixed-cost-spreading term and an implied cost-push term, and
   carries a range from the DML covariance and the capacity-attribution band.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold

from build_panel import INP, aggregate, capacity_monthly, datacenter_load
from config import CapacityConfig, ModelConfig, UncertaintyConfig
from markets import MARKETS

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"

# Structural definitions (not assumptions): which states sit in which capacity market,
# and which PJM states have vertically integrated utilities.
PJM_STATES = ["DE", "DC", "MD", "NJ", "OH", "PA", "VA", "WV"]  # fully in PJM; IL partially (config)
ISONE = ["CT", "MA", "ME", "NH", "RI", "VT"]
PJM_VERT_INT = {"VA", "WV"}

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


# =============================================================================== stage 1
def load_panel():
    """Monthly panel as built by build_panel (raw-data columns only)."""
    m = pd.read_parquet(PROC / "panel_monthly.parquet")
    drop = [c for c in ["pjm_w", "isone_w", "cap_price", "cap_dc_price", "cap_dc_price_lo",
                        "cap_dc_price_hi", "cap_restructured"] if c in m.columns]
    return m.drop(columns=drop)


def apply_capacity(m, cfg: CapacityConfig):
    """Capacity price ($/MW-day) by state-month, its data center share (by delivery year,
    with low/high band), membership weights and the restructured flag."""
    m = m.copy()
    cap = capacity_monthly()
    rto = cap[cap.zone.isin(["RTO", "RoP"])].pivot_table(index=["year", "month"], columns="market", values="cap_price").reset_index()
    m = m.merge(rto, on=["year", "month"], how="left")
    dom = cap[cap.zone == "DOM"][["year", "month", "cap_price"]].rename(columns={"cap_price": "dom"})
    m = m.merge(dom, on=["year", "month"], how="left")
    pjm_w = {s: 1 for s in PJM_STATES}
    pjm_w["IL"] = cfg.illinois_pjm_weight
    m["pjm_w"] = m.state.map(pjm_w).fillna(0)
    m["isone_w"] = m.state.isin(ISONE).astype(float)
    m["cap_price"] = m.pjm_w * m.PJM.fillna(0) + m.isone_w * m.ISONE.fillna(0)
    if cfg.use_dominion_zone:
        va = (m.state == "VA") & m.dom.notna()
        m.loc[va, "cap_price"] = m.loc[va, "dom"]
    dy = np.where(m.month >= 6, m.year, m.year - 1)  # capacity delivery year
    share = pd.Series(dy).map(cfg.dc_cap_share()).fillna(0).values * (m.pjm_w.values > 0)
    band = cfg.dc_share_band
    m["cap_dc_price"] = m.cap_price * share
    m["cap_dc_price_lo"] = m.cap_price * np.clip(share - band, 0, 1) * (share > 0)
    m["cap_dc_price_hi"] = m.cap_price * (share + band) * (share > 0)
    m["cap_restructured"] = np.where(m.state.isin(PJM_VERT_INT), 0.0, 1.0)
    return m.drop(columns=["PJM", "ISONE", "dom"])


def features(a, cfg: ModelConfig, exposure="dc_mwh"):
    cc = cfg.capacity
    a = a.copy()
    a["dc_mwh"] = a[exposure]
    a["dc_share"] = a.dc_mwh / a.tot_mwh * 100
    a["capmkt"] = ((a.pjm_w + a.isone_w) > 0).astype(float)
    a["dc_share_cap"] = a.dc_share * a.capmkt
    rho = np.where(a.cap_restructured > 0, 1.0, cc.pass_through_vert_int)
    cap_to_ckwh = cc.cap_to_ckwh()
    a["cap_ckwh"] = a.cap_price * cap_to_ckwh * rho
    for suf in ["", "_lo", "_hi"]:
        a["cap_dc" + suf] = a["cap_dc_price" + suf] * cap_to_ckwh * rho
    a["cap_nondc"] = a.cap_ckwh - a.cap_dc
    a["y_net_cap"] = a.res_price - a.cap_ckwh
    # marginal-cost gas exposure in restructured retail states
    a["gas_marg"] = a.gas_price * cfg.fuel.heat_rate / 10 * a.state.isin(set(cfg.fuel.retail_choice))
    a["ln_kwh_cust"] = np.log(a.res_kwh_per_cust)
    a["ln_nondc_load"] = np.log(a.tot_mwh - a.dc_mwh)
    a["hdd"], a["cdd"] = a.hdd / 1000, a.cdd / 1000
    return a


def add_lags(ann, ytd, m, dc, last, exposure, cfg):
    ann = ann.sort_values(["state", "year"])
    for c in ["gas_cost", "gas_marg", "dc_share"]:
        ann[c + "_l1"] = ann.groupby("state")[c].shift(1)
    # YTD lag: prior year's same-month window (2026 YTD -> 2025 YTD; 2025 YTD -> 2024 YTD)
    y24 = aggregate(m, range(1, last + 1), cfg.fuel.heat_rate).query("year==2024 and n_months==@last").merge(dc, on=["state", "year"])
    y24[["dc_mwh", "dc_mwh_v1"]] *= last / 12
    y24 = features(y24, cfg, exposure)
    lagf = pd.concat([y24, ytd[ytd.year == 2025]]).assign(year=lambda d: d.year + 1).set_index(["state", "year"])
    idx = ytd.set_index(["state", "year"]).index
    for c in ["gas_cost", "gas_marg", "dc_share"]:
        ytd[c + "_l1"] = idx.map(lagf[c])
    for f in (ann, ytd):
        f["dc_share_cap_l1"] = f.dc_share_l1 * f.capmkt
    return ann, ytd


@dataclass
class Prepared:
    m: pd.DataFrame          # monthly panel with capacity columns
    dc: pd.DataFrame         # data center load, both measures
    validation: dict         # state-specific measure vs EPRI
    ann: pd.DataFrame        # annual features 2010-2025
    ytd: pd.DataFrame        # year-to-date features 2025 and 2026
    last: int                # last month of 2026 data
    est: pd.DataFrame        # estimation sample
    D: list                  # data center regressors


def prepare(panel, cfg: ModelConfig) -> Prepared:
    m = apply_capacity(panel, cfg.capacity)
    hr = cfg.fuel.heat_rate
    ann0 = aggregate(m, heat_rate=hr)
    dc, _, validation = datacenter_load(ann0[ann0.n_months == 12], cfg.dc_load)
    exposure = cfg.estimation.exposure_col()
    ann = ann0[ann0.n_months == 12].merge(dc, on=["state", "year"])
    last = int(m[m.state == "VA"].query("year==2026").month.max())
    ytd = aggregate(m, range(1, last + 1), hr)
    ytd = ytd[ytd.year.isin([2025, 2026]) & (ytd.n_months == last)].merge(dc, on=["state", "year"])
    ytd[["dc_mwh", "dc_mwh_v1"]] *= last / 12
    ann, ytd = features(ann, cfg, exposure), features(ytd, cfg, exposure)
    ann, ytd = add_lags(ann, ytd, m, dc, last, exposure, cfg)
    D = d_cols(cfg.estimation.lag)
    years = range(cfg.estimation.est_start, cfg.estimation.est_end + 1)
    est = ann[ann.year.isin(years)].dropna(subset=["res_price"] + X_COLS + D).reset_index(drop=True)
    return Prepared(m, dc, validation, ann, ytd, last, est, D)


# =============================================================================== stage 2
def within(df, cols, iters=50):
    """Two-way (state + year) within transform by alternating projections."""
    w = df[cols].astype(float).copy()
    for _ in range(iters):
        w = w - w.groupby(df.state.values).transform("mean")
        w = w - w.groupby(df.year.values).transform("mean")
    return w


def cluster_ols(y, X, groups):
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})


def dml(a, D, n_folds=5, n_rep=5):
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
        for tr, te in GroupKFold(n_splits=n_folds).split(Xf, groups=np.round(order * 1000)):
            for c in res:
                mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=10,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=rep, verbose=-1)
                mdl.fit(Xf.iloc[tr], W[c].iloc[tr])
                res[c][te] = W[c].iloc[te] - mdl.predict(Xf.iloc[te])
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


@dataclass
class Estimates:
    theta: pd.Series          # DML coefficients on the data center regressors
    cov: pd.DataFrame         # their covariance
    betas: pd.Series          # other-factor coefficients
    betas_se: pd.Series
    ols_theta: pd.Series      # two-way FE OLS, same regressors
    ols_cov: pd.DataFrame
    ols_se: pd.Series
    n_obs: int
    n_states: int
    key: str = ""             # cfg.estimation_key() this was estimated under
    label: str = ""

    def to_dict(self):
        return {"theta": self.theta.to_dict(), "cov": self.cov.values.tolist(), "terms": list(self.theta.index),
                "betas": self.betas.to_dict(), "betas_se": self.betas_se.to_dict(),
                "ols_theta": self.ols_theta.to_dict(), "ols_cov": self.ols_cov.values.tolist(),
                "ols_se": self.ols_se.to_dict(), "n_obs": self.n_obs, "n_states": self.n_states,
                "key": self.key, "label": self.label}

    @classmethod
    def from_dict(cls, d):
        D = d["terms"]
        return cls(pd.Series(d["theta"])[D], pd.DataFrame(d["cov"], index=D, columns=D),
                   pd.Series(d["betas"])[X_COLS], pd.Series(d["betas_se"])[X_COLS],
                   pd.Series(d["ols_theta"])[D], pd.DataFrame(d["ols_cov"], index=D, columns=D),
                   pd.Series(d["ols_se"])[D], d["n_obs"], d["n_states"], d.get("key", ""), d.get("label", ""))


def estimate(prep: Prepared, cfg: ModelConfig, label="") -> Estimates:
    e = cfg.estimation
    theta, cov = dml(prep.est, prep.D, e.n_folds, e.n_rep)
    fe = ols_fe(prep.est, prep.D)
    ob = other_betas(prep.est, theta)
    D = prep.D
    return Estimates(theta, cov, ob.params, ob.bse, fe.params[D], fe.cov_params().loc[D, D], fe.bse[D],
                     len(prep.est), int(prep.est.state.nunique()), cfg.estimation_key(), label)


# =============================================================================== stage 3
def inflation_component(m, last, cfg: ModelConfig):
    """Inflation pass-through: prior national avg residential price x CPI inflation, a
    weighted average of this year's and last year's rate (rate cases lag), times the
    pass-through. Baseline: equal weights, pass-through 1, which matches the 2012-2019
    national common trend."""
    pt, w = cfg.inflation.pass_through, cfg.inflation.current_year_weight
    cpi = m.groupby(["year", "month"]).cpi.first()
    out = {}

    def window(months):
        c = cpi[cpi.index.get_level_values(1).isin(months)].groupby(level=0).mean()
        g = m[m.month.isin(months)].groupby("year")[["res_rev", "res_mwh"]].sum()
        return np.log(c).diff(), g.res_rev / g.res_mwh * 100

    def term(p_prev, pi_t, pi_t1):
        mix = (pi_t + pi_t1) / 2 if w == 0.5 else w * pi_t + (1 - w) * pi_t1
        return pt * (p_prev * mix)
    pi, p = window(range(1, 13))
    for y in range(2012, 2026):
        out[y] = term(p[y - 1], pi[y], pi[y - 1])
    pi9, p9 = window(range(1, last + 1))
    out[2026] = term(p9[2025], pi9[2026], pi9[2025])
    return out


def frames_for(ann, ytd, year):
    if year < 2026:
        return ann[ann.year == year - 1].set_index("state"), ann[ann.year == year].set_index("state")
    return ytd[ytd.year == 2025].set_index("state"), ytd[ytd.year == 2026].set_index("state")


def dc_gradient(prev, cur, D):
    """Change in each DC regressor; the local DC effect is theta . gradient."""
    return np.array([float(cur[d] - prev[d]) for d in D])


def decompose(ann, ytd, est: Estimates, infl, cfg: ModelConfig):
    theta, cov, betas = est.theta, est.cov, est.betas
    D = list(theta.index)
    fixed_share = cfg.spreading.fixed_share
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
        mech = -fixed_share * prev.res_price * (cur.dc_mwh - prev.dc_mwh) / prev.tot_mwh
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
    add_ranges(dec, cov, cfg.uncertainty)
    return dec, national


def dc_bounds(local, sd, cap, cap_lo, cap_hi, unc: UncertaintyConfig):
    """Range of the DC net effect from the local sampling SD and the capacity band."""
    z = unc.z
    if unc.combine == "add":
        return local - z * sd + cap_lo, local + z * sd + cap_hi
    net = local + cap
    return (net - np.sqrt((z * sd) ** 2 + (cap - cap_lo) ** 2),
            net + np.sqrt((z * sd) ** 2 + (cap_hi - cap) ** 2))


def add_ranges(dec, cov, unc: UncertaintyConfig):
    G = dec[[f"_g{i}" for i in range(len(cov))]].values
    sd = np.sqrt(np.einsum("ij,jk,ik->i", G, cov.values, G))
    dec["_dc_local_sd"] = sd
    dec["_dc_net_lo"], dec["_dc_net_hi"] = dc_bounds(dec._dc_local, sd, dec._dc_capacity,
                                                     dec._dc_cap_lo, dec._dc_cap_hi, unc)


def cumulative_one(rows, cov, unc: UncertaintyConfig):
    """Cumulative DC effect and range for one market over the given rows (years)."""
    gcols = [f"_g{i}" for i in range(len(cov))]
    G = rows[gcols].sum().values
    sd = float(np.sqrt(G @ np.asarray(cov) @ G))
    loc, cap = rows._dc_local.sum(), rows._dc_capacity.sum()
    lo, hi = dc_bounds(loc, sd, cap, rows._dc_cap_lo.sum(), rows._dc_cap_hi.sum(), unc)
    return {"actual": rows._actual.sum(), "dc_net": loc + cap, "dc_local": loc, "dc_capacity": cap,
            "lo": lo, "hi": hi}


def cumulative(dec, cov, years, unc: UncertaintyConfig):
    d = dec[dec.year.isin(years)]
    return {mkt: cumulative_one(g, cov, unc) for mkt, g in d.groupby("market", sort=False)}


def sustained_effect(theta, cov, terms):
    """Effect of a sustained +1 pp data center share in non-capacity-market states
    (current + lag), with its standard error."""
    v = np.array([1.0 if t in ("dc_share", "dc_share_l1") else 0.0 for t in terms])
    th = np.array([theta[t] for t in terms])
    return float(v @ th), float(np.sqrt(v @ np.asarray(cov) @ v))


def capacity_outlook(cfg: CapacityConfig, last_full=2028):
    """PJM capacity cost in residential rates by calendar year, using auction results
    already set through 2028/29: total and data-center share (with band)."""
    rows = []
    cap = pd.read_csv(INP / "capacity_prices.csv")
    pjm = cap[(cap.market == "PJM") & (cap.zone == "RTO")].set_index("delivery_year_start").price
    dom = cap[(cap.market == "PJM") & (cap.zone == "DOM")].set_index("delivery_year_start").price
    share_by_dy, band, cap_to_ckwh = cfg.dc_cap_share(), cfg.dc_share_band, cfg.cap_to_ckwh()
    groups = {"PJM: Virginia (Dominion)": (["VA"], cfg.pass_through_vert_int),
              "PJM: restructured states": (MARKETS["PJM: restructured states"], 1.0)}
    for mkt, (sts, rho) in groups.items():
        for y in range(2023, last_full + 1):
            tot = dcv = lo = hi = 0.0
            for mo in range(1, 13):
                dy = y if mo >= 6 else y - 1
                if sts == ["VA"] and cfg.use_dominion_zone:
                    price = dom.get(dy, pjm.get(dy))
                else:
                    price = pjm.get(dy)
                share = share_by_dy.get(dy, 0.0)
                c = price * cap_to_ckwh * rho / 12
                tot += c
                dcv += c * share
                lo += c * max(share - band, 0) * (share > 0)
                hi += c * (share + band) * (share > 0)
            rows.append({"market": mkt, "year": y, "capacity_ckwh": tot, "dc_capacity_ckwh": dcv,
                         "dc_lo": lo, "dc_hi": hi})
    return pd.DataFrame(rows)


def national_dc_test(m, dc, national, infl, exposure_col):
    """Does national data center growth explain the 'above inflation' national bar?
    Regress it on the change in the national DC share of US load (14 yearly points)."""
    tot = m[m.month <= 12].groupby("year").tot_mwh.sum()
    nat = dc.groupby("year")[exposure_col].sum()
    share = (nat / tot * 100).loc[2010:2025]
    above = pd.Series({y: national[y] - infl[y] for y in range(2012, 2026)})
    dshare = share.diff().loc[2012:2025]
    r = sm.OLS(above, sm.add_constant(dshare.rename("d_dc_share"))).fit(cov_type="HC1")
    implied = float(r.params["d_dc_share"] * dshare.loc[2022:2025].sum())
    return {"coef_per_pp": float(r.params["d_dc_share"]), "se": float(r.bse["d_dc_share"]),
            "p_value": float(r.pvalues["d_dc_share"]), "r2": float(r.rsquared),
            "implied_2022_2025_cents": implied, "above_inflation_2022_2025_cents": float(above.loc[2022:2025].sum()),
            "dc_share_by_year": share.round(3).to_dict()}


@dataclass
class Results:
    decomposition: pd.DataFrame
    national: dict
    inflation: dict
    outlook: pd.DataFrame
    national_test: dict
    cumulative_2022_2025: dict
    extra: dict = field(default_factory=dict)

    def national_frame(self):
        return pd.DataFrame({"national_common": self.national, "inflation": self.inflation}).rename_axis("year")


def attribute(prep: Prepared, est: Estimates, cfg: ModelConfig) -> Results:
    infl = inflation_component(prep.m, prep.last, cfg)
    dec, national = decompose(prep.ann, prep.ytd, est, infl, cfg)
    outlook = capacity_outlook(cfg.capacity)
    ntest = national_dc_test(prep.m, prep.dc, national, infl,
                             cfg.estimation.exposure_col(cfg.estimation.national_test_exposure))
    cum = cumulative(dec, est.cov, range(2022, 2026), cfg.uncertainty)
    return Results(dec, national, infl, outlook, ntest, cum)


def run(cfg: ModelConfig, panel=None, est: Estimates | None = None):
    """Full pipeline. Pass `est` to reuse cached estimates (must match cfg.estimation_key())."""
    panel = load_panel() if panel is None else panel
    prep = prepare(panel, cfg)
    if est is None:
        est = estimate(prep, cfg)
    elif est.key and est.key != cfg.estimation_key():
        raise ValueError("Estimates were computed under different estimation assumptions")
    return prep, est, attribute(prep, est, cfg)


def estimates_from_results_json(path):
    """Rebuild the published headline Estimates from output/model_results.json."""
    r = json.loads(Path(path).read_text())
    D = r["dml_terms"]
    head = r["specs"][r["headline_spec"]]
    return Estimates(pd.Series(r["dml_theta"])[D], pd.DataFrame(r["dml_cov"], index=D, columns=D),
                     pd.Series(r["other_betas"])[X_COLS], pd.Series(r["other_se"])[X_COLS],
                     pd.Series(r["ols_fe_theta"])[D], pd.DataFrame(head["ols_cov"], index=D, columns=D),
                     pd.Series(r["ols_fe_se"])[D], r["n_obs"], r["n_states"],
                     r.get("estimation_key", ModelConfig.baseline().estimation_key()), "published baseline")
