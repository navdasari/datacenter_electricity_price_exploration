"""Build a state x month panel, then state x period (annual / YTD) aggregates.

Inputs: data/raw (EIA, FRED, NOAA downloads) and data/inputs (hand-entered
capacity prices and data center anchors, each with a source column).
Output: data/processed/panel_monthly.parquet, data/processed/panel_annual.csv

Only raw-data processing happens here. Capacity prices, capacity-market membership and the
data center share of capacity costs are applied later (pipeline.apply_capacity), from
data/inputs and the model configuration, so they can change without the raw data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from config import DataCenterLoadConfig, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
RAW, INP, OUT = ROOT / "data/raw", ROOT / "data/inputs", ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

NOAA_ORDER = ("AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
              "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY").split()
STATES = NOAA_ORDER + ["DC"]  # contiguous US + DC (DC weather borrowed from MD)

def load_sales():
    d = pd.read_excel(RAW / "sales_revenue.xlsx", sheet_name="Monthly-States", header=None, skiprows=3)
    d = d.iloc[:, [0, 1, 2, 4, 5, 6, 8, 9, 12, 13, 20, 21]]
    d.columns = ["year", "month", "state", "res_rev", "res_mwh", "res_cust", "com_rev", "com_mwh",
                 "ind_rev", "ind_mwh", "tot_rev", "tot_mwh"]
    d = d[pd.to_numeric(d["year"], errors="coerce").notna()]
    for c in d.columns:
        if c != "state":
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d[["year", "month"]] = d[["year", "month"]].astype(int)
    return d[d.state.isin(STATES)]


def load_generation():
    x = pd.ExcelFile(RAW / "generation_monthly.xlsx")
    frames = []
    for s in x.sheet_names:
        if not s[:4].isdigit() or int(s[:4]) < 2008:
            continue
        g = pd.read_excel(x, s, header=4)
        g.columns = ["year", "month", "state", "producer", "source", "mwh"]
        frames.append(g[g.producer == "Total Electric Power Industry"])
    g = pd.concat(frames)
    g["mwh"] = pd.to_numeric(g["mwh"], errors="coerce")
    p = g.pivot_table(index=["year", "month", "state"], columns="source", values="mwh", aggfunc="sum").reset_index()
    out = p[["year", "month", "state"]].copy()
    out["gen_total"] = p["Total"]
    out["gen_gas"] = p["Natural Gas"].fillna(0)
    out["gen_hydro"] = p["Hydroelectric Conventional"].fillna(0)
    out["gen_coal"] = p["Coal"].fillna(0)
    return out


def _eia_xls(path):
    d = pd.read_excel(path, sheet_name="Data 1", header=2)
    d.columns = ["date", "value"]
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d["year"], d["month"] = d.date.dt.year, d.date.dt.month
    return d[["year", "month", "value"]]


def load_gas():
    hh = _eia_xls(RAW / "RNGWHHDm.xls").rename(columns={"value": "henry_hub"})
    rows = []
    for s in NOAA_ORDER:
        f = RAW / "ng" / f"{s}.xls"
        if not f.exists():
            continue
        d = _eia_xls(f)
        d["state"] = s
        # $/Mcf -> $/MMBtu (1 Mcf ~ 1.037 MMBtu)
        d["value"] = d["value"] / 1.037
        rows.append(d)
    g = pd.concat(rows).merge(hh, on=["year", "month"], how="left")
    g["basis"] = g["value"] - g["henry_hub"]
    g.loc[(g.value < 0.5) | (g.value > 60), "basis"] = np.nan  # withheld / implausible
    # Fill missing basis with the state's median basis for that calendar month
    med = g.groupby(["state", "month"]).basis.transform("median")
    g["basis"] = g["basis"].fillna(med).fillna(g.groupby("state").basis.transform("median")).fillna(0)
    g["gas_price"] = g["henry_hub"] + g["basis"]
    # Extend each state's series over the full Henry Hub month range
    full = pd.MultiIndex.from_product([NOAA_ORDER, hh[["year", "month"]].apply(tuple, axis=1)])
    full = pd.DataFrame([(s, y, m) for s, (y, m) in full], columns=["state", "year", "month"])
    full = full.merge(hh, on=["year", "month"]).merge(g[["state", "year", "month", "basis"]], on=["state", "year", "month"], how="left")
    full["basis"] = full["basis"].fillna(full.groupby(["state", "month"]).basis.transform("median")).fillna(0)
    full["gas_price"] = full["henry_hub"] + full["basis"]
    dc = full[full.state == "MD"].assign(state="DC")
    return pd.concat([full, dc])[["state", "year", "month", "gas_price", "henry_hub"]]


def load_weather():
    rows = []
    for s in NOAA_ORDER:
        for p in ["hdd", "cdd"]:
            d = pd.read_csv(RAW / "noaa" / f"{s}_{p}.csv", comment="#")
            d = d[pd.to_numeric(d["Value"], errors="coerce").notna()]
            d["year"], d["month"] = d.Date // 100, d.Date % 100
            d["state"], d["var"] = s, p
            rows.append(d[["state", "year", "month", "var", "Value"]])
    w = pd.concat(rows).pivot_table(index=["state", "year", "month"], columns="var", values="Value").reset_index()
    w = pd.concat([w, w[w.state == "MD"].assign(state="DC")])
    return w


def load_fred():
    out = None
    for f, name in [("fredgraph.csv", "cpi"), ("fred_GS10.csv", "gs10"),
                    ("fred_PCU335311335311.csv", "ppi_transformer")]:
        d = pd.read_csv(RAW / f)
        d.columns = ["date", name]
        d["date"] = pd.to_datetime(d.date)
        d[name] = pd.to_numeric(d[name], errors="coerce")
        d["year"], d["month"] = d.date.dt.year, d.date.dt.month
        d = d[["year", "month", name]]
        out = d if out is None else out.merge(d, on=["year", "month"], how="outer")
    return out.sort_values(["year", "month"]).ffill()


def capacity_monthly():
    """Capacity price ($/MW-day) applying in each month, by market (delivery year Jun-May)."""
    cp = pd.read_csv(INP / "capacity_prices.csv")
    cp.loc[cp.unit == "$/kW-month", "price"] *= 1000 * 12 / 365  # -> $/MW-day
    rows = []
    for _, r in cp.iterrows():
        for k in range(12):
            m = 6 + k
            rows.append({"market": r.market, "zone": r.zone, "year": r.delivery_year_start + (m - 1) // 12,
                         "month": (m - 1) % 12 + 1, "cap_price": r.price})
    return pd.DataFrame(rows)


def datacenter_load(annual, cfg: DataCenterLoadConfig = DataCenterLoadConfig()):
    """State data center load (MWh/yr), two measures.

    dc_mwh_v1 (national scaling): 2023 EPRI share x state load, scaled over time by the
      national LBNL series, so every state grows at the national rate.
    dc_mwh (headline): for EPRI-anchored states, 2020+ load follows the state's own
      commercial + industrial (C&I) load above its 2012-2019 non-DC trend, chained from the
      trend_end level. Used only where the implied 2023 level is within cfg.validation_tol of EPRI's
      2023 estimate (other states keep v1, since other industrial or crypto load swamps the
      signal). 2026 follows the national growth rate from 2025.
    """
    a = pd.read_csv(INP / "datacenter_anchors.csv")
    nat = a[a.kind == "national"].assign(year=lambda d: d.key.astype(int)).set_index("year").value.copy()
    nat[2025], nat[2026] = cfg.national_2025_twh, cfg.national_2026_twh  # extrapolations (assumptions)
    nat = nat.sort_index()
    yrs = np.arange(2008, 2027)
    nat_i = pd.Series(np.exp(np.interp(yrs, nat.index, np.log(nat.values))), index=yrs)
    nat_i[yrs < 2014] = nat[2014] * round(1 - cfg.backcast_decline, 12) ** (2014 - yrs[yrs < 2014])
    shares = a[a.kind == "state_share_2023"].set_index("key").value / 100
    a23 = annual[annual.year == 2023].set_index("state")
    dc23 = (shares * a23.tot_mwh).dropna()
    rest = nat_i[2023] * 1e6 - dc23.sum()
    alloc = {"commercial": a23.com_mwh, "commercial_industrial": a23.com_mwh + a23.ind_mwh,
             "total": a23.tot_mwh}[cfg.remainder_allocation]
    others = alloc.drop(dc23.index)
    epri = dc23.copy()
    dc23 = pd.concat([dc23, rest * others / others.sum()])
    v1 = pd.DataFrame([{"state": s, "year": y, "dc_mwh_v1": dc23[s] * nat_i[y] / nat_i[2023]}
                       for s in dc23.index for y in yrs])

    v1i = v1.set_index(["state", "year"]).dc_mwh_v1
    ann = annual.set_index(["state", "year"])
    ci = ann.com_mwh + ann.ind_mwh
    v2 = v1i.copy()
    validation = {}
    for s in epri.index:
        c = ci.loc[s].sort_index()
        d1 = v1i.loc[s]
        t0, t1 = cfg.trend_start, cfg.trend_end
        nondc = (c - d1.reindex(c.index)).loc[t0:t1]
        b = np.polyfit(nondc.index, np.log(nondc.values), 1)
        resid = c - np.exp(np.polyval(b, c.index))
        path = {y: max(cfg.floor_frac * d1[y], d1[t1] + resid[y] - resid[t1])
                for y in range(t1 + 1, 2026) if y in resid.index}
        ratio = path[2023] / epri[s]
        ok = abs(ratio - 1) <= cfg.validation_tol
        validation[s] = {"chained_2023_twh": path[2023] / 1e6, "epri_2023_twh": epri[s] / 1e6,
                         "ratio": ratio, "used": bool(ok)}
        if ok:
            for y, v in path.items():
                v2[(s, y)] = v
            v2[(s, 2026)] = path[2025] * nat_i[2026] / nat_i[2025]
    out = v1.merge(v2.rename("dc_mwh").reset_index(), on=["state", "year"])
    return out, nat_i, validation


def build():
    sales, gen, gas, wx, fred = load_sales(), load_generation(), load_gas(), load_weather(), load_fred()
    m = sales.merge(gen, on=["year", "month", "state"], how="left") \
        .merge(gas, on=["year", "month", "state"], how="left") \
        .merge(wx, on=["year", "month", "state"], how="left") \
        .merge(fred, on=["year", "month"], how="left")
    m.to_parquet(OUT / "panel_monthly.parquet")
    return m


def aggregate(m, months=range(1, 13), heat_rate=7.5):
    """Aggregate monthly panel to state x year over the given calendar months.
    Expects the capacity columns added by pipeline.apply_capacity."""
    d = m[m.month.isin(list(months))].copy()
    d["gas_x_share"] = d.gas_price * d.gen_gas  # for generation-weighted gas price
    sums = d.groupby(["state", "year"])[["res_rev", "res_mwh", "com_mwh", "ind_mwh", "tot_mwh", "gen_total", "gen_gas",
                                         "gen_hydro", "gen_coal", "hdd", "cdd", "gas_x_share"]].sum()
    means = d.groupby(["state", "year"])[["res_cust", "gas_price", "henry_hub", "cpi", "gs10", "ppi_transformer",
                                          "cap_price", "cap_dc_price", "cap_dc_price_lo", "cap_dc_price_hi",
                                          "cap_restructured", "pjm_w", "isone_w"]].mean()
    n = d.groupby(["state", "year"]).size().rename("n_months")
    a = sums.join(means).join(n).reset_index()
    a["res_price"] = a.res_rev / a.res_mwh * 100  # cents/kWh
    a["gas_share"] = a.gen_gas / a.gen_total
    a["hydro_share"] = a.gen_hydro / a.gen_total
    a["coal_share"] = a.gen_coal / a.gen_total
    # Fuel exposure: gas $/MMBtu x gas share of in-state generation x ~7.5 MMBtu/MWh -> cents/kWh
    a["gas_cost"] = a.gas_price * a.gas_share * heat_rate / 10
    a["res_kwh_per_cust"] = a.res_mwh * 1000 / a.res_cust / a.n_months * 12
    return a


if __name__ == "__main__":
    from pipeline import apply_capacity
    cfg = ModelConfig.baseline()
    m = apply_capacity(build(), cfg.capacity)
    a = aggregate(m, heat_rate=cfg.fuel.heat_rate)
    a = a[a.n_months == 12]
    dc, nat, val = datacenter_load(a, cfg.dc_load)
    a = a.merge(dc, on=["state", "year"], how="left")
    print(pd.DataFrame(val).T.round(2))
    a.to_csv(OUT / "panel_annual.csv", index=False)
    print(a[a.state.isin(["VA", "TX", "AZ", "OR", "WA", "MA"])].query("year>=2019")[
        ["state", "year", "res_price", "gas_cost", "hydro_share", "cap_price", "dc_mwh", "tot_mwh"]].round(3).to_string())
    print(nat.round(1).to_dict())
