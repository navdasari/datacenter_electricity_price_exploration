"""Build a state x month panel, then state x period (annual / YTD) aggregates.

Inputs: data/raw (EIA, FRED, NOAA downloads) and data/inputs (hand-entered
capacity prices and data center anchors, each with a source column).
Output: data/processed/panel_monthly.parquet, data/processed/panel_annual.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, INP, OUT = ROOT / "data/raw", ROOT / "data/inputs", ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

NOAA_ORDER = ("AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
              "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY").split()
STATES = NOAA_ORDER + ["DC"]  # contiguous US + DC (DC weather borrowed from MD)

# Organized capacity markets. Weight = rough share of the state's load in the RTO.
PJM_W = {"DE": 1, "DC": 1, "MD": 1, "NJ": 1, "OH": 1, "PA": 1, "VA": 1, "WV": 1, "IL": 0.6}
ISONE = ["CT", "MA", "ME", "NH", "RI", "VT"]
# Vertically integrated PJM states: utility owns most generation, so retail exposure to
# the capacity *price* is muted (net long). Restructured states buy capacity at market.
PJM_VERT_INT = {"VA", "WV"}


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


def datacenter_load(annual_total_mwh):
    """State data center load (MWh/yr): 2023 EPRI share x state load, scaled over time by
    the national LBNL series (a shift-share / Bartik-style exposure measure)."""
    a = pd.read_csv(INP / "datacenter_anchors.csv")
    nat = a[a.kind == "national"].assign(year=lambda d: d.key.astype(int)).set_index("year").value
    yrs = np.arange(2008, 2027)
    # log-linear interpolation; flat-ish backcast before 2014
    nat_i = pd.Series(np.exp(np.interp(yrs, nat.index, np.log(nat.values))), index=yrs)
    nat_i[yrs < 2014] = nat[2014] * 0.97 ** (2014 - yrs[yrs < 2014])
    shares = a[a.kind == "state_share_2023"].set_index("key").value / 100
    load23 = annual_total_mwh[annual_total_mwh.year == 2023].set_index("state").tot_mwh
    dc23 = (shares * load23).dropna()
    rest = nat_i[2023] * 1e6 - dc23.sum()
    # Remaining national DC load allocated to other states by commercial-sector load
    com23 = annual_total_mwh[annual_total_mwh.year == 2023].set_index("state").com_mwh
    others = com23.drop(dc23.index)
    dc23 = pd.concat([dc23, rest * others / others.sum()])
    rows = [{"state": s, "year": y, "dc_mwh": dc23[s] * nat_i[y] / nat_i[2023]} for s in dc23.index for y in yrs]
    return pd.DataFrame(rows), nat_i


def build():
    sales, gen, gas, wx, fred = load_sales(), load_generation(), load_gas(), load_weather(), load_fred()
    m = sales.merge(gen, on=["year", "month", "state"], how="left") \
        .merge(gas, on=["year", "month", "state"], how="left") \
        .merge(wx, on=["year", "month", "state"], how="left") \
        .merge(fred, on=["year", "month"], how="left")
    cap = capacity_monthly()
    rto = cap[cap.zone.isin(["RTO", "RoP"])].pivot_table(index=["year", "month"], columns="market", values="cap_price").reset_index()
    m = m.merge(rto, on=["year", "month"], how="left")
    dom = cap[cap.zone == "DOM"][["year", "month", "cap_price"]].rename(columns={"cap_price": "dom"})
    m = m.merge(dom, on=["year", "month"], how="left")
    m["pjm_w"] = m.state.map(PJM_W).fillna(0)
    m["isone_w"] = m.state.isin(ISONE).astype(float)
    m["cap_price"] = m.pjm_w * m.PJM.fillna(0) + m.isone_w * m.ISONE.fillna(0)
    va = (m.state == "VA") & m.dom.notna()
    m.loc[va, "cap_price"] = m.loc[va, "dom"]
    m["cap_restructured"] = np.where(m.state.isin(PJM_VERT_INT), 0.0, 1.0)
    m = m.drop(columns=["PJM", "ISONE", "dom"])
    m.to_parquet(OUT / "panel_monthly.parquet")
    return m


def aggregate(m, months=range(1, 13)):
    """Aggregate monthly panel to state x year over the given calendar months."""
    d = m[m.month.isin(list(months))].copy()
    d["gas_x_share"] = d.gas_price * d.gen_gas  # for generation-weighted gas price
    sums = d.groupby(["state", "year"])[["res_rev", "res_mwh", "com_mwh", "ind_mwh", "tot_mwh", "gen_total", "gen_gas",
                                         "gen_hydro", "gen_coal", "hdd", "cdd", "gas_x_share"]].sum()
    means = d.groupby(["state", "year"])[["res_cust", "gas_price", "henry_hub", "cpi", "gs10", "ppi_transformer",
                                          "cap_price", "cap_restructured", "pjm_w", "isone_w"]].mean()
    n = d.groupby(["state", "year"]).size().rename("n_months")
    a = sums.join(means).join(n).reset_index()
    a["res_price"] = a.res_rev / a.res_mwh * 100  # cents/kWh
    a["gas_share"] = a.gen_gas / a.gen_total
    a["hydro_share"] = a.gen_hydro / a.gen_total
    a["coal_share"] = a.gen_coal / a.gen_total
    # Fuel exposure: gas $/MMBtu x gas share of in-state generation x ~7.5 MMBtu/MWh -> cents/kWh
    a["gas_cost"] = a.gas_price * a.gas_share * 7.5 / 10
    a["res_kwh_per_cust"] = a.res_mwh * 1000 / a.res_cust / a.n_months * 12
    return a


if __name__ == "__main__":
    m = build()
    a = aggregate(m)
    a = a[a.n_months == 12]
    dc, nat = datacenter_load(a)
    a = a.merge(dc, on=["state", "year"], how="left")
    a.to_csv(OUT / "panel_annual.csv", index=False)
    print(a[a.state.isin(["VA", "TX", "AZ", "OR", "WA", "MA"])].query("year>=2019")[
        ["state", "year", "res_price", "gas_cost", "hydro_share", "cap_price", "dc_mwh", "tot_mwh"]].round(3).to_string())
    print(nat.round(1).to_dict())
