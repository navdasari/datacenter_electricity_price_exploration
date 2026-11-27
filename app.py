"""Streamlit dashboard: what moved residential electricity prices, and how much was data centers.

Run:  streamlit run app.py
Reads the outputs written by src/model.py, src/robustness.py and src/chart.py
(it does not re-run the models).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT, PROC = ROOT / "output", ROOT / "data/processed"
sys.path.insert(0, str(ROOT / "src"))
from markets import MARKETS  # noqa: E402

st.set_page_config(page_title="Data Centers & Power Prices", layout="wide")

NET_COLOR = "#191919"
GRID = "rgba(128,128,128,0.18)"
YTD_LABEL = "2026 YTD"


# ----------------------------------------------------------------------------- data
@st.cache_data
def load():
    chart = json.loads((OUT / "chart_data.json").read_text())
    groups = chart["groups"]
    grouped = pd.DataFrame(chart["rows"])
    detail = pd.read_csv(OUT / "decomposition.csv")
    model = json.loads((OUT / "model_results.json").read_text())
    rob = json.loads((OUT / "robustness.json").read_text())
    national = pd.read_csv(OUT / "national_effects.csv").dropna()
    outlook = pd.read_csv(OUT / "capacity_outlook.csv")
    return groups, grouped, detail, model, rob, national, outlook


@st.cache_data
def price_history():
    m = pd.read_parquet(PROC / "panel_monthly.parquet", columns=["state", "year", "month", "res_rev", "res_mwh"])
    last = int(m[m.year == m.year.max()].month.max())
    rows = []
    for mkt, sts in MARKETS.items():
        d = m[m.state.isin(sts)]
        full = d[d.year < d.year.max()].groupby("year")[["res_rev", "res_mwh"]].sum()
        for y, r in full.iterrows():
            rows.append({"market": mkt, "year": str(y), "price": r.res_rev / r.res_mwh * 100})
        ytd = d[(d.year == d.year.max()) & (d.month <= last)][["res_rev", "res_mwh"]].sum()
        rows.append({"market": mkt, "year": YTD_LABEL, "price": ytd.res_rev / ytd.res_mwh * 100})
    return pd.DataFrame(rows), last


groups, grouped, detail, model, rob, national, outlook = load()
COV = np.array(model["dml_cov"])
GCOLS = [f"_g{i}" for i in range(len(COV))]
DC_COLOR = "#6a4ce2"


def dc_range(rows):
    """Cumulative data center net effect and 95% range over the given rows (one market)."""
    G = rows[GCOLS].sum().values
    sd = float(np.sqrt(G @ COV @ G))
    loc = rows._dc_local.sum()
    return (loc + rows._dc_capacity.sum(), loc - 1.96 * sd + rows._dc_cap_lo.sum(),
            loc + 1.96 * sd + rows._dc_cap_hi.sum())
prices, ytd_months = price_history()
MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GROUP_COLORS = {g["name"]: g["color"] for g in groups}
GROUP_TEXTURE = {g["name"]: g["texture"] for g in groups}
GROUP_NAMES = [g["name"] for g in groups]
DETAIL_COLS = [c for c in detail.columns if c not in ("market", "year") and not c.startswith("_")]


def ylab(y):
    return YTD_LABEL if int(y) == 2026 else str(int(y))


def base_layout(fig, height, title=None, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        title=dict(text=title, x=0, font=dict(size=14)) if title else None,
        barmode="relative", bargap=0.35, showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, font=dict(size=11)),
        hoverlabel=dict(font_size=12), hovermode="closest",
    )
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor="rgba(128,128,128,0.6)", ticksuffix="¢")
    fig.update_xaxes(type="category", showgrid=False)
    return fig


# ----------------------------------------------------------------------------- figures
def decomposition_fig(mkt, years, view, height, legend, yrange=None, title=None, labels=False):
    if view == "Grouped":
        d = grouped[(grouped.market == mkt) & grouped.year.isin(years)]
        comps = GROUP_NAMES
    else:
        d = detail[(detail.market == mkt) & detail.year.isin(years)]
        comps = DETAIL_COLS
    labels_x = [ylab(y) for y in d.year]
    x = list(range(len(d)))
    fig = go.Figure()
    for i, c in enumerate(comps):
        color = GROUP_COLORS.get(c)
        pattern = dict(shape="/", fgcolor="rgba(255,255,255,0.9)", size=6, fillmode="overlay") if GROUP_TEXTURE.get(c) else None
        fig.add_bar(
            x=x, y=d[c], name=c, legendgroup=c, showlegend=legend,
            marker=dict(color=color, pattern=pattern, line=dict(width=1, color="rgba(255,255,255,0.9)")),
            width=0.6, customdata=labels_x,
            hovertemplate=f"<b>{c}</b><br>%{{customdata}}: %{{y:+.2f}}¢/kWh<extra></extra>",
        )
    fig.add_scatter(
        x=x, y=d["_actual"], mode="markers+text" if labels else "markers", name="Net change (actual)", showlegend=legend,
        marker=dict(size=10, color=NET_COLOR, line=dict(width=2, color="white")),
        text=[f"{v:+.2f}" for v in d["_actual"]], textposition="middle right", textfont=dict(size=12),
        customdata=np.column_stack([labels_x, d._price_prev.round(2), d._price.round(2)]),
        hovertemplate="<b>Net change</b><br>%{customdata[0]}: %{y:+.2f}¢/kWh<br>%{customdata[1]}¢ → %{customdata[2]}¢<extra></extra>",
    )
    fig.add_scatter(
        x=[v + 0.42 for v in x], y=d._dc_net, mode="markers", name="Data centers, net (95% range)", showlegend=legend,
        marker=dict(symbol="diamond", size=8, color=DC_COLOR, line=dict(width=1, color="white")),
        error_y=dict(type="data", symmetric=False, array=(d._dc_net_hi - d._dc_net).values,
                     arrayminus=(d._dc_net - d._dc_net_lo).values, color=DC_COLOR, thickness=1.5, width=0),
        customdata=np.column_stack([labels_x, d._dc_net_lo.round(2), d._dc_net_hi.round(2)]),
        hovertemplate="<b>Data centers, net</b><br>%{customdata[0]}: %{y:+.2f}¢/kWh"
                      "<br>95% range %{customdata[1]}¢ to %{customdata[2]}¢<extra></extra>",
    )
    base_layout(fig, height, title, legend)
    fig.update_xaxes(type="linear", tickvals=x, ticktext=labels_x, range=[-0.6, len(x) - 0.2])
    if yrange:
        fig.update_yaxes(range=yrange)
    return fig


def shared_range(markets, years, view):
    src = grouped if view == "Grouped" else detail
    cols = GROUP_NAMES if view == "Grouped" else DETAIL_COLS
    d = src[src.market.isin(markets) & src.year.isin(years)]
    pos = d[cols].clip(lower=0).sum(axis=1).max()
    neg = d[cols].clip(upper=0).sum(axis=1).min()
    hi = max(pos, d._actual.max(), d._dc_net_hi.max()) + 0.5
    lo = min(neg, d._actual.min(), d._dc_net_lo.min()) - 0.3
    return [lo, hi]


# ----------------------------------------------------------------------------- sidebar
st.sidebar.header("Filters")
sel_markets = st.sidebar.multiselect("Markets", list(MARKETS), default=list(MARKETS), key="markets")
yr = st.sidebar.select_slider("Years", options=list(range(2020, 2027)), value=(2022, 2026),
                              format_func=ylab, key="years")
years = list(range(yr[0], yr[1] + 1))
view = st.sidebar.radio("Components", ["Grouped", "Detailed"], key="view",
                        help="Grouped combines related factors into the chart's eight bars. "
                             "Detailed shows every modeled factor separately.")
st.sidebar.caption(f"2026 YTD = Jan–{MONTHS[ytd_months]} 2026 vs the same months of 2025. "
                   "Data-center bars are modeled estimates.")
if not sel_markets:
    st.info("Pick at least one market in the sidebar.")
    st.stop()

# ----------------------------------------------------------------------------- header
st.title("Data centers and residential electricity prices")
st.markdown(
    "How much of each year's change in the average residential price came from data centers, "
    "and how much from gas, inflation, grid costs, weather and everything else. "
    "Bars stack the estimated contributions in ¢/kWh; the dot is the actual change."
)

sel = grouped[grouped.market.isin(sel_markets) & grouped.year.isin(years)]
summary = sel.groupby("market", sort=False).agg(
    actual=("_actual", "sum"), dc_net=("_dc_net", "sum"), dc_cap=("_dc_capacity", "sum"), dc_local=("_dc_local", "sum"))
rng = {mkt: dc_range(g) for mkt, g in sel.groupby("market", sort=False)}
summary["lo"] = [rng[m][1] for m in summary.index]
summary["hi"] = [rng[m][2] for m in summary.index]
st.caption(f"Cumulative change, {ylab(years[0])}–{ylab(years[-1])} (change the range in the sidebar). "
           "Red = data centers raised the price, green = lowered it. "
           "The 95% range shows how uncertain the data center estimate is.")
cols = st.columns(min(len(summary), 4))
for i, (mkt, r) in enumerate(summary.iterrows()):
    with cols[i % len(cols)]:
        st.metric(mkt, f"{r.actual:+.2f}¢ total", f"{r.dc_net:+.2f}¢ from data centers", delta_color="inverse",
                  help=f"Cumulative {ylab(years[0])}–{ylab(years[-1])} change in the residential price, "
                       "and the modeled data center contribution (capacity market + local load).")
        st.caption(f"95% range for data centers: {r.lo:+.2f}¢ to {r.hi:+.2f}¢")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Price decomposition", "Data center effect", "National factors", "Price history", "Method & caveats"])

# ----------------------------------------------------------------------------- tab 1
with tab1:
    layout = st.radio("Layout", ["All markets", "One market"], horizontal=True, key="layout")
    if layout == "All markets":
        yrange = shared_range(sel_markets, years, view)
        ncol = 2
        for i in range(0, len(sel_markets), ncol):
            row = st.columns(ncol)
            for j, mkt in enumerate(sel_markets[i:i + ncol]):
                with row[j]:
                    st.plotly_chart(decomposition_fig(mkt, years, view, 360, legend=False, yrange=yrange, title=mkt),
                                    use_container_width=True, key=f"dec-{mkt}")
        # one shared legend, drawn as swatches
        names = GROUP_NAMES if view == "Grouped" else DETAIL_COLS
        sw = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:6px;margin:0 16px 6px 0">'
            f'<span style="width:12px;height:12px;border-radius:2px;background:{GROUP_COLORS.get(n, "#999")};'
            f'{"background-image:repeating-linear-gradient(45deg,rgba(255,255,255,.85) 0 2px,transparent 2px 5px);" if GROUP_TEXTURE.get(n) else ""}'
            f'display:inline-block"></span>{n}</span>' for n in names)
        sw += ('<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px"><span style="width:10px;height:10px;'
               'border-radius:50%;background:#191919;border:2px solid #fff;display:inline-block"></span>Net change (actual)</span>')
        sw += (f'<span style="display:inline-flex;align-items:center;gap:6px"><span style="width:9px;height:9px;'
               f'transform:rotate(45deg);background:{DC_COLOR};display:inline-block"></span>'
               'Data centers, net, with 95% range (beside each bar)</span>')
        if view == "Grouped":
            st.markdown(f'<div style="font-size:0.85rem;line-height:1.6">{sw}</div>', unsafe_allow_html=True)
        else:
            st.caption("Detailed view: hover a bar to see its factor; the one-market layout has a full legend.")
    else:
        mkt = st.selectbox("Market", sel_markets, key="one_market")
        st.plotly_chart(decomposition_fig(mkt, years, view, 560, legend=True, title=mkt, labels=True),
                        use_container_width=True, key="dec-one")

    st.subheader("Table view")
    src = grouped if view == "Grouped" else detail
    comp = GROUP_NAMES if view == "Grouped" else DETAIL_COLS
    t = src[src.market.isin(sel_markets) & src.year.isin(years)][["market", "year", *comp, "_actual"]].copy()
    t["year"] = t.year.map(ylab)
    t = t.rename(columns={"_actual": "Net change (actual)"})
    st.dataframe(t.round(2), hide_index=True, use_container_width=True)
    st.download_button("Download this table (CSV)", t.round(4).to_csv(index=False), "decomposition.csv", "text/csv")

# ----------------------------------------------------------------------------- tab 2
with tab2:
    st.markdown(
        "Data centers move residential prices through two channels. In **PJM**, data center load drives up "
        "capacity-auction prices, and those costs are spread across every PJM state. This channel is measured "
        "fairly precisely. **Locally**, data center load can raise costs (new transmission and generation) or "
        "lower rates (fixed grid costs spread over more kWh). The model cannot pin that local effect down: "
        "its 95% ranges are wide and include zero in every market."
    )
    c1, c2 = st.columns([1.1, 1])
    with c1:
        s = summary.reset_index().iloc[::-1]
        fig = go.Figure()
        fig.add_bar(y=s.market, x=s.dc_cap, orientation="h", name="Via capacity market", marker_color=DC_COLOR,
                    hovertemplate="%{y}<br>Capacity market: %{x:+.2f}¢<extra></extra>")
        fig.add_bar(y=s.market, x=s.dc_local, orientation="h", name="Via local load (net)",
                    marker=dict(color="#b3a5f0", pattern=dict(shape="/", fgcolor="rgba(255,255,255,0.9)", size=6, fillmode="overlay")),
                    hovertemplate="%{y}<br>Local load: %{x:+.2f}¢<extra></extra>")
        fig.add_scatter(y=s.market, x=s.dc_net, mode="markers", name="Net, with 95% range",
                        marker=dict(size=10, color=NET_COLOR, line=dict(width=2, color="white")),
                        error_x=dict(type="data", symmetric=False, array=(s.hi - s.dc_net).values,
                                     arrayminus=(s.dc_net - s.lo).values, color="#444", thickness=1.5, width=5),
                        customdata=np.column_stack([s.lo.round(2), s.hi.round(2)]),
                        hovertemplate="%{y}<br>Net: %{x:+.2f}¢<br>95% range %{customdata[0]}¢ to %{customdata[1]}¢<extra></extra>")
        base_layout(fig, 440, f"Cumulative data center effect, {ylab(years[0])}–{ylab(years[-1])} (¢/kWh)")
        fig.update_yaxes(type="category", ticksuffix="", gridcolor=GRID)
        fig.update_xaxes(type="linear", ticksuffix="¢", gridcolor=GRID, showgrid=True, zeroline=True,
                         zerolinecolor="rgba(128,128,128,0.6)")
        st.plotly_chart(fig, use_container_width=True, key="dc-cum")
    with c2:
        specs = model["specs"]
        head = specs[model["headline_spec"]]

        def sustained(theta, cov, terms):
            """Effect of a sustained +1 pp in non-capacity-market states (current + lag)."""
            v = np.array([1.0 if t in ("dc_share", "dc_share_l1") else 0.0 for t in terms])
            th = np.array([theta[t] for t in terms])
            return float(v @ th), float(1.96 * np.sqrt(v @ np.array(cov) @ v))

        terms = list(head["dml_theta"])
        orig = specs["national scaling, no lag (original)"]
        sc = rob["synthetic_control"]
        rows = [
            ("Double ML (headline)", *sustained(head["dml_theta"], head["dml_cov"], terms), "panel"),
            ("Two-way FE OLS", *sustained(head["ols_theta"], head["ols_cov"], terms), "panel"),
            ("Double ML, original load measure", *sustained(orig["dml_theta"], orig["dml_cov"], list(orig["dml_theta"])), "panel"),
            ("Gradient boosting + SHAP", rob["shap_dc_slope"]["non_capacity_market"], None, "panel"),
            ("Synthetic control: Arizona", sc["AZ"]["implied_effect_per_pp"], None, f"p = {sc['AZ']['placebo_p_value']:.2f}"),
            ("Synthetic control: Virginia", sc["VA"]["implied_effect_per_pp"], None, f"p = {sc['VA']['placebo_p_value']:.2f}"),
            ("Synthetic control: Oregon", sc["OR"]["implied_effect_per_pp"], None, f"p = {sc['OR']['placebo_p_value']:.2f}"),
        ]
        est = pd.DataFrame(rows, columns=["method", "est", "ci", "note"]).iloc[::-1]
        fig = go.Figure()
        fig.add_scatter(
            x=est.est, y=est.method, mode="markers",
            error_x=dict(type="data", array=est.ci.fillna(0), visible=True, color="#666", thickness=1.5, width=6),
            marker=dict(size=11, color=[DC_COLOR if n == "panel" else "#ee7fac" for n in est.note],
                        line=dict(width=2, color="white")),
            customdata=est[["note"]].values,
            hovertemplate="<b>%{y}</b><br>%{x:+.3f}¢ per +1 pp data center share<br>%{customdata[0]}<extra></extra>",
            showlegend=False)
        fig.add_vline(x=0, line_color="rgba(128,128,128,0.6)", line_width=1)
        base_layout(fig, 440, "Local effect of +1 pp data center share (¢/kWh)", legend=False)
        fig.update_yaxes(type="category", ticksuffix="", showgrid=False)
        fig.update_xaxes(type="linear", ticksuffix="¢", showgrid=True, gridcolor=GRID)
        st.plotly_chart(fig, use_container_width=True, key="dc-rob")
        st.caption("Purple = panel methods, for states outside capacity markets, with 95% intervals "
                   "(state-clustered). With the one-year lag, the effect shown is current + lagged. "
                   "Pink = synthetic control for one state, scaled by its 2019–2025 change in data center share; "
                   "p is the placebo rank across donor states.")

    st.subheader("Capacity costs already set by PJM auctions")
    st.markdown(
        "PJM's auctions for 2026/27, 2027/28 and 2028/29 have cleared at or near the price cap, so these "
        "costs are known now and reach bills over the next two years. The purple part is the share the "
        "market monitor attributes to data center load."
    )
    oc = st.columns(2)
    for col, mkt in zip(oc, ["PJM: restructured states", "PJM: Virginia (Dominion)"]):
        d = outlook[outlook.market == mkt]
        fig = go.Figure()
        fig.add_bar(x=d.year.astype(str), y=d.dc_capacity_ckwh, name="Data center share", marker_color=DC_COLOR,
                    error_y=dict(type="data", symmetric=False, array=(d.dc_hi - d.dc_capacity_ckwh).values,
                                 arrayminus=(d.dc_capacity_ckwh - d.dc_lo).values, color="#444", thickness=1.2, width=4),
                    hovertemplate="%{x}: %{y:.2f}¢/kWh from data centers<extra></extra>")
        fig.add_bar(x=d.year.astype(str), y=d.capacity_ckwh - d.dc_capacity_ckwh, name="Other capacity cost",
                    marker_color="#00b894", hovertemplate="%{x}: %{y:.2f}¢/kWh other<extra></extra>")
        base_layout(fig, 340, f"{mkt}: capacity cost in residential rates (¢/kWh)")
        col.plotly_chart(fig, use_container_width=True, key=f"outlook-{mkt}")
    st.caption("Calendar years; each auction covers June–May. Virginia assumes 20% pass-through because "
               "Dominion owns most of its capacity. 2027–2028 use auction results already published; "
               "rates may phase them in differently.")

    st.subheader("How much does the answer depend on modeling choices?")
    tbl = {}
    for name, r in specs.items():
        tbl[name] = {m: f"{v['dc_net']:+.2f}  [{v['lo']:+.2f}, {v['hi']:+.2f}]"
                     for m, v in r["cumulative_2022_2025"].items()}
    st.dataframe(pd.DataFrame(tbl), use_container_width=True)
    st.caption("Cumulative 2022–2025 data center effect (¢/kWh) with 95% range, under each combination of "
               "data center load measure and lag. The PJM restructured states are stable across all four; "
               "the local effects move around and their ranges stay wide.")

    with st.expander("Data center load measure: state-specific vs national scaling"):
        v = pd.DataFrame(model["dc_load_validation"]).T
        v = v.rename(columns={"chained_2023_twh": "State-specific 2023 (TWh)", "epri_2023_twh": "EPRI 2023 (TWh)",
                              "ratio": "Ratio", "used": "Used state-specific"})
        st.dataframe(v.astype({"Ratio": float}).round(2), use_container_width=True)
        st.caption("For states with an EPRI estimate, the state-specific measure follows commercial + industrial "
                   "load above its 2012–2019 trend from 2020 on. It is used only where its 2023 level is within "
                   "35% of EPRI's; elsewhere other industrial or crypto-mining load swamps the signal and the "
                   "national-scaling measure is kept.")

    st.subheader("Synthetic control: actual minus synthetic price")
    fig = go.Figure()
    for st_code, color in [("VA", DC_COLOR), ("OR", "#f47230"), ("AZ", "#00b894")]:
        g = pd.Series(sc[st_code]["gap_by_year"]).sort_index()
        fig.add_scatter(x=[str(k) for k in g.index], y=g.values, mode="lines+markers", name=st_code,
                        line=dict(width=2, color=color), marker=dict(size=8, line=dict(width=2, color="white")),
                        hovertemplate=f"<b>{st_code}</b> %{{x}}: %{{y:+.2f}}¢<extra></extra>")
    base_layout(fig, 340)
    # category axis: 2015 is position 0, so 2020-2025 spans 4.5-10.5
    fig.add_vrect(x0=4.5, x1=10.5, fillcolor="rgba(128,128,128,0.10)", line_width=0,
                  annotation_text="2020–2025", annotation_position="top left")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, key="synth")
    st.caption("Each state is compared with a weighted mix of low-data-center states outside capacity markets, "
               "fitted on 2010–2019 prices. Oregon's 2024–25 gap coincides with large utility rate increases "
               "(power costs, wildfire mitigation) that this method cannot separate from data center growth.")

# ----------------------------------------------------------------------------- tab 3
with tab3:
    st.markdown(
        "The national common factor is the part of each year's change that hit every state at once. "
        "It is split into **inflation** (the prior year's national average price × CPI inflation, "
        "averaged over this year and last) and the remainder **above inflation** (grid investment, "
        "interest rates, equipment costs and other national trends). Before 2020 inflation alone "
        "matches the common trend almost exactly."
    )
    n = national.copy()
    n["year"] = n.year.astype(int)
    n["above"] = n.national_common - n.inflation
    x = [ylab(y) for y in n.year]
    fig = go.Figure()
    fig.add_bar(x=x, y=n.inflation, name="Inflation", marker_color="#84b9f3",
                hovertemplate="Inflation %{x}: %{y:+.2f}¢<extra></extra>")
    fig.add_bar(x=x, y=n.above, name="Grid costs & rates above inflation", marker_color="#1757ac",
                hovertemplate="Above inflation %{x}: %{y:+.2f}¢<extra></extra>")
    fig.add_scatter(x=x, y=n.national_common, mode="markers", name="National common factor (total)",
                    marker=dict(size=10, color=NET_COLOR, line=dict(width=2, color="white")),
                    hovertemplate="Total %{x}: %{y:+.2f}¢<extra></extra>")
    fig.add_vline(x=7.5, line_dash="dot", line_color="rgba(128,128,128,0.6)")
    base_layout(fig, 440, "National common factor by year (¢/kWh, same for every market)")
    st.plotly_chart(fig, use_container_width=True, key="national")
    pre, post = n[n.year <= 2019], n[(n.year >= 2020) & (n.year <= 2025)]
    a, b, c = st.columns(3)
    a.metric("2012–2019 common trend vs inflation", f"{pre.national_common.sum():.2f}¢ vs {pre.inflation.sum():.2f}¢")
    b.metric("2020–2025 inflation", f"{post.inflation.sum():+.2f}¢")
    c.metric("2020–2025 above inflation", f"{post.above.sum():+.2f}¢")
    nt = model["national_dc_test"]
    st.markdown(
        f"**Is the above-inflation part driven by data centers nationally?** Regressing it on the yearly change "
        f"in data centers' share of US load gives {nt['coef_per_pp']:+.2f}¢ per percentage point "
        f"(SE {nt['se']:.2f}, p = {nt['p_value']:.2f}). Taken at face value that would explain "
        f"{nt['implied_2022_2025_cents']:+.2f}¢ of the {nt['above_inflation_2022_2025_cents']:+.2f}¢ above-inflation "
        "increase over 2022–2025. With only 14 yearly data points this test is weak: it finds no evidence "
        "of a national data center effect, but it could not rule out a moderate one either."
    )

# ----------------------------------------------------------------------------- tab 4
with tab4:
    st.markdown("Average residential price by market (revenue ÷ sales, EIA-861M).")
    palette = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2"]
    fig = go.Figure()
    for i, mkt in enumerate(MARKETS):
        if mkt not in sel_markets:
            continue
        d = prices[prices.market == mkt]
        fig.add_scatter(x=d.year, y=d.price, mode="lines+markers", name=mkt,
                        line=dict(width=2, color=palette[i % len(palette)]), marker=dict(size=6),
                        hovertemplate=f"<b>{mkt}</b> %{{x}}: %{{y:.2f}}¢/kWh<extra></extra>")
    base_layout(fig, 480)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, key="prices")
    st.caption(f"{YTD_LABEL} is January–{MONTHS[ytd_months]} only, so it is not directly comparable with "
               "full-year averages (residential prices vary by season).")

# ----------------------------------------------------------------------------- tab 5
with tab5:
    readme = (ROOT / "README.md").read_text()
    start, end = readme.find("## Method"), readme.find("## Reproduce")
    st.markdown(readme[start:end] if start >= 0 else readme)
    st.caption(f"Estimation sample: {model['n_obs']} state-years, {model['n_states']} states, 2011–2025.")
