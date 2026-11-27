"""Assumptions & sensitivity page: change the model's assumptions and compare with the
published baseline.

Mechanical assumptions are applied on top of existing estimates and update in about a
second. Assumptions that change the econometric estimates use, in order:
  1. the published baseline estimates (output/model_results.json),
  2. a precomputed alternative (output/scenarios/),
  3. estimates already computed on this server,
  4. otherwise a "Re-estimate" button that reruns the headline specification.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT, SCN = ROOT / "output", ROOT / "output/scenarios"
sys.path.insert(0, str(ROOT / "src"))
from config import GROUP_TITLES, ModelConfig  # noqa: E402
from display import grouped as group_display  # noqa: E402
from markets import MARKETS  # noqa: E402
from pipeline import (Estimates, attribute, cumulative_one, estimate, estimates_from_results_json,  # noqa: E402
                      load_panel, prepare, sustained_effect)
from ui import (DC_COLOR, GRID, GROUP_NAMES, NET_COLOR, base_layout, decomposition_fig,  # noqa: E402
                legend_html, shared_range, ylab)

BASE = ModelConfig.baseline()
EDITABLE_GROUPS = ["dc_load", "capacity", "spreading", "inflation", "fuel", "estimation", "uncertainty"]
OPTION_LABELS = {
    "commercial": "Commercial sales (baseline)", "commercial_industrial": "Commercial + industrial sales",
    "total": "Total sales", "state_specific": "State-specific (headline)",
    "national_scaling": "National scaling (original)", "add": "Add (conservative, baseline)",
    "quadrature": "Quadrature (independent errors)", True: "Yes", False: "No",
}
BASE_COLOR = "#a3a3a3"


# ----------------------------------------------------------------------------- data and caches
@st.cache_resource
def panel():
    return load_panel()


@st.cache_resource
def published():
    est = estimates_from_results_json(OUT / "model_results.json")
    dec = pd.read_csv(OUT / "decomposition.csv")
    return est, dec, pd.read_csv(OUT / "capacity_outlook.csv"), json.loads((OUT / "model_results.json").read_text())


@st.cache_resource
def scenario_index():
    p = SCN / "index.json"
    return json.loads(p.read_text()) if p.exists() else []


@st.cache_resource
def estimate_registry():
    """Estimates computed on this server, shared across sessions: key -> dict."""
    return {}


def find_estimates(cfg):
    key = cfg.estimation_key()
    if key == BASE.estimation_key():
        return published()[0], "published baseline"
    f = SCN / f"{key}.json"
    if f.exists():
        d = json.loads(f.read_text())
        return Estimates.from_dict(d), f"precomputed: {d.get('label', key)}"
    reg = estimate_registry()
    if key in reg:
        return Estimates.from_dict(reg[key]), "re-estimated on this server"
    return None, None


@st.cache_data(max_entries=32, show_spinner=False)
def prepared(cfg_json):
    return prepare(panel(), ModelConfig.from_json(cfg_json))


@st.cache_data(max_entries=32, show_spinner=False)
def results(cfg_json, est_json):
    cfg = ModelConfig.from_json(cfg_json)
    est = Estimates.from_dict(json.loads(est_json))
    prep = prepared(cfg_json)
    res = attribute(prep, est, cfg)
    return res.decomposition, res.outlook, res.national_test, prep.validation


# ----------------------------------------------------------------------------- controls
def wkey(group, name):
    return f"asm::{group}.{name}"


def set_widgets(cfg):
    for g, n, v, meta in cfg.items():
        if g in EDITABLE_GROUPS and meta["ui"] is not None:
            st.session_state[wkey(g, n)] = v


def control(group, name, value, meta):
    k, ui, label = wkey(group, name), meta["ui"], meta["label"]
    help_ = f"{meta['help']}  \nSource: {meta['source']}. Baseline: {fmt(value)}."
    if k not in st.session_state:
        st.session_state[k] = value
    if isinstance(ui, tuple):
        lo, hi, step = ui
        if isinstance(value, int) and not isinstance(value, bool):
            return st.slider(label, int(lo), int(hi), step=int(step), key=k, help=help_)
        return st.slider(label, float(lo), float(hi), step=float(step), key=k, help=help_)
    return st.selectbox(label, ui, key=k, help=help_, format_func=lambda o: OPTION_LABELS.get(o, str(o)))


def fmt(v):
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, tuple):
        return ", ".join(map(str, v))
    return OPTION_LABELS.get(v, str(v))


def config_from_widgets():
    vals = {}
    for g, n, v, meta in BASE.items():
        if g in EDITABLE_GROUPS and meta["ui"] is not None:
            x = st.session_state.get(wkey(g, n), v)
            if isinstance(v, float):
                x = round(float(x), 6)
            vals[f"{g}.{n}"] = x
    return BASE.with_values(**vals)


def meta_of(dotted):
    g, n = dotted.split(".")
    return dict(next(m for gg, nn, _, m in BASE.items() if gg == g and nn == n))


# ----------------------------------------------------------------------------- page
st.title("Assumptions & sensitivity")
st.markdown(
    "Every number in the results depends on modeling assumptions. Change them here and compare with the "
    "published baseline. Two kinds of assumption behave differently:"
)
c1, c2 = st.columns(2)
c1.success("**Updates instantly.** Applied after estimation: the data center share of PJM capacity "
           "costs, fixed-cost spreading, inflation pass-through and how the uncertainty ranges are combined.")
c2.warning("**Requires re-estimation.** Changes the inputs to the econometric model: where data center "
           "load is allocated, how capacity prices are converted, pass-through, fuel cost and the "
           "specification. Common alternatives are precomputed; anything else takes about 15–40 seconds.")

# ---- quick picks
idx = scenario_index()
picks = {"Baseline": {}} | {s["label"]: s["changes"] for s in idx if s["label"] != "Baseline"}


def apply_pick():
    set_widgets(BASE.with_values(**picks[st.session_state["asm_pick"]]))


st.selectbox("Start from a precomputed alternative (loads instantly)", list(picks), key="asm_pick",
             on_change=apply_pick,
             help="Each alternative changes one assumption that requires re-estimation; its estimates are "
                  "already computed. You can adjust mechanical assumptions on top.")

# ---- form
with st.form("assumptions"):
    for g in EDITABLE_GROUPS:
        items = [(n, v, m) for gg, n, v, m in BASE.items() if gg == g and m["ui"] is not None]
        if not items:
            continue
        with st.expander(GROUP_TITLES[g], expanded=g in ("dc_load", "capacity")):
            inst = [(n, v, m) for n, v, m in items if not m["reestimate"]]
            reest = [(n, v, m) for n, v, m in items if m["reestimate"]]
            for title, block in [("Updates instantly", inst), ("Requires re-estimation", reest)]:
                if not block:
                    continue
                st.caption(title)
                cols = st.columns(2)
                for i, (n, v, m) in enumerate(block):
                    with cols[i % 2]:
                        control(g, n, v, m)
    b1, b2 = st.columns([1, 1])
    submitted = b1.form_submit_button("Update results", type="primary")
    reset = b2.form_submit_button("Reset to baseline")

if reset:
    set_widgets(BASE)
    st.session_state["asm_pick"] = "Baseline"
    st.rerun()

cfg = config_from_widgets()
changes = BASE.diff(cfg)

# ---- what changed
if not changes:
    st.info("All assumptions are at their baseline values. Change some above, or start from a precomputed "
            "alternative, then press **Update results**.")
else:
    rows = []
    for k, (b, s) in changes.items():
        m = meta_of(k)
        rows.append({"Assumption": m["label"], "Baseline": fmt(b), "Scenario": fmt(s),
                     "Effect": "Re-estimation" if m["reestimate"] else "Instant"})
    st.subheader("Your scenario")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---- estimates
est, est_source = find_estimates(cfg)
if est is None:
    re_changes = {k: v for k, v in changes.items() if meta_of(k)["reestimate"]}
    st.warning(
        "This combination needs new estimates: " + "; ".join(meta_of(k)["label"] for k in re_changes) +
        ". Re-estimating runs the headline double ML specification (5 folds × 5 repetitions) and takes "
        "about 15–40 seconds. Results are kept on this server for anyone who picks the same assumptions."
    )
    if st.button("Re-estimate the model", type="primary"):
        with st.spinner("Estimating the headline specification…"):
            prep = prepared(cfg.to_json())
            new = estimate(prep, cfg, label="re-estimated in the app")
            estimate_registry()[cfg.estimation_key()] = new.to_dict()
        st.rerun()
    st.stop()

base_est, base_dec, base_outlook, base_model = published()
with st.spinner("Updating results…"):
    dec, outlook, ntest, validation = results(cfg.to_json(), json.dumps(est.to_dict(), default=float))
base_g, scn_g = group_display(base_dec), group_display(dec)
st.caption(f"Estimates used: {est_source}. Baseline values come from the published outputs.")

# ---- cumulative data center effect
st.subheader("Data center effect by market: baseline vs scenario")
yr = st.select_slider("Years", options=list(range(2020, 2027)), value=(2022, 2025), format_func=ylab, key="asm_years")
years = list(range(yr[0], yr[1] + 1))


def sgn(x):
    """Signed 2-decimal string without a negative zero."""
    return f"{round(float(x), 2) + 0.0:+.2f}"


def cum_table(dec_, cov, unc):
    d = dec_[dec_.year.isin(years)]
    return {m: cumulative_one(g, cov, unc) for m, g in d.groupby("market", sort=False)}


cb = cum_table(base_dec, base_est.cov.values, BASE.uncertainty)
cs = cum_table(dec, est.cov.values, cfg.uncertainty)
mk = list(MARKETS)[::-1]
fig = go.Figure()
for label, c, color, off in [("Baseline", cb, BASE_COLOR, 0.18), ("Scenario", cs, DC_COLOR, -0.18)]:
    yv = np.arange(len(mk)) + off
    net = np.array([c[m]["dc_net"] for m in mk])
    lo, hi = np.array([c[m]["lo"] for m in mk]), np.array([c[m]["hi"] for m in mk])
    fig.add_scatter(x=net, y=yv, mode="markers", name=label,
                    marker=dict(size=11, color=color, line=dict(width=2, color="white")),
                    error_x=dict(type="data", symmetric=False, array=hi - net, arrayminus=net - lo,
                                 color=color, thickness=1.6, width=5),
                    customdata=np.column_stack([mk, lo.round(2), hi.round(2)]),
                    hovertemplate=f"<b>%{{customdata[0]}}</b><br>{label}: %{{x:+.2f}}¢<br>"
                                  "range %{customdata[1]}¢ to %{customdata[2]}¢<extra></extra>")
fig.add_vline(x=0, line_color="rgba(128,128,128,0.6)", line_width=1)
base_layout(fig, 420, f"Cumulative data center effect, {ylab(years[0])}–{ylab(years[-1])} (¢/kWh), with range")
fig.update_yaxes(type="linear", tickvals=list(range(len(mk))), ticktext=mk, ticksuffix="", gridcolor=GRID,
                 zeroline=False)
fig.update_xaxes(type="linear", ticksuffix="¢", showgrid=True, gridcolor=GRID)
st.plotly_chart(fig, width="stretch", key="asm-cum")

tbl = pd.DataFrame([{
    "Market": m,
    "Actual change": sgn(cb[m]["actual"]),
    "Baseline: data centers": f"{sgn(cb[m]['dc_net'])}  [{sgn(cb[m]['lo'])}, {sgn(cb[m]['hi'])}]",
    "Scenario: data centers": f"{sgn(cs[m]['dc_net'])}  [{sgn(cs[m]['lo'])}, {sgn(cs[m]['hi'])}]",
    "Change": sgn(cs[m]["dc_net"] - cb[m]["dc_net"]),
    "of which capacity": sgn(cs[m]["dc_capacity"] - cb[m]["dc_capacity"]),
    "of which local": sgn(cs[m]["dc_local"] - cb[m]["dc_local"]),
} for m in MARKETS])
st.dataframe(tbl, hide_index=True, width="stretch")

# ---- decomposition side by side
st.subheader("Price decomposition: baseline vs scenario")
mkt = st.selectbox("Market", list(MARKETS), key="asm_market")
yrange = shared_range([(base_g, base_dec), (scn_g, dec)], [mkt], years, "Grouped")
l, r = st.columns(2)
l.plotly_chart(decomposition_fig(base_g, base_dec, mkt, years, "Grouped", 380, False, yrange, "Baseline"),
               width="stretch", key="asm-dec-b")
r.plotly_chart(decomposition_fig(scn_g, dec, mkt, years, "Grouped", 380, False, yrange, "Scenario"),
               width="stretch", key="asm-dec-s")
st.markdown(legend_html(GROUP_NAMES), unsafe_allow_html=True)

diff = (scn_g.set_index(["market", "year"])[GROUP_NAMES] - base_g.set_index(["market", "year"])[GROUP_NAMES])
diff = diff.loc[mkt].loc[years]
diff = diff.loc[:, (diff.abs() > 0.005).any()]
with st.expander("Change in each bar, scenario minus baseline (¢/kWh)", expanded=bool(changes)):
    if diff.empty:
        st.caption("No bar changes by more than 0.005¢ in the selected years.")
    else:
        d2 = diff.round(3)
        d2.index = [ylab(y) for y in d2.index]
        st.dataframe(d2, width="stretch")
        st.caption("Bars that don't change are hidden. The actual price change never moves; whatever an "
                   "assumption moves into or out of one bar comes out of another, usually 'Other / unexplained' "
                   "or the national bars.")

# ---- capacity outlook
st.subheader("PJM capacity costs already set by auctions")
oc = st.columns(2)
for col, m in zip(oc, ["PJM: restructured states", "PJM: Virginia (Dominion)"]):
    b, s = base_outlook[base_outlook.market == m], outlook[outlook.market == m]
    fig = go.Figure()
    fig.add_bar(x=b.year.astype(str), y=b.dc_capacity_ckwh, name="Baseline: data center share", marker_color=BASE_COLOR,
                hovertemplate="%{x}: %{y:.2f}¢/kWh (baseline)<extra></extra>")
    fig.add_bar(x=s.year.astype(str), y=s.dc_capacity_ckwh, name="Scenario: data center share", marker_color=DC_COLOR,
                error_y=dict(type="data", symmetric=False, array=(s.dc_hi - s.dc_capacity_ckwh).values,
                             arrayminus=(s.dc_capacity_ckwh - s.dc_lo).values, color="#444", thickness=1.2, width=4),
                hovertemplate="%{x}: %{y:.2f}¢/kWh (scenario)<extra></extra>")
    base_layout(fig, 320, f"{m}: data center share of capacity cost (¢/kWh)")
    fig.update_layout(barmode="group")
    col.plotly_chart(fig, width="stretch", key=f"asm-outlook-{m}")

# ---- coefficients and national test
st.subheader("Estimates")
terms_b, terms_s = list(base_est.theta.index), list(est.theta.index)
eb, sb = sustained_effect(base_est.theta, base_est.cov.values, terms_b)
es, ss = sustained_effect(est.theta, est.cov.values, terms_s)
k1, k2, k3 = st.columns(3)
k1.metric("Local effect of +1 pp data center share (states outside capacity markets)",
          f"{es:+.3f}¢ (SE {ss:.3f})", f"{es - eb:+.3f}¢ vs baseline", delta_color="off")
nb = base_model["national_dc_test"]
k2.metric("National data center test: ¢ per pp", f"{ntest['coef_per_pp']:+.3f} (p = {ntest['p_value']:.2f})",
          f"{ntest['coef_per_pp'] - nb['coef_per_pp']:+.3f} vs baseline", delta_color="off")
k3.metric("Estimation sample", f"{est.n_obs} state-years", f"{est.n_obs - base_est.n_obs:+d} vs baseline", delta_color="off")
with st.expander("Which states use the state-specific data center load measure"):
    v = pd.DataFrame(validation).T.rename(columns={"chained_2023_twh": "State-specific 2023 (TWh)",
                                                   "epri_2023_twh": "EPRI 2023 (TWh)", "ratio": "Ratio",
                                                   "used": "Used state-specific"})
    st.dataframe(v.astype({"Ratio": float}).round(2), width="stretch")

# ---- export
st.subheader("Reproduce this scenario")
st.download_button("Download scenario (JSON)", cfg.to_json(), "scenario.json", "application/json")
st.code("python3 src/model.py --config scenario.json --out output_scenario", language="bash")
st.caption("The command runs all four specifications under these assumptions and writes the same output files "
           "as the published run, in a separate folder.")

# ---- reference
with st.expander("All assumptions: values, sources and what they affect"):
    ref = [{"Group": GROUP_TITLES[g], "Assumption": m["label"] or n, "Baseline": fmt(v),
            "Scenario": fmt(getattr(getattr(cfg, g), n)),
            "Effect": "Re-estimation" if m["reestimate"] else "Instant",
            "Adjustable here": "Yes" if (m["ui"] is not None and g in EDITABLE_GROUPS) else "No",
            "Source": m["source"]} for g, n, v, m in BASE.items()]
    st.dataframe(pd.DataFrame(ref), hide_index=True, width="stretch")
    st.caption("Not adjustable here: the estimation sample and cross-fitting settings (fixed so results stay "
               "comparable), the robustness-check settings, and steps that need the raw EIA files, which "
               "aren't deployed (gas price construction and unit conversions). EPRI's published state shares "
               "are inputs, not assumptions, and are kept as published.")
