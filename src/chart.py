"""Stacked increase/decrease chart with net-change marker, one panel per market.

Writes output/decomposition_chart.png and output/chart_data.json (grouped components).
Palette: validated categorical order (dataviz reference palette, light mode).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "output"

# Display groups (stack order from the zero line outward) -> source columns
DISPLAY = [
    ("Data centers: cost push", ["Data centers: cost push"], "#6a4ce2", None),
    ("Data centers: fixed-cost spreading", ["Data centers: fixed-cost spreading"], "#b3a5f0", "///"),
    ("Capacity market (non-DC)", ["Capacity market (non-DC)"], "#00b894", None),
    ("Natural gas / fuel", ["Natural gas / fuel"], "#f47230", None),
    ("National common factors", ["National common factors (inflation, rates, equipment, etc.)"], "#257de0", None),
    ("Weather & hydro", ["Weather", "Hydro conditions"], "#f2a600", None),
    ("Load & usage (non-DC)", ["Non-DC load growth", "Lower use per customer (efficiency, rooftop solar)"], "#ee7fac", None),
    ("Other / unexplained", ["Other / unexplained (T&D, policy, storms, rate cases)"], "#a3a3a3", None),
]


def grouped(dec):
    out = dec[["market", "year", "_actual", "_price_prev", "_price", "_dc_net", "_dc_capacity", "_dc_local"]].copy()
    for name, cols, *_ in DISPLAY:
        out[name] = dec[cols].sum(axis=1)
    return out


def plot(g, path, ytd_months):
    markets = list(dict.fromkeys(g.market))
    n = len(markets)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow + 1.2), sharey=True)
    axes = axes.ravel()
    names = [d[0] for d in DISPLAY]
    for ax, mkt in zip(axes, markets):
        d = g[g.market == mkt].set_index("year")
        x = np.arange(len(d))
        pos, neg = np.zeros(len(d)), np.zeros(len(d))
        for name, _, color, hatch in DISPLAY:
            v = d[name].values
            vp, vn = np.clip(v, 0, None), np.clip(v, None, 0)
            ax.bar(x, vp, 0.62, bottom=pos, color=color, hatch=hatch, edgecolor="white", linewidth=1)
            ax.bar(x, vn, 0.62, bottom=neg, color=color, hatch=hatch, edgecolor="white", linewidth=1)
            pos += vp
            neg += vn
        ax.scatter(x, d._actual.values, s=46, color="#191919", zorder=5, edgecolor="white", linewidth=1.5)
        for xi, v in zip(x, d._actual.values):
            ax.annotate(f"{v:+.2f}", (xi, v), textcoords="offset points", xytext=(0, 8 if v >= 0 else -14),
                        ha="center", fontsize=7.5, color="#191919")
        ax.axhline(0, color="#c2c2c2", lw=1)
        ax.set_xticks(x, [str(y) if y < 2026 else f"2026\nYTD" for y in d.index], fontsize=8)
        ax.set_title(f"{mkt}\n{d._price_prev.iloc[0]:.1f}¢ (2019) → {d._price.iloc[-2]:.1f}¢ (2025)", fontsize=9.5, loc="left")
        ax.grid(axis="y", color="#e7e7e7", lw=0.8)
        ax.set_axisbelow(True)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(colors="#666666", labelsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].set_ylabel("Change in residential price (¢/kWh, year over year)", fontsize=8.5, color="#666666")
    axes[ncol].set_ylabel("Change in residential price (¢/kWh, year over year)", fontsize=8.5, color="#666666")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, hatch=h, ec="white") for _, _, c, h in DISPLAY]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="#191919", markersize=7))
    leg_ax = axes[n] if n < len(axes) else fig.add_axes([0.75, 0.05, 0.2, 0.3])
    leg_ax.legend(handles, names + ["Net change (actual)"], loc="center left", frameon=False, fontsize=8.5)
    fig.suptitle("What moved residential electricity prices, 2020–2026", fontsize=13, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.955, f"Stacked contributions to each year's change; dot = actual net change. 2026 = Jan–{['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][ytd_months]} vs same months of 2025. "
             "Data-center bars are modeled estimates (double ML + market-monitor capacity attribution).",
             fontsize=8.5, color="#666666")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    dec = pd.read_csv(OUTD / "decomposition.csv")
    res = json.loads((OUTD / "model_results.json").read_text())
    g = grouped(dec)
    plot(g, OUTD / "decomposition_chart.png", res["ytd_months"])
    rob = json.loads((OUTD / "robustness.json").read_text())
    payload = {
        "groups": [{"name": n, "color": c, "texture": bool(h)} for n, _, c, h in DISPLAY],
        "rows": g.round(4).to_dict(orient="records"),
        "model": {k: res[k] for k in ["dml_theta", "dml_se", "ols_fe_theta", "ols_fe_se", "n_obs", "n_states", "ytd_months"]},
        "robustness": {"shap_dc_slope": rob["shap_dc_slope"],
                       "synthetic_control": {k: {kk: v[kk] for kk in ["gap_2025_cents", "dc_share_change_2019_2025_pp",
                                                                     "implied_effect_per_pp", "placebo_p_value", "top_donors"]}
                                             for k, v in rob["synthetic_control"].items()}},
    }
    (OUTD / "chart_data.json").write_text(json.dumps(payload, indent=1))
    print("wrote chart + json")
