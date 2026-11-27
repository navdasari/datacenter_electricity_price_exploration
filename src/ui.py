"""Shared Streamlit / Plotly helpers for the dashboard pages."""
import numpy as np
import plotly.graph_objects as go

from display import DISPLAY

NET_COLOR = "#191919"
DC_COLOR = "#6a4ce2"
GRID = "rgba(128,128,128,0.18)"
YTD_LABEL = "2026 YTD"
MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GROUP_NAMES = [d[0] for d in DISPLAY]
GROUP_COLORS = {d[0]: d[2] for d in DISPLAY}
GROUP_TEXTURE = {d[0]: bool(d[3]) for d in DISPLAY}


def ylab(y):
    return YTD_LABEL if int(y) == 2026 else str(int(y))


def detail_cols(detail):
    return [c for c in detail.columns if c not in ("market", "year") and not c.startswith("_")]


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


def decomposition_fig(grouped, detail, mkt, years, view, height, legend, yrange=None, title=None, labels=False):
    """Stacked contributions for one market, net-change dot, and the data center net
    effect with its range beside each bar."""
    if view == "Grouped":
        d = grouped[(grouped.market == mkt) & grouped.year.isin(years)]
        comps = GROUP_NAMES
    else:
        d = detail[(detail.market == mkt) & detail.year.isin(years)]
        comps = detail_cols(detail)
    labels_x = [ylab(y) for y in d.year]
    x = list(range(len(d)))
    fig = go.Figure()
    for c in comps:
        pattern = dict(shape="/", fgcolor="rgba(255,255,255,0.9)", size=6, fillmode="overlay") if GROUP_TEXTURE.get(c) else None
        fig.add_bar(
            x=x, y=d[c], name=c, legendgroup=c, showlegend=legend,
            marker=dict(color=GROUP_COLORS.get(c), pattern=pattern, line=dict(width=1, color="rgba(255,255,255,0.9)")),
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


def shared_range(frames, markets, years, view):
    """y-axis range covering every panel; frames = list of (grouped, detail) pairs."""
    lo, hi = 0.0, 0.0
    for grouped, detail in frames:
        src = grouped if view == "Grouped" else detail
        cols = GROUP_NAMES if view == "Grouped" else detail_cols(detail)
        d = src[src.market.isin(markets) & src.year.isin(years)]
        hi = max(hi, d[cols].clip(lower=0).sum(axis=1).max(), d._actual.max(), d._dc_net_hi.max())
        lo = min(lo, d[cols].clip(upper=0).sum(axis=1).min(), d._actual.min(), d._dc_net_lo.min())
    return [lo - 0.3, hi + 0.5]


def legend_html(names):
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
    return f'<div style="font-size:0.85rem;line-height:1.6">{sw}</div>'
