"""Display groups shared by the static chart and the dashboard: how the model's detailed
components are grouped into stacked bars, and their colors (validated palette)."""

# Display groups (stack order from the zero line outward) -> source columns
DISPLAY = [
    ("Data centers: cost push", ["Data centers: cost push"], "#6a4ce2", None),
    ("Data centers: fixed-cost spreading", ["Data centers: fixed-cost spreading"], "#b3a5f0", "///"),
    ("Capacity market (non-DC)", ["Capacity market (non-DC)"], "#00b894", None),
    ("Natural gas / fuel", ["Natural gas / fuel"], "#f47230", None),
    ("National: inflation", ["National: inflation"], "#84b9f3", None),
    ("National: grid costs & rates above inflation", ["National: grid costs & interest rates above inflation"], "#1757ac", None),
    ("Weather & hydro", ["Weather", "Hydro conditions"], "#f2a600", None),
    ("Load & usage (non-DC)", ["Non-DC load growth", "Lower use per customer (efficiency, rooftop solar)"], "#ee7fac", None),
    ("Other / unexplained", ["Other / unexplained (T&D, policy, storms, rate cases)"], "#a3a3a3", None),
]

def grouped(dec):
    keep = ["market", "year", "_actual", "_price_prev", "_price", "_dc_net", "_dc_capacity", "_dc_local",
            "_dc_net_lo", "_dc_net_hi", "_dc_cap_lo", "_dc_cap_hi"] + [c for c in dec.columns if c.startswith("_g")]
    out = dec[keep].copy()
    for name, cols, *_ in DISPLAY:
        out[name] = dec[cols].sum(axis=1)
    return out
