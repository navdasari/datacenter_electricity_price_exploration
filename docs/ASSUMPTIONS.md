# Modeling assumptions

Every assumption below is a field of `ModelConfig` in `src/config.py`, where its baseline value, source and allowed range are defined. The dashboard's **Assumptions & sensitivity** page builds its controls from those definitions.

**Recompute:**
- **Mechanical:** applied after estimation. Updates in about a second and never changes the econometric estimates.
- **Re-estimate:** changes the inputs to the double ML and fixed-effects regressions, so they must rerun (about 12 s locally for the headline specification).

`cfg.estimation_key()` hashes only the re-estimate fields, so cached estimates are reused whenever only mechanical assumptions change.

## Dependency map

```
data/raw/*  ──►  build_panel.build()            raw-data steps only (gas prices, units)   [local only]
                    ▼
        data/processed/panel_monthly.parquet (in repo)
                    ▼
STAGE 1  pipeline.prepare(panel, cfg)                                              (~0.2 s)
   apply_capacity      capacity prices, PJM/ISO-NE membership, DC share by delivery year
   aggregate           annual and year-to-date state panels
   datacenter_load     data center load: national scaling and state-specific measures
   features, add_lags  outcome net of capacity cost, data center share terms, controls
                    ▼
STAGE 2  pipeline.estimate(prep, cfg)  → θ, Σ, β   cached by estimation_key()      (~12 s)
                    ▼
STAGE 3  pipeline.attribute(prep, est, cfg)                                        (~1.3 s)
   decompose           β·Δx contributions, capacity split, spreading split, national bar
   inflation_component split of the national bar
   add_ranges, cumulative   ranges on the data center effect
   capacity_outlook, national_dc_test
                    ▼
output/*  ──►  robustness.py (synthetic control donors depend on stage 1), chart.py, dashboard
```

## A. Geographic allocation of data center load (`dc_load`)

All of these feed the data center share regressors (current, lagged, and interacted with the capacity-market indicator) and the `ln_nondc_load` control. They also feed the mechanical spreading term and the synthetic-control donor pool. Code: `build_panel.datacenter_load`.

| Field | Baseline | Recompute | Limitation |
|---|---|---|---|
| `national_2025_twh`, `national_2026_twh` | 230, 275 TWh | Re-estimate | LBNL publishes through 2024; these are extrapolations at about 20% a year. They drive all recent growth under national scaling. |
| `backcast_decline` | 3%/yr before 2014 | Re-estimate | Affects only 2010–13. Weakly grounded. |
| `remainder_allocation` | by commercial sales | Re-estimate | National load not covered by EPRI's state estimates is assigned by a sales proxy. |
| `trend_start`, `trend_end` | 2012–2019 | Re-estimate | Window of the pre-boom commercial + industrial trend; the path is chained from `trend_end`. The state-specific measure credits *all* above-trend growth to data centers. |
| `floor_frac` | 30% of national scaling | Re-estimate | Floor on the state-specific path. |
| `validation_tol` | ±35% vs EPRI 2023 | Re-estimate | Threshold effect: a small change can switch a whole state between measures. |
| EPRI 2023 state shares | `data/inputs/datacenter_anchors.csv` | Re-estimate | Treated as inputs and kept as published. Several come from secondary sources. |

## B. Capacity market (`capacity`)

Code: `pipeline.apply_capacity`, `pipeline.features`, `pipeline.capacity_outlook`.

| Field | Baseline | Recompute | Limitation |
|---|---|---|---|
| `illinois_pjm_weight` | 0.6 | Re-estimate | Rough share of Illinois load in PJM (ComEd). It also decides whether Illinois counts as a capacity-market state. |
| `use_dominion_zone` | yes | Re-estimate | Virginia uses the Dominion zone price where published. |
| `pass_through_vert_int` | 20% (VA, WV) | Re-estimate | **The most influential judgment call for Virginia.** It sets the capacity cost subtracted from the price before estimation. |
| `residential_peak_load_factor` | 0.55 | Re-estimate | Converts $/MW-day to ¢/kWh. It scales every capacity number. |
| `dc_share_2025_2027`, `dc_share_2028` | 45%, 40% | Mechanical | The market monitor's attribution, mapped by delivery year (June–May). It moves cost between the "Capacity (non-DC)" and "Data centers: cost push" bars and never changes their sum or the national bar. |
| `dc_share_band` | ±5 points | Mechanical | Width of the capacity part of the range. |
| Auction prices | `data/inputs/capacity_prices.csv` | Re-estimate | Pre-2020 PJM and most ISO-NE values have low or medium confidence. |

## C. Fixed-cost spreading (`spreading`)

| Field | Baseline | Recompute | Limitation |
|---|---|---|---|
| `fixed_share` | 50% | Mechanical | Splits the local data center effect into spreading (negative) and cost push (positive). **It never changes the net data center effect, any other bar or any range.** If the estimated net is more negative than the mechanical spreading, all of it is shown as spreading. One value for every market; it ignores cost allocation between customer classes. |

## D. Inflation pass-through (`inflation`)

| Field | Baseline | Recompute | Limitation |
|---|---|---|---|
| `pass_through` | 1.0 | Mechanical | Splits the national bar into inflation and above-inflation. The national total doesn't change. Calibrated on 2012–2019, not estimated. |
| `current_year_weight` | 0.5 | Mechanical | Weight on this year's vs last year's CPI inflation (rate-case lag). |

The national data center test regresses the above-inflation remainder, so it moves with these.

## E. Uncertainty (`uncertainty`)

| Field | Baseline | Recompute | Limitation |
|---|---|---|---|
| `z` | 1.96 | Mechanical | Normal approximation. |
| `combine` | add | Mechanical | Adding the capacity band to the sampling range is conservative. `quadrature` treats them as independent. |

## F. Fuel and estimation (`fuel`, `estimation`), not adjustable in the app

| Field | Baseline | Note |
|---|---|---|
| `heat_rate` | 7.5 MMBtu/MWh | **Has no effect on any result.** It rescales the gas regressors, which tree models ignore and the linear coefficients offset (verified: 7.0 and 8.5 give identical estimates). |
| `retail_choice` | 14 states | Where the marginal gas price passes to retail rates. |
| `exposure`, `lag` | state-specific, with lag | Adjustable in the app (re-estimate). |
| `est_start`, `est_end`, `n_folds`, `n_rep` | 2010–2025, 5, 5 | Fixed so results stay comparable. |
| `national_test_exposure` | national scaling | The published national test uses the national-scaling series. |

## G. Not configurable

- **Need the raw EIA files** (gas basis filling, Mcf→MMBtu conversion). These are fixed in `build_panel.py` and can't change on the deployed app.
- **Structural definitions** in `pipeline.py`: which states are in PJM or ISO-NE, which PJM states are vertically integrated, and the market definitions in `markets.py`.
- **Decomposition conventions:** linear β·Δx contributions, prior-year residential sales as regional weights, and an unweighted mean across states for the national bar (consistent with the unweighted year effects in estimation).
- **Robustness settings** (`robustness` group): synthetic-control windows and donor rule. These are configurable in code but not rerun in the app.
