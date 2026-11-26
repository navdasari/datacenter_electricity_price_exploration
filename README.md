# Data centers and residential electricity prices

How much of the 2020–2026 change in residential electricity prices came from data centers, and how much from everything else? This project looks at five US markets:

- PJM, split into Virginia (Dominion) and the restructured PJM states (DE, DC, MD, NJ, OH, PA)
- ERCOT (Texas)
- Arizona (APS/SRP)
- New England (ISO-NE)
- Pacific Northwest, split into Oregon and Washington

![decomposition](output/decomposition_chart.png)

Each bar stacks the estimated contributions to that year's change in the average residential price (¢/kWh). The dot is the actual change. The chart shows 2022–2026 (set by `START_YEAR` in `src/chart.py`); `output/decomposition.csv` has every year from 2020. 2026 is January–September 2026 compared with the same months of 2025.

## Headline findings (cumulative 2020–2025, ¢/kWh)

| Market | Actual change | Data centers, net | of which: capacity market | of which: local load |
|---|---|---|---|---|
| PJM: restructured states | +5.57 | **+0.59** | +0.58 | +0.01 |
| PJM: Virginia (Dominion) | +3.21 | **+0.27** | +0.18 | +0.08 |
| New England | +7.74 | ~0 | 0 | ~0 |
| ERCOT (Texas) | +3.64 | **−0.56** | 0 | −0.56 |
| Arizona | +2.82 | **−0.96** | 0 | −0.96 |
| Oregon | +4.45 | **−1.44** | 0 | −1.44 |
| Washington | +3.46 | **−0.89** | 0 | −0.89 |

- **PJM is where data centers raise residential prices.** The effect runs through the capacity market. PJM's market monitor attributes about 45% of capacity charges from the 2025/26 auction onward to data center load, and those costs are spread across every PJM state. Restructured states, which buy capacity at market prices, feel this most. Data centers contribute about +0.6¢/kWh in 2025 and again in the first nine months of 2026, out of a +1.7¢ and +2.1¢ change in those periods.
- **Outside capacity markets, the panel estimates point the other way.** Data center load spreads fixed grid costs over more kWh. The double ML estimate is −0.20¢/kWh for each percentage point of data center share of state load (SE 0.10). This matches LBNL's finding that load growth has tended to lower retail prices. **But the effect is not robust** (see below).
- **Most of the increase everywhere is not data centers.** National common factors account for about +4.9¢ of the cumulative increase in every market. About **3.3¢ is general inflation** passing through to rates. The other **1.6¢ is above inflation**, concentrated in 2024 and 2026 YTD, consistent with the nationwide grid spending wave, higher interest rates feeding into rate cases, and equipment costs. Natural gas drove the 2022–23 spikes in restructured markets and the 2024 reversal (New England, PJM, ERCOT).

## Method

1. **Panel data.** 49 states (contiguous US plus DC) by year, 2010–2025, plus January–September 2026, built from public data. Sources: EIA-861M residential revenue, sales and customers; EIA-923 generation mix; EIA gas prices to power plants and Henry Hub; FRED CPI, 10-year Treasury yield and transformer PPI; NOAA heating and cooling degree days.
2. **Capacity costs (mechanical).** PJM and ISO-NE auction prices are converted to ¢/kWh of residential load. Restructured states get full pass-through; Virginia and West Virginia get 20%, since their utilities self-supply most capacity. The data center share comes from the PJM market monitor, about 45% from 2025/26 on. These costs are subtracted from the price before estimation.
3. **Data center exposure (shift-share).** Each state's 2023 data center load comes from EPRI (2024) state shares; the remaining national load is allocated by commercial sales. It is scaled over time by the LBNL national series: 58 TWh (2014), 176 TWh (2023), 192 TWh (2024), with 2025–26 extrapolated at about 20% a year.
4. **Double machine learning** (`src/model.py`). A partially linear model with state and year fixed effects, using LightGBM models for the controls and cross-fitting grouped by state (5 folds × 5 repeats, standard errors clustered by state). The treatment is data center share of state load, interacted with a capacity-market indicator.
5. **Other factors.** A two-way fixed-effects regression of the price (net of capacity and data center effects) on:
   - gas cost: in-state gas share × gas price, plus marginal gas price in restructured retail states, current and lagged
   - hydro share
   - degree days
   - use per customer
   - non-data-center load
6. **Decomposition.** Each factor's contribution is its coefficient × the year-over-year change in that factor. The change in the year fixed effect is the national common factor, split into two bars:
   - **Inflation:** the prior year's national average residential price × CPI inflation, averaged over this year and last because rate cases lag. Pass-through is set to 1, not estimated. Over 2012–2019 this explains the national common trend almost exactly: 1.66¢ predicted against 1.71¢ actual.
   - **Grid costs & rates above inflation:** the remainder.

   Inflation, interest rates and transformer prices can't be estimated separately, because they are national series and CPI and the transformer price index move almost together (correlation 0.96, 14 years). Free regressions on them gave unstable, wrong-signed coefficients.

   Whatever is left after the national bars is "other / unexplained," which covers state-specific transmission and distribution rate cases, policy charges, storm and wildfire recovery, and fuel-clause timing.
7. **Splitting the data center effect into two bars.** Fixed-cost spreading is calculated mechanically: 50% of the prior rate × the data center load growth ÷ total load. The cost push is the net effect minus the spreading. If the estimated net effect is more negative than the mechanical spreading, all of it is shown as spreading.

## Robustness: do the methods agree?

| Data center effect per +1 pp share | Estimate |
|---|---|
| Two-way fixed-effects OLS (non-capacity-market states) | −0.29 (SE 0.10) |
| Double ML (headline) | −0.20 (SE 0.10) |
| Gradient boosting + SHAP slope | −0.12 |
| Synthetic control: Arizona | −0.08 (placebo p = 0.18) |
| Synthetic control: Virginia | +0.02 (p = 0.59) |
| Synthetic control: Oregon | **+0.29 (p = 0.05)** |

The three panel methods agree on a negative sign. The synthetic control for Oregon disagrees: Oregon prices rose about 2¢ more than its synthetic comparison in 2024–25. That jump coincides with large Portland General Electric and PacifiCorp rate increases (power costs, wildfire mitigation), which this model cannot separate from data center growth. **Treat the negative data center bars in Oregon, Washington and Arizona as suggestive, not settled.** Details are in `output/robustness.json`.

## Key assumptions and limitations

- **No observed state-by-year data center load series exists.** The shift-share measure assumes every state's data center load grows at the national rate from its 2023 level. That understates recent Texas and Arizona growth.
- **Inputs with low or medium confidence** are flagged in `data/inputs/*.csv`. Pre-2020 capacity prices and some EPRI state shares come from secondary sources and should be checked against the primary reports. 2025–26 national data center load is an extrapolation.
- **Geography is state-level.** Arizona combines APS, SRP and TEP. Virginia includes Appalachian Power as well as Dominion. Texas includes some non-ERCOT load. Utility-level splits would need the EIA-861 utility files.
- **Things not modeled explicitly:**
  - the Winter Storm Uri securitization charges in Texas
  - Texas fixed-price retail contracts, which delay gas pass-through by more than a year (hence the large "other" bars in Texas in 2023–24)
  - Dominion's 2025 biennial rate review (Virginia, 2026)
  - Massachusetts and Connecticut policy charges
  - Washington's Climate Commitment Act
  - Oregon and Arizona wildfire costs
- **The two national bars assume every state got the average common shock.** Deviations from that average land in "other / unexplained." The "above inflation" bar is a remainder, not a direct measurement of grid spending or interest rates.

## Reproduce

```bash
pip install -r requirements.txt
```

```bash
src/download.sh
```

```bash
cd src && python3 build_panel.py && python3 model.py && python3 robustness.py && python3 chart.py
```

Outputs:

- `output/decomposition.csv`: every component, by market and year
- `output/model_results.json`: coefficients, standard errors and assumptions
- `output/robustness.json`: SHAP and synthetic control results
- `output/decomposition_chart.png`: the chart
