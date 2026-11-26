#!/usr/bin/env bash
# Download raw public data into data/raw (needs access to www.eia.gov, fred.stlouisfed.org, www.ncei.noaa.gov)
set -euo pipefail
cd "$(dirname "$0")/../data" && mkdir -p raw/ng raw/noaa && cd raw
curl -sSfL -o sales_revenue.xlsx https://www.eia.gov/electricity/data/eia861m/xls/sales_revenue.xlsx
curl -sSfL -o generation_monthly.xlsx https://www.eia.gov/electricity/data/state/generation_monthly.xlsx
curl -sSfL -o RNGWHHDm.xls https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDm.xls
curl -sSfL -o fredgraph.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
for id in GS10 PCU335311335311; do curl -sSfL -o fred_$id.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=$id"; done
# NOAA statewide codes are 1..48 in this (alphabetical, contiguous US) order
STATES="AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY"
i=0
for s in $STATES; do
  i=$((i+1))
  curl -sSfL -o ng/$s.xls "https://www.eia.gov/dnav/ng/hist_xls/N3045${s}3m.xls"   # gas price to power plants
  for p in hdd cdd; do
    curl -sSfL -o noaa/${s}_$p.csv "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/$i/$p/1/0/2008-2026/data.csv"
  done
done
echo "done"
