# Hase

The **Kamakura Quant Lab** analysis toolkit. Reads the data Komachi downloads
and draws it.

Named for the district where the Great Buddha sits. Komachi is where you get
things; Hase is where you go to look at them.

Design: `system/04_hase-analysis-toolkit.md` in the Hotaka repository.

```bash
pip install -e .
hase dates       --market COINCHECK:BTC_SPOT
hase plot-trades --market COINCHECK:BTC_SPOT --date 2025-07-01 --output trades.png
hase plot-book   --market COINCHECK:BTC_SPOT --date 2025-07-01 --output book.png
```

## It holds no credential

Hase reads local files. It reaches no API, holds no token, and knows nothing
about entitlement or billing. That single rule is what lets one tool serve both
as a customer product and as an analysis environment for data you already have:
point it at a Komachi download or at your own warehouse, and it behaves the
same.

If a day is missing, Hase says so and tells you the `komachi` command that
would fetch it. It will not fetch anything itself.

## Where it reads from

`--root`, then `ROOT_PATH` or `KQL_ROOT_PATH` in the environment, then
`KQL_ROOT_PATH` in a `.env` in the working directory, then `~/kql-data`. The
`.env` is the one Komachi writes on first use, so the two tools agree without
being configured twice.

```
<root>/bronze/dataset=Trade/exchange=COINCHECK/symbol=BTC_SPOT/date=2025-07-01/data.parquet
```

## Dates are Tokyo days

A `date` partition holds the rows falling in `[00:00 JST, 24:00 JST)`, which is
`15:00` to `14:59` UTC. Timestamps inside the files are UTC epochs.

Plots are labelled in JST, matching the day they were cut in. This needs saying
because matplotlib converts datetimes to plain floats and forgets the zone: a
tz-aware series plots in the right place but is *labelled* in UTC unless the
formatter is told otherwise. A chart titled JST with a UTC axis is worse than
one with no timezone at all, so `plot/style.py` owns the zone and every axis
gets it explicitly.

## What the plots do

**`plot-trades`** — price across the day, with buy and sell volume beneath.
Volume is bucketed (`--bucket`, default `5min`) because a day is tens of
thousands of trades and a bar each is an unreadable smear. Buys and sells are
drawn apart rather than combined, since their balance is the reason to look.

**`plot-book`** — best bid and ask, with the spread beneath. Resampled to `1s`
by default: a day is around 230,000 snapshots, more points than the chart has
pixels. `--no-resample` keeps every one for a short window where the detail
matters. Resampling takes the last value in each interval rather than the mean,
because a mean of neighbouring quotes is a price that never existed and the
spread between two such means can come out negative.

The spread panel clips its view to the 99.5th percentile and labels how many
points fall above the cut. A few momentary wide spreads would otherwise set the
scale and flatten the rest of the day into a line, and hiding them silently
would be worse than saying how many there were.

## Not yet built

Silver and gold derivation, spread volatility, lead-lag, InfluxDB and Grafana
export. `system/04` sections 4 and 10 cover what can be ported from Makalu and
what has to be written; the lead-lag plot is the piece with no reference
implementation.

## Tests

```bash
python -m pytest tests -q
```

Fixtures write real parquet to a temporary directory. No network, no
credentials, no display.
