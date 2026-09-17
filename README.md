# Hase

The **Kamakura Quant Lab** analysis toolkit. Reads the bronze data Komachi
downloads, derives the silver and gold datasets the articles are written on,
and draws it.

Named for the district where the Great Buddha sits. Komachi is where you get
things; Hase is where you go to look at them.

```bash
pip install -e .
hase local       --market COINCHECK:BTC_SPOT
hase derive MarketPrice --market COINCHECK:BTC_SPOT --start 2025-07-01 --end 2025-07-28
hase derive VolSpread   --market COINCHECK:BTC_SPOT --start 2025-07-01 --end 2025-07-28
hase plot-trades --market COINCHECK:BTC_SPOT --date 2025-07-01 --output trades.png
hase plot-book   --market COINCHECK:BTC_SPOT --date 2025-07-01 --output book.png
```

## What it derives

Komachi downloads `bronze/`. Hase turns it into the datasets the articles are
written on, beside it under the same root, in the layout the warehouse uses.

| Dataset | Layer | What it is |
|---|---|---|
| `BookState` | silver | Best bid and ask per snapshot, with mid, spread and the depth resting at the touch |
| `MarketPrice` | silver | What a given trade size would actually pay, by walking the book |
| `VolSpread` | gold | Mid and spread on a one-second grid, and realized volatility measured on it |

**Size is in base currency.** `--execution-size 0.002` is 0.002 BTC, and it is
the default because it means the same trade on every venue and on every day.
`--execution-notional 1000000` walks to a cash amount instead, which is a
different question and a different partition: ¥1,000,000 bought 0.0647 BTC in
July 2025 and 0.0796 BTC in September 2026, so a notional is not comparable
with itself over time, and a USDT-quoted book cannot be walked to a yen target
at all.

Deriving costs about a quarter of a second per market-day, so a market-year is
a minute and a half. `MarketPrice` is around an eighth the size of the bronze
it reads.

**Nothing is dropped silently.** A size the visible book cannot fill is `NaN`
with `filled` false, rather than a partial fill priced as a whole one. A
snapshot whose walked spread exceeds 100 bps is flagged `degenerate` rather
than removed: those are collector artefacts from a resync, they are present in
historical bronze, and how many of them a day holds is a fact about the archive
worth being able to count.

## Agreement with the pipeline that produced the data

Hase reimplements these derivations rather than importing them from the
pipeline that built the archive, because a tool you install must not depend on
trading code. That is only safe if the two agree, so the test suite checks
`MarketPrice` against the pipeline's own stored output where that warehouse is
mounted, and skips where it is not. Measured on a full day of
`COINCHECK:BTC_SPOT` and `GMO:BTC_JPY`, the largest relative difference is
4e-16 — floating-point last-bit, from accumulating the levels in a different
order.

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
export. The lead-lag plot is the piece with no reference implementation.

## Tests

```bash
python -m pytest tests -q
```

Fixtures write real parquet to a temporary directory. No network, no
credentials, no display.
