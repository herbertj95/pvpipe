# pvpipe

A small, readable pipeline for **day-ahead PV power forecasting** at a single
plant: quality checks on metered production, a physical plant model, a few
forecast models, and rolling-origin backtesting with the usual error measures.

It depends on NumPy and pandas only, and ships with a synthetic data generator
so everything can be run without any measured data.

## Install

    python -m pip install -e ".[test]"

Python 3.11 or newer.

## Quick start

    python -m pvpipe simulate --site configs/demo_site.toml --data data/demo --days 120 --seed 7
    python -m pvpipe qc       --site configs/demo_site.toml --data data/demo
    python -m pvpipe backtest --site configs/demo_site.toml --data data/demo
    python -m pvpipe forecast --site configs/demo_site.toml --data data/demo --date 2024-04-20

The backtest of that dataset prints:

    model              hours  mae_kw  rmse_kw  mbe_kw  nmae_pct  nrmse_pct  skill
    persistence         1052   39.69    63.09   -3.24      7.94      12.62   0.00
    smart_persistence   1052   39.91    63.42   -1.56      7.98      12.68  -0.01
    climatology         1052   45.80    59.85  -13.10      9.16      11.97   0.05
    physical            1052   21.59    36.23   -1.62      4.32       7.25   0.43
    ridge               1052   21.24    33.23   -1.60      4.25       6.65   0.47

## Commands

| command | what it does |
|---|---|
| `simulate` | writes a synthetic dataset; `--dirty` adds a gap and bad readings to the production |
| `qc` | counts suspicious production readings per check |
| `backtest` | forecasts every day after `--train-days` and scores each model |
| `forecast` | prints the forecast for one UTC day as `timestamp,power_kw` |

`backtest` options: `--models a,b,c`, `--train-days`, `--retrain-every`,
`--window expanding|rolling`, `--issue-hour`, `--reference`, `--all-hours`,
`--format text|markdown|csv`, `--output forecasts.csv`.

Exit code 0 on success, 1 on a data or configuration error, 2 on a usage error.

## Data

A dataset is a directory with

| file | columns |
|---|---|
| `production.csv` | `timestamp, power_kw` |
| `weather_forecast.csv` | `timestamp, ghi, temp_air, cloud_cover` |
| `weather_observed.csv` (optional, not used by the models) | `timestamp, ghi, temp_air, cloud_cover` |

Conventions, which every module relies on:

- Data is **hourly**. A timestamp labels the **start** of the hour it describes.
- Timestamps are ISO 8601 **with a UTC offset**; everything is converted to UTC.
  A timestamp without an offset is an error.
- `power_kw` is mean AC power over the hour in kW, `ghi` is W/m2, `temp_air` is
  degrees C, `cloud_cover` is a fraction from 0 to 1.
- Angles are degrees. Azimuth is clockwise from north (180 = south).

## The site file

```toml
[site]
name = "demo-rooftop"
latitude = 48.2
longitude = 11.6
capacity_kw = 500.0     # DC nameplate
ac_limit_kw = 400.0     # inverter rating; optional
tilt = 25.0
azimuth = 180.0
temp_coeff = -0.0038    # per degree C
system_losses = 0.14
```

## How a forecast is made

The forecast for every hour of UTC day D is issued on day D-1 at 12:00 UTC
(`--issue-hour`). A model may use production measured before that moment and
the weather forecast for day D. `features.py` builds one row per target hour
under that rule; `backtest.py` trains only on rows whose measurement was known
at issue time.

| model | idea |
|---|---|
| `persistence` | the newest same-hour measurement known at issue time |
| `smart_persistence` | persistence of the clear-sky index, rescaled by clear-sky power |
| `climatology` | mean measured power per month and hour |
| `physical` | plant model driven by the weather forecast |
| `ridge` | ridge regression on the physical forecast, lags and weather |

Plant model: Haurwitz clear-sky GHI, Erbs split into beam and diffuse,
isotropic-sky transposition, NOCT cell temperature, linear power with a
temperature coefficient, flat system losses, inverter clipping.

Scores: MAE, RMSE and mean bias in kW (forecast minus actual), the first two
also as a percentage of DC capacity, and skill = 1 - RMSE / RMSE of the
reference model. Daylight hours only unless `--all-hours` is given.

## Quality checks

`missing`, `negative`, `above_limit` (more than 5 % above the inverter rating),
`night_production` (more than 1 % of capacity with the sun 3 degrees below the
horizon) and `stuck` (four or more equal non-zero readings in a row, except
near the inverter rating where clipping makes that normal). Flagged hours are
treated as missing by `backtest` and `forecast`.

## Limitations

Things the package does not do, stated so nobody assumes otherwise:

- Hourly resolution only; other steps are neither resampled nor rejected.
- One plant, fixed tilt. No trackers, no bifacial gain, no horizon shading, no snow or soiling model.
- Day-ahead only. No intraday updates and no multi-day horizons beyond `lead_days` in `build_features`.
- Point forecasts only. No quantiles or prediction intervals.
- Scores are totals over the backtest. There is no breakdown by hour of day, month or weather type.
- `ridge` has a fixed penalty; nothing tunes it.
- The isotropic sky model underestimates diffuse light on a tilted plane under clear skies.
  One visible effect: at low sun behind the array the clear-sky power is not a strict upper bound.
- Quality checks do not look for time shifts (for example local time stored as UTC) or slow degradation.
- The synthetic weather has no fog, snow cover or fast cloud transients, and its forecast error does not grow with lead time.

## Tests

    python -m pytest -q

## Layout

    pvpipe/solar.py       sun position
    pvpipe/clearsky.py    clear-sky GHI, beam/diffuse split
    pvpipe/plant.py       plane-of-array irradiance to AC power
    pvpipe/synth.py       synthetic weather and production
    pvpipe/io.py          CSV reading and writing
    pvpipe/quality.py     checks on metered production
    pvpipe/features.py    model inputs under the issue-time rule
    pvpipe/models/        forecast models and their registry
    pvpipe/backtest.py    rolling-origin evaluation
    pvpipe/metrics.py     error measures
    pvpipe/report.py      table rendering
    pvpipe/cli.py         command line

See `docs/design.md` for the reasoning behind the main choices.

## Licence

MIT, see `LICENSE`.
