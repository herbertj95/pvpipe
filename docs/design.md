# Design notes

## Time

Everything is UTC and hourly. A timestamp labels the start of its hour, and
`clear_sky_frame` evaluates the sun at the middle of the hour. Using the start
instead would bias mornings low and evenings high by half an hour of sun
movement, which is visible in the plant model at sunrise and sunset.

Naive timestamps are refused at the boundary (`io.read_frame`,
`solar.require_utc`) instead of being assumed to be UTC. A silent assumption
there is the most common cause of a forecast that is shifted by an hour or two.

## What a forecast may know

One rule decides what counts as leakage: the forecast for day D is issued on
D-1 at `issue_hour` UTC, and only production measured before that instant may
be used.

- `features.lag_days` turns the rule into the lag of the newest same-hour
  measurement: one day back for target hours before the issue hour, two days
  back for the rest.
- `backtest.training_rows` applies the same rule to training data: a row can
  be trained on once its own target hour lies before the issue time.

`tests/test_features.py::test_no_feature_uses_production_from_after_the_issue_time`
and `tests/test_backtest.py::test_training_never_reaches_the_issue_time` pin
both halves down. Any new feature or model has to keep them green.

The weather forecast for day D is taken as available at issue time. The
package has no notion of forecast runs or of error growing with lead time.

## Models

A model is a `Forecaster` with `_fit` and `_predict`. The base class owns two
things so that no model can get them wrong: predictions are zero when the sun
is down, and they are clipped to what the inverter can deliver. Everything a
model needs arrives as columns of the feature table, which keeps the backtest
loop free of special cases.

Models work in kW. `ridge` divides by the DC capacity internally so its
penalty means the same for plants of different size.

## Quality checks and missing data

`quality.flag` returns one boolean column per check on the full hourly range,
so a gap in the file shows up as `missing` rows instead of going unnoticed.
`quality.clean` turns every flagged hour into NaN. From there on missing data
is simply NaN: features built from it are NaN, `ridge` fills gaps in lagged
inputs with training means, the persistence models return NaN where they have
nothing to repeat, and the error measures skip hours where forecast or actual
is missing. `hours` in the score table shows how many hours each model was
scored on, which is why it can differ between models on data with gaps.

## Scores

Errors are forecast minus actual. Normalisation is by DC capacity, which is
the convention most PV forecast studies use; normalising by mean production
gives much larger percentages and is not offered. Skill is computed on the
hours where the model, the reference and the measurement all exist, so a
model is not rewarded for having no forecast on difficult hours.

## Synthetic data

The generator exists to make tests and examples reproducible, not to be
realistic in every respect. Cloud cover is a logistic transform of an AR(1)
process, GHI follows from clear-sky GHI and the Kasten-Czeplak relation, and
production is the plant model plus multiplicative noise. Because production
comes from the same plant model that `physical` uses, the `physical` forecast
looks better here than it would on a real plant, where the plant description
is never exact.
