"""Command-line entry point."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from pvpipe import quality
from pvpipe.backtest import WINDOWS, BacktestConfig, run_backtest
from pvpipe.config import ConfigError, load_site
from pvpipe.features import build_features
from pvpipe.io import DataError, read_dataset, write_dataset, write_frame
from pvpipe.metrics import score_table
from pvpipe.models import MODELS, NotEnoughData, make_model, make_models
from pvpipe.report import FORMATS, render
from pvpipe.synth import make_dataset

DEFAULT_MODELS = "persistence,smart_persistence,climatology,physical,ridge"


def model_list(text: str) -> list[str]:
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in MODELS]
    if unknown or not names:
        named = ", ".join(unknown) or repr(text)
        raise argparse.ArgumentTypeError(f"unknown model(s): {named}; choose from {', '.join(sorted(MODELS))}")
    return list(dict.fromkeys(names))


def day(text: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(text, tz="UTC").normalize()
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a date in YYYY-MM-DD form: {text!r}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pvpipe", description="Day-ahead PV power forecasting pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub, data_help):
        sub.add_argument("--site", required=True, help="site file (TOML)")
        sub.add_argument("--data", required=True, help=data_help)

    simulate = commands.add_parser("simulate", help="write a synthetic dataset")
    common(simulate, "directory to write the dataset to")
    simulate.add_argument("--start", type=day, default=day("2024-01-01"), help="first day (default 2024-01-01)")
    simulate.add_argument("--days", type=int, default=120, help="number of days (default 120)")
    simulate.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    simulate.add_argument("--dirty", action="store_true", help="add gaps and bad readings to the production")

    qc = commands.add_parser("qc", help="count suspicious production readings")
    common(qc, "dataset directory")

    backtest = commands.add_parser("backtest", help="score models on past days")
    common(backtest, "dataset directory")
    backtest.add_argument("--models", type=model_list, default=model_list(DEFAULT_MODELS), help=f"comma-separated (default {DEFAULT_MODELS})")
    backtest.add_argument("--train-days", type=int, default=30, help="history before the first forecast (default 30)")
    backtest.add_argument("--retrain-every", type=int, default=7, help="days between refits (default 7)")
    backtest.add_argument("--window", choices=WINDOWS, default="expanding", help="training window (default expanding)")
    backtest.add_argument("--issue-hour", type=int, default=12, help="UTC hour the forecast is issued the day before (default 12)")
    backtest.add_argument("--reference", default="persistence", help="model the skill score is relative to")
    backtest.add_argument("--all-hours", action="store_true", help="score night hours too")
    backtest.add_argument("--format", choices=FORMATS, default="text", help="table format (default text)")
    backtest.add_argument("--output", help="also write every forecast to this CSV file")

    forecast = commands.add_parser("forecast", help="forecast one day")
    common(forecast, "dataset directory")
    forecast.add_argument("--date", type=day, required=True, help="UTC day to forecast, YYYY-MM-DD")
    forecast.add_argument("--model", choices=sorted(MODELS), default="ridge", help="model (default ridge)")
    forecast.add_argument("--issue-hour", type=int, default=12, help="UTC hour the forecast is issued the day before (default 12)")
    return parser


def _prepared(args):
    site = load_site(args.site)
    dataset = read_dataset(args.data)
    flags = quality.flag(dataset.production, site)
    production = quality.clean(dataset.production, flags)
    return site, dataset, flags, production


def run_simulate(args) -> int:
    site = load_site(args.site)
    dataset = make_dataset(site, args.start, args.days, seed=args.seed, dirty=args.dirty)
    write_dataset(dataset, args.data)
    print(f"wrote {len(dataset.weather_forecast)} hours for {site.name} to {args.data}")
    return 0


def run_qc(args) -> int:
    _, _, flags, _ = _prepared(args)
    counts = quality.summary(flags)
    table = counts[list(quality.FLAG_NAMES)].rename("hours").to_frame()
    table.index.name = "flag"
    print(render(table))
    share = 100.0 * counts["flagged"] / counts["hours"] if counts["hours"] else 0.0
    print(f"{counts['flagged']} of {counts['hours']} hours flagged ({share:.1f} %)")
    return 0


def run_backtest_command(args) -> int:
    site, dataset, _, production = _prepared(args)
    if args.reference not in args.models:
        raise DataError(f"reference model {args.reference!r} must be one of --models")
    features = build_features(production, dataset.weather_forecast, site, issue_hour=args.issue_hour)
    config = BacktestConfig(train_days=args.train_days, retrain_every=args.retrain_every, window=args.window)
    forecasts = run_backtest(features, production, make_models(args.models, site), config)
    scores = score_table(forecasts, site.capacity_kw, reference=args.reference, daylight_only=not args.all_hours)
    print(render(scores, args.format))
    if args.output:
        write_frame(forecasts.drop(columns="daylight"), args.output)
    return 0


def run_forecast(args) -> int:
    site, dataset, _, production = _prepared(args)
    features = build_features(production, dataset.weather_forecast, site, issue_hour=args.issue_hour)
    rows = features[features.index.normalize() == args.date]
    if rows.empty:
        raise DataError(f"no weather forecast for {args.date.date()} in {args.data}")
    known = features.index < rows["issue_time"].iloc[0]
    model = make_model(args.model, site).fit(features[known], production.reindex(features.index)[known])
    prediction = model.predict(rows).round(1)
    print("timestamp,power_kw")
    for stamp, value in prediction.items():
        print(f"{stamp.strftime('%Y-%m-%dT%H:%M:%S+00:00')},{value:.1f}")
    return 0


HANDLERS = {
    "simulate": run_simulate,
    "qc": run_qc,
    "backtest": run_backtest_command,
    "forecast": run_forecast,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return HANDLERS[args.command](args)
    except (OSError, ConfigError, DataError, NotEnoughData, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
