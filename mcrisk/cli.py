"""Command line: python -m mcrisk --demo  |  python -m mcrisk --tickers NVDA AAPL TSLA"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from . import data
from .estimate import MarketParams
from .risk import RiskSummary, summarize
from .simulate import SimulationResult, normalize_weights, simulate_pair


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mcrisk",
        description="Monte Carlo portfolio risk: simulate one-month futures, "
        "measure value at risk and expected shortfall, and show what correlation adds.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--demo", action="store_true",
                     help="Use the video's inputs (NVDA, AAPL, TSLA) without downloading anything.")
    src.add_argument("--csv", type=Path, help="CSV of daily prices: date column, then one column per ticker.")
    p.add_argument("--tickers", nargs="+",
                   help="Tickers to download from Yahoo Finance (default: NVDA AAPL TSLA), "
                   "or columns to keep from --csv (default: all).")
    p.add_argument("--years", type=float, default=5, help="Years of price history (default: 5).")
    p.add_argument("--end", help="Last date of price history, YYYY-MM-DD (default: today).")
    p.add_argument("--weights", nargs="+", type=float, help="Portfolio weights (default: equal).")
    p.add_argument("--capital", type=float, default=10_000, help="Starting value in dollars (default: 10000).")
    p.add_argument("--days", type=int, default=21, help="Horizon in trading days (default: 21 = 1 month).")
    p.add_argument("--sims", type=int, default=10_000, help="Number of simulated futures (default: 10000).")
    p.add_argument("--confidence", type=float, default=0.95, help="VaR / ES confidence level (default: 0.95).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--out", type=Path, default=Path("output"), help="Folder for charts and summary.json.")
    p.add_argument("--no-plots", action="store_true", help="Skip the charts.")
    return p


def load_inputs(args: argparse.Namespace) -> tuple[MarketParams, pd.DataFrame | None, dict[str, str]]:
    if args.demo:
        return data.demo_params(), None, dict(data.DEMO_NAMES)
    if args.csv:
        prices = data.load_prices_csv(args.csv, args.tickers)
    else:
        tickers = args.tickers or list(data.DEMO_TICKERS)
        prices = data.download_prices(tickers, years=args.years, end=args.end)
    names = {t: n for t, n in data.DEMO_NAMES.items() if t in prices.columns}
    return MarketParams.from_prices(prices), prices, names


def format_report(
    params: MarketParams,
    weights,
    together: RiskSummary,
    alone: RiskSummary,
    source: str,
) -> str:
    lines = ["Monte Carlo Portfolio Risk", "=" * 26, ""]
    lines.append(f"Inputs: {source}")
    table = params.table()
    lines.append(f"  {'ticker':<8}{'weight':>8}{'growth/yr':>12}{'swing/yr':>11}")
    for (ticker, row), w in zip(table.iterrows(), weights):
        lines.append(f"  {ticker:<8}{w:>8.1%}{row.growth_a_year:>+12.1%}{row.swing_a_year:>10.0%}")
    lines.append("")
    lines.append("Correlations")
    lines.append(params.corr_table().round(2).to_string())
    lines.append("")
    lines.append(
        f"Simulated {together.n_sims:,} futures over {together.days} trading days, "
        f"start ${together.capital:,.0f}"
    )
    lines.append(f"  ended up:   {together.n_up:,}")
    lines.append(f"  ended down: {together.n_down:,}  ({together.prob_loss:.0%} chance of a losing month)")
    sign = "-" if together.median_pnl < 0 else "+"
    lines.append(f"  median outcome: {sign}${abs(together.median_pnl):,.0f}")
    lines.append("")
    lines.append(
        f"Value at risk ({together.confidence:.0%}): 1 month in {together.one_in}, "
        f"lose more than ${together.value_at_risk:,.0f}"
    )
    lines.append(f"Expected shortfall: those worst months lose ${together.expected_shortfall:,.0f} on average")
    lines.append("")
    ratio = together.value_at_risk / alone.value_at_risk
    lines.append("If each stock moved on its own (same growth and swing, zero correlation):")
    lines.append(
        f"  value at risk ${alone.value_at_risk:,.0f}, expected shortfall ${alone.expected_shortfall:,.0f}"
    )
    lines.append(f"  -> moving together makes the bad-month loss x{ratio:.2f} bigger")
    return "\n".join(lines)


def write_outputs(
    out: Path,
    params: MarketParams,
    prices: pd.DataFrame | None,
    names: dict[str, str],
    together: SimulationResult,
    alone: SimulationResult,
    s_together: RiskSummary,
    s_alone: RiskSummary,
    plots: bool,
) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "tickers": list(params.tickers),
        "weights": together.weights.tolist(),
        "growth_a_year": params.annual_growth.tolist(),
        "swing_a_year": params.annual_swing.tolist(),
        "correlation": params.corr.tolist(),
        "moving_together": s_together.to_dict(),
        "each_on_its_own": s_alone.to_dict(),
        "var_ratio": s_together.value_at_risk / s_alone.value_at_risk,
    }
    if prices is not None:
        payload["history"] = {
            "start": prices.index[0].strftime("%Y-%m-%d"),
            "end": prices.index[-1].strftime("%Y-%m-%d"),
            "days": int(len(prices)),
        }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    written = [summary_path]
    if plots:
        from . import plots as charts

        written += [
            charts.plot_inputs(params, out / "1_inputs.png", prices, names, charts.annual_label(prices)),
            charts.plot_fan(together, out / "2_futures.png"),
            charts.plot_distribution(together, s_together, out / "3_var_es.png"),
            charts.plot_correlation_effect(together, alone, s_together, s_alone, out / "4_correlation.png"),
        ]
    return written


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        params, prices, names = load_inputs(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        print("tip: run `python -m mcrisk --demo` to try it without downloading prices.")
        return 1

    weights = normalize_weights(args.weights, params.n_assets)
    together, alone = simulate_pair(
        params, weights, capital=args.capital, days=args.days, n_sims=args.sims, seed=args.seed
    )
    s_together = summarize(together, args.confidence)
    s_alone = summarize(alone, args.confidence)

    from .plots import annual_label

    source = annual_label(prices)
    if args.demo:
        source += f"; correlations assumed at {data.DEMO_CORRELATION:.2f} (not shown in the video)"
    print(format_report(params, weights, s_together, s_alone, source))

    written = write_outputs(args.out, params, prices, names, together, alone, s_together, s_alone,
                            plots=not args.no_plots)
    print("")
    print("Saved:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
