"""Build the data file for the interactive page (web/index.html).

Downloads five years of daily prices, then stores for each stock the mean and
covariance of daily log returns (all the simulation needs) and a weekly price
line for the sparklines. Run from the repository root:

    python web/build_data.py --end 2026-11-20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mcrisk.data import download_prices
from mcrisk.estimate import MarketParams

STOCKS = {
    "TSM": "TSMC",
    "NVDA": "Nvidia",
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet (Google)",
    "META": "Meta",
    "TEVA": "Teva",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "SPY": "S&P 500 (ETF)",
    "QQQ": "Nasdaq 100 (ETF)",
    "GLD": "Or (ETF)",
}


def build(end: str, years: float = 5) -> dict:
    tickers = list(STOCKS)
    prices = download_prices(tickers, years=years, end=end)
    params = MarketParams.from_prices(prices)
    rows = list(range(0, len(prices), 5))
    if rows[-1] != len(prices) - 1:
        rows.append(len(prices) - 1)
    weekly = prices.iloc[rows]
    weekly = weekly / weekly.iloc[0]
    return {
        "source": "Yahoo Finance, cours de clôture ajustés",
        "start": prices.index[0].strftime("%Y-%m-%d"),
        "end": prices.index[-1].strftime("%Y-%m-%d"),
        "days": int(len(prices)),
        "tickers": tickers,
        "names": [STOCKS[t] for t in tickers],
        "mu": [float(x) for x in params.mu],
        "cov": [[float(x) for x in row] for row in params.cov],
        "spark": [[round(float(x), 4) for x in weekly[t].to_numpy()] for t in tickers],
        "last_price": [round(float(prices[t].iloc[-1]), 2) for t in tickers],
    }


def inject(page: Path, payload: str) -> None:
    """Put the data inside the page's <script id="stock-data"> tag, so the page works offline."""
    if not page.exists():
        return
    html = page.read_text()
    start_tag = '<script id="stock-data" type="application/json">'
    start = html.index(start_tag) + len(start_tag)
    end = html.index("</script>", start)
    page.write_text(html[:start] + payload + html[end:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", required=True, help="Last day of history, YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("data.json"))
    args = parser.parse_args()
    data = build(args.end)
    payload = json.dumps(data, separators=(",", ":"))
    args.out.write_text(payload)
    inject(Path(__file__).with_name("index.html"), payload)
    growth = np.expm1(np.array(data["mu"]) * 252)
    swing = np.sqrt(np.diag(np.array(data["cov"]))) * np.sqrt(252)
    print(f"{data['start']} -> {data['end']}, {data['days']} days, {args.out.stat().st_size:,} bytes")
    for t, g, s in zip(data["tickers"], growth, swing):
        print(f"  {t:<6}{g:+8.1%}{s:8.0%}")


if __name__ == "__main__":
    main()
