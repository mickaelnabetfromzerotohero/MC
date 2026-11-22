"""Build the data for the interactive page (web/index.html), in euros.

Every line is converted to euros with the day's exchange rate. Risk is estimated
from weekly returns (Paris and New York close at different hours, which blurs
daily co-movements) over a common window that starts when Nebius returned to
the market on 21 Oct 2024. Run from the repository root:

    python web/build_data.py --end 2026-11-20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# (Yahoo symbol, label shown on the page, quote currency)
STOCKS = [
    ("CW8.PA", "MSCI World (PEA)", "EUR"),
    ("PUST.PA", "Nasdaq 100 (PEA)", "EUR"),
    ("PAEEM.PA", "Émergents ESG (PEA)", "EUR"),
    ("NBIS", "Nebius", "USD"),
    ("TSM", "TSMC", "USD"),
    ("GLD", "Or", "USD"),
    ("SIVE.ST", "Sivers Semiconductors", "SEK"),
    ("NVDA", "Nvidia", "USD"),
    ("AAPL", "Apple", "USD"),
    ("TSLA", "Tesla", "USD"),
    ("MSFT", "Microsoft", "USD"),
    ("AMZN", "Amazon", "USD"),
    ("GOOGL", "Alphabet (Google)", "USD"),
    ("META", "Meta", "USD"),
    ("TEVA", "Teva", "USD"),
    ("KO", "Coca-Cola", "USD"),
    ("PEP", "PepsiCo", "USD"),
    ("SPY", "S&P 500 (ETF)", "USD"),
]
FX = {"USD": "EURUSD=X", "SEK": "EURSEK=X"}
START = "2024-10-21"
# Weekdays back for "last week / month / quarter" (the index holds every weekday, FX trades daily).
LOOKBACK = (5, 21, 63)


def euro_prices(end: str) -> pd.DataFrame:
    import yfinance as yf

    symbols = [s for s, _, _ in STOCKS]
    raw = yf.download(
        symbols + list(FX.values()),
        start=START,
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )["Close"].ffill()
    eur = pd.DataFrame(index=raw.index)
    for symbol, _, ccy in STOCKS:
        eur[symbol] = raw[symbol] if ccy == "EUR" else raw[symbol] / raw[FX[ccy]]
    eur = eur.dropna(how="any")
    if eur.empty:
        raise RuntimeError("No overlapping prices downloaded.")
    return eur


def build(end: str) -> dict:
    eur = euro_prices(end)
    weekly = eur.resample("W-FRI").last()
    returns = np.log(weekly).diff().dropna(how="any")
    symbols = [s for s, _, _ in STOCKS]
    last = eur.index[-1]
    moves, since = {}, {}
    for days in LOOKBACK:
        before = eur.iloc[-1 - days]
        moves[days] = [round(float(eur[s].iloc[-1] / before[s] - 1), 5) for s in symbols]
        since[days] = before.name.strftime("%Y-%m-%d")
    spark = weekly / weekly.iloc[0]
    return {
        "source": "Yahoo Finance, cours de clôture ajustés, convertis en euros",
        "currency": "EUR",
        "start": eur.index[0].strftime("%Y-%m-%d"),
        "end": last.strftime("%Y-%m-%d"),
        "weeks": int(len(returns)),
        "tickers": symbols,
        "names": [n for _, n, _ in STOCKS],
        "quote": [c for _, _, c in STOCKS],
        "mu": [float(x) for x in returns.mean().to_numpy() / 5],
        "cov": [[float(x) for x in row] for row in returns.cov().to_numpy() / 5],
        "spark": [[round(float(x), 4) for x in spark[s].to_numpy()] for s in symbols],
        "last_price": [round(float(eur[s].iloc[-1]), 2) for s in symbols],
        "moves": moves,
        "since": since,
    }


def inject(page: Path, payload: str) -> None:
    """Put the data inside the page's <script id="stock-data"> tag, so the page works offline."""
    if not page.exists():
        return
    html = page.read_text()
    start_tag = '<script id="stock-data" type="application/json">'
    start = html.index(start_tag) + len(start_tag)
    stop = html.index("</script>", start)
    page.write_text(html[:start] + payload + html[stop:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", required=True, help="Last day of history, YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("data.json"))
    args = parser.parse_args()
    data = build(args.end)
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    args.out.write_text(payload)
    inject(Path(__file__).with_name("index.html"), payload)
    vol = np.sqrt(np.diag(np.array(data["cov"])) * 252)
    print(f"{data['start']} -> {data['end']}, {data['weeks']} weeks, {args.out.stat().st_size:,} bytes")
    for t, v, m in zip(data["tickers"], vol, data["moves"][21]):
        print(f"  {t:<9}{v:7.0%}{m:+8.1%}")


if __name__ == "__main__":
    main()
