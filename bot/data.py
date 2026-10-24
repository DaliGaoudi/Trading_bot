"""Fetch price bars from Alpaca."""
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def fetch_bars(cfg, symbol: str) -> pd.DataFrame:
    client = StockHistoricalDataClient(cfg.alpaca_key, cfg.alpaca_secret)
    # Free accounts get the IEX feed; enough for paper trading.
    start = datetime.now(timezone.utc) - timedelta(days=10)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(cfg.timeframe_minutes, TimeFrameUnit.Minute),
        start=start,
        feed="iex",
    )
    df = client.get_stock_bars(req).df
    if df.empty:
        return df
    df = df.xs(symbol, level="symbol") if "symbol" in df.index.names else df
    return df[["open", "high", "low", "close", "volume"]].tail(cfg.lookback_bars)


def fetch_daily_bars(cfg, symbol: str, days: int = 400) -> pd.DataFrame:
    """Completed daily bars only: today's still-forming bar is dropped, so a live decision
    uses the same data the backtest does (yesterday's close)."""
    client = StockHistoricalDataClient(cfg.alpaca_key, cfg.alpaca_secret)
    start = datetime.now(timezone.utc) - timedelta(days=days)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, feed="iex")
    df = client.get_stock_bars(req).df
    if df.empty:
        return df
    df = df.xs(symbol, level="symbol") if "symbol" in df.index.names else df
    today = pd.Timestamp.now(tz="America/New_York").date()
    df = df[df.index.tz_convert("America/New_York").date < today]
    return df[["open", "high", "low", "close", "volume"]]
