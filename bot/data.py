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
