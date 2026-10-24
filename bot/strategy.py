"""Daily trend strategy: hold SPY/QQQ while the daily close is above its 200-day average, else cash.

This is the rule with the strongest long-run evidence (see the strategy report). The AI can only
veto a buy; it can never start a trade or block a sell.
"""
import base64

import pandas as pd

from .analyst import Decision, parse_decision

VETO_PROMPT = """You are a cautious risk reviewer. A trend-following rule wants to BUY {symbol}: \
its last daily close {close:.2f} is above its {days}-day average {sma:.2f}. The chart shows daily \
bars with SMA 50/{days} and RSI 14 (latest RSI {rsi:.1f}).
Approve the buy unless the chart shows a clear reason to wait. You can only approve or wait.
Reply with JSON only: {{"action": "buy"|"hold", "confidence": 0-1, "reason": "one sentence"}}"""


def add_trend_indicators(df: pd.DataFrame, sma_days: int = 200) -> pd.DataFrame:
    """Daily bars -> SMA 50, SMA <sma_days>, RSI 14. Column names match the chart code."""
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(50).mean()
    df["sma_slow"] = df["close"].rolling(sma_days).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
    return df


def trend_signal(df: pd.DataFrame, holding: bool) -> Decision:
    """Decide on the last completed daily bar. Buy only when flat, sell only when holding."""
    last = df.iloc[-1]
    if pd.isna(last.sma_slow):
        return Decision("hold", 0.0, "not enough daily history for the trend average")
    above = last.close > last.sma_slow
    if above and not holding:
        return Decision("buy", 1.0, f"close {last.close:.2f} above trend average {last.sma_slow:.2f}")
    if not above and holding:
        return Decision("sell", 1.0, f"close {last.close:.2f} below trend average {last.sma_slow:.2f}")
    return Decision("hold", 1.0, "in trend, holding" if above else "below trend, staying in cash")


def ai_veto(cfg, df: pd.DataFrame, symbol: str, png: bytes) -> Decision:
    """Ask the model to approve a trend buy. Anything but a confident 'buy' counts as a veto."""
    import requests

    last = df.iloc[-1]
    image_url = "data:image/png;base64," + base64.b64encode(png).decode()
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {cfg.openrouter_key}"},
        json={
            "model": cfg.model,
            "max_tokens": 300,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": VETO_PROMPT.format(symbol=symbol, close=last.close,
                                                                days=cfg.trend_sma_days,
                                                                sma=last.sma_slow, rsi=last.rsi)},
                ],
            }],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return parse_decision(resp.json()["choices"][0]["message"]["content"] or "")


def ai_veto_from_config(cfg):
    """The OpenRouter veto as a callable(df, symbol), or None when there's no key or AI_VETO=false."""
    if not (cfg.openrouter_key and cfg.ai_veto):
        return None
    from .chart import render_chart

    def veto(df, symbol):
        png = render_chart(df.tail(250), symbol, labels=("SMA 50", f"SMA {cfg.trend_sma_days}"))
        return ai_veto(cfg, df, symbol, png)
    return veto


def decide(cfg, df: pd.DataFrame, symbol: str, holding: bool, veto="auto") -> Decision:
    """Trend rule, then (for buys only) the optional AI veto.

    veto: callable(df, symbol) -> Decision, None for no veto, or "auto" for the OpenRouter
    model when a key is set and AI_VETO isn't false.
    """
    signal = trend_signal(df, holding)
    if signal.action != "buy":
        return signal
    if veto == "auto":
        veto = ai_veto_from_config(cfg)
    if veto is None:
        return signal
    review = veto(df, symbol)
    if review.action == "buy" and review.confidence >= cfg.risk.min_confidence:
        return Decision("buy", signal.confidence, f"{signal.reason}; AI approved: {review.reason}")
    return Decision("hold", signal.confidence, f"AI vetoed the buy: {review.reason}")
