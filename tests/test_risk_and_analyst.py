import numpy as np
import pandas as pd

from bot.analyst import parse_decision, rule_based
from bot.chart import render_chart
from bot.config import RiskLimits
from bot.indicators import add_indicators
from bot.risk import AccountState, check

L = RiskLimits()


def acct(**kw):
    base = dict(equity=100_000, last_equity=100_000, position_value=0, trades_today=0)
    base.update(kw)
    return AccountState(**base)


def test_buy_capped_at_max_position():
    ok, notional, _ = check(L, "SPY", "buy", 0.9, acct())
    assert ok and notional == 10_000


def test_buy_tops_up_only_to_limit():
    ok, notional, _ = check(L, "SPY", "buy", 0.9, acct(position_value=7_000))
    assert ok and notional == 3_000


def test_blocks_unlisted_symbol():
    assert not check(L, "TSLA", "buy", 0.9, acct())[0]


def test_blocks_low_confidence():
    assert not check(L, "SPY", "buy", 0.3, acct())[0]


def test_blocks_after_daily_loss():
    assert not check(L, "SPY", "buy", 0.9, acct(equity=97_000))[0]


def test_blocks_after_trade_limit():
    assert not check(L, "SPY", "buy", 0.9, acct(trades_today=4))[0]


def test_sell_needs_position():
    assert not check(L, "SPY", "sell", 0.9, acct())[0]
    assert check(L, "SPY", "sell", 0.9, acct(position_value=5_000))[0]


def test_parse_decision_handles_junk():
    assert parse_decision("no json here").action == "hold"
    d = parse_decision('Sure: {"action": "BUY", "confidence": 1.4, "reason": "trend"}')
    assert d.action == "buy" and d.confidence == 1.0


def test_pipeline_on_fake_bars():
    rng = np.random.default_rng(0)
    close = 500 + rng.normal(0, 1, 120).cumsum()
    df = pd.DataFrame({"open": close - 0.2, "high": close + 1, "low": close - 1,
                       "close": close, "volume": 1000})
    df = add_indicators(df)
    assert render_chart(df, "SPY")[:4] == b"\x89PNG"
    assert rule_based(df).action in ("buy", "sell", "hold")


def test_openrouter_request(monkeypatch):
    import requests
    from types import SimpleNamespace
    from bot.analyst import ai_decision

    sent = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"action": "sell", "confidence": 0.8, "reason": "x"}'}}]}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, body=json)
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    cfg = SimpleNamespace(openrouter_key="k", model="anthropic/claude-sonnet-5", timeframe_minutes=15)
    df = pd.DataFrame({"close": [500.0], "rsi": [55.0]})
    d = ai_decision(cfg, df, "SPY", b"png")
    assert d.action == "sell" and d.confidence == 0.8
    assert sent["url"].startswith("https://openrouter.ai/")
    assert sent["headers"]["Authorization"] == "Bearer k"
    assert sent["body"]["model"] == "anthropic/claude-sonnet-5"
    assert sent["body"]["messages"][0]["content"][0]["type"] == "image_url"
