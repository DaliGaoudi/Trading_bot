import numpy as np
import pandas as pd

from bot.analyst import Decision
from bot.backtest import count_decisions, market_hours, max_drawdown, run, summary
from bot.config import Settings


def fake_bars(days=30, seed=0, drift=0.0):
    """15-minute bars for weekdays, 09:30-15:45 New York time, as Alpaca returns them (UTC)."""
    idx = []
    for d in pd.bdate_range("2026-03-02", periods=days):
        start = pd.Timestamp(d.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        idx += [start + pd.Timedelta(minutes=15 * k) for k in range(26)]
    idx = pd.DatetimeIndex(idx).tz_convert("UTC")
    rng = np.random.default_rng(seed)
    close = 500 * np.exp((drift + rng.normal(0, 0.002, len(idx))).cumsum())
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": close * 1.001,
                         "low": close * 0.999, "close": close, "volume": 1000}, index=idx)


CFG = Settings()


def test_market_hours_mask():
    df = fake_bars(1)
    assert market_hours(df.index).all()
    assert not market_hours(df.index + pd.Timedelta(hours=7)).any()


def test_max_drawdown():
    assert max_drawdown(pd.Series([100, 120, 90, 130])) == -0.25


def test_rules_respect_position_and_order_limits():
    bars = {"SPY": fake_bars(seed=1), "QQQ": fake_bars(seed=2)}
    r = run(CFG, bars, days=25)
    assert r.trades, "expected some crossovers in 30 days of noise"
    per_day = pd.Series([t.time.tz_convert("America/New_York").date() for t in r.trades]).value_counts()
    assert per_day.max() <= 4
    for t in r.trades:
        if t.side == "buy":
            assert t.notional <= r.equity.max() * 0.50 + 1
    assert "Buy and hold SPY (100%)" in r.benchmarks
    assert "Past results" in summary(r)


def test_always_buy_never_exceeds_the_cap():
    bars = {"SPY": fake_bars(seed=3)}
    r = run(CFG, bars, days=25, decide=lambda df, s: Decision("buy", 0.9, "test"))
    # First buy fills the 50%; later ones only top up after the price dips (same as live).
    assert abs(r.trades[0].notional - 50_000) < 1
    assert sum(t.notional for t in r.trades[1:]) < 1_000


def test_low_confidence_never_trades():
    bars = {"SPY": fake_bars(seed=4)}
    r = run(CFG, bars, days=25, decide=lambda df, s: Decision("buy", 0.5, "weak"))
    assert not r.trades and r.equity.iloc[-1] == 100_000


def test_daily_loss_stop_blocks_new_orders():
    df = fake_bars(seed=5)
    crash_day = df.index.tz_convert("America/New_York").normalize().unique()[20]
    after = df.index.tz_convert("America/New_York") >= crash_day + pd.Timedelta(hours=10)
    df.loc[after, ["open", "high", "low", "close"]] *= 0.5   # 50% gap down, so a 50% position loses 25%

    r = run(CFG, {"SPY": df}, days=25, decide=lambda w, s: Decision("buy", 0.9, "dip"))
    crash = [t for t in r.trades if t.time.tz_convert("America/New_York").date() == crash_day.date()]
    assert not crash                     # the position halved, but no top-up on the loss day
    assert r.loss_stop_days >= 1


def test_count_decisions_skips_last_bar_of_day():
    bars = {"SPY": fake_bars(5)}
    # 25 per day, minus the 34 of those that fall in the 35 warm-up bars
    assert count_decisions(bars, days=100) == 5 * 25 - 34


def test_ai_decider_asks_once_per_decision_point(monkeypatch):
    import bot.analyst
    from bot.backtest import ai_decider

    calls = []

    def fake_ai(cfg, df, symbol, png):
        calls.append((symbol, df.index[-1]))
        return Decision("buy", 0.9, "ai")
    monkeypatch.setattr(bot.analyst, "ai_decision", fake_ai)
    bars = {"SPY": fake_bars(3), "QQQ": fake_bars(3, seed=1)}
    decide = ai_decider(CFG, bars, days=100, workers=4)
    assert len(calls) == len(set(calls)) == count_decisions(bars, days=100)
    r = run(CFG, bars, days=100, decide=decide)
    assert r.ai_calls == len(calls) and r.trades


def test_ai_decider_stops_on_payment_error(monkeypatch):
    import pytest
    import requests
    import bot.analyst
    from bot.backtest import ai_decider

    def no_credit(cfg, df, symbol, png):
        resp = requests.Response()
        resp.status_code = 402
        raise requests.HTTPError("402 Payment Required", response=resp)
    monkeypatch.setattr(bot.analyst, "ai_decision", no_credit)
    with pytest.raises(SystemExit, match="402"):
        ai_decider(CFG, {"SPY": fake_bars(3)}, days=100, workers=2)
