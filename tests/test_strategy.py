import numpy as np
import pandas as pd

from bot.analyst import Decision
from bot.backtest import run_trend, summary
from bot.config import Settings
from bot.strategy import add_trend_indicators, decide, trend_signal

CFG = Settings(openrouter_key="")


def daily_bars(closes, start="2024-01-02"):
    """Daily bars stamped at midnight New York time, as Alpaca returns them (UTC)."""
    idx = pd.bdate_range(start, periods=len(closes)).tz_localize("America/New_York").tz_convert("UTC")
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": close * 1.005,
                         "low": close * 0.995, "close": close, "volume": 1000}, index=idx)


def rise_then_crash(n_up=300, n_down=120):
    """Steady climb, then a long slide: the trend rule should be in, then out."""
    up = 400 * np.exp(np.linspace(0, 0.3, n_up))
    down = up[-1] * np.exp(np.linspace(0, -0.35, n_down + 1)[1:])
    return np.r_[up, down]


def frame(closes):
    return add_trend_indicators(daily_bars(closes))


def test_trend_signal_cases():
    up = frame(np.linspace(400, 500, 250))
    down = frame(np.linspace(500, 400, 250))
    assert trend_signal(up, holding=False).action == "buy"
    assert trend_signal(up, holding=True).action == "hold"
    assert trend_signal(down, holding=True).action == "sell"
    assert trend_signal(down, holding=False).action == "hold"
    assert trend_signal(frame(np.linspace(400, 500, 150)), holding=False).action == "hold"


def test_veto_blocks_buys_but_never_sells():
    up, down = frame(np.linspace(400, 500, 250)), frame(np.linspace(500, 400, 250))
    no = lambda df, s: Decision("hold", 0.9, "Fed day")
    assert decide(CFG, up, "SPY", holding=False, veto=no).action == "hold"
    assert decide(CFG, down, "SPY", holding=True, veto=no).action == "sell"
    weak_yes = lambda df, s: Decision("buy", 0.4, "maybe")
    assert decide(CFG, up, "SPY", holding=False, veto=weak_yes).action == "hold"
    yes = lambda df, s: Decision("buy", 0.9, "fine")
    assert decide(CFG, up, "SPY", holding=False, veto=yes).action == "buy"


def test_auto_veto_without_key_uses_rule_only():
    assert decide(CFG, frame(np.linspace(400, 500, 250)), "SPY", holding=False).action == "buy"


def test_run_trend_exits_the_slide():
    bars = {"SPY": daily_bars(rise_then_crash()), "QQQ": daily_bars(rise_then_crash() * 0.9)}
    r = run_trend(CFG, bars, days=300)
    sides = [t.side for t in r.trades if t.symbol == "SPY"]
    assert sides[0] == "buy" and "sell" in sides
    hold = r.benchmarks["Hold 10% SPY + 10% QQQ (same max exposure)"]
    assert r.equity.iloc[-1] > hold.iloc[-1]          # stepped aside for most of the slide
    for t in r.trades:
        if t.side == "buy":
            assert t.notional <= 100_000 * 0.10 + 1
    per_day = pd.Series([(t.time, t.symbol) for t in r.trades]).value_counts()
    assert per_day.max() == 1
    assert "Bot (trend rule)" in summary(r)


def test_run_trend_position_override():
    bars = {"SPY": daily_bars(rise_then_crash()), "QQQ": daily_bars(rise_then_crash())}
    r = run_trend(CFG, bars, days=300, max_position_pct=0.5)
    assert abs(r.trades[0].notional - 50_000) < 1
    assert "Hold 50% SPY" in "".join(r.benchmarks)


def test_vetoing_everything_never_trades_and_counts_calls():
    bars = {"SPY": daily_bars(rise_then_crash())}
    r = run_trend(CFG, bars, days=300, veto=lambda df, s: Decision("hold", 0.9, "no"))
    assert not r.trades and r.equity.iloc[-1] == 100_000
    assert r.ai_calls > 0 and r.label == "Bot (trend + AI veto)"
