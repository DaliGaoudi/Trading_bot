"""Replay the bot over past 15-minute bars with the same risk rules, and compare with buy-and-hold.

    python -m bot.backtest                      # last 365 days, rule-based analyst
    python -m bot.backtest --days 30            # shorter window
    python -m bot.backtest --analyst ai         # prints the AI call count and cost estimate, then asks

Bars are cached in data/ so a second run doesn't download them again.
Past results don't predict future returns.
"""
import argparse
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import risk
from .analyst import rule_based
from .config import settings
from .indicators import add_indicators
from .risk import AccountState

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NY = "America/New_York"
# Rough cost of one Sonnet 5 chart call: ~800 image + ~150 text tokens in at $2/M,
# 100-300 tokens out at $10/M.
AI_COST_PER_CALL = (0.002, 0.005)


# ---------- data ----------

def fetch_history(cfg, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """All 15-minute IEX bars between start and end (same feed the live bot uses)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(cfg.alpaca_key, cfg.alpaca_secret)
    req = StockBarsRequest(symbol_or_symbols=symbol,
                           timeframe=TimeFrame(cfg.timeframe_minutes, TimeFrameUnit.Minute),
                           start=start, end=end, feed="iex")
    df = client.get_stock_bars(req).df
    if "symbol" in df.index.names:
        df = df.xs(symbol, level="symbol")
    return df[["open", "high", "low", "close", "volume"]]


def load_bars(cfg, symbol: str, days: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=20)
    start = end - timedelta(days=days + 10)  # +10 days so indicators are warm on day one
    path = DATA_DIR / f"{symbol}_{cfg.timeframe_minutes}m_{days}d_{end:%Y%m%d}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = fetch_history(cfg, symbol, start, end)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(path)
    return df


def market_hours(index: pd.DatetimeIndex) -> pd.Series:
    """True for bars that start between 09:30 and 15:45 New York time on a weekday."""
    ny = index.tz_convert(NY) if index.tz is not None else index.tz_localize("UTC").tz_convert(NY)
    minutes = ny.hour * 60 + ny.minute
    return pd.Series((minutes >= 570) & (minutes < 960) & (ny.day_of_week < 5), index=index)


# ---------- simulation ----------

@dataclass
class Trade:
    time: pd.Timestamp
    symbol: str
    side: str
    price: float
    notional: float
    reason: str


@dataclass
class Result:
    start: pd.Timestamp
    end: pd.Timestamp
    start_equity: float
    equity: pd.Series                 # bot equity at every market-hours bar close
    trades: list = field(default_factory=list)
    benchmarks: dict = field(default_factory=dict)   # name -> equity Series
    loss_stop_days: int = 0
    trade_cap_days: int = 0
    ai_calls: int = 0


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def round_trips(trades):
    """Pair each sell with the buys before it; return P/L per closed position."""
    open_cost, open_shares, out = {}, {}, []
    for t in trades:
        if t.side == "buy":
            open_cost[t.symbol] = open_cost.get(t.symbol, 0) + t.notional
            open_shares[t.symbol] = open_shares.get(t.symbol, 0) + t.notional / t.price
        elif t.symbol in open_cost:
            out.append(open_shares.pop(t.symbol) * t.price - open_cost.pop(t.symbol))
    return out


def run(cfg, bars: dict, days: int, decide=None, start_equity=100_000.0, slippage_bps=1.0) -> Result:
    """Walk forward bar by bar. A decision made on a bar's close fills at the next bar's open,
    which is roughly what happens when the live bot sends a market order."""
    decide = decide or (lambda df, symbol: rule_based(df))
    is_ai = getattr(decide, "is_ai", False)
    limits = cfg.risk
    frames = {s: add_indicators(df) for s, df in bars.items()}
    open_mask = {s: market_hours(df.index) for s, df in frames.items()}

    first = max(df.index[-1] for df in frames.values()) - pd.Timedelta(days=days)
    clock = sorted(set().union(*[df.index[open_mask[s] & (df.index >= first)] for s, df in frames.items()]))
    pos = {s: {i: n for n, i in enumerate(df.index)} for s, df in frames.items()}

    cash, shares = start_equity, {s: 0.0 for s in frames}
    last_price = {s: None for s in frames}
    slip = slippage_bps / 10_000
    trades, curve = [], {}
    day, last_equity, trades_today = None, start_equity, 0
    hit_stop, hit_cap = set(), set()
    pending = {}                      # symbol -> (action, notional, reason), filled at the next open
    ai_calls = 0

    def equity_now():
        return cash + sum(shares[s] * (last_price[s] or 0) for s in frames)

    for ts in clock:
        today = ts.tz_convert(NY).date()
        if today != day:
            if day is not None:
                last_equity = curve[prev_ts]    # equity at the previous close
            day, trades_today = today, 0
        for symbol, df in frames.items():
            i = pos[symbol].get(ts)
            if i is None:
                continue
            if symbol in pending:
                action, notional, reason = pending.pop(symbol)
                if action == "buy":
                    price = df.open.iloc[i] * (1 + slip)
                    notional = min(notional, cash)
                    shares[symbol] += notional / price
                    cash -= notional
                else:
                    price = df.open.iloc[i] * (1 - slip)
                    notional = shares[symbol] * price
                    cash += notional
                    shares[symbol] = 0.0
                trades.append(Trade(ts, symbol, action, price, notional, reason))
            last_price[symbol] = df.close.iloc[i]
            # The live bot needs 35 bars and only trades while the market is open,
            # so skip the last bar of the day (its order would fill tomorrow).
            if i < 35 or i + 1 >= len(df) or not open_mask[symbol].iloc[i + 1] \
                    or df.index[i + 1].tz_convert(NY).date() != today:
                continue
            window = df.iloc[max(0, i + 1 - cfg.lookback_bars):i + 1]
            decision = decide(window, symbol)
            ai_calls += is_ai

            acct = AccountState(equity_now(), last_equity, shares[symbol] * last_price[symbol], trades_today)
            ok, notional, why = risk.check(limits, symbol, decision.action, decision.confidence, acct)
            if not ok:
                if why.startswith("daily loss"):
                    hit_stop.add(today)
                elif why.startswith("daily trade"):
                    hit_cap.add(today)
                continue
            trades_today += 1
            pending[symbol] = (decision.action, notional, decision.reason)
        curve[ts] = equity_now()
        prev_ts = ts

    equity = pd.Series(curve)
    result = Result(equity.index[0], equity.index[-1], start_equity, equity, trades,
                    loss_stop_days=len(hit_stop), trade_cap_days=len(hit_cap), ai_calls=ai_calls)
    result.benchmarks = benchmarks(frames, open_mask, equity.index, start_equity)
    return result


def benchmarks(frames, open_mask, index, start_equity):
    """Buy-and-hold curves over the same period, valued at the same bar closes."""
    def hold(weights):
        cash = start_equity * (1 - sum(weights.values()))
        out = pd.Series(cash, index=index)
        for s, w in weights.items():
            close = frames[s].close[open_mask[s]].reindex(index).ffill().bfill()
            entry = frames[s].open[open_mask[s]].reindex(index).bfill().iloc[0]
            out = out + start_equity * w * close / entry
        return out

    out = {}
    if "SPY" in frames:
        out["Buy and hold SPY (100%)"] = hold({"SPY": 1.0})
    if {"SPY", "QQQ"} <= frames.keys():
        out["Hold 10% SPY + 10% QQQ (same max exposure as the bot)"] = hold({"SPY": 0.1, "QQQ": 0.1})
    return out


# ---------- report ----------

def summary(result: Result) -> str:
    def line(name, eq):
        pnl = eq.iloc[-1] - result.start_equity
        return (f"{name:<55} {eq.iloc[-1]:>12,.2f}  {pnl:>+11,.2f}  {pnl / result.start_equity:>+7.2%}"
                f"  {max_drawdown(eq):>7.2%}")

    trips = round_trips(result.trades)
    wins = sum(1 for p in trips if p > 0)
    buys = sum(1 for t in result.trades if t.side == "buy")
    sells = len(result.trades) - buys
    rows = [
        f"Backtest {result.start.tz_convert(NY):%Y-%m-%d} to {result.end.tz_convert(NY):%Y-%m-%d}, "
        f"start equity ${result.start_equity:,.0f}",
        "",
        f"{'':<55} {'end equity':>12}  {'P/L $':>11}  {'P/L %':>7}  {'max DD':>7}",
        line("Bot (rules)" if not result.ai_calls else "Bot (AI)", result.equity),
        *[line(name, eq) for name, eq in result.benchmarks.items()],
        "",
        f"Orders: {len(result.trades)} ({buys} buys, {sells} sells); closed positions: {len(trips)}, "
        f"{wins} winners" + (f" ({wins / len(trips):.0%})" if trips else ""),
        f"Days the 2% loss stop fired: {result.loss_stop_days}; days the 4-order cap was hit: "
        f"{result.trade_cap_days}",
        "",
        "Past results don't predict future returns. Fills are simulated at the next bar's open.",
    ]
    return "\n".join(rows)


def save(result: Result, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    curves = pd.DataFrame({"bot": result.equity, **result.benchmarks})
    curves.to_csv(out_dir / "equity.csv")
    pd.DataFrame([t.__dict__ for t in result.trades]).to_csv(out_dir / "trades.csv", index=False)
    (out_dir / "summary.txt").write_text(summary(result), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = curves.plot(figsize=(10, 5), linewidth=1)
        ax.set_ylabel("equity ($)")
        ax.set_title("Bot vs buy-and-hold")
        plt.tight_layout()
        plt.savefig(out_dir / "equity.png", dpi=100)
        plt.close()
    except Exception:
        pass


def ai_decider(cfg, bars, days, workers=8):
    """Ask the AI about every decision point up front, several charts at a time.

    The AI only sees the chart, never the account, so its answers don't depend on earlier
    trades and can be fetched in parallel. Charts are drawn first (matplotlib isn't
    thread-safe), then sent with a progress line every 25 calls."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .analyst import Decision, ai_decision
    from .chart import render_chart

    charts = {}

    def collect(window, symbol):
        charts[(symbol, window.index[-1])] = (window, render_chart(window, symbol))
        return Decision("hold", 0.0, "")
    run(cfg, bars, days, collect)

    def ask(key):
        window, png = charts[key]
        for attempt in range(2):
            try:
                return ai_decision(cfg, window, key[0], png)
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (401, 402, 403):     # bad key or no credit: retrying won't help
                    raise
                err = e
                time.sleep(5)
        return Decision("hold", 0.0, f"AI call failed: {type(err).__name__}")

    answers, start = {}, time.time()
    pool = ThreadPoolExecutor(workers)
    futures = {pool.submit(ask, k): k for k in charts}
    try:
        for n, f in enumerate(as_completed(futures), 1):
            answers[futures[f]] = f.result()
            if n % 25 == 0 or n == len(charts):
                left = (time.time() - start) / n * (len(charts) - n)
                print(f"  {n:,}/{len(charts):,} AI calls done, about {left / 60:.0f} min left", flush=True)
    except BaseException as e:
        pool.shutdown(cancel_futures=True)
        raise SystemExit(f"Stopped after {len(answers):,} AI calls: {e}") from e
    pool.shutdown()
    failed = sum(1 for d in answers.values() if d.reason.startswith("AI call failed"))
    if failed:
        print(f"  {failed} calls failed twice and count as hold.")

    def decide(window, symbol):
        return answers[(symbol, window.index[-1])]
    decide.is_ai = True
    return decide


def count_decisions(bars, days):
    n = 0
    for df in bars.values():
        mask = market_hours(df.index)
        first = df.index[-1] - pd.Timedelta(days=days)
        ny_days = df.index.tz_convert(NY).date
        nxt_same_day = pd.Series(ny_days, index=df.index).shift(-1) == pd.Series(ny_days, index=df.index)
        warm = pd.Series(range(len(df)), index=df.index) >= 35
        n += int((mask & mask.shift(-1, fill_value=False) & nxt_same_day & warm & (df.index >= first)).sum())
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--equity", type=float, default=100_000)
    p.add_argument("--slippage-bps", type=float, default=1.0, help="cost per fill, in 0.01%% steps")
    p.add_argument("--analyst", choices=["rules", "ai"], default="rules")
    p.add_argument("--yes", action="store_true", help="skip the cost question for --analyst ai")
    p.add_argument("--workers", type=int, default=8, help="AI calls sent at the same time")
    p.add_argument("--out", default="backtest_results")
    args = p.parse_args()
    if not (settings.alpaca_key and settings.alpaca_secret):
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file first.")

    bars = {s: load_bars(settings, s, args.days) for s in settings.risk.symbols}
    decide = None
    if args.analyst == "ai":
        if not settings.openrouter_key:
            raise SystemExit("Set OPENROUTER_API_KEY in .env to backtest the AI analyst.")
        n = count_decisions(bars, args.days)
        lo, hi = (n * c for c in AI_COST_PER_CALL)
        print(f"{n:,} chart calls to {settings.model}, roughly ${lo:,.0f} to ${hi:,.0f}.")
        if not args.yes and input("Go ahead? [y/N] ").strip().lower() != "y":
            raise SystemExit("Cancelled, nothing was sent.")
        decide = ai_decider(settings, bars, args.days, args.workers)

    result = run(settings, bars, args.days, decide, args.equity, args.slippage_bps)
    save(result, Path(args.out))
    print(summary(result))
    print(f"\nEquity curve, trades and chart saved in {args.out}/")


if __name__ == "__main__":
    main()
