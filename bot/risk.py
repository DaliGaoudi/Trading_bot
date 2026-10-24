"""Risk checks. Every order must pass here first."""
from dataclasses import dataclass


@dataclass
class AccountState:
    equity: float
    last_equity: float          # equity at previous close
    position_value: float       # current value held in this symbol
    trades_today: int


def check(limits, symbol: str, action: str, confidence: float, acct: AccountState):
    """Return (allowed, notional_usd, reason)."""
    if symbol not in limits.symbols:
        return False, 0.0, f"{symbol} is not on the allowed list"
    if action == "hold":
        return False, 0.0, "hold"
    if confidence < limits.min_confidence:
        return False, 0.0, f"confidence {confidence:.2f} below {limits.min_confidence}"
    if acct.trades_today >= limits.max_trades_per_day:
        return False, 0.0, "daily trade limit reached"
    if action == "sell":
        # Closing a position only lowers risk, so the daily loss stop never blocks it.
        if acct.position_value <= 0:
            return False, 0.0, "nothing to sell"
        return True, acct.position_value, "close position"
    if acct.last_equity > 0 and (acct.equity - acct.last_equity) / acct.last_equity <= -limits.max_daily_loss_pct:
        return False, 0.0, "daily loss limit hit, no more buys today"
    room = acct.equity * limits.max_position_pct - acct.position_value
    if room < 1:
        return False, 0.0, "position already at max size"
    return True, round(room, 2), "ok"
