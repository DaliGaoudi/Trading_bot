"""Settings. Secrets come only from environment variables or a local .env file."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class RiskLimits:
    symbols: tuple = ("SPY", "QQQ")      # only these can ever be traded
    max_position_pct: float = 0.10       # max 10% of equity in any one symbol
    max_trades_per_day: int = 4
    max_daily_loss_pct: float = 0.02     # stop trading for the day after -2%
    min_confidence: float = 0.65         # ignore weak AI signals


@dataclass(frozen=True)
class Settings:
    alpaca_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    anthropic_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-opus-5-5"))
    dry_run: bool = field(default_factory=lambda: os.getenv("DRY_RUN", "true").lower() != "false")
    timeframe_minutes: int = 15
    lookback_bars: int = 120
    risk: RiskLimits = RiskLimits()

    # Hard-coded on purpose: this bot never talks to a live account.
    paper: bool = True


settings = Settings()
