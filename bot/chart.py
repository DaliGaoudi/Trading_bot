"""Render a candlestick-style chart as PNG bytes for the AI to look at."""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_chart(df, symbol: str, labels=("SMA 10", "SMA 30")) -> bytes:
    fig, (ax, ax_rsi) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    x = range(len(df))
    up = df["close"] >= df["open"]
    colors = ["#2a9d8f" if u else "#e76f51" for u in up]
    ax.vlines(x, df["low"], df["high"], color=colors, linewidth=1)
    ax.bar(x, (df["close"] - df["open"]).abs(), bottom=df[["open", "close"]].min(axis=1),
           color=colors, width=0.7)
    ax.plot(x, df["sma_fast"], label=labels[0], linewidth=1)
    ax.plot(x, df["sma_slow"], label=labels[1], linewidth=1)
    ax.set_title(f"{symbol} — last {len(df)} bars")
    ax.legend(loc="upper left")
    ax_rsi.plot(x, df["rsi"], color="#6a4c93", linewidth=1)
    ax_rsi.axhline(70, linestyle="--", linewidth=0.7)
    ax_rsi.axhline(30, linestyle="--", linewidth=0.7)
    ax_rsi.set_ylabel("RSI")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()
