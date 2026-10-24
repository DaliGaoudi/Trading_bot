"""Turn a chart into a decision: buy, sell or hold, with a confidence."""
import base64
import json
from dataclasses import dataclass

PROMPT = """You are a cautious technical analyst. Look at this {symbol} chart \
({minutes}-minute bars, SMA 10/30, RSI 14). Latest close: {close:.2f}, RSI: {rsi:.1f}.
Decide whether to buy, sell (close the position), or hold. Prefer hold when unclear.
Reply with JSON only: {{"action": "buy"|"sell"|"hold", "confidence": 0-1, "reason": "one sentence"}}"""


@dataclass
class Decision:
    action: str
    confidence: float
    reason: str


def parse_decision(text: str) -> Decision:
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1])
        action = str(data.get("action", "hold")).lower()
        if action not in ("buy", "sell", "hold"):
            action = "hold"
        conf = max(0.0, min(1.0, float(data.get("confidence", 0))))
        return Decision(action, conf, str(data.get("reason", "")))
    except (ValueError, TypeError):
        return Decision("hold", 0.0, "could not parse AI reply")


def rule_based(df) -> Decision:
    """Fallback when no AI key is set: SMA crossover filtered by RSI."""
    last, prev = df.iloc[-1], df.iloc[-2]
    crossed_up = prev.sma_fast <= prev.sma_slow and last.sma_fast > last.sma_slow
    crossed_down = prev.sma_fast >= prev.sma_slow and last.sma_fast < last.sma_slow
    if crossed_up and last.rsi < 70:
        return Decision("buy", 0.7, "SMA 10 crossed above SMA 30")
    if crossed_down:
        return Decision("sell", 0.7, "SMA 10 crossed below SMA 30")
    return Decision("hold", 0.5, "no crossover")


def ai_decision(cfg, df, symbol: str, png: bytes) -> Decision:
    import anthropic

    client = anthropic.Anthropic(api_key=cfg.anthropic_key)
    last = df.iloc[-1]
    msg = client.messages.create(
        model=cfg.model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(png).decode()}},
                {"type": "text", "text": PROMPT.format(symbol=symbol, minutes=cfg.timeframe_minutes,
                                                       close=last.close, rsi=last.rsi)},
            ],
        }],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return parse_decision(text)


def analyse(cfg, df, symbol: str, png: bytes) -> Decision:
    if cfg.anthropic_key:
        return ai_decision(cfg, df, symbol, png)
    return rule_based(df)
