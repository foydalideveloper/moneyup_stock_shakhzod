"""Show-don't-tell intraday analysis — candles computed from RAW ticks.

Proof that the agent reads the data itself: given a stream of raw
``(time, price, volume)`` ticks (e.g. Kiwoom 0B 체결 executions), these pure
functions BUILD the candles, measure the body/wicks, compute moving averages,
detect golden/dead crosses, and raise a safety halt when linked markets crash —
all from numbers, no charting library. The dashboard then shows the computed
values as NUMBERS next to the drawn candle, so a non-technical viewer can see the
agent is reading the data, not embedding TradingView.

Everything here is no-lookahead (a bar uses only ticks inside it; a cross uses only
the last two MA points) and deterministic (ticks are sorted by time first), so it
is unit-tested with mock ticks and no network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Tick = Tuple[object, float, float]      # (time, price, volume)

_UNIT_SECONDS = {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "hour": 3600}


def parse_interval(interval: str) -> int:
    """'1min'/'5min'/'30s'/'1h' -> bar length in seconds."""
    s = str(interval).strip().lower()
    num = "".join(ch for ch in s if ch.isdigit())
    unit = "".join(ch for ch in s if ch.isalpha()) or "min"
    n = int(num) if num else 1
    if unit not in _UNIT_SECONDS:
        raise ValueError(f"unknown interval unit '{unit}' in '{interval}'")
    return n * _UNIT_SECONDS[unit]


def _to_dt(t) -> datetime:
    """Normalize a tick time (datetime / ISO string / epoch s or ms) to UTC datetime."""
    if isinstance(t, datetime):
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    if isinstance(t, (int, float)):
        secs = t / 1000.0 if t > 1e12 else float(t)        # ms vs s
        return datetime.fromtimestamp(secs, tz=timezone.utc)
    s = str(t).replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def build_candles(ticks: Iterable[Tick], interval: str = "1min") -> List[dict]:
    """Aggregate raw ticks into OHLCV bars — THIS computes the candle from the data.

    Ticks are sorted by time, then bucketed into ``interval`` windows. Each bar is
    open (first tick), high/low (extremes), close (last tick), volume (sum), n (tick
    count). Bars come out in chronological order; no future tick affects an earlier
    bar (no-lookahead).
    """
    secs = parse_interval(interval)
    rows = sorted(((_to_dt(t), float(p), float(v or 0.0)) for t, p, v in ticks),
                  key=lambda r: r[0])
    bars: Dict[int, dict] = {}
    order: List[int] = []
    for ts, price, vol in rows:
        b = int(ts.timestamp() // secs) * secs
        bar = bars.get(b)
        if bar is None:
            bars[b] = {"t": datetime.fromtimestamp(b, tz=timezone.utc).isoformat(),
                       "open": price, "high": price, "low": price, "close": price,
                       "volume": vol, "n": 1}
            order.append(b)
        else:
            bar["high"] = max(bar["high"], price)
            bar["low"] = min(bar["low"], price)
            bar["close"] = price
            bar["volume"] += vol
            bar["n"] += 1
    return [bars[b] for b in sorted(order)]


def candle_parts(bar: dict) -> dict:
    """Body color + wick lengths from a bar's OHLC (Korean convention: RED = up).

    color: 'red' when close>open (양봉), 'blue' when close<open (음봉), 'doji' when equal.
    body = |close-open|; upper_wick = high-max(open,close); lower_wick = min(open,close)-low.
    """
    o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
    color = "red" if c > o else "blue" if c < o else "doji"
    return {"color": color, "body": abs(c - o),
            "upper_wick": h - max(o, c), "lower_wick": min(o, c) - l,
            "range": h - l, "up": c >= o}


def moving_averages(closes: Sequence[float],
                    windows: Sequence[int] = (5, 20, 60)) -> Dict[int, List[Optional[float]]]:
    """Simple moving average series per window, aligned to ``closes`` (None during
    the warmup before ``window`` closes exist). Uses only past/current closes."""
    closes = [float(c) for c in closes]
    out: Dict[int, List[Optional[float]]] = {}
    for w in windows:
        w = int(w)
        series: List[Optional[float]] = []
        run = 0.0
        for i, c in enumerate(closes):
            run += c
            if i >= w:
                run -= closes[i - w]
            series.append(run / w if i + 1 >= w else None)
        out[w] = series
    return out


def cross_signal(ma_short: Sequence[Optional[float]],
                 ma_long: Sequence[Optional[float]]) -> Optional[str]:
    """'golden' if the short MA just crossed ABOVE the long, 'dead' if just BELOW,
    else None. Looks only at the last two bars where both MAs are defined."""
    pairs = [(s, l) for s, l in zip(ma_short, ma_long) if s is not None and l is not None]
    if len(pairs) < 2:
        return None
    (ps, pl), (cs, cl) = pairs[-2], pairs[-1]
    if ps <= pl and cs > cl:
        return "golden"
    if ps >= pl and cs < cl:
        return "dead"
    return None


def regime_safety(linked_returns, threshold: float = -0.03) -> str:
    """'HALT' if any LINKED return (e.g. the stock's US peers / index) has fallen at
    or below ``threshold`` (default -3%) — the systemic kill switch — else 'OK'."""
    if isinstance(linked_returns, dict):
        vals = linked_returns.values()
    else:
        vals = linked_returns or []
    for r in vals:
        try:
            if r is not None and float(r) <= threshold:
                return "HALT"
        except (TypeError, ValueError):
            continue
    return "OK"


def analyze_ticks(ticks: Iterable[Tick], linked_returns=None, interval: str = "1min",
                  windows: Sequence[int] = (5, 20, 60), threshold: float = -0.03,
                  last_n_ticks: int = 12, last_n_candles: int = 10) -> dict:
    """Full demo pipeline: raw ticks -> candles + parts + MAs + cross + safety.

    Returns the computed numbers (rounded for display) plus the last few raw ticks,
    so the dashboard can show the data and the agent's computation side by side.
    """
    ticks = list(ticks)
    candles = build_candles(ticks, interval)
    closes = [b["close"] for b in candles]
    mas = moving_averages(closes, windows)
    short_w, long_w = int(windows[0]), int(windows[1])
    cross = cross_signal(mas.get(short_w, []), mas.get(long_w, []))
    safety = regime_safety(linked_returns, threshold)

    def _r(x, d=4):
        return round(float(x), d) if x is not None else None

    rows = sorted(((_to_dt(t), float(p), float(v or 0.0)) for t, p, v in ticks), key=lambda r: r[0])
    raw = [{"t": ts.isoformat(), "price": _r(p), "volume": _r(v, 2)}
           for ts, p, v in rows[-last_n_ticks:]]
    out_candles = []
    for i, b in enumerate(candles[-last_n_candles:]):
        parts = candle_parts(b)
        out_candles.append({
            "t": b["t"], "open": _r(b["open"]), "high": _r(b["high"]),
            "low": _r(b["low"]), "close": _r(b["close"]), "volume": _r(b["volume"], 2),
            "n": b["n"], "color": parts["color"],
            "body": _r(parts["body"]), "upper_wick": _r(parts["upper_wick"]),
            "lower_wick": _r(parts["lower_wick"])})
    return {
        "interval": interval, "n_ticks": len(ticks), "n_candles": len(candles),
        "ticks": raw, "candles": out_candles,
        "latest_candle": out_candles[-1] if out_candles else None,
        "mas": {int(w): _r(mas[w][-1]) for w in windows},
        "cross": cross, "cross_pair": [short_w, long_w],
        "safety": safety,
        "linked_returns": {k: _r(v, 4) for k, v in linked_returns.items()}
        if isinstance(linked_returns, dict) else {},
    }
