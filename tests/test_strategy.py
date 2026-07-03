from datetime import datetime, timedelta, timezone

from tagent.state import SymbolState
from tagent.strategy import evaluate

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def t(seconds):
    return BASE + timedelta(seconds=seconds)


def kinds(signals):
    return {s.kind for s in signals}


def test_rule_near_low_tight_spread_fires_buy():
    s = SymbolState("X")
    s.update_quote(99.99, 100.00, 1, 1, t(0))  # spread 0.01% -> tight
    s.update_trade(100.0, 1, t(1))             # at the session low
    sigs = evaluate(s, now=t(1))
    assert "near_low_tight_spread" in kinds(sigs)
    assert all(sig.side == "buy" for sig in sigs if sig.kind == "near_low_tight_spread")


def test_rule_breakout_fires():
    s = SymbolState("X", window_seconds=60)
    s.update_quote(100.0, 100.2, 1, 1, t(0))   # ~0.2% spread (not "tight")
    s.update_trade(100.0, 1, t(1))
    s.update_trade(101.0, 1, t(2))
    s.update_trade(100.5, 1, t(3))
    s.update_trade(101.5, 1, t(4))             # breaks above prior high (101)
    sigs = evaluate(s, now=t(4))
    assert "breakout" in kinds(sigs)


def test_rule_pullback_fires_sell():
    s = SymbolState("X")
    s.update_quote(97.0, 99.0, 1, 1, t(0))     # wide spread, blocks rule 1
    s.update_trade(100.0, 1, t(1))
    s.update_trade(98.0, 1, t(2))              # 2% below session high
    sigs = evaluate(s, now=t(2))
    assert "pullback_from_high" in kinds(sigs)
    assert all(sig.side == "sell" for sig in sigs if sig.kind == "pullback_from_high")


def test_no_signal_when_stale():
    s = SymbolState("X")
    s.update_quote(99.0, 100.0, 1, 1, t(0))
    s.update_trade(100.0, 1, t(1))
    sigs = evaluate(s, stale_after=30, now=t(1) + timedelta(seconds=120))
    assert sigs == []


def test_no_signal_without_quote():
    s = SymbolState("X")
    s.update_trade(100.0, 1, t(1))             # no quote yet
    assert evaluate(s, now=t(1)) == []
