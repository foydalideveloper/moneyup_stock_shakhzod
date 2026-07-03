"""Order-book depth-memory tests (synthetic snapshots, no network)."""

from datetime import datetime, timedelta, timezone

from tagent.feeds.base import OrderBookSnapshot
from tagent.orderbook import OrderBookMemory

T0 = datetime(2026, 6, 1, 14, 30, tzinfo=timezone.utc)


def _ts(i: int) -> datetime:
    return T0 + timedelta(seconds=i)


def _book(asks, bids, i=0, symbol="X") -> OrderBookSnapshot:
    return OrderBookSnapshot(symbol=symbol, timestamp=_ts(i), asks=asks, bids=bids)


def _reasons(deque_):
    return [reason for _p, _q, reason, _t in deque_]


def test_rising_price_plus_trade_is_filled():
    """The best ask is lifted by a print, then the book steps up -> 'filled'."""
    mem = OrderBookMemory("X")
    asks0 = [(100.0, 5), (100.1, 5), (100.2, 5)]
    bids0 = [(99.9, 5), (99.8, 5), (99.7, 5)]
    mem.update(_book(asks0, bids0, 0))

    # Price rose: the 100.0 ask is gone, a new 100.3 level appears at the top.
    asks1 = [(100.1, 5), (100.2, 5), (100.3, 5)]
    bids1 = [(100.0, 5), (99.9, 5), (99.8, 5)]
    mem.update(_book(asks1, bids1, 1), recent_trades=[100.0])

    assert _reasons(mem.vanished_asks) == ["filled"]


def test_vanished_qty_without_trade_is_cancelled():
    """An ask inside the live window disappears with no print -> 'cancelled'."""
    mem = OrderBookMemory("X")
    asks0 = [(100.0, 5), (100.1, 20), (100.2, 5)]  # fat 100.1 level
    bids0 = [(99.9, 5), (99.8, 5), (99.7, 5)]
    mem.update(_book(asks0, bids0, 0))

    # 100.1 is pulled; price did not move (best/worst ask unchanged range).
    asks1 = [(100.0, 5), (100.2, 5), (100.3, 5)]
    bids1 = [(99.9, 5), (99.8, 5), (99.7, 5)]
    mem.update(_book(asks1, bids1, 1), recent_trades=None)

    reasons = _reasons(mem.vanished_asks)
    assert reasons == ["cancelled"], reasons


def test_scrolled_when_window_shifts_without_fill_or_cancel():
    """Price drops: the highest ask falls off the far edge -> 'scrolled'."""
    mem = OrderBookMemory("X")
    asks0 = [(100.0, 5), (100.1, 5), (100.2, 5)]
    bids0 = [(99.9, 5), (99.8, 5), (99.7, 5)]
    mem.update(_book(asks0, bids0, 0))

    # Price fell: a new lower ask (99.9) enters, the old top 100.2 scrolls out.
    asks1 = [(99.9, 5), (100.0, 5), (100.1, 5)]
    bids1 = [(99.8, 5), (99.7, 5), (99.6, 5)]
    mem.update(_book(asks1, bids1, 1), recent_trades=None)

    assert "scrolled" in _reasons(mem.vanished_asks)
    # The 100.2 level specifically scrolled (above the new highest ask 100.1).
    scrolled = [p for p, _q, r, _t in mem.vanished_asks if r == "scrolled"]
    assert 100.2 in scrolled


def test_bid_hit_by_trade_is_filled():
    """A falling price prints through the best bid -> bid 'filled'."""
    mem = OrderBookMemory("X")
    asks0 = [(100.1, 5), (100.2, 5), (100.3, 5)]
    bids0 = [(100.0, 5), (99.9, 5), (99.8, 5)]
    mem.update(_book(asks0, bids0, 0))

    # Price dropped: 100.0 bid consumed, new 99.7 appears at the bottom.
    asks1 = [(100.0, 5), (100.1, 5), (100.2, 5)]
    bids1 = [(99.9, 5), (99.8, 5), (99.7, 5)]
    mem.update(_book(asks1, bids1, 1), recent_trades=[100.0])

    assert _reasons(mem.vanished_bids) == ["filled"]


def test_memory_keeps_only_last_10_per_side():
    """The deques cap at 10 entries per side regardless of churn."""
    mem = OrderBookMemory("X")
    prev = _book([(100.0 + j * 0.1, 5) for j in range(3)],
                 [(99.9 - j * 0.1, 5) for j in range(3)], 0)
    mem.update(prev)

    # 30 rounds, each cancelling one distinct ask and one distinct bid.
    for i in range(1, 31):
        ask_gone = 200.0 + i              # unique, inside-window cancel each round
        bid_gone = 50.0 - i
        asks = [(ask_gone, 7), (300.0, 5), (301.0, 5)]
        bids = [(bid_gone, 7), (40.0, 5), (39.0, 5)]
        mem.update(_book(asks, bids, i))
        # Reset prev so the *next* round sees these as resting then vanishing.
        mem.update(_book([(300.0, 5), (301.0, 5), (302.0, 5)],
                         [(40.0, 5), (39.0, 5), (38.0, 5)], i))

    assert len(mem.vanished_asks) == 10
    assert len(mem.vanished_bids) == 10


def test_derived_features_compute_correctly():
    """absorption, spoof ratio and depth imbalance match hand computation."""
    mem = OrderBookMemory("X")
    asks0 = [(100.0, 10), (100.1, 5), (100.2, 30), (100.3, 5)]
    bids0 = [(99.9, 5), (99.8, 5), (99.7, 5)]
    mem.update(_book(asks0, bids0, 0))

    # 100.0 filled by a print (qty 10). 100.2 is pulled with no trade (qty 30)
    # while 100.1 keeps resting, so 100.2 stays inside the window -> cancelled.
    asks1 = [(100.1, 5), (100.3, 5), (100.4, 5)]   # ask_depth 15
    bids1 = [(99.9, 6), (99.8, 5), (99.7, 4)]       # bid_depth 15
    mem.update(_book(asks1, bids1, 1), recent_trades=[100.0])

    snap = mem.snapshot()
    assert snap["filled_qty"] == 10.0
    assert snap["cancelled_qty"] == 30.0
    # absorption = 10 / (10 + 30) = 0.25
    assert snap["absorption_ratio"] == 0.25
    # spoof = 30 / (10 + 30 + 0) = 0.75  (no scrolled levels here)
    assert snap["spoof_ratio"] == 0.75
    # depth imbalance = (15 - 15) / 30 = 0.0
    assert snap["bid_depth"] == 15.0
    assert snap["ask_depth"] == 15.0
    assert snap["depth_imbalance"] == 0.0


def test_first_snapshot_records_nothing():
    mem = OrderBookMemory("X")
    mem.update(_book([(100.0, 5)], [(99.0, 5)], 0))
    assert not mem.vanished_asks and not mem.vanished_bids
    # depth is still cached for the imbalance feature.
    snap = mem.snapshot()
    assert snap["ask_depth"] == 5.0 and snap["bid_depth"] == 5.0
