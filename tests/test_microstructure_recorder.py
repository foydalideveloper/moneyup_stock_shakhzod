"""Forward-recorder for the new KR microstructure sessions — depth-weighted quote, append-only,
dedup, and no-lookahead. Synthetic books, no network."""

import csv
from datetime import datetime, timezone, timedelta

from tagent.feeds.base import OrderBookSnapshot
from tagent.microstructure_recorder import (
    SESSION_NAMES, MicrostructureRecorder, depth_weighted_quote, load_recorder_summary,
    session_for,
)

KST = timezone(timedelta(hours=9))


def _ob(ts, symbol="005930", bids=None, asks=None):
    bids = bids if bids is not None else [(74000.0, 10.0), (73900.0, 5.0), (73800.0, 3.0)]
    asks = asks if asks is not None else [(74100.0, 4.0), (74200.0, 6.0), (74300.0, 2.0)]
    return OrderBookSnapshot(symbol=symbol, timestamp=ts, asks=asks, bids=bids)


# --------------------------------------------------------------------------- #
# 1) depth-weighted quote — the whole visible book, NOT just the last trade/best
# --------------------------------------------------------------------------- #
def test_depth_weighted_quote_uses_full_book_not_just_top():
    bids = [(100.0, 1.0), (99.0, 9.0)]            # heavy deeper bid level
    asks = [(101.0, 1.0), (102.0, 1.0)]
    q = depth_weighted_quote(_ob(datetime(2026, 6, 11, 8, 30), bids=bids, asks=asks))
    # qty-weighted VWAP across levels, not the best price: (100*1 + 99*9)/10 = 99.1
    assert abs(q["dw_bid_price"] - 99.1) < 1e-9
    assert abs(q["dw_ask_price"] - 101.5) < 1e-9
    assert q["bid_depth"] == 10.0 and q["ask_depth"] == 2.0
    # imbalance leans to the heavier (bid) side
    assert q["depth_imbalance"] > 0 and abs(q["depth_imbalance"] - (10.0 - 2.0) / 12.0) < 1e-6
    # top-of-book microprice (equal top sizes here -> mid) and plain mid/spread
    assert abs(q["microprice"] - 100.5) < 1e-9
    assert q["mid"] == 100.5 and q["spread"] == 1.0 and q["levels"] == 2


def test_microprice_leans_to_heavier_top_size():
    # bigger bid size at the top pulls the fair price toward the ask (more buyers waiting)
    q = depth_weighted_quote(_ob(datetime(2026, 6, 11, 8, 30),
                                  bids=[(100.0, 9.0)], asks=[(101.0, 1.0)]))
    # microprice = (100*1 + 101*9)/10 = 100.9 -> above the 100.5 mid
    assert abs(q["microprice"] - 100.9) < 1e-9 and q["microprice"] > q["mid"]


def test_depth_weighted_quote_none_on_one_sided_or_nonpositive():
    base = datetime(2026, 6, 11, 8, 30)
    assert depth_weighted_quote(_ob(base, bids=[], asks=[(1.0, 1.0)])) is None
    assert depth_weighted_quote(_ob(base, bids=[(0.0, 1.0)], asks=[(1.0, 1.0)])) is None


# --------------------------------------------------------------------------- #
# 2) session calendar — NXT pre-market + KRX night (wraps past midnight)
# --------------------------------------------------------------------------- #
def test_session_detection_covers_nxt_and_wrapping_night():
    d = lambda h, m: datetime(2026, 6, 11, h, m)
    assert session_for(d(8, 0)) == "nxt-premarket"     # inclusive start
    assert session_for(d(8, 49)) == "nxt-premarket"
    assert session_for(d(8, 50)) is None               # exclusive end -> continuous market, not ours
    assert session_for(d(9, 30)) is None               # regular KRX session is NOT a new-microstructure window
    assert session_for(d(12, 0)) is None
    assert session_for(d(19, 0)) == "krx-night"
    assert session_for(d(2, 0)) == "krx-night"         # wraps past midnight
    assert session_for(d(5, 0)) is None                # night ends 05:00 exclusive
    assert set(SESSION_NAMES) == {"nxt-premarket", "krx-night"}


# --------------------------------------------------------------------------- #
# 3) append-only + dedup (including across a process restart)
# --------------------------------------------------------------------------- #
def test_record_appends_and_dedups_across_restart(tmp_path):
    p = tmp_path / "ms.csv"
    rec = MicrostructureRecorder(path=p)
    now = datetime(2026, 6, 11, 8, 40)
    t1, t2 = datetime(2026, 6, 11, 8, 30), datetime(2026, 6, 11, 8, 31)
    assert rec.record(_ob(t1), now=now) is True
    assert rec.record(_ob(t1), now=now) is False       # exact same observation -> deduped
    assert rec.record(_ob(t2), now=now) is True
    rec.close()
    rows = p.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 2                           # header + 2 distinct observations

    # a FRESH recorder reloads the seen-set from disk: no double-count, strictly append-only
    rec2 = MicrostructureRecorder(path=p)
    assert rec2.record(_ob(t1), now=now) is False       # already persisted
    assert rec2.record(_ob(t2), now=now) is False
    assert rec2.record(_ob(datetime(2026, 6, 11, 8, 32)), now=now) is True
    rec2.close()
    rows2 = p.read_text(encoding="utf-8").splitlines()
    assert len(rows2) == 1 + 3                           # only the new row appended
    assert rows2[:3] == rows                             # original header+rows untouched (append-only)


# --------------------------------------------------------------------------- #
# 4) no-lookahead — refuse future-dated and out-of-session observations
# --------------------------------------------------------------------------- #
def test_no_lookahead_refuses_future_and_out_of_session(tmp_path):
    p = tmp_path / "ms.csv"
    rec = MicrostructureRecorder(path=p)
    now = datetime(2026, 6, 11, 8, 40)
    # in-session but timestamped AFTER the clock -> refused (can't record the future)
    assert rec.record(_ob(datetime(2026, 6, 11, 8, 45)), now=now) is False
    # out-of-session even though <= now -> refused (only the new windows are recorded)
    assert rec.record(_ob(datetime(2026, 6, 11, 10, 0)), now=datetime(2026, 6, 11, 10, 5)) is False
    # in-session and not future -> recorded
    assert rec.record(_ob(datetime(2026, 6, 11, 8, 30)), now=now) is True
    rec.close()
    # every persisted observation was knowable at its clock (no lookahead leaked into the store)
    with p.open(encoding="utf-8") as fh:
        stamps = [datetime.fromisoformat(r["ts"]) for r in csv.DictReader(fh)]
    assert stamps == [datetime(2026, 6, 11, 8, 30)] and all(ts <= now for ts in stamps)


def test_tz_aware_utc_snapshot_mapped_to_kst_session(tmp_path):
    # the live Kiwoom feed stamps snapshots in UTC; 23:30 UTC == 08:30 KST (next day) -> NXT window
    p = tmp_path / "ms.csv"
    rec = MicrostructureRecorder(path=p)
    utc = datetime(2026, 6, 10, 23, 30, tzinfo=timezone.utc)
    now = datetime(2026, 6, 10, 23, 40, tzinfo=timezone.utc)
    assert rec.record(_ob(utc), now=now) is True
    rec.close()
    with p.open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["session"] == "nxt-premarket" and row["ts"].startswith("2026-06-11T08:30")


# --------------------------------------------------------------------------- #
# 5) coverage summary (no strategy — just the accrued dataset's shape)
# --------------------------------------------------------------------------- #
def test_summary_reports_per_session_symbol_coverage(tmp_path):
    p = tmp_path / "ms.csv"
    rec = MicrostructureRecorder(path=p)
    pm = datetime(2026, 6, 11, 8, 40)
    rec.record(_ob(datetime(2026, 6, 11, 8, 30), symbol="005930"), now=pm)
    rec.record(_ob(datetime(2026, 6, 11, 8, 31), symbol="000660"), now=pm)
    rec.record(_ob(datetime(2026, 6, 11, 19, 0), symbol="101S3000"),
               now=datetime(2026, 6, 11, 19, 5))
    s = rec.summary()
    assert s["total_rows"] == 3 and s["rows_added_this_run"] == 3
    assert set(s["sessions"]) == {"nxt-premarket", "krx-night"}
    assert s["by_session"]["nxt-premarket"] == {"000660": 1, "005930": 1}
    assert s["by_session"]["krx-night"] == {"101S3000": 1}
    rec.close()
    # the read-only loader reflects persisted coverage without writing
    assert load_recorder_summary(path=p)["total_rows"] == 3
