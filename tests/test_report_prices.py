"""REAL market-data price table (Kiwoom + pykrx fallback) — fully mocked (no network).

Covers: the price-row shape, None fallback when a field can't be fetched (never invent a number),
change_pct math vs prev_close, the ~5/~21 trading-day anchors, and the report integration — prices
stay SEPARATE from the video-grounded insights, the report still builds with prices=[] when no keys,
and the top-20 default universe loads."""

import json
from datetime import date
from pathlib import Path

from tagent.news.report_prices import ROW_FIELDS, build_price_row, fetch_price_table
from tagent.news.youtube_report import DEFAULT_GIANTS, build_youtube_report, report_watchlist

TODAY = date(2026, 6, 15)
# 25 ascending completed sessions strictly before TODAY; close = 70000 + i (i = 1..25)
_DAILY = [{"date": 20260500 + i, "open": 70000.0, "close": 70000.0 + i} for i in range(1, 26)]


# --------------------------------------------------------------------------- #
# 1) row shape + 2) change_pct math + ~5/~21d anchors
# --------------------------------------------------------------------------- #
def test_price_row_shape_and_change_pct_and_anchors():
    rows = fetch_price_table(
        {"005930": "삼성전자"}, now=TODAY, use_cache=False,
        minute_fn=lambda t: (72000.0, 71000.0),            # current, today_open (real-time)
        daily_fn=lambda t: list(_DAILY))
    assert len(rows) == 1
    r = rows[0]
    assert set(ROW_FIELDS).issubset(r) and r["source"] == "키움/KRX"
    assert r["name"] == "삼성전자" and r["ticker"] == "005930"
    assert r["current"] == 72000.0 and r["today_open"] == 71000.0
    assert r["prev_close"] == 70025.0                       # comp[-1]  (latest completed session)
    assert r["week_ago_close"] == 70021.0                   # comp[-5]  (~5 trading days)
    assert r["month_ago_close"] == 70005.0                  # comp[-21] (~21 trading days)
    assert r["change_pct"] == round((72000.0 - 70025.0) / 70025.0 * 100.0, 2)


def test_build_price_row_change_pct_direct():
    row = build_price_row("000660", "SK하이닉스", minute=(71000.0, None),
                          daily_rows=[{"date": 20260612, "open": 70000.0, "close": 70000.0}],
                          today_int=20260615)
    assert row["prev_close"] == 70000.0 and row["change_pct"] == 1.43   # +1000/70000


# --------------------------------------------------------------------------- #
# 2) None fallback — never invent a number when a field can't be fetched
# --------------------------------------------------------------------------- #
def test_none_fallback_when_nothing_fetched():
    rows = fetch_price_table(["005930"], now=TODAY, use_cache=False,
                             minute_fn=lambda t: (None, None), daily_fn=lambda t: [],
                             pykrx_fn=lambda t: [])
    r = rows[0]
    assert r["ticker"] == "005930" and r["name"] == "삼성전자"
    for f in ("current", "today_open", "prev_close", "week_ago_close", "month_ago_close", "change_pct"):
        assert r[f] is None                                 # no invented numbers


def test_change_pct_none_without_current():
    # daily history present but no real-time current -> change_pct stays None (not fabricated as 0)
    rows = fetch_price_table(["005930"], now=TODAY, use_cache=False,
                             minute_fn=lambda t: (None, None), daily_fn=lambda t: list(_DAILY))
    r = rows[0]
    assert r["prev_close"] == 70025.0 and r["current"] is None and r["change_pct"] is None


def test_pykrx_fallback_used_when_kiwoom_daily_empty():
    calls = {"pykrx": 0}

    def _pykrx(t):
        calls["pykrx"] += 1
        return [{"date": 20260612, "open": 1.0, "close": 100.0}]
    rows = fetch_price_table(["005930"], now=TODAY, use_cache=False,
                             minute_fn=lambda t: (110.0, None), daily_fn=lambda t: [], pykrx_fn=_pykrx)
    assert calls["pykrx"] == 1 and rows[0]["prev_close"] == 100.0 and rows[0]["change_pct"] == 10.0


# --------------------------------------------------------------------------- #
# 3) report integration — prices SEPARATE from insights; builds w/o keys; top-20 default
# --------------------------------------------------------------------------- #
def _write_one_giant(tmp_path):
    p = Path(tmp_path) / "youtube_fetch_log" / "2026-06-14"
    p.mkdir(parents=True, exist_ok=True)
    quote = "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다"
    entry = {"video_id": "v1", "channel": "한경TV", "title": "한경TV 브리핑",
             "published_at": "2026-06-14T02:00:00Z", "fetched_at": "2026-06-14T02:00:00Z",
             "segments": [{"start": 120, "text": quote}],
             "insights": [{"ticker": "000660", "stock": "000660", "stock_name": "SK하이닉스",
                           "quote": quote, "summary": "HBM 공급계약", "timestamp_mmss": "02:00",
                           "deeplink": "https://www.youtube.com/watch?v=v1&t=120s", "importance": 4.0,
                           "category": "M&A/deal"}]}
    (p / "fetch.jsonl").write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")


_WIN = dict(start="2026-06-14T00:00:00+09:00", end="2026-06-14T23:59:00+09:00",
            now="2026-06-14T15:50:00+09:00")


def test_price_section_removed_even_with_prices_fn(tmp_path):
    # §4 시세 (price table) was REMOVED: even when a prices_fn is supplied, no prices are attached.
    _write_one_giant(tmp_path)
    fake_rows = [{"name": "삼성전자", "ticker": "005930", "current": 72000.0, "today_open": 71000.0,
                  "prev_close": 70000.0, "week_ago_close": 69000.0, "month_ago_close": 68000.0,
                  "change_pct": 2.86, "source": "키움/KRX"}]
    rep = build_youtube_report(data_dir=tmp_path, prices_fn=lambda codes: fake_rows, **_WIN)
    assert rep["prices"] == []                              # price section gone; the prices_fn result is ignored
    # the video-grounded insights still build and remain free of any price rows
    assert "SK하이닉스" in rep["per_stock"]
    for ins in rep["per_stock"]["SK하이닉스"]:
        assert ins.get("source") != "키움/KRX" and ins.get("quote")
    assert all(r.get("source") != "키움/KRX" for r in rep["recommendations"])


def test_report_builds_with_empty_prices_by_default(tmp_path):
    _write_one_giant(tmp_path)
    rep = build_youtube_report(data_dir=tmp_path, **_WIN)   # no prices_fn -> offline-safe
    assert rep["prices"] == [] and "SK하이닉스" in rep["per_stock"]


def test_auto_prices_empty_when_no_kiwoom_keys(tmp_path, monkeypatch):
    from tagent.config import SETTINGS
    monkeypatch.setattr(SETTINGS, "has_kiwoom_keys", lambda: False)
    _write_one_giant(tmp_path)
    rep = build_youtube_report(data_dir=tmp_path, prices_fn="auto", **_WIN)
    assert rep["prices"] == []                              # no keys -> empty, report still builds


def test_top20_default_universe_loads(monkeypatch):
    monkeypatch.delenv("YOUTUBE_REPORT_WATCHLIST", raising=False)
    wl = report_watchlist()
    assert len(wl) == 20 and wl is not DEFAULT_GIANTS       # a copy
    for code in ("005930", "000660", "373220", "207940", "006400", "068270", "105560",
                 "055550", "028260", "034020", "042700"):
        assert code in wl
    # still env-overridable
    monkeypatch.setenv("YOUTUBE_REPORT_WATCHLIST", "005930=삼성전자,000660")
    wl2 = report_watchlist()
    assert wl2 == {"005930": "삼성전자", "000660": "SK하이닉스"}
