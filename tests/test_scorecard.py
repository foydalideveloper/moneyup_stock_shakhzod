"""Signal scorecard + simulated paper P&L — synthetic, deterministic clock."""

from datetime import datetime, timedelta, timezone

from tagent.scorecard import PaperBook, Scorecard, side_of

T0 = datetime(2026, 6, 4, 20, 0, 0, tzinfo=timezone.utc)


def _sc(tmp_path, **kw):
    return Scorecard(data_dir=tmp_path, **kw)


# --------------------------------------------------------------------------- #
# direction mapping
# --------------------------------------------------------------------------- #
def test_side_of_maps_directions():
    assert side_of("BUY") == 1 and side_of("bullish") == 1 and side_of("up") == 1
    assert side_of("SELL") == -1 and side_of("bearish") == -1 and side_of("down") == -1
    assert side_of("HOLD") == 0 and side_of("") == 0


# --------------------------------------------------------------------------- #
# de-duplication — one row per NEW signal, not every poll
# --------------------------------------------------------------------------- #
def test_dedup_logs_only_distinct_signals(tmp_path):
    sc = _sc(tmp_path, horizon_s=900)
    # same BUY polled 3x -> one logged signal
    for k in range(3):
        sc.observe("us-ml", "AAPL", "BUY", 100 + k, ts=T0 + timedelta(seconds=k))
    assert len(sc.scores) == 1
    # a HOLD does not create a signal and does not reset the side
    sc.observe("us-ml", "AAPL", "HOLD", 103, ts=T0 + timedelta(seconds=5))
    assert len(sc.scores) == 1
    # flip to SELL -> a new distinct signal
    sc.observe("us-ml", "AAPL", "SELL", 104, ts=T0 + timedelta(seconds=6))
    assert len(sc.scores) == 2
    # back to BUY -> distinct again
    sc.observe("us-ml", "AAPL", "BUY", 105, ts=T0 + timedelta(seconds=7))
    assert len(sc.scores) == 3
    # the CSV has exactly the 3 distinct rows (+header)
    rows = (tmp_path / "signal_log.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1 + 3
    assert rows[0].split(",")[:5] == ["timestamp", "source", "symbol", "direction", "price"]


def test_dedup_is_per_source_and_symbol(tmp_path):
    sc = _sc(tmp_path)
    sc.observe("us-ml", "AAPL", "BUY", 100, ts=T0)
    sc.observe("crypto-ob", "AAPL", "BUY", 100, ts=T0)     # different source -> logged
    sc.observe("us-ml", "NVDA", "BUY", 100, ts=T0)         # different symbol -> logged
    assert len(sc.scores) == 3


# --------------------------------------------------------------------------- #
# scoring: correct / wrong / pending + NO LOOKAHEAD
# --------------------------------------------------------------------------- #
def test_pending_until_horizon_then_correct(tmp_path):
    sc = _sc(tmp_path, horizon_s=900)
    sc.observe("us-ml", "AAPL", "BUY", 100.0, ts=T0)
    assert sc.scores[0]["status"] == "pending"
    # a price BEFORE the horizon must NOT resolve it (even though it's down)
    sc.observe("us-ml", "AAPL", "BUY", 90.0, ts=T0 + timedelta(minutes=5))
    assert sc.scores[0]["status"] == "pending"
    # at the horizon it resolves on THAT price (up vs entry -> correct)
    sc.observe("us-ml", "AAPL", "BUY", 105.0, ts=T0 + timedelta(minutes=15))
    assert sc.scores[0]["status"] == "correct"
    assert sc.scores[0]["resolved_price"] == 105.0


def test_no_lookahead_uses_horizon_price_not_interim(tmp_path):
    # interim price would score WRONG; horizon price scores CORRECT. The interim
    # must be ignored -> proves resolution does not peek before the horizon.
    sc = _sc(tmp_path, horizon_s=900)
    sc.observe("us-ml", "AAPL", "BUY", 100.0, ts=T0)
    sc.observe("us-ml", "AAPL", "BUY", 80.0, ts=T0 + timedelta(minutes=10))   # pre-horizon dip
    assert sc.scores[0]["status"] == "pending"
    sc.observe("us-ml", "AAPL", "BUY", 110.0, ts=T0 + timedelta(minutes=16))
    assert sc.scores[0]["status"] == "correct" and sc.scores[0]["resolved_price"] == 110.0


def test_short_signal_scored_wrong_when_price_rises(tmp_path):
    sc = _sc(tmp_path, horizon_s=600)
    sc.observe("crypto-ob", "BTCUSDT", "bearish", 100.0, ts=T0)
    sc.observe("crypto-ob", "BTCUSDT", "bearish", 120.0, ts=T0 + timedelta(minutes=11))
    assert sc.scores[0]["status"] == "wrong"             # short but price went up


def test_hit_rate_excludes_pending(tmp_path):
    sc = _sc(tmp_path, horizon_s=600)
    sc.observe("us-ml", "AAPL", "BUY", 100.0, ts=T0)                       # will be correct
    sc.observe("us-ml", "AAPL", "BUY", 110.0, ts=T0 + timedelta(minutes=11))
    sc.observe("us-ml", "NVDA", "BUY", 100.0, ts=T0 + timedelta(minutes=11))  # stays pending
    card = sc.scorecard()["sources"]["us-ml"]
    assert card["correct"] == 1 and card["pending"] == 1 and card["wrong"] == 0
    assert card["hit_rate"] == 100.0                      # 1/1 decided, pending excluded


# --------------------------------------------------------------------------- #
# P&L + MANDATORY cost math
# --------------------------------------------------------------------------- #
def test_paperbook_costs_on_entry_and_exit():
    # $10k notional, 10bps round trip => 5bps per side = $5 entry on $10k.
    b = PaperBook("us-ml", start=10_000.0, trade_pct=1.0, cost_bps_round=10.0)
    b.set_target("AAPL", 1, 100.0)                        # long 100 sh @100, entry cost $5
    assert b.trades == 1 and abs(b.costs - 5.0) < 1e-9
    assert abs(b.unrealized({"AAPL": [110.0, "t"]}) - 1000.0) < 1e-9    # +$10 * 100sh
    # equity = 10000 + 0 realized - 5 costs + 1000 unrealized
    assert abs(b.equity({"AAPL": [110.0, "t"]}) - 10_995.0) < 1e-9
    b.set_target("AAPL", -1, 110.0)                       # close long, open short
    # realized = 100*(110-100)=1000 ; exit cost = 11000*0.0005=5.5 ; short entry cost=5
    assert abs(b.realized - 1000.0) < 1e-9
    assert abs(b.costs - (5.0 + 5.5 + 5.0)) < 1e-9
    assert b.trades == 2


def test_costs_make_a_round_trip_at_flat_price_lose_money():
    b = PaperBook("ta", start=10_000.0, trade_pct=1.0, cost_bps_round=10.0)
    b.set_target("BTCUSDT", 1, 100.0)
    b.set_target("BTCUSDT", 0, 100.0)                     # close at same price
    # no price move, but two cost legs paid -> equity below start (costs are real)
    eq = b.equity({"BTCUSDT": [100.0, "t"]})
    assert eq < 10_000.0 and abs(b.costs - 10.0) < 1e-9   # ~$10 round trip
    assert abs(eq - (10_000.0 - 10.0)) < 1e-9


def test_shorting_off_goes_flat_not_short():
    b = PaperBook("us-ml", start=10_000.0, allow_short=False)
    b.set_target("AAPL", 1, 100.0)
    b.set_target("AAPL", -1, 110.0)                       # SELL with shorting off
    assert b.positions == {}                              # flat, not short
    assert abs(b.realized - 1000.0) < 1e-9                # the long was still closed for +1000


# --------------------------------------------------------------------------- #
# equity reflected through the engine + persistence
# --------------------------------------------------------------------------- #
def test_engine_equity_updates_with_marks(tmp_path):
    sc = _sc(tmp_path, horizon_s=900, trade_pct=1.0)
    sc.observe("crypto-ob", "BTCUSDT", "bullish", 100.0, ts=T0)
    base = sc.scorecard()["sources"]["crypto-ob"]
    assert base["equity"] < 10_000.0 + 1e-6               # entry cost already paid
    assert base["trades"] == 1
    # price rises -> unrealized grows, equity climbs above start net of costs
    sc.observe("crypto-ob", "BTCUSDT", "bullish", 120.0, ts=T0 + timedelta(seconds=30))
    up = sc.scorecard()["sources"]["crypto-ob"]
    assert up["unrealized_pnl"] > base["unrealized_pnl"]
    assert up["equity"] > base["equity"] and up["costs"] > 0


def test_state_persists_across_restart(tmp_path):
    sc = _sc(tmp_path, horizon_s=900)
    sc.observe("us-ml", "AAPL", "BUY", 100.0, ts=T0)
    sc.observe("us-ml", "AAPL", "BUY", 110.0, ts=T0 + timedelta(minutes=16))   # resolves correct
    eq1 = sc.scorecard()["sources"]["us-ml"]["equity"]

    # new instance over the same dir reloads scores + book + dedup state
    sc2 = _sc(tmp_path, horizon_s=900)
    card = sc2.scorecard()["sources"]["us-ml"]
    assert card["correct"] == 1 and card["trades"] == 1
    assert abs(card["equity"] - eq1) < 1e-6
    # dedup survives: re-observing the same BUY does NOT create a new signal
    sc2.observe("us-ml", "AAPL", "BUY", 111.0, ts=T0 + timedelta(minutes=17))
    assert len([r for r in sc2.scores if r["source"] == "us-ml"]) == 1


def test_state_never_resets_on_restart_or_code_change(tmp_path):
    # build up non-trivial state across symbols + sources
    sc = _sc(tmp_path, horizon_s=900)
    sc.observe("us-ml", "AAPL", "BUY", 100.0, ts=T0)
    sc.observe("us-ml", "AAPL", "BUY", 130.0, ts=T0 + timedelta(minutes=16))   # +30, correct
    sc.observe("crypto-ob", "BTCUSDT", "bullish", 200.0, ts=T0 + timedelta(minutes=16))
    before = sc.scorecard()["sources"]
    assert before["us-ml"]["equity"] != 10_000.0          # P&L accrued

    # "restart" / "code change" = a brand-new engine over the same dir
    again = _sc(tmp_path, horizon_s=900).scorecard()["sources"]
    assert again["us-ml"]["equity"] == before["us-ml"]["equity"]      # NOT reset
    assert again["us-ml"]["correct"] == 1 and again["us-ml"]["trades"] == 1
    assert again["crypto-ob"]["trades"] == 1
    # per (source, symbol) books are preserved, so per-symbol view still works
    btc = _sc(tmp_path, horizon_s=900).scorecard("BTCUSDT")["sources"]["crypto-ob"]
    assert btc["total"] == 1 and btc["equity_start"] == 10_000.0


def test_explicit_reset_clears_state(tmp_path):
    sc = _sc(tmp_path, horizon_s=900)
    sc.observe("us-ml", "AAPL", "BUY", 100.0, ts=T0)
    assert sc.scorecard()["sources"]["us-ml"]["total"] == 1
    fresh = _sc(tmp_path, horizon_s=900, reset=True)      # explicit reset only
    assert fresh.scorecard()["sources"]["us-ml"]["total"] == 0
    assert fresh.scorecard()["sources"]["us-ml"]["equity"] == 10_000.0


def test_single_instance_owns_state(tmp_path):
    sc = _sc(tmp_path)
    assert sc.owner is True
    assert (tmp_path / "scorecard.lock").exists()
    # a second instance with a (simulated) different LIVE owner does not write
    (tmp_path / "scorecard.lock").write_text("999999999")     # bogus but treated below
    import tagent.scorecard as scmod
    orig = scmod._pid_alive
    scmod._pid_alive = lambda pid: True                       # pretend it's alive
    try:
        sc2 = _sc(tmp_path)
        assert sc2.owner is False                            # won't clobber the owner
        sc2.observe("us-ml", "AAPL", "BUY", 100.0, ts=T0)    # in-memory only
        assert not (tmp_path / "signal_log.csv").exists()    # non-owner didn't append
    finally:
        scmod._pid_alive = orig


# --------------------------------------------------------------------------- #
# funding-carry agent (delta-neutral: accrue 8h funding minus costs, hold)
# --------------------------------------------------------------------------- #
def test_funding_carry_accrues_each_interval(tmp_path):
    sc = _sc(tmp_path, funding_enter_bps=1.0, funding_cost_bps=5.0)  # $10k notional
    sc.accrue_funding("BTCUSDT", 0.0002, funding_time=1, ts=T0)      # enter (no accrual yet)
    card1 = sc.scorecard()["sources"]["funding-carry"]
    assert sc.carry_state["BTCUSDT"]["in_carry"] is True
    assert card1["trades"] == 1 and card1["costs"] > 0               # paid entry cost once
    # held through the next intervals -> collects +2bp on $10k = +$2 each
    for ft in range(2, 6):
        sc.accrue_funding("BTCUSDT", 0.0002, funding_time=ft, ts=T0 + timedelta(hours=8 * ft))
    card2 = sc.scorecard()["sources"]["funding-carry"]
    assert abs(card2["realized_pnl"] - 4 * (0.0002 * 10_000)) < 1e-6  # 4 intervals * $2
    assert card2["equity"] > card1["equity"]                         # funding lifts equity
    assert card2["trades"] == 1                                      # still one position (held)


def test_funding_carry_dedups_same_interval(tmp_path):
    sc = _sc(tmp_path)
    sc.accrue_funding("ETHUSDT", 0.0002, funding_time=10, ts=T0)
    sc.accrue_funding("ETHUSDT", 0.0002, funding_time=10, ts=T0)     # same interval -> ignored
    sc.accrue_funding("ETHUSDT", 0.0002, funding_time=9, ts=T0)      # older -> ignored
    recs = [r for r in sc.scores if r["source"] == "funding-carry"]
    assert len(recs) == 1                                            # no double accrual / churn


def test_funding_carry_cost_subtracted_and_churn_costs_more(tmp_path):
    # held continuously: one entry cost only
    hold = _sc(tmp_path / "a", funding_enter_bps=1.0, funding_band_bps=4.0, funding_cost_bps=5.0)
    for ft in range(1, 13):
        hold.accrue_funding("BTCUSDT", 0.0002, funding_time=ft, ts=T0 + timedelta(hours=8 * ft))
    hb = hold.scorecard()["sources"]["funding-carry"]
    assert hb["trades"] == 1                                         # never churned

    # forced churn: funding swings hard negative (below exit) then positive, repeatedly
    churn = _sc(tmp_path / "b", funding_enter_bps=1.0, funding_band_bps=4.0, funding_cost_bps=5.0)
    seq = [0.0002, -0.0010, 0.0002, -0.0010, 0.0002, -0.0010]        # in/out/in/out...
    for ft, fr in enumerate(seq, 1):
        churn.accrue_funding("BTCUSDT", fr, funding_time=ft, ts=T0 + timedelta(hours=8 * ft))
    cb = churn.scorecard()["sources"]["funding-carry"]
    assert cb["trades"] >= 3                                         # multiple re-entries
    assert cb["costs"] > hb["costs"]                                 # churning pays more cost


def test_funding_carry_holds_through_mild_negative_no_churn(tmp_path):
    # a brief, mild negative (within the hysteresis band) must NOT trigger an exit
    sc = _sc(tmp_path, funding_enter_bps=1.0, funding_band_bps=4.0, funding_cost_bps=5.0)
    sc.accrue_funding("BTCUSDT", 0.0002, funding_time=1, ts=T0)      # enter
    sc.accrue_funding("BTCUSDT", -0.00005, funding_time=2, ts=T0)    # -0.5bp > exit(-3bp): hold
    assert sc.carry_state["BTCUSDT"]["in_carry"] is True            # still holding
    sc.accrue_funding("BTCUSDT", -0.0010, funding_time=3, ts=T0)     # -10bp < exit: now exit
    assert sc.carry_state["BTCUSDT"]["in_carry"] is False
    assert sc.scorecard()["sources"]["funding-carry"]["trades"] == 1  # one entry so far


def test_funding_carry_persists_across_restart(tmp_path):
    sc = _sc(tmp_path, funding_enter_bps=1.0, funding_cost_bps=5.0)
    for ft in range(1, 5):
        sc.accrue_funding("BTCUSDT", 0.0002, funding_time=ft, ts=T0 + timedelta(hours=8 * ft))
    eq = sc.scorecard()["sources"]["funding-carry"]["equity"]
    again = _sc(tmp_path, funding_enter_bps=1.0, funding_cost_bps=5.0)
    assert abs(again.scorecard()["sources"]["funding-carry"]["equity"] - eq) < 1e-6
    assert again.carry_state["BTCUSDT"]["in_carry"] is True          # carry state resumed
    again.accrue_funding("BTCUSDT", 0.0002, funding_time=4, ts=T0)   # stale interval -> ignored
    assert len([r for r in again.scores if r["source"] == "funding-carry"]) == 4


def test_funding_carry_shows_on_scorecard_like_others(tmp_path):
    sc = _sc(tmp_path)
    sc.accrue_funding("BTCUSDT", 0.0002, funding_time=1, ts=T0)
    card = sc.scorecard()["sources"]["funding-carry"]
    assert card["equity_start"] == 10000.0 and "equity" in card and "pct_change" in card
    assert card["hit_rate"] is None                                 # not a prediction -> no hit-rate
    assert card["recent"] and card["recent"][0]["status"] == "carry"


# --------------------------------------------------------------------------- #
# tagged signal streams (candlestick patterns) + per-pattern breakdown
# --------------------------------------------------------------------------- #
def test_tags_are_independent_signal_streams(tmp_path):
    sc = _sc(tmp_path, horizon_s=900)
    # same source+symbol, same side, but DIFFERENT tags -> both logged (not deduped)
    sc.observe("us-ta", "AAPL", "bullish", 100.0, ts=T0, tag="engulfing")
    sc.observe("us-ta", "AAPL", "bullish", 100.0, ts=T0, tag="pin_bar")
    assert len(sc.scores) == 2
    # re-firing the SAME tag+side is deduped
    sc.observe("us-ta", "AAPL", "bullish", 101.0, ts=T0, tag="engulfing")
    assert len(sc.scores) == 2
    # each tag has its own $10k book
    assert ("us-ta", "AAPL", "engulfing") in sc.books and ("us-ta", "AAPL", "pin_bar") in sc.books


def test_by_pattern_breakdown_hit_rate(tmp_path):
    sc = _sc(tmp_path, horizon_s=600)
    # engulfing: bullish then price up at horizon -> correct
    sc.observe("us-ta", "AAPL", "bullish", 100.0, ts=T0, tag="engulfing")
    sc.observe("us-ta", "AAPL", "bullish", 110.0, ts=T0 + timedelta(minutes=11), tag="engulfing")
    # pin_bar: bullish then price DOWN at horizon -> wrong
    sc.observe("us-ta", "NVDA", "bullish", 100.0, ts=T0, tag="pin_bar")
    sc.observe("us-ta", "NVDA", "bullish", 90.0, ts=T0 + timedelta(minutes=11), tag="pin_bar")
    bp = sc.scorecard()["sources"]["us-ta"]["by_pattern"]
    assert bp["engulfing"]["hit_rate"] == 100.0 and bp["engulfing"]["correct"] == 1
    assert bp["pin_bar"]["hit_rate"] == 0.0 and bp["pin_bar"]["wrong"] == 1


def test_by_pattern_excludes_untagged(tmp_path):
    sc = _sc(tmp_path)
    sc.observe("us-ml", "AAPL", "BUY", 100.0, ts=T0)             # no tag
    assert sc.scorecard()["sources"]["us-ml"]["by_pattern"] == {}


def test_tagged_books_persist_across_restart(tmp_path):
    sc = _sc(tmp_path, horizon_s=900)
    sc.observe("us-ta", "AAPL", "bullish", 100.0, ts=T0, tag="engulfing")
    again = _sc(tmp_path, horizon_s=900)
    assert ("us-ta", "AAPL", "engulfing") in again.books
    # dedup state survives: same tag+side does not re-log
    again.observe("us-ta", "AAPL", "bullish", 101.0, ts=T0, tag="engulfing")
    assert len([r for r in again.scores if r.get("tag") == "engulfing"]) == 1


def test_csv_has_distinct_rows_with_expected_columns(tmp_path):
    sc = _sc(tmp_path)
    sc.observe("ta", "BTCUSDT", "bullish", 63000.0, ts=T0)
    sc.observe("ta", "BTCUSDT", "bearish", 62000.0, ts=T0 + timedelta(minutes=1))
    text = (tmp_path / "signal_log.csv").read_text(encoding="utf-8").strip().splitlines()
    assert text[1].split(",") == [T0.isoformat(), "ta", "BTCUSDT", "bullish", "63000.0"]
    assert "bearish" in text[2]


# --------------------------------------------------------------------------- #
# accounting bug guards: $10k base, notional <= book, bounded costs/losses
# --------------------------------------------------------------------------- #
def test_every_source_card_uses_a_single_10k_base(tmp_path):
    sc = _sc(tmp_path)
    # 3 pattern sub-books under ONE source must NOT sum to 3 x $10k ("from $70,000" bug)
    for tag in ("engulfing", "pin_bar", "hammer"):
        sc.observe("crypto-ta", "BTCUSDT", "bullish", 100.0, ts=T0, tag=tag)
    card = sc.scorecard()["sources"]["crypto-ta"]
    assert card["equity_start"] == 10_000.0            # single $10k base, not N x $10k
    assert card["retired"] is True                     # killed predictor -> reference only
    # funding carry is the live agent (not retired)
    assert sc.scorecard()["sources"]["funding-carry"]["retired"] is False


# --------------------------------------------------------------------------- #
# honest DEMO framing: every agent kept live + visible, each with a one-line verdict
# --------------------------------------------------------------------------- #
def test_every_agent_exposes_label_desc_and_demo_flag(tmp_path):
    # all six agents are CONFIGURED (= running, surfaced) regardless of trade history — none
    # stopped or hidden. status() iterates the configs, so they always appear.
    sources = _sc(tmp_path).scorecard()["sources"]
    assert set(sources) == {"us-ml", "us-ta", "crypto-ml", "crypto-ob", "crypto-ta", "funding-carry"}

    # the five rigorously-killed predictors are DEMOS (live technique, NO validated edge);
    # each carries a label + an honest one-line verdict (technique + tested result).
    for s in ("us-ml", "us-ta", "crypto-ml", "crypto-ob", "crypto-ta"):
        c = sources[s]
        assert c["demo"] is True and c["retired"] is True
        assert c["label"]                                  # a human label for the card title
        assert c["desc"] and "tested:" in c["desc"]        # technique + honest verdict

    # representative verdicts (technique named AND the honest outcome)
    assert "machine learning" in sources["us-ml"]["desc"]
    assert "no edge after costs" in sources["us-ml"]["desc"]
    assert "candlestick" in sources["us-ta"]["desc"] and "~50% hit-rate" in sources["us-ta"]["desc"]
    assert "order-book imbalance" in sources["crypto-ob"]["desc"] and "bust" in sources["crypto-ob"]["desc"]

    # funding carry is NOT a demo — the one small real edge, with its own honest verdict
    carry = sources["funding-carry"]
    assert carry["demo"] is False and carry["retired"] is False
    assert carry["desc"] == "delta-neutral funding capture — small real edge"


def test_position_notional_capped_at_the_book():
    # notional is a fraction of the book and never exceeds it, even if trade_pct > 1
    b = PaperBook("crypto-ob", start=10_000.0, trade_pct=5.0)
    assert b.trade_pct == 1.0 and b._notional() == 10_000.0      # capped to the $10k base
    b.set_target("BTCUSDT", 1, 70_000.0)               # full BTC notional would be $70k...
    pos = b.positions["BTCUSDT"]
    assert pos["notional"] <= b.start                  # ...but sized to the book, not $70k
    assert abs(pos["qty"] - 10_000.0 / 70_000.0) < 1e-9  # ~0.143 BTC, not 1 BTC


def test_paperbook_bust_bounds_costs_and_losses():
    # thousands of flat-price flips: costs/losses are bounded ~by the base, not unbounded
    b = PaperBook("crypto-ob", start=10_000.0, trade_pct=1.0, cost_bps_round=10.0)
    for k in range(4000):
        b.set_target("BTCUSDT", 1 if k % 2 == 0 else -1, 100.0)
    assert b.realized_equity() <= 0                    # churned itself bust
    assert b.costs <= b.start + 20.0                   # costs ~bounded at the base (no multiples)
    assert b.trades < 4000                             # bust stops opening new positions


def test_scorecard_display_bounds_a_blown_book(tmp_path):
    # reproduce the observed bug magnitudes, then assert the corrected display is bounded
    sc = _sc(tmp_path)
    sc.observe("crypto-ob", "BTCUSDT", "bullish", 100.0, ts=T0)
    book = sc.books[("crypto-ob", "BTCUSDT", "")]
    book.costs, book.realized = 120_500.0, 13_454.0    # the real churned-out numbers
    card = sc.scorecard()["sources"]["crypto-ob"]
    assert card["equity_start"] == 10_000.0            # $10k base
    assert card["equity"] == 0.0 and card["pct_change"] == -100.0   # lose at most the base
    assert card["costs"] <= 10_000.0 and card["bust"] is True       # costs bounded by the base
