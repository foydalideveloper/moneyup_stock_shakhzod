"""Static guards on the dashboard's interactive SVG chart.

Two rendering bugs have bitten this chart:
  1. `fill="var(--up)"` — CSS var() does NOT resolve in SVG presentation
     attributes, so candles drew invisible.
  2. `y2=${yy}/>` — an UNQUOTED interpolated coordinate swallowed the
     self-closing slash, yielding `y2="6/"` which the browser rejects
     ("Expected length"), blanking the chart.

These guards inspect the chart engine source so a regression fails the suite
even without a browser. Full render-level checks — crosshair price correctness,
zoom/pan, pivot dots, alerts feed, coordinate validity, console errors — live in
scripts/verify_chart_browser.mjs (real DOM + dispatched events via node + jsdom).
"""

import re
from pathlib import Path

HTML = (Path(__file__).resolve().parent.parent
        / "tagent" / "static" / "dashboard.html").read_text(encoding="utf-8")


def _chart_engine_src() -> str:
    i = HTML.index("function createChart")
    j = HTML.index("// ---------- TA explanation line", i)
    return HTML[i:j]


# --------------------------------------------------------------------------- #
# the two historical rendering bugs
# --------------------------------------------------------------------------- #
def test_no_css_var_in_svg_attributes():
    assert "var(--" not in _chart_engine_src()


def test_every_interpolated_coordinate_is_quoted():
    # bug #2: an unquoted `attr=${...}` lets `/>` glue onto the value.
    unquoted = re.findall(r'[\w-]+=\$\{', _chart_engine_src())
    assert unquoted == [], f"unquoted interpolated attributes: {unquoted}"


def test_no_attribute_value_glued_to_self_close_slash():
    assert re.search(r'\$\{[^}]*\}\s*/>', _chart_engine_src()) is None


def test_hex_color_constants_defined():
    for c in ("C_UP", "C_DOWN", "C_SUP", "C_RES", "C_BRK"):
        assert re.search(rf"{c}\s*=\s*'#", HTML), f"{c} hex constant missing"


# --------------------------------------------------------------------------- #
# interactivity wired (crosshair, zoom/pan, expand/reset)
# --------------------------------------------------------------------------- #
def test_crosshair_and_tooltip_wired():
    src = _chart_engine_src()
    assert "crosshair" in src and "Cursor price" in src
    assert "addEventListener('mousemove'" in src
    assert 'id="cxTip"' in HTML and 'id="usTip"' in HTML


def test_zoom_and_pan_wired():
    src = _chart_engine_src()
    assert "addEventListener('wheel'" in src and "zoomAt" in src
    assert "addEventListener('mousedown'" in src        # drag to pan
    assert "invY" in src and "giAt" in src              # price-at-cursor + bar-at-cursor


def test_reset_and_expand_buttons_wired():
    assert 'id="usReset"' in HTML and 'id="usExpand"' in HTML
    assert 'id="cxReset"' in HTML and 'id="cxExpand"' in HTML
    assert "usChart.reset()" in HTML and "usChart.expand()" in HTML
    assert ".expanded" in HTML                          # fullscreen style


# --------------------------------------------------------------------------- #
# TA transparency: pivots, line detail, stamp
# --------------------------------------------------------------------------- #
def test_pivot_dots_and_line_detail_rendered():
    src = _chart_engine_src()
    assert "pivdot" in src                              # swing-pivot markers
    assert "ta.pivots" in src
    assert "trendline · pivots" in src                  # hover detail for lines


def test_candle_patterns_drawn_and_broken_down():
    src = _chart_engine_src()
    assert "ta.candle_patterns" in src and "cpat" in src    # pattern markers + labels on chart
    assert "by_pattern" in HTML and "by pattern" in HTML    # per-pattern breakdown in scorecard


def test_ta_updated_stamp_and_explanation_wired():
    assert "function taStamp" in HTML and "TA updated" in HTML
    assert 'id="usTAstamp"' in HTML and 'id="cxTAstamp"' in HTML
    assert "function taExplain" in HTML
    assert "/ta?market=us" in HTML and "/ta?market=crypto" in HTML


# --------------------------------------------------------------------------- #
# alerts feed
# --------------------------------------------------------------------------- #
def test_alerts_feed_wired():
    assert 'id="alerts"' in HTML
    assert "function renderAlerts" in HTML and "/alerts?market=us" in HTML


def test_news_panel_is_safety_light_only_list_removed():
    assert 'id="news"' in HTML and "function renderNews" in HTML       # panel + safety light kept
    assert "SAFETY" in HTML and "News SAFETY" in HTML
    # the old rule-based news LIST is gone (alerts table no longer rendered in renderNews)
    assert "(d.alerts||[]).slice" not in HTML
    assert "single source of truth" in HTML.lower()                    # consolidated into feed + briefing


def test_live_order_execution_panel_wired():
    assert 'id="orders"' in HTML and "function renderOrders" in HTML and "/orders" in HTML and "refreshOrders" in HTML
    low = HTML.lower()
    assert "live order execution" in low and "주문번호" in HTML
    assert "o.status" in HTML and "o.ord_no" in HTML and "time (kst)" in low
    assert "mock" in low and "live" in low                             # MOCK vs LIVE badge
    assert "class=\"afeed\"" in HTML or 'class="afeed"' in HTML


def test_youtube_report_generate_section_wired():
    low = HTML.lower()
    assert "유튜브 리포트 생성" in HTML                                  # Korean-first section title
    # language toggle (한국어 default / English) -> lang
    assert 'name="ytLang"' in HTML and 'value="ko"' in HTML and 'value="en"' in HTML
    assert "한국어" in HTML and "English" in HTML
    # window button + end-time picker -> /youtube_report/window
    assert 'id="ytGenWindow"' in HTML and 'id="ytEnd"' in HTML
    assert "/youtube_report/window" in HTML and "function ytGenWindow" in HTML
    # video url input + button -> /youtube_report/video
    assert 'id="ytUrl"' in HTML and 'id="ytGenVideo"' in HTML
    assert "/youtube_report/video" in HTML and "function ytGenVideo" in HTML
    # preview + download buttons + spinner; top-20 watchlist multi-select
    assert 'id="ytPreview"' in HTML and 'id="ytDocx"' in HTML and 'id="ytPdf"' in HTML
    assert 'id="ytSpin"' in HTML and "ytWatchlist" in HTML and "YT_GIANTS" in HTML


def test_youtube_fetch_log_panel_removed():
    # the YouTube Fetch Log UI panel + its endpoint wiring were removed (log file is the source of truth)
    assert 'id="ytaudit"' not in HTML
    assert "renderYtAudit" not in HTML and "refreshYtAudit" not in HTML
    assert "/youtube_audit" not in HTML
    assert "YouTube Fetch Log" not in HTML


def test_last_updated_stamp_on_each_chart():
    assert 'id="usStamp"' in HTML and 'id="cxStamp"' in HTML
    assert "nowHHMMSS" in HTML


# --------------------------------------------------------------------------- #
# signal scorecard panel
# --------------------------------------------------------------------------- #
def test_scorecard_panel_wired():
    assert 'id="scards"' in HTML
    assert "function renderScorecard" in HTML and "/scorecard" in HTML
    assert "$10,000 paper base" in HTML                 # paper P&L base label
    assert "hit-rate" in HTML and "costs paid" in HTML  # required per-source stats
    assert "0 signals yet" in HTML                      # empty-source placeholder


def test_scorecard_reframed_as_honest_demo_panel():
    # reframed from a dimmed "retired graveyard" to an explicit, honest live-DEMO panel
    assert "Live technique DEMOS" in HTML
    assert "DEMO — live technique (no validated edge)" in HTML     # the new card badge text
    assert 'class="sdesc"' in HTML and "${s.desc}" in HTML         # per-card one-line verdict
    assert "Technique demos — each runs LIVE on paper" in HTML     # honest group explainer
    assert ".scard.demo" in HTML and 'class="scard${s.demo?\' demo\':\'\'}"' in HTML  # accent, not dim
    # the old dimmed-graveyard framing is gone
    assert "Retired — no edge" not in HTML and "opacity:.6" not in HTML


def test_funding_live_panel_wired():
    assert 'id="fundlive"' in HTML
    assert "function renderFundingLive" in HTML and "/funding_live" in HTML
    assert "annualized" in HTML and "headroom" in HTML and "bps/8h" in HTML


def test_live_feed_panel_wired():
    assert 'id="feed"' in HTML
    assert "function renderFeed" in HTML and "/feed" in HTML and "refreshFeed" in HTML
    low = HTML.lower()
    assert "live news feed" in low and "newest-first" in low
    assert "feedSince" in HTML and "since=" in HTML                 # NEW marking via since watermark
    assert "no link" in low                                         # grounded: no link -> no item
    # times DISPLAYED in KST (Asia/Seoul), not UTC
    assert "detected_kst" in HTML and "published_kst" in HTML and "function kstStamp" in HTML
    assert "asia/seoul" in low and "kst" in low


def test_daily_briefing_panel_wired():
    assert 'id="briefing"' in HTML
    assert "function renderBriefing" in HTML and "/briefing" in HTML and "refreshBriefing" in HTML
    low = HTML.lower()
    # the 4 reports + breaking + cited links + YouTube mm:ss jump-links are rendered
    assert "daily briefing" in low and "breaking" in low
    assert "it.deeplink" in HTML and "it.timestamp_mmss" in HTML   # youtube jump-links
    assert "distinct_from_kiwoom" in HTML and "no_coverage" in HTML
    # honest framing on-screen
    assert "정보용" in HTML or "출처 인용" in HTML
    # breaking banner readability: light bg + dark text + colored macro/war/geopolitical tag chips
    assert "#fff3cd" in HTML and ".btag" in HTML
    assert ".btag.war" in HTML and ".btag.macro" in HTML and ".btag.geopolitical" in HTML
    assert "function" in HTML and "btag" in HTML            # tag chip rendered per breaking item
    # YouTube briefing shows the insight SUMMARY (not just a caption fragment)
    assert "it.summary" in HTML or "a.summary" in HTML or "summary" in HTML


def test_media_briefing_panel_wired_and_honestly_labeled():
    assert 'id="media"' in HTML
    assert "function renderMedia" in HTML and "/media" in HTML and "refreshMedia" in HTML
    low = HTML.lower()
    # the honest framing must be on-screen: awareness only, explicitly NOT a trading signal
    assert "awareness only" in low and "not a trading signal" in low
    assert "not fed into any trading/halt logic" in low
    # grounded, deep-linked rendering: quote + jump-link + source, nothing invented
    assert "a.deeplink" in HTML and "a.quote" in HTML and "a.timestamp_mmss" in HTML
    assert "grounded" in low and "no source" in low and "jump" in low


# --------------------------------------------------------------------------- #
# timeframe selector + crypto $value column
# --------------------------------------------------------------------------- #
def test_timeframe_selector_wired():
    assert 'id="usTf"' in HTML and 'id="cxTf"' in HTML
    for tf in ("1m", "10m", "30m", "1h", "1d"):
        assert f'value="{tf}"' in HTML
    assert "&interval=" in HTML and "usInterval" in HTML and "cxInterval" in HTML
    assert "function createChart" in HTML               # chart redraws on new data


def test_orderbook_dollar_value_column_wired():
    assert "function usd" in HTML
    assert "usd(p*q)" in HTML                           # order-book ladder value
    assert "usd(x.price*x.qty)" in HTML                 # vanished levels value


def test_per_symbol_filter_wired():
    assert 'id="scSymA"' in HTML and 'id="scSymB"' in HTML   # filter on both sections
    assert "function setScSym" in HTML and "scSymbol" in HTML
    assert "/alerts?market=us&limit=60&symbol=" in HTML      # alerts filtered by symbol
    assert "/scorecard?symbol=" in HTML                      # scorecard filtered by symbol
    assert 'value="all"' in HTML                             # "all" toggle
