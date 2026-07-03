# CLAUDE.md — moneyup_stock_shakhzod

> Orientation for any Claude (Cowork / Claude Code) working in this repo. **Read this first.**

## ⚠️ Project status — READ BEFORE DOING ANYTHING

**This project is COMPLETE and ARCHIVED.** It was an honest, exhaustive research effort, and its
central question has been **answered — negatively**:

> **머니업's stock-analysis method does not produce a general, tradeable edge.** This was tested
> every reasonable way — his picks, his specific setups, a machine-learning model on his conditions,
> his market-cap slices, his defensive timing, the contrarian (fading him), and his taught technical
> rules — all **net of costs and market**, **in-sample and out-of-sample**. Every avenue came back
> no-edge or (for the contrarian) not tradeable. The evidence is in `docs/qa_audit/`.

**Do NOT re-open the edge hunt as if it's unsolved.** The rigorous, pre-registered, out-of-sample
tests are done and logged. Re-running them "to check" is re-treading proven ground. A genuinely
*new* hypothesis or data source could be tested with the existing machine — but the 머니업-based edge
question is closed.

All scheduled automation (collector, daily report/email, dashboard) is **disabled** — the code runs
nothing on its own.

**This is a research / mock system. It is NOT investment advice and must never place trades.**

## What this project is

An AI pipeline that learned from the Korean YouTube stock-analysis channel **머니업**: it watches each
video (what he *says* + what he *shows on screen*), turns it into structured, **grounded** fact sheets,
distills his method into a "playbook," and then **falsification-tests whether that method makes money.**
It also generated a daily grounded report + email from his videos.

What remains valuable: (a) the **machine** — a reusable extraction + falsification + validation harness
that works on any strategy or source; (b) the **honest record** of what his method does and doesn't do;
(c) his content's **educational** value (not tradeable signals).

## The key finding (all returns β+size-adjusted, net of 0.50% cost, OOS-validated)

- **Raw buy calls** — significantly negative (wrong-signed). In-sample −6.7%, OOS −2.9%.
- **Specific setups** (dip-buy, 수급, 공매도 매집선, 눌림목) — all significantly negative.
- **Conditional ML model** (which of his calls win?) — could not beat a shuffled-label null.
- **Mega-cap calls** (삼성/네이버/하이닉스) — negative out-of-sample.
- **Defensive timing** ("go to cash" calls) — no predictive signal.
- **Contrarian** (fade his calls) — positive in-sample, but the effect lives in names retail can't
  short; on the shortable subset it collapses OOS. Paper-only, not tradeable.
- **Taught technical rules** as general strategies — 14/15 no edge; the lone survivor (breakout-on-volume)
  failed its pre-registered one-shot holdout.

**Conclusion: no general, tradeable edge. His value is teaching, not signals.**

## Why the results are trustworthy (keep this discipline if you extend it)

Every scoring test used: β + size adjustment; net-of-cost; **deflated significance** (Bonferroni +
deflated-Sharpe over K = tests); **sealed one-shot holdouts**; **pre-registration with locked hashes**
(no peek-then-adjust); **shuffled-label / random-entry nulls**; point-in-time features (no look-ahead);
unit-test-gated scorers. Prices come from `pykrx` (real KRX history), independent of the video OCR.

Extraction is **grounded**: OCR is the source of truth for numbers; the vision model (Qwen3-VL) is
visual/unverified and **never fabricates a number**; audio cross-checks; every fact is tagged
AGREE / AUDIO-ONLY / VIDEO-ONLY / CONFLICT.

## Architecture / code map

- `moneyup_advisor/` — extraction + analysis: video → Whisper (audio) + PaddleOCR (screen) + Qwen3-VL
  (chart) → `fuse.py` (cross-check into one timeline) → `factsheet.py` (grounded fact sheet) →
  `digest.py` / summaries (Gemini). `overnight.py` = nightly collector; `playbook.py` = distilled method;
  `phase1b.py` = falsification scorer; `dashboard/` = the 8077 dashboard + scoreboard; `live/` =
  "watch-it-extract" demo (port 8078); `kiwoom_feed.py` = Kiwoom REST real-time quotes.
- `tagent/` — daily report generator: `news/youtube_report.py`, `news/email_report.py`,
  `news/report_render.py`, `news/report_prices.py`, `report.py`, `kiwoom_report.py`.
- `scripts/` — entry points: `run_youtube_batch.py` (transcribe), `run_daily_email.py` (report + Supabase
  + email), and analysis scripts (`phase1b_*.py`, `taught_rules_backtest.py`, `defensive_timing.py`,
  `contrarian_proxy.py`, `factsheet_qa_audit.py`).
- `tests/` — unit tests (report + scoring logic).
- `docs/qa_audit/` — **the analysis record**: result JSONs, reports, and locked pre-registrations. The
  evidence behind the finding above.
- `docs/plans/` — planning / brief docs.
- `.env.example` — required environment keys (names only).

## Data, secrets, running it

- **Secrets** live in a local `.env` (Kiwoom keys, Gmail app password, Supabase, Gemini/EODHD/YouTube).
  `.env` is gitignored and was **never committed** — see `.env.example` for the key names. Never commit
  real secrets; rotate any that are ever exposed.
- **Data** (fact sheets, transcripts, frames, clips, model files) is gitignored, kept **locally only**.
  The source videos are **copyrighted** — do not commit or redistribute them.
- **Running:** extraction needs a **GPU** (Whisper + PaddleOCR + Qwen3-VL via Ollama); analysis is CPU +
  pykrx. Populate `.env` from `.env.example`, then use the scripts in `scripts/`. Automation is
  intentionally disabled — nothing runs unless you run it.

## What NOT to do

- Don't treat 머니업's calls as tradeable signals — they aren't (proven). Stays **descriptive / mock**;
  never wire it to place orders.
- Don't re-run the closed edge-hunt as if open (see status).
- Don't commit `.env`, secrets, data, media, or model weights.
- Don't weaken the discipline (no un-pre-registered result-tuning, no peeking at holdouts).

## Provenance

Fresh clean single-history repo (the older 127-commit dev history is backed up locally, outside this
repo). Mirrored identical to two **private** repos: `tripleh-aiteam/moneyup_stock_shakhzod` and
`foydalideveloper/moneyup_stock_shakhzod`.
