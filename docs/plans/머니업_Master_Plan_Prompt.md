# 머니업 — Master Plan (7 tasks, run ONE AT A TIME)

> **Paste this whole file to CC.**
> Execute the seven tasks **in order, one at a time.** After each task: report results + the
> acceptance-test status, then **STOP and wait for sign-off.** Do **not** start the next task until
> I explicitly say "go." This sequencing is deliberate — several tasks have mandatory human gates and
> later tasks depend on earlier results.

---

## Standing discipline (applies to EVERY task)
- **Grounding:** OCR owns numbers (source of truth); VLM is qualitative, tagged "visual/unverified"; **never fabricate** a number. Every figure in a report/dashboard is real or labelled `pending` — never invented.
- **Returns:** always **net of 0.50% round-trip** (KRX sell-side tax incl.) and **beta+size-adjusted** (excess vs a size-matched control). Reuse `phase1b` machinery; do not re-implement returns. Prices via **pykrx** unless a task says otherwise.
- **Anti-overfitting (any predictive test):** temporal splits only; purged + embargoed walk-forward CV; **sealed holdout**; deflated significance bar (Bonferroni + deflated-Sharpe, print K); **shuffled-label null** floor; point-in-time features (no look-ahead). **Only a one-shot sealed-holdout result may claim an edge — pre-register and STOP before running it.** No tuning on results. The words "edge/PASS/profitable/winning" appear only after a holdout passes.
- **Isolation:** don't disturb the collector, daily report (5:30/6:20), playbook, dashboard, or live demo except where a task explicitly edits them. List every file touched per task.
- **GPU:** coordinate around the daily report + nightly collector. **No 508 re-extraction** unless I approve it.
- **Secrets:** credentials/keys live in `.env`, never in chat or code.

---

## TASK 1 — Chart-reader refinements (extraction)
**Goal:** make the visual reader report what the speaker is *pointing at*, and flag audio↔chart disagreement.
**Do:**
1. **Lock onto the speaker's feature.** Parse audio cues ("lower high/lower low", "below the 50MA", "gapped below") and have fusion surface the **near-term feature the speaker emphasizes** distinctly from the overall multi-month structure — report both, labelled, not blended.
2. **Flag audio↔chart conflict.** Compare audio direction/sentiment vs the VLM chart stance; when they oppose (e.g. Gareth bearish audio vs bullish chart), tag **CONFLICT** (was 0).
3. **Kill raw noise.** Apply the plausibility filter to the **raw `[screen]` token dump** too, not just the grounded levels — no "999.9" runs.
4. **Generic label for non-머니업 creators.** Detect non-머니업 channels → use a neutral "technical approach" label instead of forcing the "머니업 플레이" framing.
**Acceptance (re-run Gareth `2R7Kkjiy0no` + one 머니업 video):** summary reflects Gareth's near-term bearish thesis (or explicitly flags the timeframe split); audio↔chart CONFLICT now raised on Gareth; no 999.9 in the raw dump; non-머니업 label is generic; **머니업 regression intact**. **STOP.**

## TASK 2 — Provenance tables in the Full summary (reporting)
**Goal:** make every line of the Full summary traceable to its source — showing only what was used.
**Do:** add to the Full summary three **per-source tables** — Whisper/audio, OCR/screen, VLM/chart — each listing **only the items actually cited in the summary**, mapped to the summary point + `[mm:ss]` they support. **Exclude every extracted-but-unused item.** Derivation is **computed** (match summary citations to source items), not LLM-written.
**Acceptance:** on a 머니업 sheet + the Gareth sheet, each table contains only referenced items (zero unused); every summary claim traces to a listed source row; counts reconcile. **STOP.**

## TASK 3 — Market-cap / per-ticker slice (analysis · existing data, no re-extraction)
**Goal:** answer "do his calls work better on the majors than on theme stocks?"
**Do:** on the existing corrected Phase 1B call set — (a) **count ex-ante buy calls per ticker**; report n for Samsung `005930`, SK Hynix `000660`, NAVER `035420`, and the top names; (b) slice **net-vs-market returns by market-cap tier** (large / mid / small) and by ticker; (c) only call a tier's result real if **n ≥ 30**.
**Acceptance:** a calls-per-ticker table + a market-cap-tier breakdown (n, mean net, t per tier); honest verdict — majors better / same / underpowered. **STOP.**

## TASK 4 — Final out-of-sample confirmation (analysis)
**Goal:** confirm the buy-call verdict on untouched data + test the contrarian sign OOS.
**Do:** run the corrected long-side scorer on the **sealed 152-video holdout** (`holdout_manifest.json`) → confirm or deny the loss. Test the **contrarian** sign on the holdout (paper, net of cost; flag shortability as unresolved). This is confirmation, **no tuning**.
**Acceptance:** holdout results (n, mean net, t) for the long side + the contrarian paper signal; a clear statement of whether the in-sample finding held OOS. **STOP.**

## TASK 5 — Offense hunt (the edge test) — *the mandatory-gate task*
**Goal:** find a genuine long edge — his powered setups + a model learned from his signals.
**Do:**
- **Inventory** data coverage first (Supabase tables + ticker count; which pykrx endpoints actually return data here).
- **Track A — powered setups:** pool his concrete mechanizable BUY setups by TYPE (breakout-with-volume, 매물대 돌파, OBV+핵심물량, 공매도 매집선 도달, …) into buckets with **n ≥ 30**; score each (beta+size-adj, net cost, deflated bar).
- **Track B — full-feature model:** build the **widest feature set the data supports** (price/return, MA family, momentum oscillators, volume/money-flow, volatility, support/resistance/매물대, relative-strength/regime, investor flows, short-selling, fundamentals, calendar, **plus his-call features**: setup type, first-mention vs repeat, triggers firing). Primary = **conditional selection on his calls** (can the features separate his winners from losers?); secondary = standalone signals on the broad universe. LightGBM + a regularized linear baseline.
- Full anti-overfit discipline above. **Report every model/horizon tried + the shuffled-label floor.**
**Acceptance / HARD STOP:** deliver the data inventory + feature list + Track A/B **CV** results + the **locked pre-registration** for the one-shot holdout test — then **STOP for sign-off. Do NOT run the holdout test or state any edge.** (Precondition for a *tradeable* contrarian: resolve KRX shortability/borrow data.)

## TASK 6 — Kiwoom real-time feed (infra · 5090 box)
**Goal:** live intraday prices for the dashboard.
**Do:** wire the **Kiwoom OpenAPI** on the 5090 (Windows/OCX): real-time current price + key live fields for the watchlist (majors + his theme names). Credentials in `.env` (I provide; never in chat). Expose a clean internal interface (`get_quote(ticker) → price/change%/volume/…`) with login + reconnect handling. **Isolated** — must not disturb the collector or daily report.
**Acceptance:** live quotes for `005930 / 000660 / 035420 / 086520 / 028300` (and a few peers); reconnect/login works; setup documented; isolation confirmed. **STOP.** *(Depends on nothing, but feeds Task 7.)*

## TASK 7 — Live strategy scoreboard (dashboard) — *depends on Tasks 3–6*
**Goal:** one glanceable, auto-updating board of every strategy's real win/lose.
**Do:** add a new section to the 8077 dashboard with two panels —
- **Live stocks:** price + features (Kiwoom real-time + computed indicators) for majors + theme names.
- **Strategy scoreboard:** each registered strategy → **net-vs-market (after costs)**, status, **backtest vs live-forward shown separately**, and a **versioned improvement-delta** ("vs prev"). Driven by a **strategy registry** so new strategies/improvements appear automatically.
**Honest design (required):** every % is net of market + costs; backtest and live-forward are distinct columns; **only mechanized strategies get a number** (vague discretionary ones marked "not scorable"); mostly-red is expected and fine; `pending` shown where not yet measured — **never a fabricated number.**
**Acceptance:** section renders; numbers labelled net-of-market/costs and backtest vs live; the registry drives the rows; adding a strategy/version updates the board; live stocks pull from the Kiwoom feed. **STOP.**

---

## Dependencies & open decision
- **7** needs **6** (feed) and **3–5** (strategy results). **1, 2** are independent — fine to do first.
- Open (not a build): whether to **re-extract the 508** with the improved reader — optional, not needed for any conclusion; decide later.

**Reminder: after each task, report + STOP. Do not chain tasks without sign-off.**
