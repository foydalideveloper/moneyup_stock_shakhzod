# 머니업 Phase 1B — Loss-Decomposition Diagnostic (EXPLORATORY)

> **This is exploratory hypothesis-generation, NOT a verdict. Nothing here establishes a tradeable
> result of any kind.** Outputs are ranked, falsifiable hypotheses to be tested LATER on untouched data.
> No confirmatory test was run.

## Discipline confirmation
- **Holdout sealed BEFORE any analysis.** The most-recent **152 videos** by publish datetime
  (publish ≥ **2026-06-01**) are sealed in `holdout_manifest.json` and were **never loaded, scored, or
  read** in this diagnostic. Analysis ran ONLY on the older **355 videos** (571 scored long-side rows).
- **Returns reuse `phase1b.py`** (`score_call` + `_ohlcv5` + the exact `sign·(gs − β·gi) − 0.50%` formula).
  Nothing re-implemented; every number is comparable to the verdict.
- **Every slice recorded:** **37 slices examined** (full list in `phase1b_diagnostic.json`). Deterministic;
  no LLM in any number. Read-only; wrote only to `qa_audit/diagnostic/`.
- **Market context (real in this dataset):** the index (KODEX 200, 069500) rose **~+280%** over the
  exploratory window — a strong semiconductor-led market. 머니업's picks (battery/bio/small-cap themes)
  fell against it. The per-call β-control uses only the 1–20-day **holding-window** index move, not the
  full trend.

---

## Loss map (D1–D5)

### D1 — Where in time does the money leak? (gap vs drift)  [LONG-GENERIC, n=510]
| component | mean % | t | reading |
|---|--:|--:|---|
| overnight **gap** (prev close → entry open) | **+1.19** | +6.64 | follower overpays only ~1.2% on the open |
| post-entry **drift** (open → 5d, mkt-adj) | **−7.92** | −14.12 | **the loss is the drift — the pick fades** |
| entry (a) next-open → 5d | −7.92 | −14.12 | baseline |
| entry (b) next-close → 5d | −7.49 | −14.90 | no help |
| entry (c) open+1 → 5d | −7.90 | −15.30 | no help |
| entry (d) publish-day close → 5d | +17.14 | — | **n=1, ignore** (only 1 call published intraday) |
**Answer:** the loss is **not** the gap/fill (only +1.2%) — it is the **post-entry drift**; the pick itself
fades. No entry-timing variant rescues it.

### D2 — Holding-horizon curve (market-adjusted, ±8% barrier ignored)
| h (td) | LONG-GENERIC mean % | t | DIP-BUY mean % | t |
|--:|--:|--:|--:|--:|
| 1 | −2.50 | −8.37 | −3.81 | −4.24 |
| 2 | −3.67 | −9.96 | −5.99 | −3.67 |
| 3 | −5.22 | −12.37 | −7.07 | −4.28 |
| 5 | −7.92 | −14.12 | −7.28 | −2.99 |
| 10 | −14.32 | −20.82 | −16.66 | −7.06 |
| 20 | −30.78 | −34.87 | −34.43 | −13.43 |
**Answer:** **negative at EVERY horizon**, monotonically worsening, for both segments. **No early positive
window** — it is a robust fade from day 1 (already −2.5% at h=1).

### D3 — First-mention vs repeat, and the "already-run" test  [LONG-GENERIC]
| slice | n | mean % | t |
|---|--:|--:|--:|
| first-mention (net, barrier) | 26 | −4.46 | −2.41 |
| repeat 2+ (net, barrier) | 545 | −6.73 | −15.10 |
| first-mention LG (open→5d) | 23 | −3.93 | −1.53 |
| repeat LG (open→5d) | 487 | −8.11 | −14.14 |
| prior-runup **low** tercile → fwd5 | 170 | −8.17 | −9.86 |
| prior-runup **high** tercile → fwd5 | 170 | −7.81 | −6.66 |

Prior-runup regression (fwd5 on trailing-20d return): slope ≈ 0.004, **corr ≈ 0.009** (≈ zero).
**Answer:** **no "already-run" effect** — forward return is independent of prior run-up. First-mentions are
*less* negative (−3.9%, n=23, |t|<2, below floor) than repeats (−8.1%) — but **still negative**. Repeats
(the recurring boilerplate "defended on the way down") are the worst.

### D4 — Concentration & regime  [all 571 scored rows]
| slice | n | mean % | t |
|---|--:|--:|--:|
| headline (all) | 571 | −6.63 | — |
| excl top-3 contributors | — | −4.69 | — |
| excl top-10 contributors | — | **−2.40** | — |
| battery (2차전지) names | 68 | −7.53 | −6.21 |
| non-battery names | — | −6.4× | — |

Top contributors (sum net %): **전진건설로봇 −1792% over 201 calls** (recurring boilerplate), 에코프로 −334%/39,
삼천당제약 −294%/40, 알테오젠 −263%/27, HLB −249%/34, 카카오 −167%/11. **Answer:** the loss is **broad** — even
after removing the worst 10 names it is **still −2.4%** — but **amplified by a few disaster names** (one
ticker, 전진건설로봇, is ~25% of total negative PnL) and the **2차전지 regime** (−7.5%). A different-period OOS
test is therefore decisive for separating "broad fade" from "one regime / one name."

### D5 — Is the "opposite" even tradeable? (cost-aware contrarian)
| slice | n | mean % | t |
|---|--:|--:|--:|
| mirror short, all, **gross** → 5d | 571 | +7.31 | +13.66 |
| mirror short, all, **net 0.50% rt** → 5d | 571 | +6.81 | +12.73 |
| mirror short on **shortable subset** (net + borrow) | **0** | — | — |
**Answer (with a hard data caveat):** the mirror (short the calls, cover ~5d) is large on **paper**
(+6.8% net of the 0.50% round-trip). **BUT shortability could NOT be determined:** in this environment the
KRX market-cap / ticker-list / shorting-balance endpoints are unavailable (only the OHLCV endpoint
responds), so `% shortable = undeterminable` (the proxy returned 0). Strong qualitative prior: Korea's
short-sale universe is restricted (~KOSPI200 + KOSDAQ150 ≈ 350 borrowable names; retail access narrower),
and 머니업's picks are dominated by small/mid-cap theme names (전진건설로봇, 알테오젠, HLB, 삼천당제약 …) that are
**mostly outside the borrowable universe**. **The contrarian is therefore a paper signal until shortability
+ borrow are established.**

---

## Ranked hypotheses (EXPLORATORY — in-sample only; none established)

**H1 — Cost-aware contrarian on the SHORTABLE subset (primary candidate, shortability-gated).**
*Rule:* short at next-session open the LONG-GENERIC/DIP-BUY calls **restricted to retail-shortable names**,
cover at a fixed horizon; judge net of 0.50% round-trip **and actual borrow**, β+size-adjusted.
*In-sample (whole set, paper):* +6.8% net/5d (t=12.7) — but on an **undetermined** shortable subset.
*Why ranked #1:* it is the only sub-pocket with a large, horizon-robust in-sample sign. *Why not more:* its
entire viability hinges on the **unanswered** shortability/borrow question — likely a small subset.

**H2 — Contrarian horizon/decay (secondary).**
*Rule:* same mirror, but compare exit at h ∈ {1, 3, 5} to trade off the day-1 drop (gross +2.5%/1d) against
borrow that grows with horizon. *In-sample:* −2.5%/1d → −7.9%/5d long ⇒ mirror grows with h; shortest
horizon minimises borrow but the 0.50% round-trip eats more of a small move. *Purpose:* find the
horizon that best survives realistic short costs **on the shortable subset only.**

**No long-side hypothesis.** The long side loses at **every** horizon, under **every** entry variant,
**independent of prior run-up**, and even after removing the worst 10 names (−2.4%). First-mentions are
less-bad but still negative and below the n-floor. **There is no candidate long pocket worth an OOS test.**

---

## Draft pre-registrations (DO NOT RUN NOW — for untouched data only)

**PR-1 (for H1).** *Precondition:* first build the retail-shortable universe at each call date (KRX
borrowable list + 공매도 과열종목 restriction flag + a real borrow-fee source) — none obtainable in this
environment. *Then:* segment = {shortable LONG-GENERIC + DIP-BUY}; entry = short at next-open; exit = h=5
(and h=1,3 as the H2 grid); β+size-adjusted; **net of 0.50% round-trip + per-name borrow**; deflated bar
**t ≥ 4.242** (reuse K from the verdict); **n-floor ≥ 30 on the shortable subset**. *Test on:* the **sealed
30% holdout** (152 videos, publish ≥ 2026-06-01) AND future uploads. *Kill criterion:* if < ~30 calls are
shortable, or net-of-borrow ≤ 0, abandon.

**PR-2 (for H2).** Same as PR-1 but pre-register the **single** horizon (choose h before seeing holdout
returns) to avoid horizon mining; report all three but gate the decision on the pre-chosen h.

**PR-3 (regime control).** Repeat PR-1 with the headline recomputed **excluding 전진건설로봇** and
**excluding 2차전지 names**, to confirm any holdout result is not one-name / one-regime. Pre-register both the
full and the trimmed estimates.

---

## Honest framing (§4)
Primarily **(i): the long-side loss is broad and robust at all horizons** — no long pocket; it is the pick's
post-entry drift, not the fill, not prior run-up, and survives removing the worst names. Plus a
**conditional (iii): a candidate contrarian** with a large *paper* in-sample sign that **cannot be called
tradeable** until retail shortability + borrow are established (undeterminable here) — most picks are
likely non-shortable. **No tradeable result of any kind is claimed; this only tells us where to look next,
with proof, on untouched data.**
