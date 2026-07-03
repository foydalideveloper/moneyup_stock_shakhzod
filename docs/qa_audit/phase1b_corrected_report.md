# Phase 1B — Corrected falsification re-run (REPORT)

_Research / mock only — NOT a trading signal. Pre-registered spec (§1.1–1.4), fixed BEFORE returns, not
re-tuned on results. Human sample-audit signed off prior to computing any returns._

Price source: pykrx (full KRX). Index proxy: KODEX 200 (069500). Beta- AND size-adjusted, net of 0.50%
round-trip. **K = 21** tests (segments × horizons + Test B) → Bonferroni **deflated t-bar = 4.242** (base 3.5).
Floor: n ≥ 30 per segment else INCONCLUSIVE.

## Call set
1,884 ex-ante calls → **944 scored** / 940 excluded-but-counted. Entered **818** · non-triggered 126.
Inferred-level rows (symmetric ±8% default barrier, no spoken target/stop): **801 / 818** — flagged.

## Per-segment result (§1.2 uniform gate: PASS ⇔ mean net > 0 AND t ≥ +4.242; t ≤ −4.242 ⇒ WRONG-SIGNED)

| segment | n | mean net % | t | hit | control %¹ | verdict |
|---|--:|--:|--:|--:|--:|---|
| **LONG-GENERIC** | **741** | **−5.60** | **−15.56** | 0.39 | +3.65 | **WRONG-SIGNED** |
| **DIP-BUY** | **47** | **−7.48** | **−7.46** | 0.21 | +2.39 | **WRONG-SIGNED** |
| SCHEDULE | 20 | −0.39 | −0.23 | 0.65 | +2.90 | INCONCLUSIVE (n<30) |
| ROTATION | 8 | −3.95 | −2.67 | 0.25 | −0.55 | INCONCLUSIVE (n<30) |
| BREAKOUT | 0 | — | — | — | — | INCONCLUSIVE (never triggered) |
| BEARISH | 2 | +20.48 | 11.07 | 1.00 | +12.98 | INCONCLUSIVE (n<30, accepted) |

¹ control % = the size-matched market (β-expected) return over the same window — what a passive index
position earned. His long calls returned **−5.6% net while the market made +3.6%**.

## Test B — event study (CAAR t0…t+5, event-clustered SE)
n=740 events (14 clusters) · **CAAR(t+5) = −7.36%** · t = −3.5 · drift = **reversal** · bar 4.242 →
**no edge** (negative but |t| below the deflated bar) · pass=False.

## §5 DECISION
**WRONG-SIGNED — LONG-GENERIC, DIP-BUY significantly AGAINST the stated direction.** The two powered
long segments lose, with very large negative t-stats; Test B confirms a negative (reversal) drift. Per the
locked rule: this is a **contrarian lead — it needs its own out-of-sample test and is NOT traded from this
run**. No segment and Test B clear the deflated bar with the correct (positive) sign. The method, as he
states his calls, has **no demonstrated positive edge**; entering his long calls at the next open and
holding has been significantly unprofitable vs a size-matched market position over this sample.

### Caveats (do not over-read)
- **801/818 rows use the symmetric ±8% default barrier** (he rarely gives an explicit target/stop). The
  magnitude depends on that default; the SIGN (negative) is robust across the 1/5/20-day horizons + Test B.
- Entry is next-session open after publish (§1.4). The strong negative is consistent with calling stocks
  *after* an intraday run, then fading — but a contrarian strategy needs its own pre-registered OOS test.
- SCHEDULE/ROTATION/BREAKOUT/BEARISH are below the n≥30 floor → INCONCLUSIVE, not evidence either way.

---

## What changed vs the original run (call-set delta + the verdict-flipping fixes)

**Why the verdict differs — and why it is NOT tuning:** two pre-registered sign-convention fixes, plus the
classification corrections from the QA audit. Each was applied before returns and would be applied
identically regardless of the result's sign.

1. **Call set (classification):**
   - **+732 LONG-GENERIC** restored (old scorer DROPPED plain longs as UNCLASSIFIED → the primary test had
     almost no long sample). This is now the dominant, powered segment.
   - **SCHEDULE date-gated** — 374 boilerplate "스케줄 매매" misfires routed away; only 20 real dated events remain.
   - **§1.1 AVOID split** → BEARISH=2 / HOLD-WAIT-CASH=300 / TRIM=72 (the old single AVOID-as-short bucket
     conflated wait/cash/trim with genuine down-calls; there are essentially **no** genuine down-calls).
   - **Tier-1 removed** (170): holdings-substring avoids, market-structure noise, ex-post recaps, negations.
   - **Tier-3 (141) excluded** for human review (direction/ticker ambiguous).
2. **§1.2 verdict-gate sign bug FIXED** — the old gate flipped the AVOID segment's sign (`t ≤ −bar` for a
   "pass"), so a wrong-signed result could read as an edge. Now uniform: PASS ⇔ mean net > 0 AND t ≥ +bar
   for every segment; t ≤ −bar ⇒ WRONG-SIGNED. No per-segment sign flip.
3. **Test B sign bug FIXED** — the old Test B passed on `abs(t) ≥ 3.5`, so a **negative** CAAR (−7.36%)
   was being reported as "PASS — Test B clears the deflated bar." Now Test B uses the same §1.2 convention
   (positive CAAR in his favour, at the deflated bar). The headline flips from a **false PASS** to the
   honest **WRONG-SIGNED**.
4. **Symmetric ±8% default barriers** (was the biased −5/+8), so neither side is structurally favoured.

Both sign fixes are pinned by the test suite (5b/5e/5f/5g/5h) so they cannot silently regress.
