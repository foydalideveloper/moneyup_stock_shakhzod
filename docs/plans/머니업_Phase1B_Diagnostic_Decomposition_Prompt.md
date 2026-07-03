# 머니업 Phase 1B — Loss-Decomposition Diagnostic (EXPLORATORY)

> **Paste this whole file to CC.**
> This is an **exploratory diagnostic, not a verdict.** It may **not** claim any trading edge.
> Its only job: find exactly **where the −5.6% is made and lost**, and output **ranked, testable
> hypotheses** — each with a draft pre-registration to be run **later, on untouched data.**

---

## 0. Why this exists

The corrected Phase 1B re-run found 머니업's buy-calls **WRONG-SIGNED**:
LONG-GENERIC n=741, **−5.60%** net, t=−15.56 (vs **+3.65%** size-matched market control);
DIP-BUY n=47, **−7.48%**, t=−7.46; Test B CAAR(t+5) = **−7.36%**.

The fixes were verified twice (43/43 tests, all-zero regression), and **removing bugs only made the
result more negative** — so this is **NOT** a hunt for a bug that's hiding a win. It is a decomposition
to understand the loss and to find whether any **sub-pocket** — a timing/horizon window, first-mention
longs, or a cost-aware contrarian — is worth a **separate, pre-registered, out-of-sample test.**

### Discipline (non-negotiable)
- **Exploratory ≠ confirmatory.** Everything here is in-sample hypothesis generation. **Nothing here
  establishes an edge.** The words **"edge" / "PASS" / "profitable" / "winning"** must not appear in any
  conclusion — only *"hypothesis worth an OOS test."*
- **Seal a holdout NOW.** Before any analysis, reserve the **most recent 30% of videos by publish date**
  as a **sealed holdout**. Do not read, plot, or fit on it. It exists so a later confirmatory test has
  untouched data.
- **Report every slice you try** (good *and* bad) and the **total count of slices examined**, so the
  multiple-testing surface is visible and nothing is cherry-picked.
- **No re-running until clean.** Reuse `phase1b.py`'s entry/exit/return/cost machinery so every number
  is comparable to the verdict; do not re-implement returns.

---

## 1. Sources of truth / scope

- `moneyup_advisor/phase1b.py` — **reuse** its entry/exit/return/cost/beta-size functions.
- `data/_moneyup_advisor/qa_audit/phase1b_callscore.json` — per-call scored results (corrected classifier). **Start here.**
- `data/_moneyup_advisor/qa_audit/phase1b_corrected_report.md` — the segment verdict.
- `머니업_Phase1B_Corrected_Amendment.md` — locked rules (entry next-open, 0.50% round-trip cost incl. KRX sell-side tax, beta+size adjustment, deflated bar).
- Prices: **pykrx** (same source the scorer uses).
- **Read-only.** Do not modify fact sheets, collector, daily report, playbook, dashboard, or live demo.
  No GPU. No video downloads. **Deterministic** (no LLM in any number).
  Write outputs only to `data/_moneyup_advisor/qa_audit/diagnostic/`.

---

## 2. The five decompositions

### D1 — Where in time does the money leak? (entry timing)
For every entered call, split the outcome into:
- **gap component** = the move from the last close he is reacting to (before publish) → the **entry open**
  (what a follower *cannot* capture), and
- **post-entry drift** = entry open → each horizon (what a follower *does* experience).

Then recompute mean net return under alternative entry points: **(a)** next-open *[baseline]*,
**(b)** next-session close, **(c)** open + 1 day, **(d)** publish-day close (if published during market hours).
**Question:** is the loss mostly the **gap** (the crowd front-runs you → bad fill, his pick may be fine) or
the **drift** (the pick itself fades)? Report the split with n and t for each entry variant.

### D2 — The holding-horizon curve
Ignore the ±8% barrier. Compute **market-adjusted** mean net return at fixed horizons
**h ∈ {1, 2, 3, 5, 10, 20}** trading days from next-open entry. Tabulate mean net, t, n at each h,
**separately for LONG-GENERIC and DIP-BUY.**
**Question:** is there an **early window that is positive** (fast momentum to exit quickly) before it turns
negative, or is it negative at **every** horizon (robust fade)?

### D3 — First-mention vs repeat, and the "already-run" test
- Group calls by ticker, order by publish date, tag occurrence index (1st mention, 2nd, …). Compare mean
  net for **first-mentions** vs **repeats (2nd+)**.
- Compute each stock's **prior run-up** = trailing 20-trading-day return **before** the call. Regress the
  forward 5-day market-adjusted return on prior run-up.
**Question:** does he call stocks **after they've already run** (high prior run-up → worse forward return)?
Are **first** calls (before the run) different from **repeat** calls (defending it on the way down)?

### D4 — Concentration & regime
Decompose total PnL contribution by **ticker**, **sector**, and **calendar month**. Show the **top-10 names
by absolute contribution**; recompute the headline result **excluding the top-3 and top-10** contributors.
Tag the market regime over the sample (index trend; any sector crash, e.g. 2차전지).
**Question:** is the −5.6% **broad**, or driven by a few disaster names / one regime (which would make the
OOS test in a different period decisive)?

### D5 — Is the "opposite" even tradeable? (cost-aware contrarian)
Compute the **mirror** strategy (short at next-open, cover at horizon/barrier) and report **gross** vs
**net of realistic Korean short costs**:
- **borrow fee** (annualized; flag hard-to-borrow / non-borrowable names),
- **0.50%+ round-trip** incl. sell-side tax,
- **shortability flags**: how many tickers were **retail-shortable at all** (most KOSDAQ small-caps are not),
  and how many sat on the KRX **공매도 과열종목** (short-overheating) restriction list at the call date
  (flag where determinable from pykrx).
**Question:** the fade may be real — but is it **shortable by a retail follower net of costs**, or only a
paper signal? Report **% of calls actually shortable** and the net contrarian return on that shortable subset.

---

## 3. Output — then STOP (do **not** run a confirmatory test)

Produce `data/_moneyup_advisor/qa_audit/diagnostic/phase1b_diagnostic.md` + a machine-readable `.json` with:

1. **Loss map** — D1–D5 results, with **every slice tried** and its n / mean / t (good and bad), plus the
   **count of slices examined**.
2. **Ranked hypotheses** — the 1–4 most promising sub-pockets (e.g. *"H1: first-mention longs, exit day 2"*;
   *"H2: cost-aware contrarian on shortable large-caps"*), each stated as a **falsifiable rule** with its
   in-sample effect — explicitly labelled **exploratory, not established.**
3. **A draft pre-registration per hypothesis** — exact rule, segment, entry/exit, horizon, deflated bar,
   n-floor, and **which untouched data it must be tested on** (the sealed 30% holdout and/or future uploads).
   **Do not run these now.**
4. **Discipline confirmation** — holdout sealed before analysis and never touched; no "edge/PASS/profitable"
   claim made; isolation held (only the diagnostic folder written).

---

## 4. Honest framing for the conclusion

State plainly which is true:
- **(i)** the loss is **broad and robust at all horizons** → no easy pocket, likely **no tradeable edge
  either way**; or
- **(ii)** a **candidate long pocket** (timing / horizon / first-mention) worth an OOS test; or
- **(iii)** a **candidate contrarian** that survives costs, worth an OOS test.

**Any of the three is an acceptable, useful answer. Do not manufacture a positive.**
The point is to *know* — with proof — not to find a win.
