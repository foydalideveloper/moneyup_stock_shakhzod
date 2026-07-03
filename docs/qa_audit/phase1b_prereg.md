# LOCKED PRE-REGISTRATION — 머니업 one-shot sealed-holdout test (Task 5 gate)

**Locked:** 2026-06-30 (before any holdout access). **Holdout:** the sealed 152-video set in
`holdout_manifest.json` — NOT touched by the offense hunt. **Scorer:** the UNCHANGED
`phase1b.score_call` (net 0.50% round-trip, beta+size-adjusted). **Discipline:** one-shot; no tuning;
no peek-then-adjust; the words "edge/PASS/profitable/winning" are used ONLY if a rule below fires.

## Why so little is pre-registered
The offense hunt (exploratory 70%, CV) found **NO long edge worth confirming**:
- **Track A** — every powered setup bucket (n≥30) is **significantly NEGATIVE**: LONG-GENERIC −6.7% (t −14.4),
  수급_기관외인 −5.9% (t −7.2), 공매도_매집선 −7.0% (t −6.5), 눌림목_지지 −7.2% (t −6.0), DIP-BUY −8.7% (t −8.7).
  None is positive; none clears the deflated bar (3.95, K=6) on the positive side.
- **Track B** — the conditional model does **not** beat the shuffled-label null floor: logistic AUC 0.520 vs
  null-95 floor 0.572 (p=0.32); LightGBM AUC 0.399 vs floor 0.554. Features do not separate his
  winners from losers OOS.

So there is **no long-setup and no model hypothesis to pre-register** — confirming a non-signal is
pointless. Only the two narrow hypotheses below are registered.

## H1 — "true mega-cap calls" slice  (the ONE pre-registered hypothesis from Task 3)
- **Universe:** his ex-ante BUY calls on **005930 / 035420 / 000660** within the sealed holdout.
- **Metric:** pooled `net_abnormal` (beta+size-adj, net 0.50% round-trip) from the unchanged scorer; n, mean, Newey-West t.
- **Decision rule (fixed now):** declare a mega-cap long edge **iff** holdout `n ≥ 30` AND `mean_net > 0`
  AND `t ≥ +3.5` (base bar; with K=2 hypotheses the Bonferroni two-sided bar is **t ≥ +3.92**).
- **Stated prior:** exploratory mega-cap slice = **−2.46% (t −1.96, n 44, NS)** → expectation is this
  **FAILS** (mildly negative, not significant). Run once; report as-is whether it fails or not.

## H2 — contrarian (short) signal  — CONDITIONAL, NOT runnable yet
- **Hard precondition:** resolve KRX **shortability** (borrowable universe + borrow fee + 공매도 과열종목 status)
  for his picks. Until resolved, this test is **NOT run** and **no contrarian edge may be claimed.**
- **If resolved:** on the *borrowable* subset of holdout buy calls, test mirror net (net of round-trip + borrow).
  Declare a contrarian edge **iff** that subset has `n ≥ 30` AND `mean_net > 0` AND `t ≥ +3.92` (Bonferroni, K=2).
- **Flag:** most picks are mid/small theme names likely outside the retail-borrowable universe → the
  borrowable subset may be `n < 30` → **inconclusive**, not an edge.

## Multiple-testing budget
K = **2** hypotheses on the holdout (H1 now; H2 only if its precondition is met). Bonferroni two-sided
α = 0.025 each → **t ≥ 3.92**. No other holdout test is permitted under this lock. Any new hypothesis
requires a NEW pre-registration before touching the holdout again.
