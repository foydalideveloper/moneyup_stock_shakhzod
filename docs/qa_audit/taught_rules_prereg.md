# LOCKED PRE-REGISTRATION — taught-rule one-shot holdout (R4 only)

**Locked:** 2026-07-02 (before any holdout access). **Holdout:** entries with next-open date **≥ 2025-06-17**
(the sealed recent ~30%), NOT touched in the explore stage. **Scorer:** the same backtest engine
(`scripts/taught_rules_backtest.py`) unchanged. One-shot; no tuning; no peek-then-adjust.

## Why only R4 is pre-registered
Explore stage tested K = 5 rules × 3 horizons = 15 (deflated bar t=4.166, α_bonf=0.0033). **14 of 15
failed** (see `taught_rules_backtest.json`). Only **one** cleared BOTH null floors at the deflated bar:

- **R4 — resistance breakout on volume, @ 20 trading days.** Explore: n=2,229, β-adj net **+2.31%**,
  **t=4.85** (≥4.166), beats buy-and-hold (raw), beats random-entry null (p=0.0 < 0.0033). Its 10d cell
  also beat random (p=0.0007) but failed the t-bar (t=2.65) — the signal builds monotonically with horizon
  (5d +0.21% → 10d +0.83% → 20d +2.31%), which argues against a single-cell fluke.

All others — R1 50MA break-fail (short, significantly negative), R2 reclaim-50MA, R3 정배열, R5 눌림목/SMA20
pullback — showed no edge (none beat both floors at the bar). They are NOT pre-registered.

## H1 — R4 resistance-breakout-on-volume @ 20d (the ONLY hypothesis)
- **Rule (fixed):** trigger on day t when `close_t > max(high[t-60 … t-1])` (60-day-high breakout, a proxy
  for 주요 매물대/저항) **AND** `volume_t ≥ 1.5 × mean(volume[t-20 … t-1])` (volume expansion). Enter LONG at
  next open (t+1); exit at the close 20 trading days later. Point-in-time; no look-ahead.
- **Universe:** the SAME top-200 KOSPI + top-150 KOSDAQ by FDR cap.
- **Metric:** β-adjusted net return (vs KODEX200, β over 250d), net 0.50% round-trip.
- **Decision rule (fixed now):** declare a real strategy **iff** holdout `n ≥ 30` AND `mean_βadj_net > 0`
  AND **`t ≥ 4.166`** (the same deflated bar — conservative, since it was selected from 15) AND it **beats
  buy-and-hold** (raw, same universe/horizon) AND **beats the random-entry null** (p < 0.0033, same trade
  count). Run once; report as-is.
- **Holdout coverage:** 2,531 R4@20d triggers already exist in the sealed window (well above the n-floor).

## Caveats that MUST accompany any positive result (state them, don't bury them)
1. **Survivorship / look-ahead in the universe** — membership is **current** top-350-by-cap, not
   point-in-time. A breakout rule on today's large-caps can be inflated by stocks that grew INTO the
   universe. A clean confirmation should rebuild the universe with **point-in-time KOSPI200/KOSDAQ150
   membership** (needs data not available here). This is the single biggest threat to R4.
2. **Thin edge over buy-and-hold** — R4@20d’s raw return barely exceeds B&H (+2.31% β-adj is the real
   claim; the raw margin over just holding is small). Much of the 20d move is market beta.
3. **Proxy definitions** — 매물대/저항 ≈ 60-day-high; volume filter = 1.5× (documented defaults, not his exact
   thresholds). Sensitivity to these should be checked before trading.

## Multiple-testing budget
K = **1** hypothesis on the holdout (R4@20d). No other holdout test permitted under this lock; a new
hypothesis needs a new pre-registration. **If it validates, the strategy is credited to 머니업 as the source
(his taught 매물대/저항 돌파 + 거래량 rule).** Prior: generic technical rules usually don't beat buy-and-hold net
of cost — 14/15 here didn't — so treat R4 as a candidate, not a result, until the holdout passes.
