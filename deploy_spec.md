# Deploy Spec — KR trend-futures core (PRE-REGISTERED)

**Status:** pre-registered before running the v2 backtest and the drawdown diagnostic.
The constants and rules below are LOCKED; results are reported as-is (even if worse),
never tuned back into the design. Paper only — **do not deploy** until the ops shakedown
below completes.

The deployable product is the **200d-MA trend-filtered KOSPI-200 index future**, not a
stock-picking book (see `alpha-vs-beta-decompositions` memory / `index_calibration`).
The cross-sectional ranking sleeve is **frozen** (Part 3) — a low-conviction satellite,
not deployed.

---

## A. Sizing — off the REAL drawdown, not the comfortable one

The long-history (1990→2026) filtered core's worst drawdown is **≈ −49.7%**, **not** the
~−23% of a single benign sub-period. **Sizing must survive −49.7%.**

DD diagnostic finding (the pre-registered "likely 1997 IMF" hypothesis was REFUTED):
the binding episode is the **1990–92 post-launch Korean bear** — peak at the index
launch (1990-01-03), trough 1992-10, not recovering until 1999. The 1997 IMF crisis was
actually *well handled* by the filter (it went to cash). The early-history drawdown is
partly warmup-influenced (the filter is forced in-market before its 200d MA exists), and
the realistic trade-next-bar execution lag *worsens* it (idealised same-bar maxDD ≈ −37%
→ realistic ≈ −46% → committed 2-bar-lag product −49.7%). We size off the most
conservative realistic figure, −49.7%.

Pre-commitment: pick exposure so the account's tolerable drawdown is not breached even
if the strategy repeats its worst historical DD.

```
size_multiplier = tolerable_account_DD / |strategy_maxDD|
```

With tolerable account DD = **−15%** and strategy maxDD = **−49.7%**:
`0.15 / 0.497 ≈ 0.30x`. → **cap core exposure at ~0.30x** notional (one KOSPI-200 mini
sized to ≤0.3x of the account), or the **vol-target equivalent** (target ≈ 0.30 × the
index's full-exposure vol). This is the hard size cap; never exceed it on conviction.

## B. No discretionary override

Once live, the exposure is whatever the pre-registered rule outputs. **No manual
overrides** — not on macro views, not on "obvious" tops/bottoms, not on a bad month.
The only permitted changes are: (1) a pre-registered rule change that is itself
re-registered here with a new dated version and re-backtested on full history before it
goes live; (2) the hard size cap in (A). A discretionary trade voids the evaluation.

## C. Evaluation horizon — the decade table IS the contract

Evaluate over **3–5 years**, judged against the historical by-decade table — not against
a P&L target. The contract (binary 200d filter, 1990→2026, net of ~12bps/yr roll):

| Decade | Sharpe | Read |
|---|---|---|
| 1990s | +0.31 | crisis era |
| 2000s | +0.50 | |
| **2010s** | **−0.06** | **Boxpi range — the known failure mode** |
| 2020s | +1.07 | trend-rich |
| **Full** | **+0.45** | CAGR +6.58%, maxDD −49.7% |

**A flat-to-negative multi-year stretch is WITHIN contract** (the 2010s row): trend
following churns to ~nothing (even a loss) in range-bound years — that is the *price* of
crash protection, not a failure. We abandon the core only if it underperforms the
**−0.06-to-+0.50 non-bull-decade band by a wide margin AND the drawdown protection
itself stops working** (e.g. a >−55% DD in a normal correction), not because of a quiet
range stretch.

## D. Roll-out plan (operational confidence, not P&L)

1. **1–3 months paper ops-shakedown** on live feeds (signal generation, roll handling,
   fills, monitoring) — prove the plumbing, not the edge.
2. **One mini contract** at the ≤0.30x cap once the shakedown is clean.
3. **Scale on operational confidence** (clean fills, no missed rolls, monitoring holds),
   **not on P&L**. P&L over 1–3 years is noise vs the decade table.

---

## Frozen satellite (Part 3) — promotion / kill (pre-registered)

The ranking sleeve runs as an **automatic paper track only** with the frozen book
(12-1, monthly, top-quintile, 30% sector cap). No more variants.

- **PROMOTE** if EITHER:
  - (a) the **combined backtest+forward spanning-alpha 95% CI excludes zero**, OR
  - (b) **concentration reverses** — EW basket beats the CW index on trailing 12mo, OR
    the index top-2 weight falls ≥ **5 percentage points** — **while** the forward
    ranking contribution stays **positive**.
- **KILL → archive** if the **forward ranking contribution is negative over 24 months**.
- Otherwise **FROZEN — continue paper-tracking**. Never auto-deploys.
