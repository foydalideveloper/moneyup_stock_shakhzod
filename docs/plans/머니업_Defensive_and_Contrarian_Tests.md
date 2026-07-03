# 머니업 Edge Tests — Defensive-Timing + Contrarian (best-effort)

> **Paste to CC.** Two tests. Run TEST 1, report + STOP; then TEST 2, report + STOP.
> Standing discipline: net of 0.50% cost + beta/size-adjusted where relevant; deflated
> significance across the multiple horizons; n≥30 floor; exploratory 70% first (note holdout
> availability for anything that survives); read-only; no 508 re-extraction; no GPU;
> no tradeable-edge claim without a real holdout / real data behind it.

---

## TEST 1 — Defensive-timing: does his "go to cash" call predict market drops?

**Question:** when 머니업 tells people to step aside (관망 / 현금비중 / 보유·don't-add / 대기 / 쉬어가다),
does the market actually fall afterward — i.e., does his *timing* have predictive value even though his
buy-*picking* doesn't?

**Do:**
- Pull his defensive calls (the **HOLD-WAIT-CASH** bucket from the corrected classifier) with dates. Split
  **market-level** defensive calls (about the market/index) from **stock-level** "hold, don't add" (a specific name).
- For each **market-level** defensive call, measure the **FORWARD** return of **KOSPI and KOSDAQ** over
  h = 1 / 5 / 10 / 20 trading days **after** the call.
- Compare to the **unconditional index return** over the same horizons (baseline): is the post-call forward
  return significantly **below** baseline?
- **Control the key confound:** does his "cash" call **predict** a future drop, or just **describe** a market
  that's already falling? Report the index return in the days **before** each call too. If the market was
  already down and there's no *forward* underperformance, that's reactive, not skill. The test is **forward
  return vs baseline**, not contemporaneous.
- **Secondary:** compare forward index returns after his **BUY** calls vs after his **DEFENSIVE** calls — does
  he actually lean cautious before drops and bullish before rises, or is there no timing signal?
- Stat: deflate the bar for the 4 horizons; n ≥ 30; exploratory 70% (note holdout availability if it survives).

**Verdict (honest):** would going to cash on his defensive calls have avoided drawdowns vs staying invested —
with forward significance, after controlling for the reactive confound? Yes / no / underpowered.

---

## TEST 2 — Contrarian tradeability (best-effort PROXY; real KRX data pending)

**Question:** does the contrarian (short his buy calls — +1.93% OOS on paper) survive real shortability +
borrow costs?

**Hard constraint:** real KRX shortability / borrow / 공매도 과열 data is **not available in this environment.**
So this is a **PROXY**, labelled as such throughout — the definitive test needs that data (user to source).

**Do:**
- Take the existing contrarian mirror returns (short his buy calls; exploratory + holdout, already computed).
- **Proxy the borrowable universe** with FDR market caps: likely-borrowable = KOSPI-listed large-cap
  (e.g. market cap ≥ ~2조, KOSPI200-style); likely-NOT = small/mid KOSDAQ theme names. Report the split —
  what % of his calls are even on plausibly-shortable names.
- Restrict the contrarian to the **likely-borrowable subset**: report n, mean net, t (before borrow).
- **Borrow-cost sensitivity:** subtract an annualized borrow fee at **3% / 6% / 12% / 20%**, prorated over the
  actual holding period; report the contrarian net at each level, on the borrowable subset.
- **공매도 과열 flag:** note as unavailable here — caveat that some borrowable names may still have been
  short-banned on the call date, which the proxy can't see.

**Verdict (honest):** on the names you could plausibly short, does the contrarian stay positive net of
realistic borrow? State the finding candidly — the borrowable large-caps have the **weakest** fade (mega-caps
were only −2.6%), so it may well be marginal or negative once borrow is charged. Label **PROXY** everywhere and
state exactly what real KRX shortability/borrow data would change.

---

## Both
Read-only on existing data + FDR/pykrx; no 508 re-extraction; no GPU. Run TEST 1 → report → STOP; then
TEST 2 → report → STOP. If either result survives exploratory, propose (do not run) a one-shot sealed-holdout
confirmation, pre-registered, for sign-off.
