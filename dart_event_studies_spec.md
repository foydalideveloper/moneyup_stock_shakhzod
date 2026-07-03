# DART Event Studies — PRE-REGISTERED SPEC (Track B)

**Status:** locked BEFORE fetching/running. Free data via OpenDART. Results reported
as-is (flat/negative is a fine, expected outcome). Two separate trials (cancellation,
contract) — counted for multiple testing. Beta autopsy is built into v1 (the same lens
that killed PEAD: positive corporate events cluster in bull markets).

## Hygiene (both studies) — locked

- **No raw-return headline.** Primary metric = **size-matched abnormal return** = event
  return − same-period return of the EW portfolio of the stock's **market-cap quintile**
  within the PIT universe (quintiles recomputed daily among active members; cap = cached
  shares × close). Secondary = **calendar-time portfolio Newey-West t** (each day: EW of
  active event positions − EW of their matched size-quintile controls; NW lag = hold).
  Raw return is a **footnote only**.
- **Beta autopsy:** also report event return − β·KOSPI-200 over the held window
  (`pead_beta_adjust`), so we see how much "drift" is market beta.
- **No-lookahead:** enter the FIRST tradable open strictly AFTER the filing date.
- **Non-overlapping** events per name (one position per name at a time).
- **Holds pre-registered:** 20 / 40 / 60 trading days. No post-hoc horizon search.
- **Costs:** base **0.20%** cash round trip; stress **0.51%** and **0.71%**. **Extra
  gap-entry slippage** of **+0.30%** when the entry open gaps up (r0 = entry_open/prev_close − 1
  > **+1.0%**) — gap-up opens are hard to fill. Report the **gap (r0) vs post-open
  (entry_open→exit) share** of the drift.

## Study 1 — Cancellation (소각) drift  [PRIMARY]

- **Cancellation** = title contains `소각` AND not `정정` (correction) AND not `자회사`
  (subsidiary's filing). Credible/irreversible — shares destroyed.
- **Acquisition** = title contains `자기주식취득` AND `결정`, excluding `신탁` (trust),
  `처분` (disposal), `정정`. Announcement only — SECONDARY (Korean firms historically HOLD
  treasury as a control tool, so weaker). **Reported separately; expect cancellation >
  acquisition.**
- "Scale by size vs market cap": report abnormal drift **by size quintile** (the matched
  control already conditions on size). Per-event cancellation value/mcap is NOT parsed
  (no clean free structured endpoint for 소각) — flagged as a refinement.

## Study 2 — Supply-contract (단일판매·공급계약) drift

- **Contract** = title contains `단일판매` OR `공급계약`, excluding `정정`.
- **Materiality:** the contract value / market-cap-and-revenue filter is **not available**
  on free structured OpenDART for 단일판매·공급계약 (no JSON value endpoint). **Proxy:**
  test **by size quintile** — the smallest-cap quintile is where a given contract is most
  material vs cap/revenue. Reported by quintile; the contract-value filter limitation is
  flagged. **Caveat:** the PIT universe is large-cap KOSPI; the classic KOSDAQ
  retail-pump small-caps are under-represented, so the dramatic gap-pump may be muted here.
- Report with conservative gap-entry slippage and **flag gap vs post-open share** of the
  drift (expect the pump concentrated in the overnight gap, which we do NOT capture by
  entering next open).

## Verdicts — PRE-REGISTERED

An edge clears the bar ONLY if **ALL** hold:
1. **size-matched abnormal return POSITIVE** (not just raw), and
2. **calendar-time portfolio Newey-West t > ~2.5–3** (use **t > 3** given 2 trials), and
3. **survives conservative slippage** (net positive at 0.71% + gap slippage), and
4. (autopsy) **not explained away by β·KOSPI-200** (beta-adjusted abnormal still positive).

Cancellation and contract are **separate trials** — both counted. Expect, a priori, that
positive-event drift is mostly the bull-market beta trap (as with PEAD); a flat/negative
verdict is the honest baseline.
