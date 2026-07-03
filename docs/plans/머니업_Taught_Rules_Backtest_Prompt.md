# 머니업 Taught-Rules Backtest — test his TEACHING as general market strategies

> **Paste to CC.** Goal: extract 머니업's concrete, taught **technical** rules and backtest each as a
> **general market strategy** on the broad universe with real historical prices — decoupled from the
> specific stocks he named. This tests his *teaching* ("when X happens, do Y, here's why"), not his picks.
>
> Standing discipline: real pykrx OHLCV; net of 0.50% round-trip; beta/size-adjusted (excess vs size-matched
> market); temporal split + sealed holdout; deflated significance (Bonferroni over #rules × #horizons);
> a random-entry null floor; point-in-time features (no look-ahead); n ≥ 30 triggers/rule; report EVERY rule
> tried; no edge claim without the one-shot holdout; STOP before running it.

## STEP 1 — Extract the taught rules (from the Playbook)
From the 머니업 Playbook, select the **concrete, mechanizable, PRICE-based** rules — the ones stated as
"when [price / MA / structure condition] → [action], because [reason]". For each, write an exact mechanical
definition (trigger, direction, exit). The *type* of rule (use his actual wording/thresholds where given):
- **50MA break-and-fail** — closes below the 50-day MA and, on the next bounce, fails to reclaim it → bearish.
- **Reclaim above 50MA** — gaps/closes back above the 50-day MA and holds → bullish.
- **정배열 (MA alignment)** — price > SMA20 > SMA60 > SMA120 → trend-long.
- **주요 매물대 / 저항 돌파** — breakout above a defined resistance / volume node on rising volume → long.
- **눌림목 / 지지** — pullback to a rising MA or prior support that holds → long.

**EXCLUDE:** (a) vague/discretionary rules that can't be mechanized (e.g. "catch the 타점"); (b) **flow-based**
rules (수급 / 외국인 / 공매도 / program) — the broad-universe flow data doesn't exist in this environment, so they
can't be tested generally, and Track A already tested them on his picks (negative). List each kept rule with its
exact trigger / direction / exit; where his threshold is ambiguous, pick a documented default and flag it.

## STEP 2 — Backtest each rule as a GENERAL strategy (not his picks)
- **Universe:** a defined liquid set — **KOSPI200 + KOSDAQ150** constituents (point-in-time membership if
  obtainable; else current membership with a stated caveat). Explicitly NOT limited to stocks he named.
- **Execution:** whenever a rule's condition triggers on ANY stock in the universe, enter at next open; exit per
  the rule's own logic or a triple-barrier with a symmetric default; test the relevant horizons.
- **Returns:** real pykrx OHLCV; net of 0.50% round-trip; beta+size-adjusted.
This is the honest test of "does his taught rule work as a general strategy," independent of his stock selection.

## STEP 3 — Discipline (multiple-testing is the main risk)
Testing many rules WILL throw up false winners, so:
- **Temporal split:** explore on the older ~70%; seal a recent ~30% holdout.
- **Deflated bar:** Bonferroni + deflated-Sharpe over K = (#rules × #horizons) — print K and the threshold.
- **Null floors:** each rule must beat BOTH (a) buy-and-hold on the same universe and (b) a **random-entry null**
  (same trade count, random dates/names) by the deflated bar. A rule that can't beat random entry is noise.
- Point-in-time only (no look-ahead); n ≥ 30 triggers/rule (else mark underpowered).
- **Report every rule tried** with its explore-stage result — winners AND losers — plus K.

## STEP 4 — Output, then STOP
- The mechanized rule list; the explore-stage results table (all rules); the null-floor comparisons.
- For any rule that survives explore, a **locked pre-registration** for the one-shot holdout (rule, universe,
  horizon, deflated bar, n-floor, decision rule) — then **STOP for sign-off. Do NOT run the holdout or claim an edge.**
- **Honest framing:** state the prior up front (generic technical rules usually don't beat buy-and-hold net of
  cost) and report candidly. If a rule validates → it's a real strategy, **credited to 머니업 as the source**. If
  none → his taught rules, like his picks, don't produce a general edge — and his value is teaching, not signals.

## Scope
Read-only on existing data + pykrx; no GPU; no 508 re-extraction. No tradeable-edge claim without the one-shot
holdout behind it.
