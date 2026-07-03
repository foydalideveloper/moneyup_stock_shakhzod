# 머니업 OCR Price-Field Fix — read the REAL on-screen price, not axis gridlines

> **Paste to CC.** Goal: make the OCR capture the **exact price shown on screen** — the number the speaker is
> actually discussing (e.g. Bitcoin **58,486**) — instead of axis tick labels (124,000…) and "999.9" noise.
> The OCR-read exact price is the **source of truth** (that's why OCR exists); **audio is a cross-check /
> fallback**, not the primary; the VLM stays structure-only. Shared reader — generalize WITHOUT regressing the
> 머니업 HTS path. Grounding: never fabricate, and never present an axis gridline as "the price."

## 0. Reproduce first
On the crypto video `5yBrQbMgiwY`, confirm the current failure: the summary reports the axis ladder
(124,000 → 88,000) as "key price structure" and **misses the real BTC price (~58,486)** — which is right there
in the chart header (`C 58,646`) and the right-axis last-price label (`58,486.35`) — with "999.9" noise
throughout the [screen] dump.

## 1. Read the actual price fields (the core fix)
On a TradingView-style chart the exact price is NOT on the axis — it's in:
- the **OHLC header** (top-left: `O … H … L … C …` + change %), and
- the **highlighted last-price label** on the right axis (the colored box, e.g. 58,486.35).

Target those regions and OCR the exact values. Also read the on-screen **symbol** (top-left: BTCUSD, MSFT,
SNDK) and **timeframe**. That header/last-price value is the number that matters.

## 2. Axis ladder = scale, not levels
Detect an **evenly-spaced numeric ladder** on the right edge (near-constant step: 124,000 / 120,000 /
116,000…) → tag it as the **axis scale** and DO NOT promote it to "key price levels" in the summary or the
delta. Real support/resistance comes from price structure or from what the speaker states — never from the
even gridline.

## 3. Attribute every price to its instrument
Read the on-screen symbol so each captured price/level is tagged to its instrument (BTC vs MSFT vs SNDK). When
the chart switches symbol (he flips through many), switch attribution. The summary must read "BTC ~58,486 /
MSFT ~451," never a mixed, unlabelled list of numbers.

## 4. Audio = cross-check / fallback (NOT the primary)
- OCR reads an exact price + audio states a rounded one ("around 58,500") → tag **AGREE**, keep the **exact OCR
  value (58,486)** as the figure.
- OCR fails to read a price field → **fall back to the audio's stated price**, tagged as audio — never report an
  axis tick or garbage as the price.
- Keep the plausibility filter so "999.9" / one-decimal artifacts never surface as a price.

## 5. Grounding (unchanged discipline)
OCR-read exact price = source of truth; audio = cross-check; VLM = chart structure only (visual/unverified);
no fabricated numbers.

## 6. Acceptance (reproduce → fix → verify on 3 videos)
- **Crypto `5yBrQbMgiwY`:** captures BTC's real price (~58,486 / 58,523 from header/last-label) attributed to
  BTCUSD; MSFT's price attributed to MSFT; axis ladders NOT reported as "key levels"; the summary's price
  statements match the on-screen header, not axis ticks; "999.9" gone.
- **NASDAQ Gareth `2R7Kkjiy0no`:** the current price is **OCR-confirmed** (~25,778 from header/last-label), not
  VLM-estimated.
- **머니업 regression (one Korean video):** HTS price + watchlist reads unchanged — no regression.

## 7. Scope
Shared visual reader — GPU (PaddleOCR); coordinate around the daily report / nightly collector. Test on **3
videos only**; no 508 re-extraction. Reproduce first, fix per stage, report the **before/after** (the [screen]
price fields, the summary's price statements, the delta), then **STOP for review** — do not wire into nightly
until I sign off.
