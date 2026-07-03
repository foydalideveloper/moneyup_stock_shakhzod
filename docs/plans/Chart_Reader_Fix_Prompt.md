# Fix the Chart-Reading Pipeline — make the visual reader actually read financial charts (grounded)

> **Paste this whole file to CC.**
> Goal: capture what a speaker **shows** on a chart but does **not say** — visible price levels
> (the axis) + technical structure (trend, MA position/breaks, gaps, support/resistance) — and
> surface it as **grounded** `[screen]`/`[chart]` content the audio-only summary lacks.
> **Without hallucinating** anything, and **without breaking** the Korean 머니업 path.

---

## 0. The failing case — reproduce FIRST, don't fix blind

Test video (English, chart-only, no spoken numbers): `https://www.youtube.com/watch?v=2R7Kkjiy0no`
(Gareth Soloway — talks the NASDAQ chart: "lower high / lower low", a 50-day MA break, gaps).

**Current broken output (confirm you reproduce it):**
- VLM `[chart]` = `안 보임` ("nothing visible") on **all 5** reads — captured none of the structure.
- OCR `[screen]` = `999.9 / 989.9 / 889.9` noise ×hundreds — never read the real axis (chart shows ~25,778 / 27,600).
- Subject hallucinated as Korean **"LS (006260)"** on a **NASDAQ** video.
- Whisper **translated ~[01:52]–[02:50] English → Korean** in the transcript.
- Net: the "Full" summary == the audio-only summary; the visual layer added **nothing**.

Root-cause each stage, then fix.

---

## 1. Scope / safety
- This is the **shared visual reader** (used by the 머니업 nightly collector + live demo). **Generalize** it;
  **do not regress** the Korean path.
- **GPU:** coordinate — do not collide with the nightly collector or the 5:30/6:20 daily report. Test on
  **only 2 videos** (the Gareth one + one 머니업 one). **No 508 re-extraction.**
- **Grounding is non-negotiable:** OCR'd numbers = source of truth; the **VLM reports qualitative structure
  only, tagged "visual / unverified", and may NEVER fabricate a price.** Keep AGREE / AUDIO-ONLY /
  VIDEO-ONLY / CONFLICT.
- Don't touch playbook / dashboard / report logic except where the visual reader feeds them.

---

## 2. Fix per stage

### 2.1 Whisper — language handling
Auto-detect source language; transcribe **verbatim in-language** (English stays English). Do **not** force
Korean or `task=translate` on the **transcript**. Translation happens only in the summary layer (as today).
**Pass:** the Gareth transcript is 100% English, zero Korean segments.

### 2.2 Frame selection — chart-aware
Sample frames where a chart is **stable and full** (skip UI transitions / talking-head-only frames); detect
the largest plot region; ensure enough chart frames on chart-centric videos; drop near-duplicates.

### 2.3 VLM — real technical chart-reading (the core fix)
Replace the prompt with a **structured English schema the model must fill**, per chart frame:
- `instrument` (what's charted, as shown — e.g. "NASDAQ Composite / ^IXIC")
- `timeframe` (1D/1W… as labelled)
- `trend_structure` (higher-highs/higher-lows · lower-highs/lower-lows · range)
- `moving_averages` (which MAs visible · price above/below · recent cross/break · slope)
- `key_levels` (support/resistance/round levels **read off the axis**, with values **only if legible**)
- `recent_event` (gap up/down · breakout/breakdown · test/bounce of a level or MA)
- `visible_price` (current/last price **only if shown**)
- per-field `confidence`, and "not legible" allowed **per field** — but **NOT a blanket `안 보임`** when a
  clear chart is present.

Rules: describe structure **even on a clean line chart**; report a numeric level **only if legible** (else
qualitative, e.g. "below the 50-day MA"); **never invent** a number. Tag all VLM output as
**visual-interpretation (unverified)**.

### 2.4 OCR — read the price axis, kill the garbage
Add a **chart-axis mode**: detect the right-side price-axis region and OCR its labels (real values, e.g.
27,600 / 27,200 / 25,778). **Plausibility filter:** an index axis is monotonic, evenly spaced, 4–5 digits —
use that to validate and **drop impossible tokens** (`999.9` repeats, single digits). If the axis isn't
legible, **emit nothing** rather than noise.

### 2.5 Subject resolution — don't force the Korean universe
Detect US-index / non-Korean subjects (audio "NASDAQ/S&P/Dow" + chart label) and label accordingly
(e.g. NASDAQ Composite / ^IXIC). **Do not** map to a Korean 6-digit code when no Korean stock exists. No
confident match in the relevant universe → label the **named index/symbol**, not a forced ticker.

### 2.6 Grounding cross-check (keep it honest)
Where possible, cross-check the VLM's `trend_structure` against a **deterministic** price-series slope
(reuse the existing OpenCV candle/trend check, or derive from OCR'd levels) and tag AGREE / CONFLICT — so a
hallucinated trend is caught, not trusted.

---

## 3. Fusion / summary — make the visual layer contribute
The Full summary must include **grounded visual content the audio-only lacks**: the read-off axis levels, the
trend structure, the MA-break/gap events — each tagged `[screen]`/`[chart]` + timestamp. The **delta (d)**
must show the visual layer added real structure (not 0), still **computed, not LLM**. Numeric levels only if
axis-legible; no fabrication; keep CONFLICT when audio and chart disagree.

---

## 4. Acceptance tests (must pass before done)

**On the Gareth video (the failing case):**
1. Transcript fully English (no Korean segments).
2. Subject = NASDAQ / index — **not** "LS 006260".
3. VLM returns **real structure** on the clear-chart frames (e.g. "lower highs / lower lows", "broke below
   the 50-day MA", "gap below the MA") — **no** `안 보임` when a chart is present.
4. OCR captures the **real axis levels** (5-digit index values); the `999.9` garbage is gone.
5. The Full summary contains **≥3 grounded visual items absent from the audio-only summary** (the levels he
   points at, the 50MA-break structure), each `[screen]`/`[chart]` + timestamp — i.e. it now captures what he
   **shows but doesn't say**.
6. **No fabricated numbers** — every numeric level traceable to a legible axis/label.

**Regression — the 머니업 path must not break:**
7. Re-run **one** 머니업 video; confirm the Korean watchlist OCR, candle read, ticker resolution, and the
   existing summary still work as before. No regression.

---

## 5. Deliverables
1. The visual-reader diff (Whisper language, frame selection, VLM schema/prompt, OCR axis mode + plausibility,
   subject resolution, grounding cross-check).
2. **Before/after on the Gareth video** — the `[screen]`/`[chart]` block, the delta (d), and Full vs
   audio-only — showing the visual layer now adds grounded structure.
3. The **머니업 regression** result (unchanged).
4. A short note: what generalized, what's still limited (e.g. hand-drawn annotations), GPU/time cost.
5. Isolation: every file touched; confirm collector/report/playbook/dashboard logic unchanged except the
   shared visual reader.

## 6. Discipline
Ground everything — no fabricated levels; VLM = interpretation, OCR = numbers. Test on **2 videos only**, no
508 re-extraction, coordinate GPU around the daily report. **Stop and report with the before/after** — do not
wire into nightly until we review.
