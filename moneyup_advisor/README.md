# 머니업 AI Advisor — `moneyup_advisor/`

A **NEW, fully isolated** module. It does **not** modify the daily YouTube report pipeline, its
scheduler, the email/Supabase push, or the existing dashboard (port 8000). It only *imports*
`tagent.news.youtube_source`'s Whisper + yt-dlp helpers **read-only**.

## Two hard rules
1. **PaddleOCR is the single source of truth for every number.** Qwen3-VL never overrides an OCR'd
   value; any VLM claim not visible in the frame is tagged `VIDEO-ONLY` / `AUDIO-ONLY` or dropped —
   never promoted to a fact (e.g. the 호가 ladder Qwen hallucinated in Step 0).
2. **Every fact is keyed on the 6-digit KRX ticker**, never the OCR'd Korean name (names can lose a
   glyph: 하나머티리얼즈 → "해나머티리"). The name is a display label only.

## Runtime (two conda envs, on purpose)
- **base** (py3.13): orchestration, yt-dlp, Whisper (faster-whisper large-v3), Qwen3-VL via HTTP.
- **`moneyup`** (py3.12): PaddleOCR + OpenCV — invoked as a subprocess **vision worker**.

The only cross-env hop is `vision_worker.py` (run in the `moneyup` env). Set
`MONEYUP_ENV_PYTHON` if its path differs from the default.

## Pipeline (per video)
download → Whisper transcript (timestamped) → ffmpeg frame sampling (scene-change + dense on call
segments) → **vision worker**: PaddleOCR exact numbers + OpenCV chart geometry → Qwen3-VL
pattern/context → ex-ante/ex-post call split → **timeline fusion** (`AGREE` / `AUDIO-ONLY` /
`VIDEO-ONLY` / `CONFLICT`) → one fused fact sheet (`.json` + `.md`).

## Run
```bash
# base env
python run_moneyup.py --max 4              # extract 4 recent 머니업 videos
python run_moneyup.py --video Q9JHwbZ591E  # a specific video
python run_moneyup.py --max 3 --no-vlm     # skip Qwen (faster)
python run_moneyup_dashboard.py            # http://127.0.0.1:8077
```

Outputs (git-ignored, under `data/_moneyup_advisor/`): `videos/`, `frames/<id>/`, `cache/`,
`factsheets/<id>.json|.md`.

## Ex-ante call schema (for the later Phase-1 falsification)
`ticker` (6-digit) · `direction` (long/short/avoid) · `stated_price` (if any) · `in_video_t` /
`mmss` · `publish_datetime`. Ex-post commentary (past picks / realized returns) is kept separate.

Mock-first · grounding-only · **no live trading**. Final reasoning/fusion layer (Gemini 3.5 Pro,
paid) is intentionally **not** wired in Phase 0.
