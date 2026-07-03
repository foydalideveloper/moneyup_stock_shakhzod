"""머니업 AI Advisor — a NEW, fully isolated module.

Phase 0 (this package): grounded, multi-modal extraction of 머니업 stock-call videos into
one FUSED FACT SHEET per video. It does NOT touch the daily YouTube report pipeline, its
scheduler, the email/Supabase push, or the existing dashboard. It only *imports* the Whisper
and yt-dlp helpers from ``tagent.news.youtube_source`` READ-ONLY (never modifies them).

Two hard rules (from the Step-0 smoke test):
  1. PaddleOCR is the SINGLE SOURCE OF TRUTH for every number. Qwen3-VL never overrides an
     OCR'd value; any VLM claim not visible in the frame is tagged VIDEO-ONLY / AUDIO-ONLY or
     dropped — never promoted to a fact.
  2. Every fact is keyed on the 6-digit KRX ticker, never the OCR'd Korean name (names can lose
     a glyph, e.g. 하나머티리얼즈 -> "해나머티리"). The name is a display label only.

Runtime is split across two conda envs on purpose (isolation):
  * base env (py3.13): orchestration, yt-dlp, Whisper (faster-whisper large-v3), Qwen via HTTP.
  * ``moneyup`` env (py3.12): PaddleOCR + OpenCV — invoked as a subprocess vision worker.

Mock-first, grounding-only, NO live trading.
"""

__all__ = ["config"]
