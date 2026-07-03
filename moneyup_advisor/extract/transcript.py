"""Timestamped Korean transcript via the existing pipeline's Whisper — READ-ONLY reuse.

Calls ``tagent.news.youtube_source._whisper_transcribe`` (faster-whisper) and forces the
``large-v3`` model on cuda/float16 *process-locally* (env vars set in THIS process only — the
daily pipeline's config is never modified). Results are cached to a git-ignored JSON.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from moneyup_advisor import config


def _detect_language(audio_path: str) -> Optional[str]:
    """Cheap language id (faster-whisper 'base' encoder on the first 60s). 'ko'/'en'/… or None."""
    try:
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio
        from tagent.news.youtube_source import _whisper_device_compute
        dev, ctype = _whisper_device_compute()
        m = WhisperModel("base", device=dev, compute_type=ctype)
        wav = decode_audio(audio_path)
        lang, prob, _ = m.detect_language(wav[:16000 * 60])
        return lang if prob and prob >= 0.5 else None
    except Exception:
        return None


def transcribe(audio_path: str, video_id: str, language: Optional[str] = None,
               refresh: bool = False) -> List[Dict]:
    """[{text, start, duration}] timestamped segments (cached). Empty list on failure.

    AUTO-DETECTS the source language and transcribes VERBATIM in-language (English stays English;
    Korean stays Korean) — never forces Korean or task=translate on the transcript. The Korean finance
    glossary is applied ONLY for Korean audio, so it can't bias a non-Korean video toward Korean."""
    cache = config.CACHE_DIR / f"{video_id}.transcript.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))

    # Force large-v3 on GPU for THIS process only (read-only wrt the pipeline's config).
    os.environ["WHISPER_MODEL"] = config.WHISPER_MODEL
    os.environ.setdefault("WHISPER_DEVICE", "cuda")
    os.environ.setdefault("WHISPER_COMPUTE_TYPE", "float16")

    lang = language or _detect_language(audio_path)          # None caller -> auto-detect
    prev_prompt = os.environ.get("WHISPER_PROMPT")
    if lang and lang != "ko":                                # non-Korean: drop the Korean glossary
        os.environ["WHISPER_PROMPT"] = ""
    try:
        from tagent.news.youtube_source import _whisper_transcribe   # READ-ONLY reuse
        segs = _whisper_transcribe(audio_path, language=lang) or []   # lang=None -> faster-whisper auto-detect
    finally:
        if lang and lang != "ko":                            # restore the env (don't leak to other calls)
            if prev_prompt is None:
                os.environ.pop("WHISPER_PROMPT", None)
            else:
                os.environ["WHISPER_PROMPT"] = prev_prompt
    # normalize + add end time + mm:ss for convenience
    out = []
    for s in segs:
        start = float(s.get("start", 0.0) or 0.0)
        dur = float(s.get("duration", 0.0) or 0.0)
        out.append({"text": str(s.get("text", "")).strip(), "start": round(start, 2),
                    "end": round(start + dur, 2), "duration": round(dur, 2)})
    out = [s for s in out if s["text"]]
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    return out
