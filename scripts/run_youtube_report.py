"""Build the grounded daily YouTube report and render it to .docx + .pdf (bilingual).

Reads the already-logged grounded transcripts/insights (no re-transcription), attaches the real
Kiwoom/KRX price table when KIWOOM keys exist (prices_fn="auto"), and renders downloadable files.
Email/dashboard delivery come in later phases.

Usage:
    python scripts/run_youtube_report.py                       # KST: yesterday 00:00 -> now, lang ko
    python scripts/run_youtube_report.py --lang en
    python scripts/run_youtube_report.py --start 2026-06-14T00:00:00+09:00 --end 2026-06-15T15:50:00+09:00
    python scripts/run_youtube_report.py --out ./out
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.config import SETTINGS                       # noqa: E402
from tagent.news.report_render import render_report      # noqa: E402
from tagent.news.youtube_report import build_youtube_report  # noqa: E402


def _translate_fn():
    """A KO->EN BATCH translator (one call) on the fast model when a Gemini key is set, else None."""
    if not SETTINGS.has_gemini_key():
        return None
    try:
        from tagent.gemini import GeminiClient, translate_batch_fn
        model = getattr(SETTINGS, "gemini_interactive_model", None) or SETTINGS.gemini_model
        return translate_batch_fn(GeminiClient(SETTINGS.gemini_api_key, model=model))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="KST ISO (default: yesterday 00:00 KST)")
    ap.add_argument("--end", default=None, help="KST ISO (default: now)")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--out", default=None, help="output dir (default: data/reports)")
    args = ap.parse_args()

    translate = _translate_fn() if args.lang == "en" else None
    report = build_youtube_report(start=args.start, end=args.end, lang=args.lang,
                                  translate_fn=translate, prices_fn="auto")   # Kiwoom prices if keys

    m = report["meta"]
    print(f"Report: {m['n_videos']} videos · {m['n_insights']} grounded insights · "
          f"{len(report.get('prices', []))} price rows · window "
          f"{m['window_start']} → {m['window_end']}")
    paths = render_report(report, lang=args.lang, out_dir=args.out)
    print(f"DOCX: {paths['docx']}")
    print(f"PDF : {paths['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
