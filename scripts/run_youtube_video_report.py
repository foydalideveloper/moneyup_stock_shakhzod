"""Build the grounded SINGLE-VIDEO YouTube report (fetch + Whisper transcribe + extract) and render
it to .docx + .pdf. Transcript + extracted insights are CACHED per video_id so KO/EN and repeat
requests reuse ONE deterministic extraction (only the prose is translated).

Extraction uses the STRONGER batch model (gemini-2.5-pro) for specific, named, well-grounded
insights; translation stays on the fast model.

Usage:
    python scripts/run_youtube_video_report.py --url https://youtu.be/9RatueY9jfQ
    python scripts/run_youtube_video_report.py --url 9RatueY9jfQ --lang en
    python scripts/run_youtube_video_report.py --url 9RatueY9jfQ --refresh   # cache-bust: re-extract
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="YouTube URL or bare 11-char video id")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--out", default=None, help="output dir (default: data/reports)")
    ap.add_argument("--refresh", action="store_true",
                    help="cache-bust: ignore any cached extraction and re-extract (overwrites cache)")
    ap.add_argument("--no-render", action="store_true", help="skip docx/pdf rendering")
    args = ap.parse_args()

    from tagent.dashboard import _default_report_video      # noqa: E402
    from tagent.news.report_render import render_report      # noqa: E402

    report = _default_report_video(url=args.url, lang=args.lang, prices_fn="auto", refresh=args.refresh)
    m = report["meta"]
    print(f"Report: {m['n_insights']} grounded insights · {len(report.get('prices', []))} price rows")
    print("\nRECOMMENDATIONS:")
    for r in report.get("recommendations", []):
        tp = r.get("target_price") or ""
        print(f"  - {r.get('stock')} {r.get('action')}" + (f"  | 목표가 {tp}" if tp else ""))

    if not args.no_render:
        paths = render_report(report, lang=args.lang, out_dir=args.out)
        print(f"\nDOCX: {paths['docx']}")
        print(f"PDF : {paths['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
