"""Run the Gemini YouTube insight pipeline on a LOCAL timestamped transcript (.srt) — no quota,
no IP-block.

Proves the YouTube -> Gemini -> grounded-insight pipeline end-to-end while BYPASSING the two
network steps that hit the YouTube Data API quota / get IP-blocked:
  * NO search/listing call (we already know the video id),
  * NO transcript fetch (we inject the local .srt as the transcript).

It feeds the parsed segments straight into ``youtube_extract_fn`` (= llm_extract_fn ∘ Gemini
youtube_analyze_fn), which reads the FULL timestamped transcript and returns, per watchlist stock,
a grounded insight: a summary + a VERBATIM quote (hallucinated quotes are dropped) snapped to the
real segment start, with an mm:ss deep-link. The audit log is enabled, so the raw transcript +
insights are saved under ``data/youtube_fetch_log/<date>/`` for verifiable proof.

Falls back to the no-LLM grounded extractor if GEMINI_API_KEY is unset (and says so). Secrets are
never printed.

Usage:
    python scripts/run_youtube_transcript_insight.py
    python scripts/run_youtube_transcript_insight.py --srt path/to/transcript.srt --video-id dMVpoiK36EU
"""

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.config import SETTINGS                       # noqa: E402
from tagent.news.youtube_source import YouTubeSource, mmss  # noqa: E402
from tagent.youtube_audit import AuditWriter, fetch_log_path  # noqa: E402

DEFAULT_SRT = pathlib.Path.home() / "Downloads" / "transcript.srt"


def _ts_to_seconds(ts: str) -> float:
    """'HH:MM:SS,mmm' (or '.mmm') -> float seconds."""
    ts = ts.strip().replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_srt(text: str):
    """Parse an .srt into [{text, start, duration}] (start = float seconds). Robust to blank lines
    and multi-line cues; no external deps."""
    segments = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
    arrow = re.compile(r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})")
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        ti = next((i for i, ln in enumerate(lines) if arrow.search(ln)), None)
        if ti is None:
            continue
        m = arrow.search(lines[ti])
        start, end = _ts_to_seconds(m.group(1)), _ts_to_seconds(m.group(2))
        body = " ".join(ln.strip() for ln in lines[ti + 1:]).strip()
        if not body:
            continue
        segments.append({"text": body, "start": start, "duration": max(0.0, end - start)})
    return segments


def _extract_fn():
    """Gemini-backed grounded extractor if a key is set (the real pipeline), else the no-LLM
    grounded default. Returns (fn, label). The key is never printed."""
    from tagent.gemini import build_extractor, extractor_label
    ex = build_extractor(SETTINGS)
    if ex is not None:
        return ex, extractor_label(SETTINGS)
    from tagent.news.youtube_source import default_extract_catalysts
    return default_extract_catalysts, "grounded default (no LLM key — set GEMINI/OPENAI key)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", default=str(DEFAULT_SRT))
    ap.add_argument("--video-id", default="dMVpoiK36EU")
    ap.add_argument("--title", default="네이버(035420) 분석 — 로컬 전사본(.srt)")
    ap.add_argument("--channel", default="YouTube (로컬 .srt)")
    ap.add_argument("--published", default="")
    args = ap.parse_args()

    srt_path = pathlib.Path(args.srt)
    if not srt_path.exists():
        print(f"[ERROR] transcript not found: {srt_path}")
        return 1
    segments = parse_srt(srt_path.read_text(encoding="utf-8", errors="replace"))
    if not segments:
        print(f"[ERROR] no segments parsed from {srt_path}")
        return 1

    extract_fn, backend = _extract_fn()
    dur = segments[-1]["start"] + segments[-1]["duration"]
    print(f"Transcript: {srt_path.name}  ({len(segments)} segments, ~{mmss(dur)})")
    print(f"Backend   : {backend}")
    print(f"Video     : {args.video_id}  '{args.title}'")
    print("Bypassing : YouTube search/listing + transcript fetch (no quota, no IP-block)\n")

    # inject the local transcript -> NO transcript fetch; we never call listing -> NO quota.
    src = YouTubeSource(SETTINGS.youtube_api_key or "UNUSED",
                        transcript_fn=lambda vid, langs: segments, extract_fn=extract_fn)
    video = {"video_id": args.video_id, "title": args.title, "published_at": args.published}
    audit = AuditWriter()                                # audit ON -> raw transcript + insights saved
    insights = src.video_catalysts(video, channel_name=args.channel, audit=audit)

    if not insights:
        print("No grounded watchlist insights extracted (nothing met the catalyst/grounding bar).")
    else:
        print(f"GROUNDED INSIGHTS ({len(insights)}):\n")
        for i, ins in enumerate(insights, 1):
            print(f"{i}. {ins['stock_name']} ({ins['stock']})  ·  {ins.get('category', '')}"
                  f"  ·  {ins.get('sentiment', '')}")
            print(f"   summary : {ins['summary']}")
            print(f"   quote   : \"{ins['quote']}\"")
            print(f"   moment  : {ins['timestamp_mmss']}  ->  {ins['deeplink']}")
            ref = ins.get("fetch_log_ref", {})
            print(f"   source  : {ref.get('log', '')}  (span @ {ref.get('span_start')}s)\n")

    print(f"-> audit log: {fetch_log_path()}  (raw transcript + insights persisted for proof)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
