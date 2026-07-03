"""BATCH-transcribe every recent video on the configured channels into the fetch log.

Lists each channel's recent uploads (quota-cheap channels.list + playlistItems.list), then for EACH
video fetches its FULL transcript through the fallback chain that bypasses the IP-block:
    (a) youtube-transcript-api (Korean first; uses PROXY_URL / Webshare from .env -> seconds/video)
    (b) yt-dlp auto-subtitles (also proxied -> seconds/video)
    (c) yt-dlp audio + local Whisper (no proxy needed; slow — minutes/video)
stopping at the first success. Every video's full transcript + insights is written to
``data/youtube_fetch_log/<date>/fetch.jsonl`` (one line per video — the single source of truth), then
Gemini extracts grounded insights. The daily briefing #3 replays those logged insights.

Reports per video: method (api/ytdlp/whisper) + segments + insights. ``--max-videos`` caps the TOTAL
processed (Whisper without a proxy is slow). Secrets (API keys, proxy creds) are never printed.

Usage:
    python scripts/run_youtube_batch.py                       # all channels, proxy-fast if configured
    python scripts/run_youtube_batch.py --max-videos 3        # cap total (e.g. when on Whisper)
    python scripts/run_youtube_batch.py --lookback-hours 72 --max-per-channel 8
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
from tagent.news.youtube_source import YouTubeSource, channels_from_env  # noqa: E402
from tagent.youtube_audit import AuditWriter, fetch_log_path  # noqa: E402


def _extract_fn():
    """Gemini grounded extractor if a key is set, else the no-LLM grounded default. Key not printed."""
    from tagent.gemini import build_extractor, extractor_label
    ex = build_extractor(SETTINGS)                            # provider per LLM_PROVIDER (gemini|openai)
    if ex is not None:
        return ex, extractor_label(SETTINGS)
    from tagent.news.youtube_source import default_extract_catalysts
    return default_extract_catalysts, "grounded default (no LLM key)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-videos", type=int, default=None, help="cap TOTAL videos processed")
    ap.add_argument("--max-per-channel", type=int, default=6)
    ap.add_argument("--lookback-hours", type=int, default=48)
    ap.add_argument("--reextract", action="store_true",
                    help="re-run Gemini insights on CACHED transcripts (no re-transcription)")
    args = ap.parse_args()

    if not SETTINGS.has_youtube_key():
        print("[ERROR] YOUTUBE_API_KEY not set — cannot list channel uploads.")
        return 1
    channels = [c for c in channels_from_env(SETTINGS.youtube_channels) if c.channel_id]
    if not channels:
        print("[ERROR] no configured channels with ids (set YOUTUBE_CHANNELS=name=UC...,...).")
        return 1

    extract_fn, backend = _extract_fn()
    proxy = "on" if SETTINGS.has_youtube_proxy() else "off (Whisper fallback — slow)"   # URL never printed
    print(f"Channels  : {', '.join(c.name for c in channels)}")
    print(f"Extractor : {backend}")
    print(f"Fallbacks : api -> yt-dlp subs (proxy: {proxy}) -> yt-dlp+Whisper")
    print(f"Cap       : max_videos={args.max_videos}, max_per_channel={args.max_per_channel}, "
          f"lookback={args.lookback_hours}h\n")

    src = YouTubeSource(SETTINGS.youtube_api_key, extract_fn=extract_fn,
                        enable_fallbacks=True, proxy_url=SETTINGS.youtube_proxy_url,
                        webshare=SETTINGS.webshare_proxy(),
                        pace_seconds=SETTINGS.youtube_transcript_pace_seconds)
    reports = src.batch_transcribe(channels, lookback_hours=args.lookback_hours,
                                   max_per_channel=args.max_per_channel, max_videos=args.max_videos,
                                   audit=AuditWriter(), reextract=args.reextract)

    n_tx = sum(1 for r in reports if r["n_segments"] > 0)
    n_ins = sum(r["n_insights"] for r in reports)
    n_skip = sum(1 for r in reports if str(r.get("status", "")).startswith("skipped"))
    cur = None
    for r in reports:
        if r["channel"] != cur:
            cur = r["channel"]
            print(f"=== {cur} ===")
        print(f"  • {str(r['title'])[:60]}  [{r['video_id']}]")
        print(f"      {r.get('status', '')}  ·  {r['n_segments']} seg via {r['method'] or 'NONE'}  ·  "
              f"{r['n_insights']} insight(s)")
        for ins in r["insights"]:
            print(f"      → {ins['stock_name']} ({ins['stock']})  {ins['timestamp_mmss']}  {ins['deeplink']}")
            print(f"        {ins['summary']}")

    print(f"\nSummary: {len(reports)} videos ({n_skip} skipped/cached), {n_tx} with transcripts, "
          f"{n_ins} grounded insights.")
    print(f"-> log: {fetch_log_path()}  (one line per video: full transcript + insights)")
    print("   The daily briefing #3 replays these logged insights (no re-transcription).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
