"""Korean/global MEDIA-MONITORING (YouTube / TV finance shows) — GROUNDED briefing.

The boss's ask: don't paraphrase — cite the EXACT video, the EXACT minute, and link to the
moment. For each recent video from a configurable channel list we pull the transcript WITH
timestamps and extract, per watchlist stock, the SENSITIVE / actionable catalysts actually
STATED in the transcript (named deals / M&A / partnerships, earnings surprises, regulatory /
geopolitical events, the reason a stock moved) — suppressing generic "AI 잠재력" filler. Every
extracted claim carries the verbatim transcript QUOTE, the segment start-second, a mm:ss label
and a deep-link to that moment: ``youtube.com/watch?v=<id>&t=<int(start)>s``.

Output per claim: ``{stock, channel, video_title, quote, timestamp_mmss, deeplink, ...}`` —
newest / most-important first. If a video says nothing about a stock it is OMITTED — never
invented. Every claim is GROUNDED in a real transcript segment + link; no source -> no claim.

IMPORTANT — INFORMATIONAL briefing, **not a trading signal** and **not a validated edge**.
Kept OUT of every trading / HALT / kill-switch path (``alerts.safety_signal`` never sees it).

Design: everything network/heavy is lazy + injectable so unit tests run fully mocked.
  * video listing -> YouTube Data API ``search.list`` (key from .env, never logged),
  * transcripts   -> ``youtube-transcript-api`` ([{text, start, duration}], Korean first),
                     OPTIONAL yt-dlp + Whisper fallback only when captions are missing,
  * extraction    -> an injectable ``extract_fn`` (plug in an LLM); the default is a
                     transparent, GROUNDED rule-based pass that only ever quotes real segments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence

from tagent.news.finnhub_source import sentiment_label

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
PLAYLISTITEMS_URL = "https://www.googleapis.com/youtube/v3/playlistItems"
WATCH_URL = "https://www.youtube.com/watch?v="

# Awareness-only label reused everywhere so the honest framing can't drift.
MEDIA_LABEL = "awareness only — not a trading signal (every claim grounded in a real quote + link)"


# --------------------------------------------------------------------------- #
# channel list (configurable; IDs come from .env so none are hard-coded wrong)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Channel:
    name: str
    channel_id: str = ""        # YouTube channelId (UC...); empty -> skipped until configured


DEFAULT_CHANNELS: List[Channel] = [
    Channel("한국경제TV"), Channel("매일경제 MBN"), Channel("Bloomberg Television"),
]


def channels_from_env(value: str) -> List[Channel]:
    """Parse YOUTUBE_CHANNELS ("name=UCid,name2=UCid2") into Channels (order preserved)."""
    out: List[Channel] = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, cid = part.partition("=")
        out.append(Channel(name.strip(), cid.strip()))
    return out


# --------------------------------------------------------------------------- #
# ticker mapping — map mentions (Korean / English names, bare codes) to our watchlist
# --------------------------------------------------------------------------- #
WATCHLIST_TICKERS: Dict[str, str] = {
    "삼성전자": "005930", "samsung electronics": "005930", "005930": "005930",
    "sk하이닉스": "000660", "sk 하이닉스": "000660", "하이닉스": "000660",
    "sk hynix": "000660", "000660": "000660",
    "현대차": "005380", "현대자동차": "005380", "hyundai motor": "005380", "005380": "005380",
    "네이버": "035420", "naver": "035420", "035420": "035420",
    "lg화학": "051910", "lg chem": "051910", "051910": "051910",
    "포스코": "005490", "posco": "005490", "005490": "005490",
    "기아": "000270", "kia": "000270", "000270": "000270",
    "삼성바이오로직스": "207940", "samsung biologics": "207940", "207940": "207940",
    "한미반도체": "042700", "hanmi semiconductor": "042700", "042700": "042700",
    "엔비디아": "NVDA", "nvidia": "NVDA", "마이크론": "MU", "micron": "MU",
    "브로드컴": "AVGO", "broadcom": "AVGO", "테슬라": "TSLA", "tesla": "TSLA",
    "애플": "AAPL", "apple": "AAPL",
}

# Canonical display name per ticker (for the "STOCK — channel: ..." line).
TICKER_NAMES: Dict[str, str] = {
    "005930": "삼성전자", "000660": "SK하이닉스", "005380": "현대차", "035420": "네이버",
    "051910": "LG화학", "005490": "포스코", "000270": "기아", "207940": "삼성바이오로직스",
    "042700": "한미반도체", "NVDA": "엔비디아", "MU": "마이크론", "AVGO": "브로드컴",
    "TSLA": "테슬라", "AAPL": "애플",
}

_EN_ALIASES = {a: t for a, t in WATCHLIST_TICKERS.items() if re.fullmatch(r"[a-z0-9 .&-]+", a)}
_NON_EN_ALIASES = {a: t for a, t in WATCHLIST_TICKERS.items() if a not in _EN_ALIASES}


def extract_tickers(text: str) -> List[str]:
    """Watchlist tickers mentioned in ``text`` (sorted, unique). Korean names, English names,
    and bare codes all resolve to the same ticker."""
    if not text:
        return []
    low = text.lower()
    hits = set()
    for alias, ticker in _NON_EN_ALIASES.items():
        if alias in low:
            hits.add(ticker)
    for alias, ticker in _EN_ALIASES.items():
        if re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", low):
            hits.add(ticker)
    return sorted(hits)


# --------------------------------------------------------------------------- #
# sentiment (bilingual, transparent) + the old one-line summary (kept, back-compat)
# --------------------------------------------------------------------------- #
_BULLISH_KW = ["상승", "급등", "강세", "호재", "최고", "사상최대", "돌파", "수혜", "기대",
               "수주", "공급계약", "흑자전환", "호실적",
               "surge", "rally", "beat", "beats", "record", "upgrade", "jump", "soar", "strong"]
_BEARISH_KW = ["하락", "급락", "약세", "악재", "우려", "경고", "부진", "위기", "충격", "폭락",
               "적자전환", "리콜", "감자",
               "miss", "plunge", "downgrade", "warning", "weak", "slump", "fall", "drop", "cut"]


def media_sentiment(text: str) -> float:
    """Bilingual (KO/EN) rule-based sentiment in [-1, 1] from keyword counts. Transparent —
    NOT a price model."""
    t = str(text or "").lower()
    bull = sum(t.count(k) for k in _BULLISH_KW)
    bear = sum(t.count(k) for k in _BEARISH_KW)
    if bull == bear:
        return 0.0
    return max(-1.0, min(1.0, (bull - bear) / float(bull + bear)))


def one_line_summary(text: str, max_len: int = 140) -> str:
    """First sentence/clause, trimmed (heuristic; an LLM extract_fn does better)."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return ""
    first = re.split(r"(?<=[.!?。])\s|[\n·]", s)[0].strip() or s
    return first if len(first) <= max_len else first[:max_len - 1].rstrip() + "…"


def default_analyze(text: str, title: str = "") -> dict:
    """Back-compat sentiment+summary pass (no longer the catalyst extractor)."""
    score = media_sentiment(f"{title} {text}")
    return {"sentiment_score": round(score, 3), "sentiment": sentiment_label(score),
            "summary": one_line_summary(text or title)}


# --------------------------------------------------------------------------- #
# deep-link helpers — exact moment in the video
# --------------------------------------------------------------------------- #
def deeplink(video_id: str, start_seconds) -> str:
    """youtube.com/watch?v=<id>&t=<int(start)>s — jumps to the cited moment."""
    if not video_id:
        return ""
    return f"{WATCH_URL}{video_id}&t={int(max(0.0, float(start_seconds or 0.0)))}s"


def mmss(start_seconds) -> str:
    """Seconds -> 'mm:ss' (or 'h:mm:ss' past an hour)."""
    s = int(max(0.0, float(start_seconds or 0.0)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


# --------------------------------------------------------------------------- #
# transcript fetching WITH timestamps ([{text, start, duration}])
# --------------------------------------------------------------------------- #
def _normalize_segments(segs) -> List[dict]:
    """Coerce raw transcript output into [{text, start, duration}]. A bare string (e.g. a
    Whisper fallback with no timing) becomes a single segment at t=0."""
    if isinstance(segs, str):
        return [{"text": segs.strip(), "start": 0.0, "duration": 0.0}] if segs.strip() else []
    out: List[dict] = []
    for s in segs or []:
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        try:
            start = float(s.get("start", 0.0) or 0.0)
        except (TypeError, ValueError):
            start = 0.0
        out.append({"text": text, "start": start, "duration": float(s.get("duration", 0.0) or 0.0)})
    return out


def join_transcript(segments: Sequence[dict]) -> str:
    return " ".join(s["text"] for s in _normalize_segments(segments)).strip()


# ---- timed-text (WebVTT / SRT) parsing — shared by yt-dlp subtitle parsing ------------ #
_TS_RE = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


def _ts_seconds(ts: str) -> float:
    """'HH:MM:SS.mmm' / 'MM:SS,mmm' -> float seconds (first timestamp found)."""
    m = _TS_RE.search(ts or "")
    if not m:
        return 0.0
    h = int(m.group(1) or 0)
    return h * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4).ljust(3, "0")) / 1000.0


def parse_timed_text(text: str) -> List[dict]:
    """Parse WebVTT or SRT into [{text, start, duration}]: strips inline ``<...>`` caption tags and
    drops consecutive duplicate lines (auto-subs roll the same text). Pure / no deps."""
    out: List[dict] = []
    last = None
    for block in re.split(r"\n\s*\n", (text or "").replace("\r\n", "\n").strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        ti = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if ti is None:
            continue
        a, _, b = lines[ti].partition("-->")
        start, end = _ts_seconds(a), _ts_seconds(b)
        body = re.sub(r"<[^>]+>", "", " ".join(lines[ti + 1:]))    # strip <c>/<00:00:00.000> tags
        body = re.sub(r"\s+", " ", body).strip()
        if not body or body == last:                              # drop empties + rolling duplicates
            continue
        last = body
        out.append({"text": body, "start": start, "duration": max(0.0, end - start)})
    return out


# ---- strategy (a): youtube-transcript-api (Korean first), optional proxy --------------- #
def _api_transcript(video_id: str, languages: Sequence[str] = ("ko", "en"),
                    proxy_url: str = "", webshare: Optional[dict] = None) -> List[dict]:
    """youtube-transcript-api transcript ([{text, start, duration}]); supports the 0.6.x static and
    1.x instance APIs, with an OPTIONAL proxy (generic PROXY_URL or Webshare) to bypass the IP-block.
    Raises on failure so the caller falls through the chain. The proxy creds are never logged."""
    from youtube_transcript_api import YouTubeTranscriptApi
    langs = list(languages)
    if hasattr(YouTubeTranscriptApi, "get_transcript"):           # 0.6.x static API
        kw = {"proxies": {"http": proxy_url, "https": proxy_url}} if proxy_url else {}
        return YouTubeTranscriptApi.get_transcript(video_id, languages=langs, **kw)
    proxy_config = None                                           # 1.x instance API
    if webshare:
        from youtube_transcript_api.proxies import WebshareProxyConfig
        proxy_config = WebshareProxyConfig(**webshare)
    elif proxy_url:
        from youtube_transcript_api.proxies import GenericProxyConfig
        proxy_config = GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
    api = YouTubeTranscriptApi(proxy_config=proxy_config) if proxy_config else YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=langs)
    if hasattr(fetched, "to_raw_data"):
        return fetched.to_raw_data()
    return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]


def _ytdlp_argv(*extra) -> List[str]:
    """yt-dlp argv via ``python -m yt_dlp`` (robust even when the yt-dlp.exe shim isn't on PATH)."""
    import sys
    return [sys.executable, "-m", "yt_dlp", *extra]


def _run_cmd(cmd: Sequence[str]) -> None:
    """Run a subprocess quietly (yt-dlp). Never raises on a non-zero exit; never prints output."""
    import subprocess
    subprocess.run(list(cmd), check=False, capture_output=True, timeout=420)


def webshare_proxy_url(webshare: Optional[dict]) -> str:
    """A yt-dlp/requests proxy URL for Webshare rotating-residential creds, or "". The username gets
    the ``-rotate`` suffix Webshare uses for its rotating endpoint. Creds are never logged."""
    u = (webshare or {}).get("proxy_username")
    p = (webshare or {}).get("proxy_password")
    return f"http://{u}-rotate:{p}@p.webshare.io:80" if (u and p) else ""


# ---- strategy (b): yt-dlp auto-subtitles (a different endpoint — often NOT IP-blocked) - #
def _ytdlp_subs(video_id: str, languages: Sequence[str] = ("ko", "en"), run_fn=None,
                proxy_url: str = "") -> List[dict]:
    """yt-dlp ``--write-auto-subs --skip-download`` -> parse the .vtt/.srt. Korean preferred,
    English next. Uses ``proxy_url`` (``--proxy``) when set so captions load past the IP-block in
    seconds. Returns [] if no subtitles are produced."""
    import glob
    import os
    import tempfile
    runner = run_fn or _run_cmd
    langs = list(languages) + ["en"]
    proxy = ["--proxy", proxy_url] if proxy_url else []
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "%(id)s.%(ext)s")
        runner(_ytdlp_argv(*proxy, "--write-auto-subs", "--write-subs", "--sub-langs",
                           ",".join(langs), "--sub-format", "vtt/srt/best", "--skip-download",
                           "--no-warnings", "--no-playlist", "-o", out, f"{WATCH_URL}{video_id}"))
        files = glob.glob(os.path.join(td, "*.vtt")) + glob.glob(os.path.join(td, "*.srt"))

        def _rank(p):                                             # prefer a requested-language file
            low = os.path.basename(p).lower()
            return next((i for i, l in enumerate(langs) if f".{l}." in low), 99)

        for f in sorted(files, key=_rank):
            try:
                segs = parse_timed_text(open(f, encoding="utf-8", errors="replace").read())
            except Exception:
                segs = []
            if segs:
                return segs
    return []


# ---- strategy (c): yt-dlp audio + local Whisper (no caption endpoint -> IP-block-immune) #
def _cuda_available() -> bool:
    """True if a CUDA GPU is usable (torch present + a visible device). Never raises."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# Seed Whisper with correct KR finance terms / proper nouns so ASR garble (휴머노이드 etc.) is reduced.
WHISPER_GLOSSARY = ("삼성전자, SK하이닉스, 그룹, 휴머노이드, 보스턴다이내믹스, 아틀라스, 밸류에이션, "
                    "컨센서스, 나스닥, 파운드리, PER, 차별성, 수급, 소버린 AI, 메모리, HBM, TSMC, "
                    "엔비디아, 마이크론, 목표주가, 점유율")


def _whisper_prompt() -> str:
    """The faster-whisper ``initial_prompt`` glossary (WHISPER_PROMPT env, default WHISPER_GLOSSARY)."""
    import os as _os
    v = _os.getenv("WHISPER_PROMPT")
    return v if v is not None else WHISPER_GLOSSARY


WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")


def _whisper_model(model: Optional[str] = None) -> str:
    """The Whisper model size, from WHISPER_MODEL (.env, default "small"). Valid: tiny/base/small/
    medium/large-v3 — bigger = more accurate but slower. Anything unrecognized falls back to "small"."""
    import os as _os
    m = (model if model is not None else _os.getenv("WHISPER_MODEL", "small")) or "small"
    m = str(m).strip().lower()
    return m if m in WHISPER_MODELS else "small"


def _whisper_vad(value: Optional[str] = None) -> bool:
    """Whether to enable Silero VAD on the Whisper pass (drops silence/music so it can't loop on
    non-speech). From WHISPER_VAD (.env), default ON; "0/false/no/off" turn it off."""
    import os as _os
    v = value if value is not None else _os.getenv("WHISPER_VAD", "1")
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


def _whisper_device_compute(device: Optional[str] = None, compute_type: Optional[str] = None):
    """Resolve ``(device, compute_type)`` for faster-whisper. ``device`` defaults to WHISPER_DEVICE
    (.env, default "auto"); "auto" -> "cuda" if a CUDA GPU is available, else "cpu". ``compute_type``
    defaults to WHISPER_COMPUTE_TYPE (.env); when blank it is "float16" on cuda, "int8" on cpu —
    cuda/float16 is ~50x faster than cpu/int8. Pure: imports torch only to probe, loads no model."""
    import os as _os
    dev = (device if device is not None else _os.getenv("WHISPER_DEVICE", "auto")) or "auto"
    ctype = (compute_type if compute_type is not None
             else (_os.getenv("WHISPER_COMPUTE_TYPE") or _os.getenv("WHISPER_COMPUTE") or ""))
    if dev == "auto":
        dev = "cuda" if _cuda_available() else "cpu"
    if not ctype:
        ctype = "float16" if dev == "cuda" else "int8"
    return dev, ctype


def _whisper_transcribe(audio_path: str, language: str = "ko") -> List[dict]:
    """Local Whisper transcription -> [{text, start, duration}]. GPU-aware: faster-whisper runs on
    cuda/float16 when a CUDA GPU is available (device "auto"), else cpu/int8; if CUDA init fails it
    falls back safely to cpu/int8. Model size (WHISPER_MODEL, default "small"), device + compute
    (WHISPER_DEVICE / WHISPER_COMPUTE_TYPE) and VAD (WHISPER_VAD, default on) are configurable via
    .env. Anti-hallucination: Silero VAD drops silence/music and condition_on_previous_text=False
    stops repetition loops feeding themselves across windows. Falls back to openai-whisper, then []
    (graceful) if neither is installed."""
    try:
        from faster_whisper import WhisperModel
        model_size = _whisper_model()
        device, compute_type = _whisper_device_compute()
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception:
            if device == "cuda":                         # GPU init failed -> safe CPU fallback
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
            else:
                raise
        segs, _ = model.transcribe(audio_path, language=language,
                                   vad_filter=_whisper_vad(),          # Silero VAD: no looping on non-speech
                                   condition_on_previous_text=False,   # no self-feeding repetition loops
                                   initial_prompt=_whisper_prompt())   # glossary: cut KR finance ASR garble
        return [{"text": s.text.strip(), "start": float(s.start), "duration": float(s.end - s.start)}
                for s in segs if str(s.text).strip()]
    except Exception:
        pass
    try:
        import whisper
        model = whisper.load_model(_whisper_model())
        res = model.transcribe(audio_path, language=language, initial_prompt=_whisper_prompt())
        return [{"text": str(s.get("text", "")).strip(), "start": float(s.get("start", 0.0) or 0.0),
                 "duration": float((s.get("end", 0.0) or 0.0) - (s.get("start", 0.0) or 0.0))}
                for s in res.get("segments", []) if str(s.get("text", "")).strip()]
    except Exception:
        return []


# Skip live / over-long videos on the (slow) Whisper path so a livestream can't hang the batch.
# (A configured proxy makes the cheap caption paths handle these in seconds instead.)
WHISPER_MAX_DURATION_S = 5400        # 90 min — beyond this, transcription is impractical on CPU


def _ytdlp_whisper(video_id: str, languages: Sequence[str] = ("ko", "en"),
                   run_fn=None, transcribe_fn=None) -> List[dict]:
    """yt-dlp audio download + local Whisper. Hits NO caption endpoint, so it is immune to the
    transcript IP-block. Skips live streams and videos longer than ~90 min (which would hang CPU
    Whisper). Returns [] if audio or Whisper is unavailable, or the video is live/too long."""
    import glob
    import os
    import tempfile
    runner = run_fn or _run_cmd
    transcribe = transcribe_fn or _whisper_transcribe
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "%(id)s.%(ext)s")
        runner(_ytdlp_argv("-x", "--audio-format", "mp3", "--no-warnings", "--no-playlist",
                           "--match-filter", f"!is_live & duration < {WHISPER_MAX_DURATION_S}",
                           "-o", out, f"{WATCH_URL}{video_id}"))
        audio = (glob.glob(os.path.join(td, "*.mp3")) + glob.glob(os.path.join(td, "*.m4a"))
                 + glob.glob(os.path.join(td, "*.webm")))
        if not audio:
            return []
        return transcribe(audio[0], "ko" if "ko" in languages else "en")


# ---- the fallback chain: run strategies in order, STOP at first success ---------------- #
def _run_transcript_chain(video_id: str, strategies, languages: Sequence[str] = ("ko", "en"), *,
                          cache=None, pace: float = 0.0, sleep_fn=None, paced_state=None):
    """Run ``strategies`` (a list of ``(name, fn_or_None)``) in order and STOP at the first that
    returns segments. Returns ``(segments, method)`` with method in {api, ytdlp, whisper, ""}.
    Whisper segments are labeled ``source="whisper"`` (machine-transcribed). Caches by video id and
    paces between real fetches (a small delay so we don't re-trip the IP-block)."""
    if cache is not None and video_id in cache:
        return cache[video_id]
    if paced_state is not None and paced_state.get("any") and pace and pace > 0:
        (sleep_fn or _sleep)(pace)                                # delay BETWEEN videos (not the first)
    if paced_state is not None:
        paced_state["any"] = True
    result = ([], "")
    for name, fn in strategies:
        if fn is None:
            continue
        try:
            segs = _normalize_segments(fn(video_id, languages))
        except Exception:
            segs = []
        if segs:
            if name == "whisper":
                for s in segs:
                    s["source"] = "whisper"                       # machine-transcribed, labeled
            result = (segs, name)
            break
    if cache is not None:
        cache[video_id] = result
    return result


def _sleep(seconds: float) -> None:
    import time
    time.sleep(seconds)


TRANSCRIPT_MODES = ("auto", "whisper", "api")


def _transcript_mode(mode: Optional[str] = None) -> str:
    """Normalize the transcript mode (defaults from YOUTUBE_TRANSCRIPT_MODE, .env, "auto"):
      * "auto"    — chain api -> yt-dlp subs -> Whisper (stop at first success);
      * "whisper" — Whisper FIRST (full video transcript regardless of captions / IP-blocks), with
                    api + subs only as a fallback if Whisper fails;
      * "api"     — youtube-transcript-api only.
    Anything unrecognized falls back to "auto"."""
    import os as _os
    m = (mode if mode is not None else _os.getenv("YOUTUBE_TRANSCRIPT_MODE", "auto")) or "auto"
    m = str(m).strip().lower()
    return m if m in TRANSCRIPT_MODES else "auto"


def fetch_segments(video_id: str, languages: Sequence[str] = ("ko", "en"),
                   transcript_fn: Optional[Callable] = None, ytdlp_fn: Optional[Callable] = None,
                   whisper_fn: Optional[Callable] = None) -> List[dict]:
    """Timestamped transcript segments via the fallback chain (Korean first): youtube-transcript-api
    -> yt-dlp auto-subs -> yt-dlp+Whisper, stopping at the first success. Strategies are injectable
    for tests; ``transcript_fn`` defaults to the real youtube-transcript-api. ``whisper_fn`` takes a
    video id (back-compat). Returns [] only if every strategy fails."""
    wfn = (lambda vid, langs: whisper_fn(vid)) if whisper_fn is not None else None
    strategies = [
        ("api", transcript_fn or (lambda vid, langs: _api_transcript(vid, langs))),
        ("ytdlp", ytdlp_fn),
        ("whisper", wfn),
    ]
    return _run_transcript_chain(video_id, strategies, languages)[0]


def fetch_transcript(video_id: str, languages: Sequence[str] = ("ko", "en"),
                     transcript_fn: Optional[Callable] = None,
                     whisper_fn: Optional[Callable] = None) -> str:
    """Back-compat: the joined transcript text (no timestamps)."""
    return join_transcript(fetch_segments(video_id, languages, transcript_fn=transcript_fn,
                                          whisper_fn=whisper_fn))


# --------------------------------------------------------------------------- #
# GROUNDED catalyst extraction — only quotes that actually state a catalyst
# --------------------------------------------------------------------------- #
# Ordered by priority; the first matching category wins. Generic hype ("AI 잠재력", "유망",
# "장기 성장") contains NONE of these, so such segments yield NO claim (filler suppressed).
_CATALYST_RULES = [
    ("M&A/deal", ["인수합병", "인수", "합병", "m&a", "지분", "매각", "파트너십", "협력",
                  "공급계약", "납품계약", "수주", "수출계약", "계약 체결", "조인트벤처", "jv",
                  "merger", "acquisition", "acquire", "partnership", "stake", "supply deal",
                  "supply contract", "contract win", "order win", "joint venture"]),
    ("regulatory/geopolitical", ["규제", "제재", "관세", "반독점", "공정위", "수출규제",
                                 "수출통제", "소송", "조사", "지정학", "미중", "보조금", "칩스법",
                                 "tariff", "sanction", "antitrust", "lawsuit", "probe",
                                 "export control", "subsidy", "chips act", "ban"]),
    ("earnings", ["어닝서프라이즈", "어닝쇼크", "실적", "영업이익", "순이익", "호실적", "흑자전환",
                  "적자전환", "잠정실적", "가이던스", "컨센서스", "earnings", "guidance",
                  "beat", "miss", "profit", "revenue", "surprise"]),
    ("price-move reason", ["급등", "급락", "폭등", "폭락", "강세", "약세", "상한가", "하한가",
                           "신고가", "신저가"]),
]
_CAT_WEIGHT = {"M&A/deal": 4.0, "regulatory/geopolitical": 3.0, "earnings": 3.0,
               "price-move reason": 2.0}


def detect_catalyst(text: str) -> Optional[str]:
    """The SENSITIVE catalyst category stated in ``text`` (highest priority first), or None if
    the text is generic/hype with no concrete catalyst — so filler is suppressed, not surfaced."""
    low = str(text or "").lower()
    for cat, kws in _CATALYST_RULES:
        if any(k in low for k in kws):
            return cat
    return None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _grounded_summary(text: str, max_len: int = 160) -> str:
    """Extractive 1-2 sentence summary of the stated point (the no-LLM default; verbatim-grounded).
    An injected LLM analyze_fn produces a richer abstractive summary instead."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    parts = re.split(r"(?<=[.!?。])\s", s)
    summ = " ".join(parts[:2]).strip() or s
    return summ if len(summ) <= max_len else summ[:max_len - 1].rstrip() + "…"


def _grounded_summary_from_segments(span: Sequence[dict], max_sentences: int = 3,
                                    max_len: int = 320) -> str:
    """A 2-3 SENTENCE grounded summary built from a context span (lead-in + catalyst + follow-on):
    each transcript segment becomes one sentence (a trailing period is added between segments only),
    so the summary reads as claim → context → implication WITHOUT inventing any facts. The verbatim
    quote still anchors it."""
    sents = []
    for seg in span:
        t = re.sub(r"\s+", " ", str(seg.get("text", ""))).strip()
        if not t:
            continue
        if t[-1] not in ".!?。…":
            t += "."
        sents.append(t)
        if len(sents) >= max_sentences:
            break
    summ = " ".join(sents).strip() or _grounded_summary(" ".join(s.get("text", "") for s in span))
    return summ if len(summ) <= max_len else summ[:max_len - 1].rstrip() + "…"


def _resolve_ticker(stock: str) -> str:
    """A watchlist ticker for a code/name string (else "")."""
    s = str(stock or "").strip()
    if s in TICKER_NAMES:
        return s
    hits = extract_tickers(s)
    return hits[0] if hits else ""


def _make_insight(ticker: str, summary: str, quote: str, start, category: str, score: float,
                  video_id: str, video_title: str, channel: str, published_at: str,
                  action=None, target_price=None, name=None) -> dict:
    """One GROUNDED insight: a summary of the actual point + a supporting transcript QUOTE SPAN,
    deep-linked to ``start`` (where the discussion begins). ``action`` (매수/매도/…) and
    ``target_price`` (목표가/적정가) are carried when the analyst stated them — both grounded. For a
    stock NOT in the ticker map, pass ``name`` (the spoken name) so it's kept correctly named with an
    empty ticker (used by the single-video report which doesn't filter to the giant universe)."""
    link = deeplink(video_id, start)
    disp = TICKER_NAMES.get(ticker) or name or ticker        # display name (mapped name, else spoken)
    return {
        "stock": ticker or (name or ""), "stock_name": disp, "ticker": ticker,
        "channel": channel, "video_title": video_title,
        "summary": str(summary or "").strip(), "quote": str(quote or "").strip(),
        "start": int(max(0.0, float(start or 0.0))), "timestamp_mmss": mmss(start),
        "deeplink": link, "source_link": link, "category": category,
        "action": (str(action).strip() if action else None),
        "target_price": (str(target_price).strip() if target_price else None),
        "sentiment": sentiment_label(score), "sentiment_score": round(score, 3),
        "importance": round(_CAT_WEIGHT.get(category, 1.0) + abs(score), 3),
        "video_id": video_id, "published_at": published_at, "source": "youtube",
    }


def default_extract_catalysts(segments: Sequence[dict], video: dict,
                              channel_name: str = "") -> List[dict]:
    """GROUNDED, context-aware default (no LLM). For each segment that STATES a catalyst, attribute
    it to the watchlist stock(s) NAMED in that segment, summarize the stated point, and attach a
    multi-segment QUOTE SPAN (lead-in + catalyst + follow-on) deep-linked to where that stock's
    discussion starts. Passing mentions (a stock named with no catalyst) and filler are dropped;
    one insight per stock per video; nothing is invented (the span is verbatim transcript)."""
    segs = _normalize_segments(segments)
    if not segs:
        return []
    vid = video.get("video_id", "")
    title = video.get("video_title") or video.get("title", "")
    pub = video.get("published_at", "")
    seg_tks = [extract_tickers(s["text"]) for s in segs]
    out, seen = [], set()
    for i, s in enumerate(segs):
        category = detect_catalyst(s["text"])
        if category is None or not seg_tks[i]:           # no catalyst here, or no stock named -> skip
            continue
        lo, hi = max(0, i - 1), min(len(segs), i + 2)    # lead-in + catalyst + follow-on (context span)
        span = segs[lo:hi]
        quote = " ".join(x["text"] for x in span).strip()
        for tk in seg_tks[i]:
            if (tk, vid) in seen:                        # one insight per stock per video
                continue
            seen.add((tk, vid))
            start = next((x["start"] for x in span if tk in extract_tickers(x["text"])), s["start"])
            out.append(_make_insight(tk, _grounded_summary_from_segments(span), quote, start, category,
                                     media_sentiment(quote), vid, title, channel_name, pub))
    out.sort(key=lambda x: x["importance"], reverse=True)
    return out


def _ground_quote(segs: Sequence[dict], quote: str) -> Optional[float]:
    """Anchor an LLM-returned quote to the real transcript: return the start-second of the segment
    where the quote (or a substantial prefix of it) actually appears, or None if it is NOT in the
    transcript (a hallucination) — so an unsupported claim is DROPPED, never surfaced. ALL whitespace
    is stripped from both the quote and the transcript before matching, so a quote that SPANS several
    short (whisper) segments — or whose spacing differs from the captions — still matches (the prior
    space-sensitive version rejected most spanning quotes -> 0 insights)."""
    def _strip(s):
        return re.sub(r"\s+", "", str(s or "")).lower()      # remove ALL whitespace (spacing-agnostic)
    q = _strip(quote)
    if len(q) < 8:
        return None
    joined, starts = "", []
    for s in segs:
        piece = _strip(s["text"])
        joined += piece
        starts.extend([s["start"]] * len(piece))
    pos = joined.find(q)
    if pos == -1:
        pos = joined.find(q[:max(12, len(q) // 2)])      # the LLM may extend/trim the exact wording
    if pos == -1:
        return None
    return starts[min(pos, len(starts) - 1)]


def _extract_debug(msg: str) -> None:
    """One-line extraction diagnostic to stderr, opt-in via YT_EXTRACT_DEBUG. Never prints secrets."""
    import os
    import sys
    if os.getenv("YT_EXTRACT_DEBUG"):
        print(f"[yt-extract] {msg}", file=sys.stderr)


def llm_extract_fn(analyze_fn: Callable[[str, dict], list], *, watchlist: Optional[dict] = None):
    """Build a GROUNDED ``extract_fn`` from an injected LLM ``analyze_fn(full_transcript, watchlist)``
    that READS the full timestamped transcript and returns, per substantively-discussed stock,
    ``{stock, summary, quote, category?}``. Each insight is grounded: the quote must appear in the
    transcript (else dropped) and ``start`` is snapped to that segment. The summary is kept only when
    its quote checks out — never invented. Plug in via ``YouTubeSource(extract_fn=llm_extract_fn(fn))``."""
    names = dict(watchlist or TICKER_NAMES)

    def _extract(segments: Sequence[dict], video: dict, channel_name: str = "") -> List[dict]:
        segs = _normalize_segments(segments)
        if not segs:
            return []
        vid = video.get("video_id", "")
        title = video.get("video_title") or video.get("title", "")
        pub = video.get("published_at", "")
        transcript = "\n".join(f"[{mmss(s['start'])}] {s['text']}" for s in segs)   # FULL, with timestamps
        try:
            raw = analyze_fn(transcript, names) or []
        except Exception as e:
            _extract_debug(f"analyze_fn raised: {str(e)[:120]}")
            return []
        _extract_debug(f"analyze parsed {len(raw)} raw item(s) from {len(segs)} segments")
        out, seen_q, seen_s, per_stock = [], set(), set(), {}
        drops = {"empty": 0, "dup": 0, "cap": 0, "ungrounded": 0}
        for r in raw:
            raw_stock = str(r.get("stock") or r.get("ticker") or "").strip()
            tk = _resolve_ticker(raw_stock)
            summary = str(r.get("summary", "")).strip()
            quote = str(r.get("quote", "")).strip()
            if not raw_stock or not summary or not quote:    # need a named stock + content
                drops["empty"] += 1
                continue
            ident = tk or _norm(raw_stock)                   # group unmapped stocks by their spoken name
            qn, sn = _norm(quote), _norm(summary)
            # NO DUPLICATE-PER-STOCK TEXT: drop a summary already emitted (a point applying to several
            # stocks must be ONE combined insight that names them, not the same sentence per stock).
            if (ident, qn) in seen_q or sn in seen_s:
                drops["dup"] += 1
                continue
            if per_stock.get(ident, 0) >= 6 or len(out) >= 40:   # richer cap (several distinct points/stock)
                drops["cap"] += 1
                continue
            start = _ground_quote(segs, quote)           # GROUNDING: drop if the quote isn't real
            if start is None:
                drops["ungrounded"] += 1
                _extract_debug(f"DROP ungrounded [{raw_stock}] quote={quote[:60]!r}")
                continue
            seen_q.add((ident, qn))
            seen_s.add(sn)
            per_stock[ident] = per_stock.get(ident, 0) + 1
            cat = r.get("category") or detect_catalyst(quote) or detect_catalyst(summary) or "discussed"
            out.append(_make_insight(tk, summary, quote, start, cat,
                                     media_sentiment(f"{summary} {quote}"), vid, title, channel_name, pub,
                                     action=r.get("action"), target_price=r.get("target_price"),
                                     name=raw_stock))      # keep non-watchlist stocks correctly named
        _extract_debug(f"kept {len(out)} insight(s); dropped {drops}")
        out.sort(key=lambda x: x["importance"], reverse=True)
        return out

    return _extract


# --------------------------------------------------------------------------- #
# the source: list videos -> timestamped transcript -> grounded catalyst claims
# --------------------------------------------------------------------------- #
def _parse_search_items(data: dict) -> List[dict]:
    items = (data or {}).get("items") if isinstance(data, dict) else None
    out: List[dict] = []
    for it in items or []:
        vid = (((it.get("id") or {}).get("videoId")) if isinstance(it.get("id"), dict)
               else it.get("id"))
        sn = it.get("snippet") or {}
        if not vid:
            continue
        out.append({"video_id": vid, "title": sn.get("title", ""),
                    "channel": sn.get("channelTitle", ""), "published_at": sn.get("publishedAt", "")})
    return out


def _parse_playlist_items(data: dict) -> List[dict]:
    """playlistItems.list -> [{video_id, title, channel, published_at}], newest-first, deduped by
    video id. Each item's videoId lives in contentDetails.videoId (or snippet.resourceId.videoId)."""
    items = (data or {}).get("items") if isinstance(data, dict) else None
    out: List[dict] = []
    seen = set()
    for it in items or []:
        cd = it.get("contentDetails") or {}
        sn = it.get("snippet") or {}
        vid = cd.get("videoId") or ((sn.get("resourceId") or {}).get("videoId"))
        if not vid or vid in seen:
            continue
        seen.add(vid)
        out.append({"video_id": vid, "title": sn.get("title", ""),
                    "channel": sn.get("channelTitle", ""),
                    "published_at": cd.get("videoPublishedAt") or sn.get("publishedAt", "")})
    return out


def _cached_complete(entry: dict, video: dict, min_fraction: float = 0.5) -> bool:
    """Whether a prior log ``entry`` is a SUCCESSFUL, complete transcript we can reuse (skip). False
    for a missing/0-segment prior (failure -> retry) or one that looks truncated — its last timestamp
    far below the known video duration (``video["duration"]`` seconds, when available)."""
    if not entry or int(entry.get("n_segments", 0) or 0) <= 0:   # missing / 0-seg failure -> retry
        return False
    try:
        dur = float(video.get("duration") or 0.0)
    except (TypeError, ValueError):
        dur = 0.0
    if dur > 0:
        segs = entry.get("segments") or []
        last = max((float(s.get("start") or 0.0) for s in segs), default=0.0)
        if last < dur * min_fraction:                           # truncated/partial -> retry
            return False
    return True


class YouTubeSource:
    """List recent videos, fetch timestamped transcripts, extract grounded catalyst claims.
    The API key travels only in the query string and is never logged."""

    def __init__(self, api_key: str, session=None,
                 transcript_fn: Optional[Callable] = None,
                 extract_fn: Optional[Callable[..., List[dict]]] = None,
                 whisper_fn: Optional[Callable] = None, ytdlp_fn: Optional[Callable] = None,
                 enable_fallbacks: bool = False, proxy_url: str = "",
                 webshare: Optional[dict] = None, pace_seconds: float = 0.0,
                 sleep_fn=None, languages: Sequence[str] = ("ko", "en"),
                 mode: Optional[str] = None):
        self.api_key = api_key
        self._session = session
        self._transcript_fn = transcript_fn          # explicit strategy fns win; (vid, langs) -> segs
        self._ytdlp_fn = ytdlp_fn
        self._whisper_fn = whisper_fn
        self._mode_arg = mode                        # None -> YOUTUBE_TRANSCRIPT_MODE (.env) at call time
        self._extract_fn = extract_fn or default_extract_catalysts
        self._enable_fallbacks = bool(enable_fallbacks)   # only then do REAL yt-dlp/Whisper run
        self._proxy_url = proxy_url
        self._webshare = webshare
        self._languages = tuple(languages)
        self._pace_seconds = float(pace_seconds or 0.0)
        self._sleep_fn = sleep_fn
        self._handle_cache: Dict[str, str] = {}
        self._uploads_cache: Dict[str, str] = {}     # channel -> "uploads" playlist id (UU…); cached
        self._tx_cache: Dict[str, tuple] = {}        # video id -> (segments, method); fetched once
        self._tx_state: Dict[str, bool] = {}         # pacing state across videos

    def _get(self, url: str, params: dict):
        session = self._session
        if session is None:
            import requests  # lazy
            session = requests
        resp = session.get(url, params={**params, "key": self.api_key}, timeout=10)
        try:
            return resp.json()
        except Exception:
            return None

    def resolve_channel_id(self, channel: str) -> str:
        """A UC… channelId for ``channel``. Pass-through for a UC id; an ``@handle`` is resolved
        via channels.list?forHandle (cached). Returns "" if it can't be resolved (never guessed),
        so callers surface "no coverage" rather than inventing."""
        c = str(channel or "").strip()
        if not c or c.startswith("UC"):
            return c
        if c in self._handle_cache:
            return self._handle_cache[c]
        data = self._get(CHANNELS_URL, {"part": "id", "forHandle": c.lstrip("@")})
        items = (data or {}).get("items") if isinstance(data, dict) else None
        cid = (items[0].get("id") if items else "") or ""
        self._handle_cache[c] = cid
        return cid

    def resolve_uploads_playlist(self, channel: str) -> str:
        """The channel's "uploads" playlist id (UU…), cached. Accepts a UC id or an @handle and
        costs ONE channels.list?part=contentDetails unit the FIRST time per channel, then 0 units
        (served from cache). Returns "" if it can't be resolved (never guessed)."""
        c = str(channel or "").strip()
        if not c:
            return ""
        if c in self._uploads_cache:
            return self._uploads_cache[c]
        params = {"part": "contentDetails"}
        if c.startswith("UC"):
            params["id"] = c
        else:
            params["forHandle"] = c.lstrip("@")
        data = self._get(CHANNELS_URL, params)
        items = (data or {}).get("items") if isinstance(data, dict) else None
        uploads = ""
        if items:
            it = items[0]
            uploads = (((it.get("contentDetails") or {}).get("relatedPlaylists") or {})
                       .get("uploads")) or ""
            cid = it.get("id") or ""
            if cid:                                  # remember handle->id + cache by resolved id too
                self._handle_cache.setdefault(c, cid)
                self._uploads_cache.setdefault(cid, uploads)
        self._uploads_cache[c] = uploads
        return uploads

    def list_recent_videos(self, channel_id: str, published_after: Optional[str] = None,
                           max_results: int = 5) -> List[dict]:
        """Recent uploads for one channel (newest first), via the QUOTA-CHEAP path: the channel's
        uploads playlist (channels.list, cached) + playlistItems.list — ~1-2 units vs search.list's
        100. Deduped by video id, filtered to ``published_after``. ``channel_id`` may be a UC id or
        an @handle; an unresolvable channel yields []."""
        if not channel_id:
            return []
        uploads = self.resolve_uploads_playlist(channel_id)
        if not uploads:
            return []
        items = _parse_playlist_items(self._get(
            PLAYLISTITEMS_URL,
            {"part": "snippet,contentDetails", "playlistId": uploads, "maxResults": max_results}))
        if published_after:                          # uploads are newest-first; drop older than the window
            items = [v for v in items if not v.get("published_at") or v["published_at"] >= published_after]
        return items

    def _transcript_strategies(self):
        """The ordered (name, fn) fallback chain for the active YOUTUBE_TRANSCRIPT_MODE. Injected fns
        always win; the REAL yt-dlp / Whisper defaults fill in only when ``enable_fallbacks`` is set —
        so tests never hit the network. Read at call time, so a mutated ``_transcript_fn`` / env mode
        is honoured.
          * auto    -> [api, ytdlp, whisper]   (cheap captions first, Whisper last)
          * whisper -> [whisper, api, ytdlp]   (Whisper first; api/subs only if Whisper fails)
          * api     -> [api]                   (youtube-transcript-api only)"""
        fb = self._enable_fallbacks
        cap_proxy = self.caption_proxy_url()             # proxy for the cheap caption paths (a)+(b)
        api = ("api", self._transcript_fn or ((lambda vid, langs: _api_transcript(
            vid, langs, self._proxy_url, self._webshare)) if fb else None))
        ytdlp = ("ytdlp", self._ytdlp_fn or ((lambda vid, langs: _ytdlp_subs(vid, langs, proxy_url=cap_proxy))
                                             if fb else None))
        whisper = ("whisper", self._whisper_fn or (_ytdlp_whisper if fb else None))  # Whisper: no-proxy
        mode = _transcript_mode(self._mode_arg)
        if mode == "whisper":                            # Whisper FIRST; api/subs only as a fallback
            return [whisper, api, ytdlp]
        if mode == "api":                                # youtube-transcript-api only
            return [api]
        return [api, ytdlp, whisper]                     # auto

    def caption_proxy_url(self) -> str:
        """The proxy URL used for the cheap caption paths (youtube-transcript-api + yt-dlp subs):
        an explicit PROXY_URL/YOUTUBE_PROXY_URL, else built from Webshare creds, else ""."""
        return self._proxy_url or webshare_proxy_url(self._webshare)

    def fetch_video_transcript(self, video_id: str):
        """The full transcript for one video via the fallback chain (cached, paced). Returns
        ``(segments, method)`` — method in {api, ytdlp, whisper, ""}."""
        return _run_transcript_chain(video_id, self._transcript_strategies(), self._languages,
                                     cache=self._tx_cache, pace=self._pace_seconds,
                                     sleep_fn=self._sleep_fn, paced_state=self._tx_state)

    def video_report(self, video: dict, channel_name: str = "", audit=None):
        """Fetch one video's FULL transcript (fallback chain), extract grounded insights, and — if an
        ``audit`` writer is given — log the raw transcript + insights for EVERY video (even 0-segment).
        Returns ``(insights, segments, method)`` for reporting."""
        vid = video.get("video_id", "")
        segments, method = self.fetch_video_transcript(vid)
        meta = {"video_id": vid, "video_title": video.get("title", ""),
                "published_at": video.get("published_at", "")}
        try:
            insights = self._extract_fn(segments, meta, channel_name or video.get("channel", "")) or []
        except Exception:
            insights = []
        if audit is not None:
            try:
                audit.record(meta, channel_name or video.get("channel", ""), segments, insights,
                             method=method)
            except Exception:
                pass
        return insights, segments, method

    def video_catalysts(self, video: dict, channel_name: str = "", audit=None) -> List[dict]:
        """All GROUNDED catalyst claims in one video (possibly none -> []). Never invents. If an
        ``audit`` writer is given, the raw transcript + insights are logged and each insight is
        tagged with its fetch-log reference (so it's traceable to its raw source)."""
        return self.video_report(video, channel_name, audit)[0]

    def media_briefing(self, channels: Sequence[Channel] = (), lookback_hours: int = 36,
                       max_per_channel: int = 4, now: Optional[datetime] = None, audit=None) -> List[dict]:
        """Flattened, grounded catalyst claims across all channels — newest / most-important
        first. A failing channel/video is skipped, never fatal. A video with nothing on a stock
        contributes nothing (omitted, never invented). ``audit`` (a youtube_audit.AuditWriter) logs
        every fetched video's listing + transcript + insights for verifiable proof."""
        channels = list(channels) or DEFAULT_CHANNELS
        now = now or datetime.now(timezone.utc)
        published_after = (now - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows: List[dict] = []
        for ch in channels:
            if not ch.channel_id:
                continue
            try:
                vids = self.list_recent_videos(ch.channel_id, published_after, max_per_channel)
            except Exception:
                continue
            for v in vids:
                rows.extend(self.video_catalysts(v, ch.name, audit=audit))
        rows.sort(key=lambda r: (str(r.get("published_at", "")), r.get("importance", 0.0)), reverse=True)
        return rows

    def _batch_row(self, ch, v, method, n_segments, insights, status):
        return {"channel": ch.name, "video_id": v.get("video_id"), "title": v.get("title", ""),
                "published_at": v.get("published_at", ""), "method": method, "n_segments": n_segments,
                "n_insights": len(insights), "insights": insights, "status": status}

    def batch_transcribe(self, channels: Sequence[Channel] = (), lookback_hours: int = 48,
                         max_per_channel: int = 6, max_videos: Optional[int] = None,
                         now: Optional[datetime] = None, audit=None, reextract: bool = False):
        """INCREMENTAL / idempotent batch: list recent videos, then per video either SKIP (a prior
        log entry already has a complete transcript — reuse its transcript + insights, no work),
        RE-EXTRACT (``reextract``: re-run Gemini insights on the cached transcript WITHOUT
        re-transcribing — cheap refresh after a watchlist/prompt change), or TRANSCRIBE (a NEW
        video id, or a prior 0-segment / truncated failure -> re-fetch via the fallback chain). The
        log stays append-only; the report builder dedupes by video id keeping the newest success.

        ``max_videos`` caps the TOTAL videos in the result (None = no cap). Each row carries a
        ``status``: "transcribed (new)" / "skipped (cached NNN seg)" / "re-transcribed (prior
        failed/partial)" / "re-extracted (cached NNN seg)"."""
        from tagent.youtube_audit import latest_entries_by_video
        channels = list(channels) or DEFAULT_CHANNELS
        now = now or datetime.now(timezone.utc)
        published_after = (now - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        cached = latest_entries_by_video(getattr(audit, "data_dir", None))   # video_id -> newest entry
        if reextract:                                        # refresh EVERY cached video in the window
            return self._reextract_window(cached, published_after, audit, max_videos)
        reports: List[dict] = []
        for ch in channels:
            if not ch.channel_id or (max_videos is not None and len(reports) >= max_videos):
                continue
            try:
                vids = self.list_recent_videos(ch.channel_id, published_after, max_per_channel)
            except Exception:
                continue
            for v in vids:
                if max_videos is not None and len(reports) >= max_videos:
                    break
                prior = cached.get(v.get("video_id"))
                if _cached_complete(prior, v):
                    ins = prior.get("insights") or []           # SKIP: reuse logged transcript+insights
                    reports.append(self._batch_row(ch, v, prior.get("method", ""),
                                   prior.get("n_segments", 0), ins,
                                   f"skipped (cached {prior.get('n_segments', 0)} seg)"))
                else:                                           # NEW id, or prior failed/partial -> transcribe
                    insights, segments, method = self.video_report(v, ch.name, audit=audit)
                    status = "re-transcribed (prior failed/partial)" if prior is not None else "transcribed (new)"
                    reports.append(self._batch_row(ch, v, method, len(segments), insights, status))
        return reports

    def _reextract_window(self, cached, published_after, audit, max_videos=None):
        """Re-run insight extraction on EVERY cached transcript whose published_at is in the window —
        NO re-transcription, NO channel-listing cap. Force-appends a fresh log entry per video so the
        newer (action/target-price-aware) insights supersede the stale ones. Returns per-video rows."""
        entries = [e for e in cached.values()
                   if not e.get("published_at") or str(e.get("published_at")) >= published_after]
        entries.sort(key=lambda e: str(e.get("published_at", "")), reverse=True)
        if max_videos is not None:
            entries = entries[:max_videos]
        reports: List[dict] = []

        class _V:                                            # adapt a log entry to the _batch_row shape
            def __init__(s, e): s.e = e
            def get(s, k, d=None): return s.e.get(k, d)
        for e in entries:
            ch = type("C", (), {"name": e.get("channel", "")})()
            segs = e.get("segments") or []
            if not segs:                                     # nothing cached to re-extract
                reports.append(self._batch_row(ch, _V(e), e.get("method", ""), 0, [],
                               "skipped (no cached transcript)"))
                continue
            meta = {"video_id": e.get("video_id"), "video_title": e.get("title", ""),
                    "published_at": e.get("published_at", "")}
            try:
                ins = self._extract_fn(segs, meta, e.get("channel", "")) or []
            except Exception:
                ins = []
            if audit is not None:
                try:
                    audit.record(meta, e.get("channel", ""), segs, ins, method=e.get("method", ""),
                                 force=True)                  # supersede the stale entry (newest wins)
                except Exception:
                    pass
            reports.append(self._batch_row(ch, _V(e), e.get("method", ""), len(segs), ins,
                           f"re-extracted (cached {len(segs)} seg)"))
        return reports


# --------------------------------------------------------------------------- #
# rendering — "STOCK — channel: 'quote' [mm:ss -> jump-link]" for dashboard + daily report
# --------------------------------------------------------------------------- #
def render_catalyst_line(item: dict) -> str:
    """One catalyst as the boss wants it read: grounded quote + jump-link + source."""
    name = item.get("stock_name") or item.get("stock", "")
    return (f"{name} — {item.get('channel', '')}: \"{item.get('quote', '')}\" "
            f"[{item.get('timestamp_mmss', '')} → {item.get('deeplink', '')}]")


def render_catalysts_report(items: Sequence[dict]) -> str:
    """A grounded markdown block for a daily report/email. No items -> an honest empty line."""
    rows = sorted(items or [], key=lambda r: (str(r.get("published_at", "")), r.get("importance", 0.0)),
                  reverse=True)
    if not rows:
        return "Media catalysts (grounded): none in the lookback window."
    head = "Media catalysts (grounded — each links to the exact moment; awareness only, not a signal):"
    return head + "\n" + "\n".join(f"- {render_catalyst_line(it)}" for it in rows)


def build_media_payload(items: Sequence[dict]) -> dict:
    """Dashboard payload. DISPLAY-ONLY and GROUNDED: no safety/halt field (media never feeds
    the trading or kill-switch logic), and every item maps to a real quote + deep-link."""
    rows = sorted(items or [], key=lambda r: (str(r.get("published_at", "")), r.get("importance", 0.0)),
                  reverse=True)
    return {"enabled": True, "label": MEDIA_LABEL, "grounded": True, "signal": False,
            "n": len(rows), "items": rows,
            # ready-to-paste grounded lines for the dashboard AND a daily report/email
            "report": render_catalysts_report(rows)}
