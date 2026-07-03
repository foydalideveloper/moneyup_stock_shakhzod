"""Whisper accuracy experiment (isolated; no daily-pipeline changes).

(1) STRUGGLE LIST — scan cached transcripts for name-like tokens that don't exactly match a KRX
    name, fuzzy-map to the nearest official name/ticker, and OCR-cross-check (spoken name vs the
    on-screen OCR ticker per video). Ranked CSV: misheard term -> correct KRX name/ticker -> freq.
(2) GLOSSARY — seed finance jargon + expand from the corpus (most-mentioned stock names + jargon).
(3) IMPROVED PASS — Whisper large-v3 with KRX names + jargon in initial_prompt, then a fuzzy
    post-correction that maps transcribed names -> official KRX name/ticker (captures avg_logprob).
(4) A/B — baseline (cached transcript) vs improved, on ~10 videos. Metric = % of primary-stock name
    mentions that resolve to the on-screen OCR ticker (answer key), + unmapped/conflict counts.

Reuses the cached pykrx name<->ticker map (~3,920) + youtube_source aliases. Outputs under
data/_moneyup_advisor/.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from moneyup_advisor import config, tickers

_TOK = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9]{1,6}")

# common transcript words that look name-ish but aren't stocks (suppress false mishears)
_STOP = set("""그리고 그래서 그러면 그런데 지금 오늘 어제 내일 우리 여러분 가지고 때문에 봤을 보시면
같은 같이 입니다 습니다 거든요 인데요 이라고 그거 저거 이거 여기 저기 거기 시장 종목 주식 코스피 코스닥
지수 상승 하락 거래 외국인 기관 개인 수급 차트 지표 가격 매수 매도 보유 관망 정도 부분 상태 진행 마이너스
플러스 퍼센트 포인트 그게 근데 이제 약간 조금 많이 계속 다시 한번 일단 결국 사실 물론 아마 만약""".split())

# (2) seed finance-jargon glossary
SEED_JARGON = ["공매도", "호가", "숏스퀴즈", "수급", "외국인 순매수", "외국인 순매도", "장대양봉",
               "장대음봉", "도지", "업틱룰", "대차거래", "대차잔고", "공매도 잔고", "거래량", "거래대금",
               "시가총액", "이동평균선", "이평선", "골든크로스", "데드크로스", "지지선", "저항선",
               "추세선", "매물대", "양봉", "음봉", "윗꼬리", "아랫꼬리", "횡보", "박스권", "돌파", "이탈",
               "갭상승", "갭하락", "단타", "스윙", "분할매수", "분할매도", "손절", "익절", "목표가",
               "적정주가", "컨센서스", "어닝서프라이즈", "어닝쇼크", "기관 순매수", "프로그램 매매",
               "선물", "옵션", "콜옵션", "풋옵션", "변동성", "만", "억", "조"]


# --------------------------------------------------------------------------- #
# corpus access
# --------------------------------------------------------------------------- #
def load_transcripts() -> Dict[str, list]:
    out = {}
    for p in glob.glob(str(config.CACHE_DIR / "*.transcript.json")):
        vid = Path(p).name.split(".")[0]
        try:
            out[vid] = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            pass
    return out


def answer_key(vid: str) -> Tuple[Optional[str], Optional[str]]:
    """(primary_ticker, primary_name) from the OCR-grounded fact sheet."""
    p = config.SHEET_DIR / f"{vid}.json"
    if not p.exists():
        return None, None
    s = json.loads(p.read_text(encoding="utf-8"))
    return s.get("primary_ticker"), s.get("primary_name")


def watchlist_names(vid: str) -> List[str]:
    p = config.SHEET_DIR / f"{vid}.json"
    if not p.exists():
        return []
    s = json.loads(p.read_text(encoding="utf-8"))
    return [r.get("name") for r in s.get("watchlist_ocr", []) if r.get("name")]


def _norm(s: str) -> str:
    return tickers._norm(s)


def _tokens(text: str) -> List[str]:
    return _TOK.findall(text or "")


def _is_mishearing(tn: str, nn: str, r: float) -> bool:
    """A near-full-length wrong spelling of a name — not an abbreviation/substring/correct mention."""
    return (0.6 <= r < 1.0 and tn != nn and len(tn) >= len(nn) - 1
            and tn not in nn and nn not in tn)


# --------------------------------------------------------------------------- #
# (1) struggle list
# --------------------------------------------------------------------------- #
def struggle_list() -> List[dict]:
    tx = load_transcripts()
    name2code = tickers.ensure_name_map()["name2code"]
    exact_names = set(name2code)                          # normalized official names
    rows: Dict[str, dict] = {}                            # heard -> row

    def add(heard, code, score, src):
        if not code:
            return
        key = _norm(heard)
        r = rows.get(key)
        if not r:
            rows[key] = {"heard": heard, "corrected_name": tickers.display_name(code),
                         "ticker": code, "match_score": score, "frequency": 0, "source": src}
            r = rows[key]
        r["frequency"] += 1
        r["match_score"] = max(r["match_score"], score)

    # (a) OCR cross-check: spoken attempts at the PRIMARY name that mis-resolve / don't match the
    #     on-screen OCR ticker for that video.
    ocr_errors = 0
    for vid, segs in tx.items():
        code, name = answer_key(vid)
        if not code or not name:
            continue
        pn = _norm(name)
        for seg in segs:
            for tok in _tokens(seg.get("text", "")):
                tn = _norm(tok)
                if len(tn) < 2 or tn in _STOP:
                    continue
                r = SequenceMatcher(None, tn, pn).ratio()
                if _is_mishearing(tn, pn, r) and tickers.resolve(tok) != code:
                    add(tok, code, round(r, 3), "ocr_crosscheck")   # misheard the on-screen stock
                    ocr_errors += 1

    # (b) corpus-wide misheard names: tokens that fuzzy-match an official name but aren't exact.
    freq = Counter()
    for segs in tx.values():
        for seg in segs:
            for tok in _tokens(seg.get("text", "")):
                tn = _norm(tok)
                if len(tn) >= 2 and tn not in _STOP and tn not in exact_names:
                    freq[tok] += 1
    for tok, f in freq.items():
        if f < 2:
            continue
        fz = tickers.fuzzy_name_to_code(tok)              # (code, score) if >= cutoff
        if not fz:
            continue
        nn = _norm(tickers.display_name(fz[0]))
        if _is_mishearing(_norm(tok), nn, fz[1]):         # full-length wrong spelling only
            add(tok, fz[0], fz[1], "fuzzy_corpus")

    ranked = sorted(rows.values(), key=lambda r: (-r["frequency"], -r["match_score"]))
    out = config.DATA_DIR / "whisper_struggle_list.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["heard", "corrected_name", "ticker",
                                          "match_score", "frequency", "source"])
        w.writeheader()
        w.writerows(ranked)
    print(f"[struggle] {len(ranked)} misheard term(s) -> {out}  (OCR-crosscheck hits={ocr_errors})")
    return ranked


# --------------------------------------------------------------------------- #
# (2) glossary
# --------------------------------------------------------------------------- #
def corpus_top_names(tx: Dict[str, list], k: int = 60) -> List[str]:
    """Most-mentioned official KRX names across the corpus (exact-name hits)."""
    name2code = tickers.ensure_name_map()["name2code"]
    cnt = Counter()
    for segs in tx.values():
        text = " ".join(s.get("text", "") for s in segs)
        for code in tickers.names_in_text(text, min_len=2):
            cnt[code] += 1
    return [tickers.display_name(c) for c, _ in cnt.most_common(k)]


def build_glossary() -> dict:
    tx = load_transcripts()
    jargon_freq = Counter()
    blob = " ".join(s.get("text", "") for segs in tx.values() for s in segs)
    for term in SEED_JARGON:
        jargon_freq[term] = blob.count(term)
    expanded_jargon = [t for t, _ in sorted(jargon_freq.items(), key=lambda x: -x[1])]
    top_names = corpus_top_names(tx, 60)
    glo = {"jargon": expanded_jargon, "jargon_corpus_freq": dict(jargon_freq),
           "top_stock_names": top_names}
    out = config.DATA_DIR / "whisper_glossary.json"
    out.write_text(json.dumps(glo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[glossary] {len(expanded_jargon)} jargon + {len(top_names)} top names -> {out}")
    return glo


# --------------------------------------------------------------------------- #
# (3) improved transcribe (GPU) + post-correction
# --------------------------------------------------------------------------- #
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        _MODEL = WhisperModel("large-v3", device="cuda", compute_type="float16")
    return _MODEL


def build_prompt(vid: str, glossary: dict) -> str:
    """initial_prompt: jargon + this video's OCR primary name + OCR watchlist peers + top names.
    Kept short (Whisper uses ~the last 224 tokens)."""
    _, pname = answer_key(vid)
    peers = watchlist_names(vid)[:10]
    names = ([pname] if pname else []) + peers + glossary.get("top_stock_names", [])[:25]
    seen, names_u = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            names_u.append(n)
    terms = glossary.get("jargon", [])[:22] + names_u
    return "한국 주식 방송. 종목명과 용어: " + ", ".join(terms) + "."


def transcribe(audio: str, initial_prompt: str) -> List[dict]:
    segs, _ = _model().transcribe(audio, language="ko", vad_filter=True,
                                  condition_on_previous_text=False, initial_prompt=initial_prompt)
    return [{"text": s.text.strip(), "start": float(s.start), "end": float(s.end),
             "avg_logprob": round(float(s.avg_logprob), 3)} for s in segs if s.text.strip()]


def post_correct(segments: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Fuzzy-map name-like tokens -> official KRX names; return (corrected_segments, corrections)."""
    name2code = tickers.ensure_name_map()["name2code"]
    exact = set(name2code)
    corrections = []
    out = []
    for seg in segments:
        text = seg.get("text", "")
        for tok in set(_tokens(text)):
            tn = _norm(tok)
            if len(tn) < 2 or tn in _STOP or tn in exact:
                continue
            fz = tickers.fuzzy_name_to_code(tok)
            if fz and fz[1] >= 0.78:
                official = tickers.display_name(fz[0])
                if official and official != tok:
                    text = re.sub(re.escape(tok), official, text)
                    corrections.append({"from": tok, "to": official, "ticker": fz[0],
                                        "score": fz[1], "t": seg.get("start")})
        s2 = dict(seg)
        s2["text"] = text
        out.append(s2)
    return out, corrections


# --------------------------------------------------------------------------- #
# (4) A/B
# --------------------------------------------------------------------------- #
def _resolve_metric(segments: List[dict], code: str, name: str) -> dict:
    """For primary-name ATTEMPTS (tokens fuzzy>=0.55 to the official name): how many resolve to the
    correct on-screen ticker (exact-name and via-resolver), how many unmapped/conflict."""
    pn = _norm(name)
    attempts = exact_hits = resolved_correct = unmapped = conflict = 0
    for seg in segments:
        for tok in _tokens(seg.get("text", "")):
            tn = _norm(tok)
            if len(tn) < 2 or tn in _STOP:
                continue
            if SequenceMatcher(None, tn, pn).ratio() >= 0.55:
                attempts += 1
                if tn == pn:
                    exact_hits += 1
                r = tickers.resolve(tok)
                if r == code:
                    resolved_correct += 1
                elif r is None:
                    unmapped += 1
                else:
                    conflict += 1
    return {"attempts": attempts, "exact_name_hits": exact_hits,
            "resolved_to_ocr_ticker": resolved_correct, "unmapped": unmapped, "conflict": conflict}


def _rate(d: dict, num: str) -> Optional[float]:
    return round(d[num] / d["attempts"], 3) if d["attempts"] else None


def ab_test(video_ids: List[str]) -> dict:
    glossary = build_glossary()
    tx = load_transcripts()
    per_video = []
    agg = {"baseline": Counter(), "improved": Counter()}
    for vid in video_ids:
        code, name = answer_key(vid)
        audio = config.VIDEO_DIR / f"{vid}.m4a"
        if not code or not name or not audio.exists() or vid not in tx:
            print(f"  skip {vid} (need answer key + audio + cached transcript)")
            continue
        base_segs = tx[vid]
        prompt = build_prompt(vid, glossary)
        imp_segs = transcribe(str(audio), prompt)
        imp_segs, corr = post_correct(imp_segs)
        bm = _resolve_metric(base_segs, code, name)
        im = _resolve_metric(imp_segs, code, name)
        low_lp = sum(1 for s in imp_segs if s.get("avg_logprob", 0) < -0.8)
        per_video.append({"video_id": vid, "ticker": code, "name": name,
                          "baseline": bm, "improved": im, "n_corrections": len(corr),
                          "low_logprob_segments": low_lp,
                          "sample_corrections": corr[:5]})
        for k in bm:
            agg["baseline"][k] += bm[k]
        for k in im:
            agg["improved"][k] += im[k]
        print(f"  {vid} [{name} {code}] base resolve={_rate(bm,'resolved_to_ocr_ticker')} "
              f"-> improved={_rate(im,'resolved_to_ocr_ticker')}  (+{len(corr)} corrections)")
    report = {
        "test": "Whisper A/B — baseline (deployed) vs improved (KRX+jargon prompt + post-correction)",
        "n_videos": len(per_video),
        "metric": "% of primary-name mentions that resolve to the on-screen OCR ticker (answer key)",
        "baseline": dict(agg["baseline"]), "improved": dict(agg["improved"]),
        "baseline_resolve_rate": _rate(agg["baseline"], "resolved_to_ocr_ticker"),
        "improved_resolve_rate": _rate(agg["improved"], "resolved_to_ocr_ticker"),
        "baseline_exact_rate": _rate(agg["baseline"], "exact_name_hits"),
        "improved_exact_rate": _rate(agg["improved"], "exact_name_hits"),
        "per_video": per_video,
    }
    out = config.DATA_DIR / "whisper_ab_result.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[A/B] {len(per_video)} videos -> {out}")
    return report


def main():
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["analyze", "ab"], help="analyze=struggle+glossary (CPU); ab=A/B (GPU)")
    ap.add_argument("--videos", help="comma-separated ids for ab; default = first 10 with audio+key")
    a = ap.parse_args()
    if a.cmd == "analyze":
        struggle_list()
        build_glossary()
    else:
        if a.videos:
            ids = a.videos.split(",")
        else:
            tx = load_transcripts()
            ids = [v for v in tx if answer_key(v)[0] and (config.VIDEO_DIR / f"{v}.m4a").exists()][:10]
        print(f"A/B on {len(ids)} videos: {ids}")
        rep = ab_test(ids)
        print(f"\nBASELINE resolve={rep['baseline_resolve_rate']} exact={rep['baseline_exact_rate']} "
              f"| IMPROVED resolve={rep['improved_resolve_rate']} exact={rep['improved_exact_rate']}")


if __name__ == "__main__":
    main()
