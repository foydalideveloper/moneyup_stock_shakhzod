"""Phase 2 — 머니업 Playbook: a structured, evidence-linked map of his METHOD.

Uses Gemini 3.1 Pro (paid REST API, reasoning) — NOT the local GPU. Read-only on the fact sheets.
Per video Gemini extracts his setups/rules, indicators, buy/sell/avoid triggers, vocabulary and
reasoning chains — each cited to a real timestamp + verbatim transcript quote (quotes are GROUNDED
against the transcript; ungrounded items are dropped, never invented). Then a corpus-level pass
clusters & ranks rules by how many videos use each.

DISCIPLINE: descriptive only. Nothing here is labeled profitable or treated as a trading signal —
whether his rules make money is Phase 1's job (kept separate). Per-video extractions are cached, so
the playbook AUTO-UPDATES incrementally as collection adds sheets.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

from moneyup_advisor import config

MODEL = os.getenv("MONEYUP_GEMINI_MODEL", "gemini-3.1-pro-preview")
FALLBACKS = ["gemini-3-pro-preview", "gemini-pro-latest"]
PLAYBOOK_DIR = config.DATA_DIR / "playbook"
PLAYBOOK_DIR.mkdir(parents=True, exist_ok=True)
MIN_VIDEOS = int(os.getenv("MONEYUP_PLAYBOOK_MIN_VIDEOS", "3"))   # floor: a rule needs >=N distinct videos w/ VERIFIED quotes
MERGE_TEMPERATURE = 0.0                                           # the Gemini label-merge is deterministic (temp 0)
KST = timezone(timedelta(hours=9))
# Content-addressed merge cache (day-to-day STABILITY): a Gemini label-merge decision is keyed by the
# sorted set of raw cluster labels it grouped — so identical facts -> identical labels/groupings with
# ZERO Gemini calls; only new/changed clusters are re-decided, then frozen. Counts always come from the
# deterministic union of video_ids, so they keep updating as videos arrive (caching fixes only the one
# nondeterministic step — the label-merge — not the counts).
MERGE_CACHE = PLAYBOOK_DIR / "merge_cache.json"
BUILD_STATE = PLAYBOOK_DIR / "build_state.json"                   # previous run's signature set + content hash
RELEASE_DIR = PLAYBOOK_DIR / "releases"                           # immutable boss-facing snapshots
RELEASED_JSON = PLAYBOOK_DIR / "playbook_released.json"           # pointer to the latest release (boss-facing)
RELEASED_MD = PLAYBOOK_DIR / "playbook_released.md"

_DISCIPLINE = ("서술적으로만 추출하라. 어떤 규칙도 '수익이 난다/좋다/매수신호'라고 평가하거나 매매 신호로 "
               "승격하지 말 것(수익성은 별도 단계가 검증함). 규칙이나 인용을 절대 지어내지 말 것. "
               "모든 항목은 제공된 자막에 실제로 존재하는 mm:ss 타임스탬프와 '그대로의' 한국어 인용으로 근거를 달 것.")


def _gemini_key() -> str:
    k = os.getenv("GEMINI_API_KEY")
    if k:
        return k.strip()
    try:
        for line in (config.REPO_ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


def _client(model: str = None, timeout: float = 120.0):
    from tagent.gemini import GeminiClient
    return GeminiClient(_gemini_key(), model=model or MODEL, timeout=timeout)


def _gen(prompt: str, system: str, timeout: float = 120.0, temperature: float = 0.2) -> str:
    """Gemini generate with model fallback if the chosen model 404s / returns empty."""
    for m in [MODEL] + FALLBACKS:
        try:
            out = _client(m, timeout=timeout).generate(prompt, system=system, temperature=temperature)
            if out and out.strip():
                return out
        except Exception:
            continue
    return ""


def _json(text: str):
    """Parse a JSON object/array from a model response, tolerating ```json fences."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"[\{\[].*[\}\]]", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #
def _full_transcript(vid: str) -> List[dict]:
    p = config.CACHE_DIR / f"{vid}.transcript.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _mmss(sec) -> str:
    s = int(max(0.0, float(sec or 0)))
    return f"{s//60:02d}:{s%60:02d}"


def _mmss_to_sec(mmss: str) -> int:
    parts = [int(x) for x in re.findall(r"\d+", str(mmss or ""))]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] if parts else 0


def corpus(limit: Optional[int] = None) -> List[dict]:
    items = []
    for p in sorted(config.SHEET_DIR.glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        vid = s.get("video_id")
        segs = _full_transcript(vid)
        if not vid or not segs:
            continue
        items.append({"video_id": vid, "title": s.get("title"),
                      "publish": s.get("publish_date"), "segments": segs,
                      "exante_calls": s.get("exante_calls", []),
                      "primary_numbers": s.get("primary_numbers", []),
                      "vlm": s.get("vlm_observations", [])})
    items.sort(key=lambda x: x.get("publish") or "", reverse=True)
    return items[:limit] if limit else items


def _strip(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()


def _grounded(quote: str, joined: str) -> bool:
    q = _strip(quote)
    if len(q) < 6:
        return False
    return q in joined or q[: max(10, len(q) // 2)] in joined


def _verbatim(quote: str, joined_norm: str) -> bool:
    """STRICT firewall check (stricter than _grounded): the quote must appear VERBATIM in the
    transcript — an exact substring after whitespace-normalization (Whisper spacing varies, so we
    normalize spaces only; we do NOT accept _grounded's half-prefix leniency). A quote that fails
    this is treated as a hallucination and its evidence clip is dropped."""
    q = _strip(quote)
    return len(q) >= 6 and q in joined_norm


# --------------------------------------------------------------------------- #
# (1) per-video extraction (cached, grounded)
# --------------------------------------------------------------------------- #
_PV_SYS = ("너는 한국 주식 유튜버 '머니업'의 한 방송을 분석한다. 그의 매매 방법론을 추출한다. " + _DISCIPLINE +
           " 반드시 아래 JSON 스키마의 객체 하나만 출력하라.")


def _pv_prompt(item: dict) -> str:
    tx = "\n".join(f"[{_mmss(s['start'])}] {s['text']}" for s in item["segments"])
    calls = "; ".join(f"{c.get('name')}({c.get('ticker')}) {c.get('direction')}@{c.get('mmss')}"
                      for c in item["exante_calls"]) or "없음"
    vlm = "; ".join(f"{o.get('parsed',{}).get('chart_pattern')}/{o.get('parsed',{}).get('points_at')}"
                    for o in item["vlm"][:5]) or "없음"
    schema = ('{"setups_rules":[{"rule":"규칙 요약","kind":"setup|pattern|principle",'
              '"evidence":[{"mmss":"mm:ss","quote":"자막 그대로"}],"reasoning":"관찰->결론"}],'
              '"indicators":[{"name":"지표명","evidence":[{"mmss":"","quote":""}]}],'
              '"triggers":{"buy":[{"condition":"매수 조건","evidence":[{"mmss":"","quote":""}]}],'
              '"sell":[{"condition":"","evidence":[]}],"avoid":[{"condition":"","evidence":[]}]},'
              '"vocabulary":[{"term":"용어/입버릇","meaning":"뜻","evidence":[{"mmss":"","quote":""}]}],'
              '"reasoning_chains":[{"observation":"관찰","conclusion":"결론","evidence":[{"mmss":"","quote":""}]}]}')
    return (f"영상 제목: {item['title']}\n사전 추출 콜: {calls}\n화면(차트 VLM): {vlm}\n\n"
            f"=== 타임스탬프 자막 ===\n{tx}\n\n위 자막만 근거로, 이 JSON 스키마로 출력하라:\n{schema}")


def _ground_extraction(ext: dict, joined: str) -> dict:
    """Drop evidence clips whose quote isn't in the transcript; drop items left with no evidence."""
    def fix_ev(evs):
        return [e for e in (evs or []) if _grounded(e.get("quote", ""), joined)]

    out = {"setups_rules": [], "indicators": [], "triggers": {"buy": [], "sell": [], "avoid": []},
           "vocabulary": [], "reasoning_chains": []}
    for r in ext.get("setups_rules", []):
        ev = fix_ev(r.get("evidence"))
        if ev:
            out["setups_rules"].append({**r, "evidence": ev})
    for i in ext.get("indicators", []):
        ev = fix_ev(i.get("evidence"))
        if ev:
            out["indicators"].append({**i, "evidence": ev})
    for side in ("buy", "sell", "avoid"):
        for t in ext.get("triggers", {}).get(side, []):
            ev = fix_ev(t.get("evidence"))
            if ev:
                out["triggers"][side].append({**t, "evidence": ev})
    for v in ext.get("vocabulary", []):
        ev = fix_ev(v.get("evidence"))
        if ev:
            out["vocabulary"].append({**v, "evidence": ev})
    for c in ext.get("reasoning_chains", []):
        ev = fix_ev(c.get("evidence"))
        if ev:
            out["reasoning_chains"].append({**c, "evidence": ev})
    return out


def extract_video(item: dict, refresh: bool = False) -> Optional[dict]:
    vid = item["video_id"]
    cache = config.CACHE_DIR / f"{vid}.playbook.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    raw = _gen(_pv_prompt(item), _PV_SYS)
    ext = _json(raw)
    if not isinstance(ext, dict):
        return None
    joined = _strip(" ".join(s["text"] for s in item["segments"]))
    grounded = _ground_extraction(ext, joined)
    grounded["video_id"] = vid
    cache.write_text(json.dumps(grounded, ensure_ascii=False), encoding="utf-8")
    return grounded


# --------------------------------------------------------------------------- #
# (2) corpus aggregation (cluster + rank)
# --------------------------------------------------------------------------- #
_AGG_SYS = ("너는 여러 방송에서 추출한 머니업의 규칙/지표/트리거/어휘를 하나의 'playbook'으로 통합한다. " +
            _DISCIPLINE + " 의미가 같은 규칙은 canonical 한 문장으로 합치고, 서로 다른 영상 수(n_videos)로 "
            "정렬한다. 입력에 있는 근거(video_id,mmss,quote)만 그대로 옮기고 새 인용을 만들지 마라. JSON 객체 하나만 출력.")


def _agg_prompt(rules_blob: str) -> str:
    schema = ('{"rules":[{"canonical":"통합 규칙 한 문장","kind":"setup|pattern|principle|trigger",'
              '"n_videos":0,"video_ids":[],"evidence":[{"video_id":"","mmss":"","quote":""}],'
              '"reasoning":"관찰->결론"}],'
              '"indicators_ranked":[{"name":"","n_videos":0,"video_ids":[],"evidence":[{"video_id":"","mmss":"","quote":""}]}],'
              '"triggers":{"buy":[{"condition":"","n_videos":0,"evidence":[{"video_id":"","mmss":"","quote":""}]}],'
              '"sell":[],"avoid":[]},'
              '"vocabulary":[{"term":"","meaning":"","examples":[{"video_id":"","mmss":"","quote":""}]}]}')
    return (f"아래는 영상별로 추출한 머니업의 항목들이다(JSON lines, 각 줄에 video_id 포함):\n{rules_blob}\n\n"
            f"의미가 같은 것은 반드시 하나로 합쳐라(예: '일정 매매'/'스케줄 매매'/'매집선 기반 타점'은 한 규칙). "
            f"다음 스키마로 출력하되 분량을 제한하라: rules 최대 15개(n_videos 내림차순, 가장 자주 반복되는 것 위주), "
            f"각 항목 evidence 최대 2개, indicators_ranked 최대 15개, vocabulary 최대 20개:\n{schema}")


def _ev1(evs):
    """One evidence clip with the quote truncated — keeps the aggregator input small/fast."""
    if not evs:
        return []
    e = evs[0]
    return [{"video_id": e.get("video_id", ""), "mmss": e.get("mmss", ""),
             "quote": (e.get("quote", "") or "")[:60]}]


def aggregate(per_video: List[dict]) -> dict:
    # condense per-video items into compact lines (1 short evidence clip each) so the single
    # clustering call stays well under the request timeout.
    lines = []
    for pv in per_video:
        vid = pv["video_id"]

        def ev(evs):
            out = _ev1(evs)
            for e in out:
                e["video_id"] = vid
            return out
        for r in pv.get("setups_rules", []):
            lines.append(json.dumps({"vid": vid, "type": "rule", "kind": r.get("kind"),
                                     "text": r.get("rule"), "evidence": ev(r.get("evidence"))},
                                    ensure_ascii=False))
        for side in ("buy", "sell", "avoid"):
            for t in pv.get("triggers", {}).get(side, []):
                lines.append(json.dumps({"vid": vid, "type": f"trigger_{side}",
                                         "text": t.get("condition"), "evidence": ev(t.get("evidence"))},
                                        ensure_ascii=False))
        for i in pv.get("indicators", []):
            lines.append(json.dumps({"vid": vid, "type": "indicator", "text": i.get("name"),
                                     "evidence": ev(i.get("evidence"))}, ensure_ascii=False))
        for v in pv.get("vocabulary", []):
            lines.append(json.dumps({"vid": vid, "type": "vocab", "text": v.get("term"),
                                     "meaning": v.get("meaning"), "evidence": ev(v.get("evidence"))},
                                    ensure_ascii=False))
    raw = _gen(_agg_prompt("\n".join(lines)), _AGG_SYS, timeout=300.0)   # big reasoning call -> 5 min
    agg = _json(raw)
    return agg if isinstance(agg, dict) and agg.get("rules") else {}


def _nrm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()


def aggregate_programmatic(per_video: List[dict]) -> dict:
    """Deterministic clustering of the (Gemini-extracted) per-video items by text similarity, ranked
    by # distinct videos. Reliable fallback when the single large Gemini clustering call is flaky."""
    def evs(vid, lst):
        return [{"video_id": vid, "mmss": e.get("mmss", ""), "quote": e.get("quote", "")}
                for e in (lst or [])]

    clusters = []                                          # rules
    for pv in per_video:
        vid = pv["video_id"]
        for r in pv.get("setups_rules", []):
            text = r.get("rule", "")
            n = _nrm(text)
            if len(n) < 4:
                continue
            hit = next((c for c in clusters
                        if any(SequenceMatcher(None, n, _nrm(t)).ratio() > 0.6 for t in c["texts"][:6])), None)
            if not hit:
                hit = {"texts": [], "vids": set(), "evidence": [], "kinds": Counter(), "reasonings": []}
                clusters.append(hit)
            hit["texts"].append(text)
            hit["vids"].add(vid)
            hit["evidence"] += evs(vid, r.get("evidence"))
            hit["kinds"][r.get("kind") or "rule"] += 1
            if r.get("reasoning"):
                hit["reasonings"].append(r["reasoning"])
    rules = [{"canonical": max(c["texts"], key=len), "kind": c["kinds"].most_common(1)[0][0],
              "n_videos": len(c["vids"]), "video_ids": sorted(c["vids"]),
              "evidence": c["evidence"], "reasoning": c["reasonings"][0] if c["reasonings"] else ""}
             for c in clusters]                                # keep FULL evidence; the firewall verifies + trims
    rules.sort(key=lambda r: (-r["n_videos"], r["canonical"]))   # deterministic tie-break

    def merge(key_field, getlist, ev_field="evidence"):
        d = {}
        for pv in per_video:
            vid = pv["video_id"]
            for it in getlist(pv):
                k = _nrm(it.get(key_field, ""))
                if not k:
                    continue
                e = d.setdefault(k, {"label": it.get(key_field), "meaning": it.get("meaning"),
                                     "vids": set(), "evidence": []})
                e["vids"].add(vid)
                e["evidence"] += evs(vid, it.get(ev_field))
        return d

    ind = merge("name", lambda pv: pv.get("indicators", []))
    indicators = sorted([{"name": v["label"], "n_videos": len(v["vids"]),
                          "video_ids": sorted(v["vids"]), "evidence": v["evidence"][:3]}
                         for v in ind.values()], key=lambda x: (-x["n_videos"], x["name"]))
    triggers = {}
    for side in ("buy", "sell", "avoid"):
        tc = merge("condition", lambda pv, s=side: pv.get("triggers", {}).get(s, []))
        triggers[side] = sorted([{"condition": v["label"], "n_videos": len(v["vids"]),
                                  "video_ids": sorted(v["vids"]), "evidence": v["evidence"][:2]}
                                 for v in tc.values()], key=lambda x: (-x["n_videos"], x["condition"]))
    voc = merge("term", lambda pv: pv.get("vocabulary", []))
    vocabulary = sorted([{"term": v["label"], "meaning": v["meaning"], "n_videos": len(v["vids"]),
                          "examples": v["evidence"][:2]} for v in voc.values()],
                        key=lambda x: (-x["n_videos"], x["term"]))
    return {"rules": rules, "indicators_ranked": indicators, "triggers": triggers,
            "vocabulary": vocabulary}


# --------------------------------------------------------------------------- #
# verify aggregated evidence still grounds (anti-fabrication, second pass)
# --------------------------------------------------------------------------- #
def _verify_agg(agg: dict, joined_by_vid: Dict[str, str]) -> dict:
    def fix(evs):
        out = []
        for e in evs or []:
            j = joined_by_vid.get(e.get("video_id", ""), "")
            if j and _grounded(e.get("quote", ""), j):
                out.append(e)
        return out
    real = set(joined_by_vid)                              # real corpus video ids (anti-hallucination)
    for r in agg.get("rules", []):
        r["evidence"] = fix(r.get("evidence"))
        r["video_ids"] = sorted(v for v in (r.get("video_ids") or []) if v in real)
        # true frequency = distinct REAL videos the aggregator merged (not the trimmed evidence count)
        r["n_videos"] = max(len(r["video_ids"]), len({e["video_id"] for e in r["evidence"]}))
    agg["rules"] = [r for r in agg.get("rules", []) if r.get("evidence")]
    agg["rules"].sort(key=lambda r: -r.get("n_videos", 0))
    for i in agg.get("indicators_ranked", []):
        i["evidence"] = fix(i.get("evidence"))
        i["video_ids"] = sorted(v for v in (i.get("video_ids") or []) if v in real)
        i["n_videos"] = max(len(i["video_ids"]), len({e["video_id"] for e in i["evidence"]}))
    agg["indicators_ranked"] = sorted(agg.get("indicators_ranked", []),
                                      key=lambda x: -x.get("n_videos", 0))
    for side in ("buy", "sell", "avoid"):
        for t in agg.get("triggers", {}).get(side, []):
            t["evidence"] = fix(t.get("evidence"))
    for v in agg.get("vocabulary", []):
        v["examples"] = fix(v.get("examples"))
    return agg


# --------------------------------------------------------------------------- #
# content-addressed merge cache (stability) + helpers
# --------------------------------------------------------------------------- #
def _sig(section: str, labels) -> str:
    """Stable signature for a merge GROUP, from the INPUT only: the section namespace + the sorted set
    of raw label strings it groups. Independent of Gemini's prior wording, so identical facts reproduce
    the same key and the cached decision is reused (no Gemini call)."""
    key = "".join([section] + sorted(str(l) for l in labels))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _load_cache() -> dict:
    try:
        return json.loads(MERGE_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        MERGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    except Exception as e:
        print(f"[cache] save failed: {str(e)[:100]}")


def _content_hash(agg: dict) -> str:
    """Hash of the SUBSTANTIVE content (rules/indicators/triggers/vocabulary) — excludes meta.generated
    (the timestamp), so two builds over the same facts hash identically."""
    payload = {k: agg.get(k) for k in ("rules", "indicators_ranked", "triggers", "vocabulary")}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def _valid_merge(g, n_items: int) -> bool:
    """Schema-validate the label-merge output STRUCTURE: an object whose 'groups' is a non-empty list
    containing at least one well-formed group (a dict with a non-empty 'canonical' string and a
    non-empty 'members' list). Individual out-of-range / non-int member ids are NOT fatal here — the
    merge loop coerces/skips them and keeps unassigned items as singletons — so a single stray id can't
    discard the whole otherwise-good merge (that brittleness made the 80-label indicator merge
    intermittently no-op). Returns False only on genuinely unusable output (caller then keeps input)."""
    if not isinstance(g, dict) or not isinstance(g.get("groups"), list) or not g["groups"]:
        return False
    for grp in g["groups"]:
        if (isinstance(grp, dict) and str(grp.get("canonical") or "").strip()
                and isinstance(grp.get("members"), list) and grp["members"]):
            return True                                       # at least one usable group -> proceed
    return False


def _gemini_group(items: List[dict], label_key: str, what: str, want_kind: bool, examples: str):
    """ONE temp-0, schema-validated Gemini call grouping equivalent LABELS among ``items``. Returns a
    list of groups [{"members":[item,...], "canonical": str, "kind": str|None}] covering every item
    exactly once (ungrouped items become singletons; all-singletons on any failure). Label-merge ONLY —
    never invents an item / edits meaning."""
    singletons = [{"members": [it], "canonical": it.get(label_key, ""), "kind": it.get("kind")} for it in items]
    if len(items) < 2:
        return singletons
    listing = "\n".join(f"{i}\t{it.get('n_videos', 0)}\t{it.get(label_key, '')}"
                        for i, it in enumerate(items))
    kindfld = '"kind":"setup|pattern|principle|trigger",' if want_kind else ''
    schema = '{"groups":[{"canonical":"통합 라벨",' + kindfld + '"members":[0]}]}'
    sysp = (f"아래는 머니업 방송에서 추출한 '{what}' 클러스터 목록이다(각 줄: id<TAB>영상수<TAB>라벨). "
            f"표현만 다르고 의미가 같은 항목{examples}을 하나의 그룹으로 병합하라. " + _DISCIPLINE +
            " 오직 라벨 중복 제거만 하라(의미 편집·새 항목/근거 생성 금지). 모든 id를 정확히 한 그룹에만 배정하라. "
            "JSON 객체 하나만 출력.")
    g = _json(_gen(f"{listing}\n\n스키마:\n{schema}", sysp, timeout=180.0, temperature=MERGE_TEMPERATURE))
    if not _valid_merge(g, len(items)):                       # temp-0, schema-validated JSON (else all-singletons)
        return singletons
    used, groups = set(), []
    for grp in g["groups"]:
        idxs = []
        for i in grp.get("members", []):
            try:
                i = int(i)
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(items) and i not in used:
                idxs.append(i)
                used.add(i)
        members = [items[i] for i in idxs]
        if not members:
            continue
        groups.append({"members": members,
                       "canonical": grp.get("canonical") or max((m.get(label_key, "") for m in members), key=len),
                       "kind": (grp.get("kind") if want_kind else None) or members[0].get("kind")})
    for i, it in enumerate(items):                            # ungrouped -> singleton (never lose content)
        if i not in used:
            groups.append({"members": [it], "canonical": it.get(label_key, ""), "kind": it.get("kind")})
    return groups


def _merge_labels_gemini(items: List[dict], label_key: str, what: str, *,
                         want_kind: bool = False, examples: str = "", section: str = "",
                         head: int = 80, cache: Optional[dict] = None, sigs: Optional[set] = None) -> List[dict]:
    """SHARED, content-addressed label-merge — used IDENTICALLY for rules / indicators / triggers.

    Deterministic programmatic clustering already gives each item a TRUE distinct-video set. This step
    only merges equivalent LABELS. With a ``cache``: any group whose member labels are all still present
    is REUSED verbatim (canonical + grouping) with ZERO Gemini calls; only the leftover (new/changed)
    labels are sent to ONE temp-0, schema-validated Gemini call, whose decisions are written back. So
    identical facts -> identical output; a new video updates only the affected entries. Counts are ALWAYS
    the union of the members' current video_ids (a video contributing two member-labels counts once), so
    they update naturally. Large sections are capped to the top-``head`` for one reliable call (stable
    sort -> same head every run); the rare tail is kept verbatim. ``cache=None`` forces a full re-merge
    (the --no-cache path). ``sigs`` (if given) collects every group signature for the stability guard."""
    if len(items) < 2:
        if sigs is not None and items:
            sigs.add(_sig(section, [items[0].get(label_key, "")]))
        return items
    if head is not None and len(items) > head:                # cap only LARGE sections (one reliable call)
        ranked = sorted(items, key=lambda x: (-x.get("n_videos", 0), x.get(label_key, "")))  # stable head
        pool, tail = ranked[:head], ranked[head:]
    else:
        pool, tail = list(items), []                          # small section (e.g. rules) -> merge in full
    if len(pool) < 2:
        if sigs is not None:
            for it in pool:
                sigs.add(_sig(section, [it.get(label_key, "")]))
        return items

    by_label = {}
    for it in pool:
        by_label.setdefault(it.get(label_key, ""), it)        # exact label -> cluster (programmatic dedups)
    label_set, used_labels, groups = set(by_label), set(), []

    # 1) REUSE cached decisions whose member labels are all still present (deterministic order)
    n_cached = 0
    if cache is not None:
        for _sigk, entry in sorted((cache.get(section) or {}).items()):
            mlabels = entry.get("labels") or []
            if mlabels and all((l in label_set and l not in used_labels) for l in mlabels):
                groups.append({"members": [by_label[l] for l in mlabels],
                               "canonical": entry.get("canonical"), "kind": entry.get("kind")})
                used_labels.update(mlabels)
                n_cached += 1

    # 2) Gemini decides ONLY the leftover (new/changed) labels; write decisions back to the cache
    leftover = [by_label[l] for l in by_label if l not in used_labels]
    if leftover:
        for grp in _gemini_group(leftover, label_key, what, want_kind, examples):
            groups.append(grp)
            if cache is not None:
                mlabels = sorted(m.get(label_key, "") for m in grp["members"])
                entry = {"labels": mlabels, "canonical": grp["canonical"]}
                if grp.get("kind"):
                    entry["kind"] = grp["kind"]
                cache.setdefault(section, {})[_sig(section, mlabels)] = entry
    if section:
        print(f"  [merge:{section}] reused {n_cached} cached · {len(leftover)} labels->Gemini · {len(groups)} groups")

    # 3) build merged items (counts/evidence from CURRENT members; members sorted for determinism)
    merged = []
    for grp in groups:
        members = sorted(grp["members"], key=lambda m: m.get(label_key, ""))
        if sigs is not None:
            sigs.add(_sig(section, [m.get(label_key, "") for m in members]))
        vids = sorted({v for m in members for v in (m.get("video_ids") or [])})   # union = TRUE count
        out = {label_key: grp.get("canonical") or max((m.get(label_key, "") for m in members), key=len),
               "n_videos": len(vids), "video_ids": vids,
               "evidence": [e for m in members for e in (m.get("evidence") or [])]}   # full; firewall trims
        if want_kind:
            out["kind"] = grp.get("kind") or members[0].get("kind")
        if any(m.get("reasoning") for m in members):
            out["reasoning"] = next((m.get("reasoning") for m in members if m.get("reasoning")), "")
        if section and len(members) > 1:                      # print the merge mapping for review
            raw = " + ".join(f"{(m.get(label_key) or '')[:30]}({m.get('n_videos', 0)})" for m in members)
            print(f"    {raw} -> {(out[label_key] or '')[:46]} [{out['n_videos']}v]")
        merged.append(out)
    merged.sort(key=lambda x: (-x.get("n_videos", 0), x.get(label_key, "")))   # deterministic tie-break
    return merged + tail                                      # re-ranked merged head + untouched rare tail


def _firewall(agg: dict, joined_by_vid: Dict[str, str], floor: int = MIN_VIDEOS):
    """Anti-hallucination firewall on the aggregated playbook:
      (1) VERBATIM-verify every cited quote against ITS OWN video transcript; drop quotes that don't
          match (whitespace-normalized exact substring — no half-prefix leniency).
      (2) A rule's n_videos becomes the count of DISTINCT videos that still have >=1 verified quote,
          so the ranking count is itself backed by real quotes.
      (3) Floor: drop any rule supported by < `floor` distinct verified videos.
      (4) Confidence tier from that verified count: high >=15, medium 5-14, low 3-4.
    Indicators/triggers/vocabulary evidence is verbatim-filtered too (GIGO), without the floor/tier.
    Returns (agg, stats); stats is the hallucination meter (quotes checked vs failed)."""
    real = set(joined_by_vid)
    checked = failed = 0

    def verify(evs):
        nonlocal checked, failed
        out = []
        for e in evs or []:
            checked += 1
            vid = e.get("video_id", "")
            j = joined_by_vid.get(vid, "")
            if vid in real and j and _verbatim(e.get("quote", ""), j):
                out.append(e)
            else:
                failed += 1
        return out

    n_before = len(agg.get("rules", []))
    kept = []
    for r in agg.get("rules", []):
        ev = verify(r.get("evidence"))                        # verbatim-verified clips only
        vids = sorted({e["video_id"] for e in ev})            # distinct videos with a verified quote
        if len(vids) < floor:
            continue                                          # below the minimum-evidence floor -> drop
        r["n_videos"] = len(vids)
        r["video_ids"] = vids
        r["confidence"] = "high" if len(vids) >= 15 else "medium" if len(vids) >= 5 else "low"
        seen, sample = set(), []                              # display sample: up to 4 distinct videos
        for e in ev:
            if e["video_id"] in seen:
                continue
            seen.add(e["video_id"])
            sample.append(e)
            if len(sample) >= 4:
                break
        r["evidence"] = sample
        kept.append(r)
    kept.sort(key=lambda r: -r.get("n_videos", 0))
    agg["rules"] = kept
    rule_checked, rule_failed = checked, failed               # headline meter = RULE quotes only

    for i in agg.get("indicators_ranked", []):               # GIGO-clean the rest (no floor/tier)
        i["evidence"] = verify(i.get("evidence"))[:3]
    for side in ("buy", "sell", "avoid"):
        for t in agg.get("triggers", {}).get(side, []):
            t["evidence"] = verify(t.get("evidence"))[:2]
    for v in agg.get("vocabulary", []):
        v["examples"] = verify(v.get("examples"))[:2]

    stats = {"rule_quotes_checked": rule_checked, "rule_quotes_failed": rule_failed,
             "all_quotes_checked": checked, "all_quotes_failed": failed,
             "rules_before_floor": n_before, "rules_after_floor": len(kept), "floor_min_videos": floor}
    return agg, stats


def _section_diff(o: list, n: list, key: str) -> dict:
    """Added / dropped / reordered between two ranked sections. Items are matched by VIDEO-MEMBERSHIP
    overlap (Jaccard of video_ids) first — robust to Gemini re-wording the canonical label between runs
    — falling back to label-text similarity when membership is empty/unavailable."""
    def jac(a, b):
        sa, sb = set(a or []), set(b or [])
        return len(sa & sb) / len(sa | sb) if sa and sb else 0.0

    def best(oi, orl, pool):
        if not pool:
            return -1, 0.0, 0.0
        cand = []
        for j, r in enumerate(pool):
            jc = jac(orl.get("video_ids"), r.get("video_ids"))
            lab = SequenceMatcher(None, _nrm(orl.get(key, "")), _nrm(r.get(key, ""))).ratio()
            cand.append((jc, lab, -abs(j - oi), -j, j))       # high jac, high label-sim, NEAREST to oi
        cand.sort(reverse=True)
        jc, lab, _, _, j = cand[0]                            # tie-break toward same index -> identical -> empty diff
        return j, jc, lab

    matched, reordered, dropped = set(), [], []
    for oi, orl in enumerate(o):
        bi, jc, lab = best(oi, orl, n)
        if bi >= 0 and (jc >= 0.3 or lab >= 0.5):
            matched.add(bi)
            if oi != bi or orl.get("n_videos") != n[bi].get("n_videos"):
                reordered.append({"label": n[bi].get(key), "old_rank": oi + 1, "new_rank": bi + 1,
                                  "old_n_videos": orl.get("n_videos"), "new_n_videos": n[bi].get("n_videos"),
                                  "match": "videos" if jc >= 0.3 else "label", "overlap": round(max(jc, lab), 2)})
        else:
            dropped.append({"label": orl.get(key), "old_rank": oi + 1, "old_n_videos": orl.get("n_videos")})
    added = [{"label": nr.get(key), "new_rank": j + 1, "n_videos": nr.get("n_videos")}
             for j, nr in enumerate(n) if j not in matched]
    return {"counts": {"old": len(o), "new": len(n), "added": len(added),
                       "dropped": len(dropped), "reordered": len(reordered)},
            "added": added, "dropped": dropped, "reordered": reordered}


def _playbook_diff(old: dict, new: dict) -> dict:
    """Structured diff vs the PREVIOUS playbook — rules + indicators + each trigger bucket, in the same
    added/dropped/reordered format — for review. The new playbook is derived FRESH from the fact sheets;
    the old one is only read here to COMPARE, never to seed."""
    o = old if isinstance(old, dict) else {}
    return {
        "generated": new.get("meta", {}).get("generated"),
        "previous_generated": o.get("meta", {}).get("generated"),
        "rules": _section_diff(o.get("rules", []), new.get("rules", []), "canonical"),
        "indicators": _section_diff(o.get("indicators_ranked", []),
                                    new.get("indicators_ranked", []), "name"),
        "triggers": {side: _section_diff(o.get("triggers", {}).get(side, []),
                                         new.get("triggers", {}).get(side, []), "condition")
                     for side in ("buy", "sell", "avoid")},
    }


def build(limit: Optional[int] = None, refresh: bool = False, workers: int = 6,
          use_cache: bool = True, rebuild_cache: bool = False) -> dict:
    items = corpus(limit)
    print(f"[playbook] corpus = {len(items)} videos; extracting (model {MODEL}) …")
    per_video = []

    def work(it):
        try:
            return extract_video(it, refresh=refresh)
        except Exception as e:
            print(f"  extract fail {it['video_id']}: {str(e)[:80]}")
            return None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(work, items):
            if r:
                per_video.append(r)
    print(f"[playbook] per-video extractions = {len(per_video)}; aggregating …")
    joined_by_vid = {it["video_id"]: _strip(" ".join(s["text"] for s in it["segments"])) for it in items}
    # COUNTS come from deterministic programmatic clustering (true distinct-video membership). Gemini then
    # merges equivalent cluster LABELS only — and that one nondeterministic step is STABILIZED by the
    # content-addressed merge cache: identical facts -> identical labels/groupings with zero Gemini calls.
    prog = aggregate_programmatic(per_video)
    cache = {} if (rebuild_cache or not use_cache) else _load_cache()
    cache_arg = cache if use_cache else None                    # --no-cache -> always re-merge, don't persist
    sigs: set = set()
    if not use_cache:
        print("[playbook] --no-cache: forcing full Gemini re-merge (cache ignored, not written)")
    elif rebuild_cache:
        print("[playbook] --rebuild-cache: discarding merge_cache.json and re-merging from scratch")
    # ONE shared label-merge PER SECTION (rules / indicators / triggers per bucket). Triggers merge
    # WITHIN each bucket only — the volume+credit pattern legitimately lives in SELL AND AVOID.
    rules_in = prog.get("rules", [])
    merged = dict(prog)
    merged["rules"] = _merge_labels_gemini(                     # rules: NO head cap — merge in full
        rules_in, "canonical", "매매 규칙", want_kind=True, section="rules", head=None,
        examples="(예: '일정 매매'/'스케줄 매매', '순환매 대응'의 중복 표현)", cache=cache_arg, sigs=sigs)
    merged["indicators_ranked"] = _merge_labels_gemini(
        prog.get("indicators_ranked", []), "name", "지표", section="indicators",
        examples="(예: 'OBV'/'OBV (On Balance Volume)', '프로그램 순매수 증감폭'/'프로그램 매매 추이')",
        cache=cache_arg, sigs=sigs)
    merged["triggers"] = {
        side: _merge_labels_gemini(prog.get("triggers", {}).get(side, []), "condition",
                                   f"{side.upper()} 트리거 조건", section=f"triggers:{side}",
                                   examples="(예: '거래량 폭발+신용잔고 급증'의 표현 변형들)",
                                   cache=cache_arg, sigs=sigs)
        for side in ("buy", "sell", "avoid")}
    if use_cache:
        _save_cache(cache)
    method = "programmatic+gemini-merge" + ("+cached" if use_cache else "")
    # FIREWALL: verbatim-verify every cited quote, recount distinct VERIFIED videos, drop rules below
    # the >=N-video floor, assign confidence tiers. Counts are now backed by real, verbatim quotes.
    agg, fw = _firewall(merged, joined_by_vid)
    print(f"[playbook] firewall: {fw['rule_quotes_failed']}/{fw['rule_quotes_checked']} rule quotes failed "
          f"verbatim verification; rules {fw['rules_before_floor']} -> {fw['rules_after_floor']} "
          f"after >={fw['floor_min_videos']}-video floor")
    agg["meta"] = {"model": MODEL, "aggregation": method, "n_videos": len(per_video),
                   "generated": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
                   "confidence_tiers": "high >=15 · medium 5-14 · low 3-4 distinct videos with verified quotes",
                   "firewall": {**fw, "quote_match": "verbatim (whitespace-normalized exact substring)",
                                "merge_temperature": MERGE_TEMPERATURE,
                                "merge_schema_validated": "gemini-merge" in method},
                   "merge_cache": {"enabled": use_cache, "rebuilt": rebuild_cache, "signatures": len(sigs)},
                   "discipline": "Descriptive only — NOT trading signals; profitability is Phase 1."}
    agg["meta"]["content_hash"] = _content_hash(agg)
    # structured diff vs the PREVIOUS playbook (review artifact). Read the old file BEFORE overwriting;
    # the new playbook is derived fresh from the fact sheets — the old one is compared, never reused.
    old = {}
    pj = PLAYBOOK_DIR / "playbook.json"
    if pj.exists():
        try:
            old = json.loads(pj.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    pj.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    (PLAYBOOK_DIR / "playbook.md").write_text(render_markdown(agg), encoding="utf-8")
    diff = _playbook_diff(old, agg)
    (PLAYBOOK_DIR / "playbook_diff.json").write_text(json.dumps(diff, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    rc, ic = diff["rules"]["counts"], diff["indicators"]["counts"]
    tc = {s: diff["triggers"][s]["counts"] for s in ("buy", "sell", "avoid")}
    print(f"[playbook] saved -> {pj}  ({len(agg.get('rules', []))} rules) · diff vs previous: "
          f"rules +{rc['added']}/-{rc['dropped']}/~{rc['reordered']} · "
          f"indicators +{ic['added']}/-{ic['dropped']}/~{ic['reordered']} · "
          + " ".join(f"trig:{s}+{tc[s]['added']}/-{tc[s]['dropped']}/~{tc[s]['reordered']}"
                     for s in ("buy", "sell", "avoid")))
    _stability_guard(sigs, diff, agg["meta"]["content_hash"], use_cache)
    return agg


def _stability_guard(sigs: set, diff: dict, content_hash: str, use_cache: bool) -> None:
    """If the set of cluster signatures is UNCHANGED vs the previous working run (i.e. no fact change),
    assert the new-vs-old diff is EMPTY and log STABLE. Persists this run's signatures + content hash
    for next time. This is the day-to-day stability contract: same facts -> no playbook change."""
    cur = sorted(sigs)
    prev = {}
    if BUILD_STATE.exists():
        try:
            prev = json.loads(BUILD_STATE.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    if use_cache and prev.get("signatures") is not None and set(prev["signatures"]) == set(cur):
        sec_counts = [diff["rules"]["counts"], diff["indicators"]["counts"]] + \
                     [diff["triggers"][b]["counts"] for b in ("buy", "sell", "avoid")]
        empty = all(c["added"] == 0 and c["dropped"] == 0 and c["reordered"] == 0 for c in sec_counts)
        if empty and prev.get("content_hash") == content_hash:
            print("[playbook] STABLE: no fact change → no playbook change.")
        else:
            print("[playbook] WARNING: cluster signatures unchanged but output differs — investigate "
                  f"(diff empty={empty}, hash match={prev.get('content_hash') == content_hash}).")
    try:
        BUILD_STATE.write_text(json.dumps({"signatures": cur, "content_hash": content_hash},
                                          ensure_ascii=False, indent=0), encoding="utf-8")
    except Exception as e:
        print(f"[playbook] build_state save failed: {str(e)[:100]}")


def release() -> Optional[dict]:
    """MANUAL: promote the current WORKING playbook (playbook.json/md) to an immutable, auditable
    snapshot under playbook/releases/, and update the boss-facing playbook_released.{json,md} pointer.
    Never auto-run by the 06:50 job. Does NOT rebuild — it freezes whatever the working copy currently
    holds. Records released-at, corpus size, model, firewall stats, and content hash."""
    pj, pm = PLAYBOOK_DIR / "playbook.json", PLAYBOOK_DIR / "playbook.md"
    if not pj.exists():
        print("[release] no working playbook.json — run a build first.")
        return None
    data = pj.read_text(encoding="utf-8")
    md = pm.read_text(encoding="utf-8") if pm.exists() else ""
    meta = (json.loads(data) or {}).get("meta", {})
    now = datetime.now(KST)
    stamp = now.strftime("%Y%m%d-%H%M")
    rel = {"released_at": now.strftime("%Y-%m-%d %H:%M KST"), "snapshot": f"playbook_{stamp}",
           "generated": meta.get("generated"), "corpus_n_videos": meta.get("n_videos"),
           "model": meta.get("model"), "firewall": meta.get("firewall"),
           "content_hash": meta.get("content_hash"),
           "file_sha256": hashlib.sha256(data.encode("utf-8")).hexdigest()}
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    (RELEASE_DIR / f"playbook_{stamp}.json").write_text(data, encoding="utf-8")
    (RELEASE_DIR / f"playbook_{stamp}.md").write_text(md, encoding="utf-8")
    released = dict(json.loads(data) or {})
    released["release"] = rel
    RELEASED_JSON.write_text(json.dumps(released, ensure_ascii=False, indent=2), encoding="utf-8")
    header = (f"<!-- RELEASED {rel['released_at']} · snapshot releases/playbook_{stamp}.json · "
              f"corpus {rel['corpus_n_videos']} videos · model {rel['model']} · "
              f"content_hash {str(rel['content_hash'])[:12]} -->\n")
    RELEASED_MD.write_text(header + md, encoding="utf-8")
    print(f"[release] snapshot -> {RELEASE_DIR / ('playbook_' + stamp)}.{{json,md}}")
    print(f"[release] pointer  -> {RELEASED_JSON.name} / {RELEASED_MD.name}  "
          f"(corpus {rel['corpus_n_videos']} videos · content_hash {str(rel['content_hash'])[:12]})")
    return rel


def render_markdown(a: dict) -> str:
    m = a.get("meta", {})
    fw = m.get("firewall", {})
    L = [f"# 머니업 Playbook (descriptive doctrine)", "",
         f"_{m.get('generated','')} · model {m.get('model','')} · {m.get('n_videos',0)} videos · "
         f"agg {m.get('aggregation','')}_  ",
         f"> ⚠️ {m.get('discipline','')}  "]
    if fw:
        L.append(f"> 🔒 firewall: {fw.get('rule_quotes_failed')}/{fw.get('rule_quotes_checked')} rule quotes "
                 f"failed verbatim · rules {fw.get('rules_before_floor')}→{fw.get('rules_after_floor')} "
                 f"(≥{fw.get('floor_min_videos')} videos) · label-merge temp {fw.get('merge_temperature')}  ")
    L += ["", f"_Confidence tiers — {m.get('confidence_tiers','')}_", "",
          "## Top rules (ranked by # videos with verified quotes)"]
    for i, r in enumerate(a.get("rules", [])[:25], 1):
        ev = r.get("evidence", [])
        tier = (r.get("confidence") or "").upper()
        L.append(f"{i}. **{r.get('canonical')}**  _( {r.get('kind')} · {r.get('n_videos')} videos · "
                 f"**{tier}** )_  ")
        if r.get("reasoning"):
            L.append(f"   reasoning: {r['reasoning']}  ")
        for e in ev[:2]:
            L.append(f"   - `{e['video_id']}` @{e['mmss']}: \"{e['quote'][:120]}\"")
    L += ["", "## Indicators he watches (ranked)"]
    for i in a.get("indicators_ranked", [])[:15]:
        L.append(f"- **{i.get('name')}** ({i.get('n_videos',0)} videos)")
    L += ["", "## Triggers"]
    for side, lbl in (("buy", "BUY"), ("sell", "SELL"), ("avoid", "AVOID")):
        L.append(f"### {lbl}")
        for t in a.get("triggers", {}).get(side, [])[:8]:
            L.append(f"- {t.get('condition')} ({t.get('n_videos',0)})")
    L += ["", "## Vocabulary / catchphrases"]
    for v in a.get("vocabulary", [])[:20]:
        L.append(f"- **{v.get('term')}** — {v.get('meaning')}")
    return "\n".join(L)


def main():
    import argparse
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-cache", action="store_true",
                    help="force a full Gemini re-merge: ignore AND do not write merge_cache.json")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="discard merge_cache.json and re-merge from scratch, then repopulate it")
    ap.add_argument("--release", action="store_true",
                    help="MANUAL: freeze the current working playbook to an immutable boss-facing snapshot")
    a = ap.parse_args()
    if a.release:
        release()
        return
    agg = build(limit=a.limit, refresh=a.refresh, use_cache=not a.no_cache, rebuild_cache=a.rebuild_cache)
    fw = agg.get("meta", {}).get("firewall", {})
    print(f"\nfirewall: {fw.get('rule_quotes_failed')}/{fw.get('rule_quotes_checked')} rule quotes dropped "
          f"(verbatim); {fw.get('rules_after_floor')} rules after >={fw.get('floor_min_videos')}-video floor")
    print("Top 10 rules:")
    for i, r in enumerate(agg.get("rules", [])[:10], 1):
        print(f"  {i}. [{r.get('n_videos')}v {(r.get('confidence') or '').upper()}] {r.get('canonical')}")


if __name__ == "__main__":
    main()
