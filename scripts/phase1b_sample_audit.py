# -*- coding: utf-8 -*-
"""Gate 2 — human sample-audit table for the CORRECTED Phase 1B classifier.

Emits, for 15-20 calls (weighted to the powered segments AND drawn deliberately from the newly-included
LONG-GENERIC and the Tier-3 ambiguous): quote → corrected class → entry rule → exit rule. NO RETURNS.
Also prints the disposition delta vs the original run. Classification only — no pykrx, no returns.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from moneyup_advisor import config, tickers, phase1b as P

OUT = config.DATA_DIR / "qa_audit" / "phase1b_sample_audit.md"

ENTRY_RULE = {
    "SCHEDULE": "enter NEXT session open (≤10d window)",
    "ROTATION": "enter NEXT session open (≤10d window)",
    "LONG-GENERIC": "enter NEXT session open (≤10d window)",
    "DIP-BUY": "enter AT stated support if a low touches it ±0.5% within 10d (else NON-TRIGGERED)",
    "BREAKOUT": "enter AT close if close>stated level on volume ≥2× 20d-avg (else NON-TRIGGERED)",
    "BEARISH": "SHORT at NEXT session open (≤10d window), sign −1",
}
EXIT_RULE = {
    "SCHEDULE": "hold to the parsed event date (D-N/relative); else 20d cap",
    "ROTATION": "triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap",
    "LONG-GENERIC": "triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap",
    "DIP-BUY": "triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap",
    "BREAKOUT": "triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap",
    "BEARISH": "SYMMETRIC ±8% triple-barrier (sign −1), else 20d cap",
}


def _tier3_cat():
    cat = {}
    try:
        fl = json.loads((config.DATA_DIR / "qa_audit" / "factsheet_qa_flags.json").read_text(encoding="utf-8"))
        for f in fl.get("track_a_flags", []):
            if f.get("category") in ("direction_mismatch", "ticker_resolution"):
                cat[(f.get("sheet"), f.get("call_index"))] = f.get("category")
    except Exception:
        pass
    return cat


def main():
    calls = P.load_calls()
    t3cat = _tier3_cat()
    rows = []
    for c in calls:
        if c.get("ambiguous"):
            disp = {"bucket": "AMBIGUOUS", "segment": None, "scored": False,
                    "reason": t3cat.get((c["video_id"], c["call_index"]), "Tier-3 ambiguous") + " — surfaced for review, excluded"}
        else:
            disp = P.classify_call(c)
        rows.append({**c, "disp": disp})

    # ---- disposition counts ----
    from collections import Counter
    bc = Counter()
    for r in rows:
        bc[r["disp"]["segment"] or r["disp"]["bucket"]] += 1
    scored = sum(1 for r in rows if r["disp"]["scored"])

    # ---- select the sample (deterministic) ----
    picked_ids = set()
    ordered = sorted(rows, key=lambda x: (x["video_id"], x["call_index"]))
    def take(pred, n):
        out = []
        for r in ordered:
            rid = (r["video_id"], r["call_index"])
            if rid not in picked_ids and pred(r):
                out.append(r); picked_ids.add(rid)
                if len(out) >= n:
                    break
        return out

    def seg_is(*names): return lambda r: (r["disp"]["segment"] in names)
    def bucket_is(*names): return lambda r: (r["disp"]["bucket"] in names) and not r["disp"]["scored"]
    sample = []
    sample += [("LONG-GENERIC (newly included, §1.3)", r) for r in take(seg_is("LONG-GENERIC"), 6)]
    sample += [("powered play", r) for r in take(seg_is("DIP-BUY", "BREAKOUT", "ROTATION", "SCHEDULE"), 4)]
    sample += [("BEARISH (AVOID split, §1.1)", r) for r in take(seg_is("BEARISH"), 2)]
    sample += [("Tier-3 ambiguous (review)", r) for r in take(lambda r: r["disp"]["bucket"] == "AMBIGUOUS", 4)]
    sample += [("excluded (Tier-1 filter)", r) for r in take(bucket_is("TRIM", "NOISE", "EXPOST", "HOLD-WAIT-CASH"), 4)]

    # ---- write ----
    L = ["# Phase 1B — Gate 2 human sample-audit (NO RETURNS)\n",
         "_Classification + entry/exit RULE only. Returns are computed ONLY after this table is signed off._\n",
         "## Disposition of all 1,884 ex-ante calls under the corrected classifier\n",
         f"- **Scored** (directional test): **{scored}** — " +
         " · ".join(f"{s}={bc[s]}" for s in P.SEGMENTS if bc[s]),
         f"- **Excluded but counted**: " + " · ".join(f"{b}={bc[b]}" for b in P.EXCLUDED_BUCKETS if bc[b]),
         "\n### What changed vs the original run (call-set delta)\n",
         f"- **+{bc['LONG-GENERIC']} LONG-GENERIC** — plain longs the old scorer DROPPED as UNCLASSIFIED "
         "(§1.3); now entered next-open + symmetric triple-barrier.",
         "- **SCHEDULE is date-gated** — the 374 boilerplate '스케줄 매매' misfires are routed to "
         f"LONG-GENERIC/excluded; only {bc['SCHEDULE']} calls with a real dated event remain SCHEDULE.",
         f"- **AVOID split (§1.1)** → BEARISH={bc['BEARISH']} (scored, sign −1) / HOLD-WAIT-CASH="
         f"{bc['HOLD-WAIT-CASH']} (excluded) / TRIM={bc['TRIM']} (excluded).",
         f"- **Tier-1 removed**: NOISE={bc['NOISE']} · EXPOST={bc['EXPOST']} · CANCELLED(negation)={bc['CANCELLED']} "
         "(holdings-substring avoids no longer fire).",
         f"- **Tier-3 ambiguous excluded for review**: AMBIGUOUS={bc['AMBIGUOUS']} (direction/ticker).",
         "- **Sign gate fixed (§1.2)**: PASS ⇔ mean net>0 AND t≥+deflated_bar for EVERY segment; "
         "t≤−bar ⇒ WRONG-SIGNED. Default barriers SYMMETRIC ±8% (never −5/+8).\n",
         "## Sample (15–20 calls) — verify class + rules; NO returns yet\n",
         "| # | source | video [mm:ss] | ticker | corrected class | scored | entry rule | exit rule | quote (verbatim) |",
         "|--:|---|---|---|---|:--:|---|---|---|"]
    for i, (src, r) in enumerate(sample, 1):
        d = r["disp"]; seg = d["segment"]
        cls = seg or d["bucket"]
        nm = tickers.display_name(r["ticker"]) or r["ticker"]
        entry = ENTRY_RULE.get(seg, "—  (excluded: " + (d["reason"][:54]) + ")")
        exit_ = EXIT_RULE.get(seg, "—")
        q = (r["quote"] or "").replace("\n", " ").replace("|", "/")[:180]
        L.append(f"| {i} | {src} | `{r['video_id']}` {r.get('mmss','')} | {nm}({r['ticker']}) | "
                 f"**{cls}** | {'✅' if d['scored'] else '—'} | {entry} | {exit_} | {q} |")
    L.append("\n**STOP — awaiting human sign-off.** No returns/verdict computed until this sample is approved.\n")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"scored={scored} | " + " ".join(f"{s}={bc[s]}" for s in P.SEGMENTS if bc[s]))
    print(f"excluded: " + " ".join(f"{b}={bc[b]}" for b in P.EXCLUDED_BUCKETS if bc[b]))
    print(f"sample rows: {len(sample)}  -> {OUT}")


if __name__ == "__main__":
    main()
