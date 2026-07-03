"""Order-execution log + dashboard panel — the at-a-glance proof the agent places orders via the
Kiwoom API automatically.

Each place/cancel appends ONE structured JSON line to ``data/order_log.jsonl`` (written UTF-8 with
``ensure_ascii=False`` so the Korean return messages read cleanly). The panel reads them back into a
human table: time (KST), stock, side (BUY/SELL), qty, price, 주문번호, status (ACCEPTED / CANCELLED /
FILLED / REJECTED), return_code, and a MOCK vs LIVE badge. Pure / no network.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from tagent.config import DATA_DIR
from tagent.intraday_feed import kst_stamp

ORDER_LOG = "order_log.jsonl"


def order_log_path(data_dir=None) -> Path:
    return (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / ORDER_LOG


def log_order_event(*, action: str, stock: str, side: str, qty, price, ord_no: str,
                    return_code, return_msg: str, env: str, accepted: bool,
                    ts: Optional[str] = None, data_dir=None) -> dict:
    """Append one order event (UTF-8). ``action`` "buy"/"sell"/"cancel"; ``env`` "mock"/"live"."""
    ev = {"ts": ts or datetime.now(timezone.utc).isoformat(), "action": str(action),
          "stock": str(stock), "side": str(side), "qty": qty, "price": price,
          "ord_no": str(ord_no or ""), "return_code": return_code, "return_msg": str(return_msg or ""),
          "env": str(env or "mock"), "accepted": bool(accepted)}
    p = order_log_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:                # UTF-8 so 주문 메시지 read cleanly
        fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return ev


def load_order_events(data_dir=None) -> List[dict]:
    p = order_log_path(data_dir)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():   # read UTF-8 (clean Korean)
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def order_status(ev: dict) -> str:
    """ACCEPTED / CANCELLED / FILLED / REJECTED from an event."""
    action = str(ev.get("action", "")).lower()
    if not ev.get("accepted"):
        return "REJECTED"
    if action == "cancel":
        return "CANCELLED"
    if action in ("fill", "filled"):
        return "FILLED"
    return "ACCEPTED"


def order_panel(events: Optional[List[dict]] = None, *, limit: int = 50, data_dir=None) -> dict:
    """Human-readable order table for the dashboard (newest-first). Honest empty when no orders."""
    events = events if events is not None else load_order_events(data_dir)
    rows = []
    for ev in list(reversed(events))[:limit]:
        rows.append({
            "time_kst": kst_stamp(ev.get("ts", "")), "stock": ev.get("stock", ""),
            "side": str(ev.get("side", "")).upper(), "qty": ev.get("qty"), "price": ev.get("price"),
            "ord_no": ev.get("ord_no", ""), "status": order_status(ev),
            "return_code": ev.get("return_code"), "return_msg": ev.get("return_msg", ""),
            "env": str(ev.get("env", "mock")).lower(), "live": str(ev.get("env", "")).lower() == "live",
        })
    return {"enabled": True, "n": len(rows), "orders": rows,
            "note": "Kiwoom API order execution — MOCK 모의투자 unless badged LIVE" if rows
                    else "no orders yet (runs scripts/run_kiwoom_order_proof.py during KR market hours)"}
