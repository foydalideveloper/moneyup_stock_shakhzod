"""Unified news alert stream + a transparent, rule-based SAFETY kill-switch.

Combines KR disclosures (OpenDART) and US news (Finnhub) into one newest-first feed,
and computes a HALT/OK signal the trading logic can read: if the linked US tape is
strongly negative (mean news sentiment below a threshold) OR the US overnight return
is sharply down, emit HALT. Rule-based and explainable — no price model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from tagent.news.finnhub_source import aggregate_sentiment


@dataclass
class NewsAlertConfig:
    overnight_halt: float = -0.03      # HALT if US overnight return <= this (-3%)
    sentiment_halt: float = -0.40      # HALT if mean linked US news sentiment <= this


def combine_alerts(kr_items: Sequence[dict], us_items: Sequence[dict],
                   limit: int = 50) -> List[dict]:
    """Merge KR + US alerts into one list, newest first (by ts), capped at ``limit``."""
    merged = list(kr_items) + list(us_items)
    merged.sort(key=lambda a: a.get("ts", 0.0), reverse=True)
    return merged[:limit]


def safety_signal(linked_sentiment: Optional[float], overnight_return: Optional[float],
                  cfg: Optional[NewsAlertConfig] = None) -> dict:
    """Rule-based kill-switch. HALT when the linked US tape is strongly negative on
    EITHER axis (overnight return or news sentiment). Returns the state + reasons."""
    cfg = cfg or NewsAlertConfig()
    reasons: List[str] = []
    if overnight_return is not None and overnight_return <= cfg.overnight_halt:
        reasons.append(f"US overnight {overnight_return*100:+.2f}% <= {cfg.overnight_halt*100:.0f}%")
    if linked_sentiment is not None and linked_sentiment <= cfg.sentiment_halt:
        reasons.append(f"linked US news sentiment {linked_sentiment:+.2f} <= {cfg.sentiment_halt:.2f}")
    state = "HALT" if reasons else "OK"
    return {"state": state, "halt": state == "HALT", "reasons": reasons,
            "overnight_return": overnight_return, "linked_sentiment": linked_sentiment,
            "thresholds": {"overnight_halt": cfg.overnight_halt,
                           "sentiment_halt": cfg.sentiment_halt}}


def build_news_payload(kr_items: Sequence[dict], us_items: Sequence[dict],
                       overnight_return: Optional[float] = None,
                       cfg: Optional[NewsAlertConfig] = None, limit: int = 50) -> dict:
    """Assemble the dashboard/runner payload: combined newest-first alerts + the
    safety light (driven by the US overnight return and mean linked US sentiment)."""
    cfg = cfg or NewsAlertConfig()
    linked_sentiment = aggregate_sentiment(us_items) if us_items else None
    safety = safety_signal(linked_sentiment, overnight_return, cfg)
    alerts = combine_alerts(kr_items, us_items, limit=limit)
    return {
        "enabled": True, "safety": safety,
        "n_kr": len(kr_items), "n_us": len(us_items),
        "alerts": alerts,
    }
