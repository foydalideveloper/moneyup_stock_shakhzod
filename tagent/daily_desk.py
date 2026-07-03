"""Daily Desk — a semiconductor trading desk's morning briefing + monitor, on-screen.

Three honestly-labelled layers:
  1. PRE-OPEN RECOMMENDATIONS — grounded in the VALIDATED momentum edge (12-1,
     point-in-time): rank the watchlist by momentum, emit BUY candidates (with reason
     + suggested stop) and a SELL/REDUCE list for paper-holdings that dropped out of
     the momentum top group, broke their stop, or hit a downtrend regime.
  2. IN-SESSION TIMING MONITOR — DECISION-SUPPORT ONLY (no proven intraday edge):
     live price / computed candles / RSI / volume-vs-avg + the US-shock ARMED state
     and the news HALT/OK safety light, so the user sees timing context. No orders.
  3. CONSOLIDATED BRIEFING — one table of price / volume / 공매도 / SOX context / news
     sentiment for the ~9 names (the boss's briefing on-screen instead of email).

Plus a PAPER TRACK of the BUY recommendations so we forward-test whether the desk adds
value. Pure functions over cached/injected daily bars + Kiwoom minute bars; no orders.

No-lookahead: every recommendation/briefing uses only bars dated <= the as-of date; the
paper track books a recommendation's forward return only once it is realised.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from tagent.config import DATA_DIR
from tagent.intraday_backtest import CostModel
from tagent.strategies.earnings_drift import market_regime
from tagent.xs_momentum import align_close, momentum_signal

# Default KR semiconductor watchlist (6-digit codes). Daily bars come from the
# authoritative split-adjusted pykrx source (data/<code>_1d.csv). US context for the briefing.
SEMI_WATCHLIST = ("000660", "005930", "042700", "009150")   # SK Hynix, Samsung, Hanmi Semi, Samsung E-M
US_CONTEXT = ("NVDA", "AMD", "MU", "AVGO", "SOXX")     # SOXX ~ SOX index proxy

# Data-quality guard. KR equities are limit-bounded at ±30%/day (since 2015), so any
# adjacent-day move beyond ±35% in a cached daily series is a split/adjustment artifact
# (mixed adjusted+unadjusted, a missing split factor), not real price action — surface it
# as an error rather than letting a corrupt level inflate a displayed price or paper P&L.
MAX_DAILY_MOVE = 0.35


def sanity_check(panel, max_daily_move: float = MAX_DAILY_MOVE) -> Dict[str, list]:
    """Scan a daily panel for implausible adjacent-day moves (split/adjustment artifacts).

    Returns ``{symbol: [(date, ret), ...]}`` for every name with an out-of-bound move;
    an empty dict means the data is clean. ``panel`` may be a {sym: DataFrame} dict or a
    close DataFrame. The default ±35% bound sits just outside KR's ±30% daily price limit,
    so it trips on data corruption, not on real (even limit-up) volatility."""
    close = align_close(panel) if isinstance(panel, dict) else panel
    flagged: Dict[str, list] = {}
    for s in close.columns:
        r = close[s].dropna().pct_change()
        bad = r[r.abs() > max_daily_move]
        if len(bad):
            flagged[s] = [(str(pd.Timestamp(d).date()), round(float(v), 4)) for d, v in bad.items()]
    return flagged


def assert_sane(panel, max_daily_move: float = MAX_DAILY_MOVE) -> None:
    """Raise ``ValueError`` if any daily series has an implausible (artifact) move, so
    corrupted prices fail loudly instead of producing fake levels or P&L."""
    flagged = sanity_check(panel, max_daily_move)
    if flagged:
        worst = {s: v[0] for s, v in flagged.items()}
        raise ValueError(
            f"Daily price sanity check failed — implausible moves (>{max_daily_move:.0%}, "
            f"likely split/adjustment artifacts): {worst}")


@dataclass
class DeskConfig:
    watchlist: tuple = SEMI_WATCHLIST
    top_n: int = 5                    # BUY candidates surfaced
    top_group: int = 5               # "momentum top group" for the sell-out rule
    lookback: int = 252              # 12-1 momentum formation
    skip: int = 21
    recent_window: int = 21          # "recent return" context (~1 month)
    stop_pct: float = 0.08           # suggested stop distance
    regime_ma: int = 200
    per_trade_frac: float = 0.05     # paper sizing per recommendation
    slippage_bps: float = 15.0       # conservative KR round trip ~0.51%

    def cost(self) -> CostModel:
        return CostModel(slippage_bps=self.slippage_bps)


# --------------------------------------------------------------------------- #
# 1) pre-open recommendations (validated momentum edge)
# --------------------------------------------------------------------------- #
def momentum_ranks(panel, cfg: DeskConfig, asof=None,
                   membership: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Rank the watchlist by 12-1 momentum AS OF ``asof`` (no-lookahead). Returns
    [symbol, mom, recent_return, rank] sorted best-first."""
    close = align_close(panel)
    if asof is not None:
        close = close.loc[close.index <= pd.Timestamp(asof)]
    if close.empty:
        return pd.DataFrame(columns=["symbol", "mom", "recent_return", "rank", "price"])
    need = cfg.lookback + cfg.skip + 1
    wl = [s for s in cfg.watchlist if s in close.columns]
    rows = []
    for s in wl:
        cs = close[s].dropna()                             # the name's own valid history <= asof
        if len(cs) < need:
            continue
        if membership is not None:
            mser = membership.reindex(index=[cs.index[-1]], columns=[s])
            if not bool(mser.fillna(False).iloc[0, 0]):
                continue
        mom = float(cs.iloc[-1 - cfg.skip] / cs.iloc[-1 - cfg.skip - cfg.lookback] - 1.0)  # 12-1
        rw = min(cfg.recent_window, len(cs) - 1)
        recent = float(cs.iloc[-1] / cs.iloc[-1 - rw] - 1.0) if rw > 0 else 0.0
        rows.append({"symbol": s, "mom": mom, "recent_return": recent, "price": float(cs.iloc[-1])})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "mom", "recent_return", "rank", "price"])
    df = df.sort_values("mom", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def regime_is_on(panel, cfg: DeskConfig, asof=None,
                 membership: Optional[pd.DataFrame] = None) -> bool:
    """Is the KR market above its regime MA as of ``asof`` (the validated gate)?"""
    reg = market_regime(panel, ma_window=cfg.regime_ma, membership=membership)
    if asof is not None:
        reg = reg.loc[reg.index <= pd.Timestamp(asof)]
    return bool(reg.iloc[-1]) if len(reg) else True


def buy_candidates(panel, cfg: DeskConfig, asof=None,
                   membership: Optional[pd.DataFrame] = None) -> dict:
    """Top-N BUY candidates by momentum, each with a reason + suggested stop. In a
    confirmed downtrend the validated edge says CASH, so BUYs are withheld."""
    ranks = momentum_ranks(panel, cfg, asof, membership)
    on = regime_is_on(panel, cfg, asof, membership)
    if not on:
        return {"regime": "cash", "note": "KR market below its 200d MA — momentum edge says CASH; "
                "no new BUYs.", "buys": []}
    buys = []
    for r in ranks.head(cfg.top_n).itertuples():
        buys.append({
            "symbol": r.symbol, "rank": int(r.rank), "price": round(r.price, 2),
            "mom_12_1": round(r.mom, 4), "recent_return": round(r.recent_return, 4),
            "stop": round(r.price * (1.0 - cfg.stop_pct), 2),
            "reason": f"momentum rank #{int(r.rank)} of watchlist · 12-1 {r.mom:+.1%} · "
                      f"1-mo {r.recent_return:+.1%}",
            "basis": "validated momentum edge (12-1, point-in-time)",
        })
    return {"regime": "in-market", "buys": buys}


def sell_list(holdings, panel, cfg: DeskConfig, asof=None,
              membership: Optional[pd.DataFrame] = None,
              entry_stops: Optional[Dict[str, float]] = None) -> List[dict]:
    """SELL/REDUCE list for paper-holdings: out of the momentum top group, stop broken,
    or downtrend regime. ``holdings`` = {sym: ...} or a list; ``entry_stops`` = {sym: stop_price}."""
    ranks = momentum_ranks(panel, cfg, asof, membership)
    top = set(ranks.head(cfg.top_group)["symbol"]) if len(ranks) else set()
    close = align_close(panel)
    if asof is not None:
        close = close.loc[close.index <= pd.Timestamp(asof)]
    on = regime_is_on(panel, cfg, asof, membership)
    entry_stops = entry_stops or {}
    out = []
    for sym in list(holdings):
        if sym not in close.columns:
            continue
        price = float(close[sym].iloc[-1])
        reasons = []
        if not on:
            reasons.append("downtrend regime (market < 200d MA) — reduce")
        stop = entry_stops.get(sym)
        if stop is not None and price < stop:
            reasons.append(f"broke stop ({price:,.0f} < {stop:,.0f})")
        if top and sym not in top:
            reasons.append("dropped out of the momentum top group")
        if reasons:
            out.append({"symbol": sym, "price": round(price, 2), "reasons": reasons,
                        "basis": "validated momentum edge (exit rules)"})
    return out


# --------------------------------------------------------------------------- #
# 2) in-session timing monitor (decision-support, NOT a proven edge)
# --------------------------------------------------------------------------- #
def _rsi(closes: Sequence[float], period: int = 14) -> float:
    s = pd.Series(closes, dtype=float)
    if len(s) <= period:
        return float("nan")
    d = s.diff()
    up = d.clip(lower=0).rolling(period).mean()
    dn = (-d.clip(upper=0)).rolling(period).mean()
    rs = up / dn.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def session_signals(symbol: str, minute_bars, vol_window: int = 20,
                    us_armed: Optional[bool] = None, safety: Optional[str] = None,
                    interval: str = "1min") -> dict:
    """Assemble the in-session monitor row from minute bars (Kiwoom feed) + context.

    DECISION-SUPPORT ONLY — there is no proven intraday edge; this is for timing
    context. Computes price, candle, RSI, volume-vs-average from the raw bars; the
    US-shock ARMED state + news safety light are passed in."""
    from tagent.intraday_demo import candle_parts
    out = {"symbol": str(symbol), "label": "decision-support (no proven intraday edge)",
           "us_shock": "ARMED" if us_armed else ("OK" if us_armed is not None else None),
           "safety": safety}
    if isinstance(minute_bars, pd.DataFrame) and not minute_bars.empty:
        df = minute_bars
        closes = df["close"].astype(float).tolist()
        vols = df["volume"].astype(float)
        last = df.iloc[-1]
        out.update({
            "price": round(float(last["close"]), 2),
            "rsi": round(_rsi(closes), 1) if not np.isnan(_rsi(closes)) else None,
            "vol_vs_avg": round(float(vols.iloc[-1] / vols.rolling(vol_window).mean().iloc[-1]), 2)
            if len(vols) > vol_window and vols.rolling(vol_window).mean().iloc[-1] > 0 else None,
            "n_bars": int(len(df)),
            "candle": candle_parts({"open": float(last["open"]), "high": float(last["high"]),
                                    "low": float(last["low"]), "close": float(last["close"])}),
        })
    else:
        out.update({"price": None, "rsi": None, "vol_vs_avg": None, "n_bars": 0, "candle": None})
    return out


# --------------------------------------------------------------------------- #
# 3) consolidated briefing table
# --------------------------------------------------------------------------- #
def briefing_row(symbol: str, daily_df: pd.DataFrame, news_sentiment: Optional[str] = None,
                 short_ratio: Optional[float] = None, vol_window: int = 20) -> dict:
    """One briefing row: price, day change, volume + volume-vs-avg, 공매도 (if given),
    latest news sentiment."""
    if daily_df is None or daily_df.empty:
        return {"symbol": symbol, "price": None, "change": None, "volume": None,
                "vol_vs_avg": None, "short_ratio": short_ratio, "news": news_sentiment}
    df = daily_df.sort_index()
    last = df.iloc[-1]
    prev = df["close"].iloc[-2] if len(df) > 1 else last["close"]
    vol = df["volume"] if "volume" in df.columns else None
    vv = None
    if vol is not None and len(vol) > vol_window and vol.rolling(vol_window).mean().iloc[-1] > 0:
        vv = round(float(vol.iloc[-1] / vol.rolling(vol_window).mean().iloc[-1]), 2)
    return {
        "symbol": symbol, "price": round(float(last["close"]), 2),
        "change": round(float(last["close"] / prev - 1.0), 4) if prev else None,
        "volume": int(last["volume"]) if "volume" in df.columns and pd.notna(last.get("volume")) else None,
        "vol_vs_avg": vv, "short_ratio": short_ratio, "news": news_sentiment,
    }


def briefing_table(symbols: Sequence[str], panels: Dict[str, pd.DataFrame],
                   news_by_sym: Optional[Dict[str, str]] = None,
                   shorts: Optional[Dict[str, float]] = None) -> List[dict]:
    news_by_sym = news_by_sym or {}
    shorts = shorts or {}
    return [briefing_row(s, panels.get(s), news_by_sym.get(s), shorts.get(s)) for s in symbols]


# --------------------------------------------------------------------------- #
# 4) paper track of the recommendations
# --------------------------------------------------------------------------- #
class DeskTracker:
    """Paper-tracks BUY recommendations: did they go up, net of cost? (per-trade
    fractional sizing; deduped by (date, symbol); persistent)."""

    CSV_FIELDS = ["entry", "symbol", "gross", "net", "equity"]

    def __init__(self, cfg: Optional[DeskConfig] = None, data_dir=None, clock=None):
        self.cfg = cfg or DeskConfig()
        self.dir = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        self.state_path = self.dir / "daily_desk_track.json"
        self.csv_path = self.dir / "daily_desk_track.csv"
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.equity = 10_000.0
        self.trades = 0
        self.wins = 0
        self.sum_net = 0.0
        self.booked: set = set()
        self.last_date: Optional[str] = None
        self.first_ts: Optional[str] = None
        self._load()

    def book(self, entry_date, symbol: str, gross_ret: float) -> dict:
        ds = str(pd.Timestamp(entry_date).date())
        key = f"{ds}|{symbol}"
        if key in self.booked:
            return self.status()
        net = float(gross_ret) - self.cfg.cost().round_trip_frac()
        self.equity *= (1.0 + self.cfg.per_trade_frac * net)
        self.trades += 1
        self.wins += int(net > 0)
        self.sum_net += net
        self.booked.add(key)
        if self.first_ts is None:
            self.first_ts = ds
        if self.last_date is None or ds > self.last_date:
            self.last_date = ds
        self._append_csv(ds, symbol, gross_ret, net)
        self._save()
        return self.status()

    def status(self) -> dict:
        wr = (self.wins / self.trades) if self.trades else 0.0
        exp = (self.sum_net / self.trades) if self.trades else 0.0
        return {"equity": round(self.equity, 2), "n_trades": self.trades,
                "win_rate_pct": round(wr * 100, 1), "expectancy_pct": round(exp * 100, 4),
                "round_trip_cost_pct": round(self.cfg.cost().round_trip_frac() * 100, 3),
                "last_date": self.last_date, "first_ts": self.first_ts}

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        state = {"cfg": asdict(self.cfg), "equity": self.equity, "trades": self.trades,
                 "wins": self.wins, "sum_net": self.sum_net, "booked": sorted(self.booked),
                 "last_date": self.last_date, "first_ts": self.first_ts}
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            s = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.equity = float(s.get("equity", 10_000.0))
        self.trades = int(s.get("trades", 0))
        self.wins = int(s.get("wins", 0))
        self.sum_net = float(s.get("sum_net", 0.0))
        self.booked = set(s.get("booked", []))
        self.last_date = s.get("last_date")
        self.first_ts = s.get("first_ts")

    def _append_csv(self, ds, symbol, gross, net) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(self.CSV_FIELDS)
            w.writerow([ds, symbol, f"{gross*100:.3f}", f"{net*100:.3f}", round(self.equity, 2)])


# --------------------------------------------------------------------------- #
# snapshot for the dashboard (pre-open recos + briefing + track)
# --------------------------------------------------------------------------- #
def desk_snapshot_path(data_dir=None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / "daily_desk.json"


def write_desk_snapshot(payload: dict, data_dir=None) -> Path:
    path = desk_snapshot_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_desk_snapshot(data_dir=None) -> dict:
    """Dashboard reader: the latest pre-open desk snapshot, or {'enabled': False}."""
    path = desk_snapshot_path(data_dir)
    if not path.exists():
        return {"enabled": False}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"enabled": False}
