"""RL execution/filtering layer ON TOP of the crypto order-book signal.

The order-book signal (depth imbalance etc.) has a tiny, real, *sub-cost* edge:
acting on every flip churns and bleeds fees. This layer does NOT try to invent
alpha — it learns *when it's worth acting* so the net-of-cost result stops being
dominated by trading costs.

Framing — a small, fully-observed MDP solved offline:
  * **state** = recent order-book features (signal direction, |imbalance| strength,
    spoof bucket) + current position {-1,0,+1};
  * **actions** = {ACT on signal, HOLD current position, DO NOTHING (go flat)};
  * **reward** = NET-of-cost P&L for the step =
        new_position * forward_one_step_return  −  cost * |Δposition|
    so every trade explicitly pays a cost and over-trading is penalised.

The exogenous part of the state (features) follows the recorded sequence and the
forward return is observed, so the controlled dynamics (position) are
deterministic and known — we solve it with tabular value iteration over the
TRAIN slice only (strict chronological split, no lookahead) and report the greedy
policy OOS against the raw "act on every signal" baseline.

Pure numpy/pandas; unit-tested on synthetic episodes (no network, no recording).
Offline experiment only — not wired to live trading.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from tagent.microstructure import information_coefficient, time_split

# actions
ACT, HOLD, NOTHING = 0, 1, 2
N_ACTIONS = 3
_POS = (-1, 0, 1)


def transition(pos: int, action: int, signal: int) -> int:
    """Next position from an action: ACT -> take the signal, HOLD -> keep, NOTHING -> flat."""
    if action == ACT:
        return int(signal)
    if action == HOLD:
        return int(pos)
    return 0


def learn_signal_sign(imbalance, fwd_ret) -> int:
    """Orientation of the order-book signal, learned on TRAIN only: does positive
    depth-imbalance precede UP (+1) or DOWN (-1) moves? Defaults to +1 when the IC
    is undefined (too few distinct values) — only flips on a clear negative IC."""
    ic = information_coefficient(imbalance, fwd_ret)
    return -1 if (np.isfinite(ic) and ic < 0) else 1


class ExecEnv:
    """Offline order-book execution episode built from a recorded slice.

    Precomputes, per row t: the signal direction, the one-step forward return
    (mid[t+1]/mid[t]-1, the realised return of a position held over the step), and
    the discretised exogenous state bucket. Rows are consecutive samples.
    """

    N_EXO = 3 * 2 * 2          # signal{-1,0,1} x |imb| strength{0,1} x spoof{0,1}
    N_STATES = N_EXO * 3       # x position {-1,0,1}

    def __init__(self, df: pd.DataFrame, signal_sign: int = 1, cost_bps: float = 10.0,
                 imb_dead: float = 0.05, imb_strong: float = 0.30, spoof_hi: float = 0.5):
        d = df.reset_index(drop=True)
        mid = pd.to_numeric(d["mid"], errors="coerce").to_numpy(float)
        imb = pd.to_numeric(d.get("depth_imbalance", 0.0), errors="coerce").to_numpy(float)
        spoof = pd.to_numeric(d.get("spoof_ratio", 0.0), errors="coerce").to_numpy(float)
        imb = np.nan_to_num(imb)
        spoof = np.nan_to_num(spoof)
        T = len(mid)
        fwd = np.zeros(T)
        if T >= 2:
            fwd[:-1] = mid[1:] / np.where(mid[:-1] == 0, np.nan, mid[:-1]) - 1.0
        self.fwd = np.nan_to_num(fwd)
        # signal direction (oriented by the train-learned sign, with a dead zone)
        raw = np.sign(imb) * int(signal_sign)
        raw[np.abs(imb) < imb_dead] = 0
        self.signal = raw.astype(int)
        self._imb_strong = (np.abs(imb) >= imb_strong).astype(int)
        self._spoof_hi = (spoof >= spoof_hi).astype(int)
        self.cost_frac = cost_bps / 1e4
        self.T = T
        # cache exogenous bucket per row
        self.exo = np.array([self._exo(i) for i in range(T)], dtype=int)
        self.pos = 0
        self.i = 0

    # --- discretisation ---
    def _exo(self, i: int) -> int:
        sig = self.signal[i] + 1                       # 0,1,2
        return (sig * 2 + self._imb_strong[i]) * 2 + self._spoof_hi[i]

    def encode(self, exo: int, pos: int) -> int:
        return int(exo) * 3 + (int(pos) + 1)

    # --- gym-ish interface (used by tests / rollouts) ---
    def reset(self) -> int:
        self.pos, self.i = 0, 0
        return self.encode(self.exo[0], self.pos) if self.T else 0

    def step(self, action: int):
        sig = self.signal[self.i]
        new_pos = transition(self.pos, action, sig)
        reward = new_pos * self.fwd[self.i] - self.cost_frac * abs(new_pos - self.pos)
        self.pos = new_pos
        self.i += 1
        done = self.i >= self.T - 1
        nxt = self.encode(self.exo[self.i], self.pos) if not done else -1
        return nxt, float(reward), bool(done)


def train_q(env: ExecEnv, epochs: int = 60, gamma: float = 0.0, alpha: float = 0.5) -> np.ndarray:
    """Tabular value iteration over the TRAIN episode (model-based: the forward
    return is observed and position dynamics are deterministic, so we can back up
    every (state, position, action) at each step)."""
    Q = np.zeros((ExecEnv.N_STATES, N_ACTIONS))
    fwd, exo, sig, cf, T = env.fwd, env.exo, env.signal, env.cost_frac, env.T
    for _ in range(epochs):
        for t in range(T - 1):
            for pos in _POS:
                s = env.encode(exo[t], pos)
                for a in range(N_ACTIONS):
                    npos = transition(pos, a, sig[t])
                    r = npos * fwd[t] - cf * abs(npos - pos)
                    s2 = env.encode(exo[t + 1], npos)
                    Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
    return Q


def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """Argmax action per state, tie-broken toward LESS trading (HOLD/NOTHING)."""
    pol = np.zeros(Q.shape[0], dtype=int)
    for s in range(Q.shape[0]):
        best = Q[s].max()
        # prefer NOTHING, then HOLD, then ACT among (near-)ties -> fewer trades
        for a in (NOTHING, HOLD, ACT):
            if Q[s, a] >= best - 1e-12:
                pol[s] = a
                break
    return pol


def rollout(env: ExecEnv, act: Callable[[int, int, int], int]) -> Dict[str, float]:
    """Run a policy through the episode. `act(state, pos, signal) -> action`."""
    pos, net, gross, costs, trades = 0, 0.0, 0.0, 0.0, 0
    for t in range(env.T - 1):
        s = env.encode(env.exo[t], pos)
        a = act(s, pos, env.signal[t])
        npos = transition(pos, a, env.signal[t])
        c = env.cost_frac * abs(npos - pos)
        g = npos * env.fwd[t]
        if npos != pos:
            trades += 1
        net += g - c
        gross += g
        costs += c
        pos = npos
    steps = max(1, env.T - 1)
    return {"net": net, "gross": gross, "costs": costs, "n_trades": trades,
            "steps": env.T - 1, "net_bps": net * 1e4,
            "net_bps_per_step": net / steps * 1e4,
            "trades_per_step": trades / steps}


def evaluate_policy(env: ExecEnv, Q: np.ndarray) -> Dict[str, float]:
    pol = greedy_policy(Q)
    return rollout(env, lambda s, pos, sig: int(pol[s]))


def evaluate_baseline(env: ExecEnv) -> Dict[str, float]:
    """The raw strategy: ACT on every signal (the thing we're trying to beat)."""
    return rollout(env, lambda s, pos, sig: ACT)


def run_experiment(df: pd.DataFrame, train_frac: float = 0.6, cost_bps: float = 10.0,
                   epochs: int = 60, gamma: float = 0.0,
                   imb_dead: float = 0.05, imb_strong: float = 0.30) -> dict:
    """Full offline experiment on one recording: chronological split, learn the
    signal orientation + policy on TRAIN, evaluate both policy and baseline OOS.
    """
    d = df.dropna(subset=["mid"]).reset_index(drop=True)
    n = len(d)
    tr, te = time_split(n, train_frac)
    train_df = d.iloc[tr].reset_index(drop=True)
    test_df = d.iloc[te].reset_index(drop=True)

    # orientation learned on TRAIN only (no lookahead)
    tmp = ExecEnv(train_df, signal_sign=1, cost_bps=cost_bps,
                  imb_dead=imb_dead, imb_strong=imb_strong)
    sign = learn_signal_sign(
        pd.to_numeric(train_df.get("depth_imbalance", 0.0), errors="coerce").to_numpy()[:-1],
        tmp.fwd[:-1])

    env_tr = ExecEnv(train_df, signal_sign=sign, cost_bps=cost_bps,
                     imb_dead=imb_dead, imb_strong=imb_strong)
    env_te = ExecEnv(test_df, signal_sign=sign, cost_bps=cost_bps,
                     imb_dead=imb_dead, imb_strong=imb_strong)
    Q = train_q(env_tr, epochs=epochs, gamma=gamma)

    pol_test = evaluate_policy(env_te, Q)
    base_test = evaluate_baseline(env_te)
    pol_train = evaluate_policy(env_tr, Q)
    base_train = evaluate_baseline(env_tr)
    return {
        "signal_sign": int(sign), "cost_bps": cost_bps,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test": {"policy": pol_test, "baseline": base_test},
        "train": {"policy": pol_train, "baseline": base_train},
        "Q": Q,
    }


def summarize(res: dict) -> dict:
    """Honest OOS comparison: net, trades, and whether the policy beats baseline."""
    p, b = res["test"]["policy"], res["test"]["baseline"]
    return {
        "oos_policy_net_bps": round(p["net_bps"], 3),
        "oos_baseline_net_bps": round(b["net_bps"], 3),
        "oos_policy_trades": p["n_trades"],
        "oos_baseline_trades": b["n_trades"],
        "trade_reduction_pct": round((1 - p["n_trades"] / max(1, b["n_trades"])) * 100.0, 1),
        "beats_baseline_net": p["net"] > b["net"],
        "policy_near_breakeven": abs(p["net_bps_per_step"]) < abs(b["net_bps_per_step"]) + 1e-9,
    }
