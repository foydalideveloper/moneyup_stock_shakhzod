"""RL execution/filtering layer — synthetic episodes, no network."""

import numpy as np
import pandas as pd

from tagent.rl_execution import (
    ACT, HOLD, NOTHING, ExecEnv, evaluate_baseline, evaluate_policy, rollout,
    run_experiment, summarize, train_q, transition,
)


def _df(mid, imb, spoof=0.0):
    n = len(mid)
    ts = pd.date_range("2026-01-01", periods=n, freq="s", tz="UTC").astype(str)
    return pd.DataFrame({"ts": ts, "mid": np.asarray(mid, float),
                         "depth_imbalance": np.asarray(imb, float),
                         "absorption_ratio": 0.5,
                         "spoof_ratio": np.full(n, spoof, float)})


def _alt_signal(n):
    return np.where(np.arange(n) % 2 == 0, 0.6, -0.6)


def _churn_episode(n=400, seed=0):
    """Strong alternating signal but tiny random moves (<< cost) -> acting churns."""
    rng = np.random.default_rng(seed)
    mid = 100 + np.cumsum(rng.normal(0, 0.001, n))     # ~0.1 bp steps
    return _df(mid, _alt_signal(n))


def _predictive_episode(n=400, move=0.002, seed=1):
    """Persistent-sign signal (continuous magnitudes) that genuinely predicts a
    move BIGGER than cost -> the IC is well-defined and positive."""
    rng = np.random.default_rng(seed)
    sgn = np.r_[np.full(n // 2, 1.0), np.full(n - n // 2, -1.0)]
    imb = sgn * rng.uniform(0.3, 0.9, n)               # continuous, sign predicts move
    rets = move * sgn
    mid = 100 * np.cumprod(1 + np.r_[0.0, rets[:-1]])
    return _df(mid, imb)


# --------------------------------------------------------------------------- #
# transitions
# --------------------------------------------------------------------------- #
def test_transition_semantics():
    assert transition(0, ACT, 1) == 1 and transition(1, ACT, -1) == -1
    assert transition(1, HOLD, -1) == 1                 # keep current
    assert transition(1, NOTHING, 1) == 0              # go flat


# --------------------------------------------------------------------------- #
# reward includes costs; over-trading penalised
# --------------------------------------------------------------------------- #
def test_reward_subtracts_cost_on_trade_only():
    env = ExecEnv(_df([100, 101, 101, 101], [0.6, 0.6, 0.6, 0.6]), signal_sign=1, cost_bps=10.0)
    env.reset()
    _, r0, _ = env.step(ACT)                            # 0 -> +1 : pays cost on 1 unit
    assert np.isclose(r0, 1 * env.fwd[0] - env.cost_frac * 1)
    _, r1, _ = env.step(HOLD)                           # +1 -> +1 : no trade, no cost
    assert np.isclose(r1, 1 * env.fwd[1])              # cost term is zero
    assert env.cost_frac == 10.0 / 1e4


def test_overtrading_is_penalised_on_flat_prices():
    env = ExecEnv(_df(np.full(200, 100.0), _alt_signal(200)), signal_sign=1, cost_bps=10.0)
    churn = evaluate_baseline(env)                      # ACT every step on alternating signal
    nothing = rollout(env, lambda s, pos, sig: NOTHING)
    assert churn["costs"] > 0 and churn["net"] < 0      # pure cost bleed
    assert np.isclose(nothing["net"], 0.0) and nothing["n_trades"] == 0
    assert churn["net"] < nothing["net"]               # over-trading strictly worse


# --------------------------------------------------------------------------- #
# forward return alignment (no lookahead in the env)
# --------------------------------------------------------------------------- #
def test_forward_return_is_one_step_ahead():
    mid = [100.0, 102.0, 101.0, 105.0]
    env = ExecEnv(_df(mid, [0.6, 0.6, 0.6, 0.6]), signal_sign=1)
    assert np.isclose(env.fwd[0], 102 / 100 - 1)
    assert np.isclose(env.fwd[1], 101 / 102 - 1)
    assert env.fwd[-1] == 0.0                           # last row has no future


def test_chronological_split_no_leak():
    res = run_experiment(_predictive_episode(300), train_frac=0.6)
    assert res["n_train"] + res["n_test"] == 300
    assert res["n_train"] == 180 and res["n_test"] == 120   # train strictly precedes test


# --------------------------------------------------------------------------- #
# the policy cuts churn (the whole point)
# --------------------------------------------------------------------------- #
def test_policy_reduces_trades_and_not_worse_when_signal_is_noise():
    res = run_experiment(_churn_episode(500), train_frac=0.6, cost_bps=10.0)
    pol, base = res["test"]["policy"], res["test"]["baseline"]
    assert pol["n_trades"] < base["n_trades"]           # trades far less
    assert pol["net"] >= base["net"]                    # net no worse (baseline bleeds cost)
    assert base["net"] < 0                              # acting on every flip loses to cost
    s = summarize(res)
    assert s["trade_reduction_pct"] > 50.0 and s["beats_baseline_net"]


def test_policy_still_acts_when_signal_truly_pays():
    # when the move clears cost, the filter should NOT just sit out -> it trades and earns
    res = run_experiment(_predictive_episode(400, move=0.003), train_frac=0.6, cost_bps=10.0)
    pol = res["test"]["policy"]
    assert pol["n_trades"] > 0 and pol["net"] > 0       # acts, and nets positive


def test_train_env_only_sees_train_rows():
    df = _predictive_episode(200)
    res = run_experiment(df, train_frac=0.6)
    # policy learned on 120 train rows; evaluated on the 80 held-out rows
    assert res["n_train"] == 120 and res["n_test"] == 80
    assert res["test"]["policy"]["steps"] == 80 - 1


# --------------------------------------------------------------------------- #
# Q / policy plumbing
# --------------------------------------------------------------------------- #
def test_train_q_shapes_and_greedy_runs():
    env = ExecEnv(_predictive_episode(120), signal_sign=1)
    Q = train_q(env, epochs=20)
    assert Q.shape == (ExecEnv.N_STATES, 3)
    out = evaluate_policy(env, Q)
    assert set(out) >= {"net", "n_trades", "steps", "net_bps"}
