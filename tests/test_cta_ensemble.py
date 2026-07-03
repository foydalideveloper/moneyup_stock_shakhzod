"""CTA ensemble — multi-lookback TSMOM blend, vol-target, risk-parity, portfolio scaling.
Synthetic series, no network."""

import numpy as np
import pandas as pd

from tagent.cta_ensemble import (
    COSTS, LOOKBACKS, cta_combined, load_sleeves, portfolio_vol_scale, tsmom_signal,
    vol_targeted_sleeve,
)


def _close(drift, n=360, vol=0.01, seed=0, start="2015-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="B")
    r = drift + rng.normal(0.0, vol, n)
    return pd.Series(100 * np.cumprod(1 + np.r_[0.0, r[:-1]]), index=idx)


# --------------------------------------------------------------------------- #
# 1) multi-lookback TSMOM blend (1/3/12mo)
# --------------------------------------------------------------------------- #
def test_tsmom_blend_signs_and_range():
    up = tsmom_signal(_close(0.003, vol=0.002))      # all horizons up
    dn = tsmom_signal(_close(-0.003, vol=0.002))     # all horizons down
    assert abs(up.iloc[-1] - 1.0) < 1e-9 and abs(dn.iloc[-1] + 1.0) < 1e-9
    assert ((up.dropna() >= -1.0) & (up.dropna() <= 1.0)).all()
    # blend = mean of three sign(trailing L) -> in {-1,-1/3,1/3,1}
    assert set(np.unique(np.round(up.dropna().to_numpy(), 6))) <= {-1.0, -1 / 3, 1 / 3, 1.0, 0.0}


def test_tsmom_uses_all_three_lookbacks():
    # down over 12mo but up over the last ~1mo -> horizons disagree -> partial signal
    idx = pd.date_range("2015-01-01", periods=360, freq="B")
    close = pd.Series(np.r_[np.linspace(100, 60, 339), np.linspace(60, 72, 21)], index=idx)
    s = tsmom_signal(close, LOOKBACKS).iloc[-1]
    assert -1.0 < s < 1.0                            # 1mo up, 12mo down -> blend = +1/3
    assert abs(s - 1 / 3) < 1e-9


# --------------------------------------------------------------------------- #
# 2) vol-targeted sleeve: same rule, cost dispatch, no-lookahead
# --------------------------------------------------------------------------- #
def test_vol_target_position_and_cost_and_nolookahead():
    c = _close(0.002, vol=0.01)
    sw, roll = COSTS["index"]
    s_lowcost = vol_targeted_sleeve(c, sw, roll)
    s_highcost = vol_targeted_sleeve(c, COSTS["crypto"][0], COSTS["crypto"][1])
    assert s_highcost.mean() < s_lowcost.mean()      # crypto cost/funding drags net below index
    # no-lookahead: appending future bars doesn't change the past sleeve returns
    ext = pd.concat([c, _close(0.002, n=40, seed=7, start="2016-05-23").iloc[:40].set_axis(
        pd.date_range(c.index[-1] + pd.offsets.BDay(1), periods=40, freq="B"))])
    s2 = vol_targeted_sleeve(ext, sw, roll)
    common = s_lowcost.index[:-2]
    assert np.allclose(s_lowcost.reindex(common).to_numpy(), s2.reindex(common).to_numpy())


# --------------------------------------------------------------------------- #
# 3) portfolio vol scaling: targets vol, lagged (no-lookahead), capped
# --------------------------------------------------------------------------- #
def test_portfolio_vol_scale_lagged_and_capped():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2015-01-01", periods=300, freq="B")
    r = pd.Series(rng.normal(0.0, 0.02, 300), index=idx)
    sc = portfolio_vol_scale(r, port_target=0.10, window=63, cap=3.0)
    assert (sc <= 3.0 + 1e-9).all() and (sc >= 0).all()
    # lagged: scalar[t] uses realized vol through t-1 -> appending future leaves past unchanged
    r2 = pd.concat([r, pd.Series(rng.normal(0, 0.05, 50),
                                 index=pd.date_range(r.index[-1] + pd.offsets.BDay(1), periods=50, freq="B"))])
    sc2 = portfolio_vol_scale(r2, 0.10, 63, 3.0)
    assert np.allclose(sc.to_numpy(), sc2.reindex(sc.index).to_numpy())


# --------------------------------------------------------------------------- #
# 4) combined accounting + same params on every market (no per-market tuning)
# --------------------------------------------------------------------------- #
def test_cta_combined_same_params_everywhere():
    sleeves = pd.DataFrame({
        "A": vol_targeted_sleeve(_close(0.002, seed=1), *COSTS["index"]),
        "B": vol_targeted_sleeve(_close(0.001, seed=2, vol=0.02), *COSTS["index"]),
    })
    comb = cta_combined(sleeves, ["A", "B"], vol_window=30)
    assert len(comb) > 0 and np.isfinite(comb.to_numpy()).all()
    # the signal/vol params are module-level constants (identical across markets by construction)
    assert LOOKBACKS == (21, 63, 252)
    # crypto sleeve in MARKETS bundles BTC+ETH; cost dispatch differs only by KIND, not market
    assert COSTS["index"] != COSTS["crypto"] and COSTS["fx"][1] == 0.0
