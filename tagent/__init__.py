"""trading-agent: a modular, real-time stock monitoring + ML prediction agent.

Layers (see README):
  feeds/      live market data adapters (Alpaca now; Kiwoom/Nautilus later)
  data/       historical data loaders for training (yfinance/OpenBB)
  state.py    per-symbol live state
  strategy.py rule-based "best opportunity" conditions
  risk.py     position sizing, stops, daily-loss limit, kill switch
  features.py feature engineering for ML
  labels.py   triple-barrier labeling for ML
  ml/         LightGBM training + inference
  backtest.py vectorized backtest with realistic costs
  alerts.py   debounced alert delivery
"""

__version__ = "0.1.0"
