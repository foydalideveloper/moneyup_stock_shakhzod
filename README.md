# trading-agent

A modular, real-time stock monitoring + ML prediction agent. This is the
robust foundation (Phases 1–2 of our plan): a clean, tested core you extend
toward OpenBB data, TradingAgents-style reasoning, and NautilusTrader execution.

## What works today
- **Local web dashboard** (FastAPI + one white page): two independent search
  boxes — US stocks (Alpaca → ML BUY/HOLD/SELL + paper account) and crypto
  (Binance → live 10-level book + memory gauges). (`scripts/run_dashboard.py`)
- **Live paper trading** on Alpaca (US): the trained model runs each new daily
  bar, the risk layer sizes/stops/halts, and **paper** orders are placed and
  logged — paper-only, idle off-hours, reconnects. (`scripts/run_paper_trader.py`)
- **Live monitoring** of ~10 symbols via Alpaca (free IEX feed): tracks bid/ask,
  spread, session highs/lows, and a rolling window — and fires **debounced alerts**.
- **Live KR feed via Kiwoom** (REST API + real-time WebSocket): streams 10-level
  호가 → `OrderBookSnapshot`, 체결 → `Trade`, best bid/ask → `Quote`, feeding the
  same Monitor/OrderBookMemory unchanged. Mock-first, keys from `.env`.
- **Free crypto order-book feed** (Binance public WebSocket, **no API key**):
  depth → `OrderBookSnapshot`, trades → `Trade`, best bid/ask → `Quote`, with one
  `OrderBookMemory` per coin and **runtime** `subscribe_symbol`/`unsubscribe_symbol`
  so a dashboard can search/switch coins on the fly. (`scripts/record_orderbook.py`)
- **Level-2 depth memory**: remembers the visible 10-level order book per symbol
  and classifies *why* price levels disappear — **filled** (consumed by trades /
  price advancing), **cancelled** (pulled with no trade → possible spoof), or
  **scrolled** (window shifted) — then derives absorption / spoof / depth-imbalance
  features for the strategy and future agents.
- **Korean short-selling (공매도) data layer**: loads daily/EOD short-selling
  stats and engineers **leak-free** features (short ratio, its change, balance
  change, rolling z-score) that optionally merge into the training set.
- **Korean 수급 (foreign/institutional flow) data layer**: daily investor
  net-buying via pykrx → **leak-free** flow features (net, rolling sum, change,
  z-score), with combined + broad OOS studies to test for edge.
- **Risk layer**: position sizing, stop-loss/take-profit, daily-loss limit, kill switch.
- **ML pipeline**: download history (yfinance) → engineer features → triple-barrier
  labels → train **LightGBM** with **walk-forward validation** → **backtest with costs**
  (both an in-sample backtest and a leak-free **out-of-sample** one vs buy-and-hold).
- **A real pytest suite** for all the pure logic.

## Project layout
```
trading-agent/
  tagent/
    config.py        # all settings (env-overridable)
    state.py         # per-symbol live state (incremental, tested)
    orderbook.py     # Level-2 depth memory: fills vs cancels vs scrolls
    strategy.py      # rule-based "best opportunity" conditions  <- edit me
    risk.py          # sizing, stops, daily limit, kill switch
    alerts.py        # debounced alerts (screen + optional Telegram)
    monitor.py       # orchestrator: feed -> state -> strategy -> risk -> alert
    feeds/           # base.py (Quote/Trade/OrderBookSnapshot interface) + alpaca_feed.py
                     #   + kiwoom_auth.py (OAuth2) + kiwoom_feed.py (KR live WS)
                     #   + crypto_feed.py (Binance public WS, no key, dynamic symbols)
    data/historical.py      # yfinance OHLCV loader (training data)
    data/krx_source.py      # REAL KR data via pykrx (OHLCV + 공매도), free, CSV-cached
    data/short_selling.py   # KR short-selling (공매도) canonical loader (CSV or fetch=)
    data/flows_source.py    # KR 수급 (foreign/institutional flow) loader via pykrx
    data/krx_signals.py     # gated KR signals registry (lending/margin/program/investor-detail/short-balance)
    data/funding.py         # Binance public funding/perp/spot for cash-and-carry (no key)
    features.py      # feature engineering (no lookahead)
    features_short.py   # leak-free short-selling features (EOD/delayed alignment)
    features_flows.py   # leak-free 수급 flow features (EOD/delayed alignment)
    features_signals.py # leak-free generic features for the gated KR signals
    labels.py        # triple-barrier labeling
    cross_sectional.py  # cross-sectional long-short (panel, purged CV, portfolio)
    microstructure.py   # order-book edge experiment (forward returns, IC, OOS, verdict)
    funding_study.py    # crypto cash-and-carry simulation (funding capture, net of costs)
    paper.py         # live PAPER trader: decide_long + PaperTrader (Alpaca, paper-only)
    dashboard.py     # FastAPI backend: /signals /account /orderbook /memory + sources
    static/dashboard.html  # one clean white page (two independent search boxes)
    ml/train.py      # purged+embargoed walk-forward (+ walk_forward_predict) + select_features
    ml/models.py     # regularized LightGBM/XGBoost/CatBoost, calibrated, + ensemble factory
    ml/predict.py    # inference wrapper
    backtest.py      # vectorized backtest with costs
  scripts/           # run_monitor / download_data / train_model / run_backtest / run_backtest_oos
                     #   + alpaca_smoke_test / run_paper_trader (live US paper trading)
                     #   + run_dashboard (local web UI: stocks + crypto)
                     #   + download_krx / download_krx_universe / download_krx_signals (KRX login)
                     #   + run_short_study / run_combined_study / run_broad_study (4-arm OOS)
                     #   + run_cross_sectional / run_robust_study (ensemble + purged CV)
                     #   + record_orderbook / run_microstructure_study (order-book edge)
                     #   + run_funding_study (crypto cash-and-carry, public Binance, no key)
                     #   + kiwoom_smoke_test (auth + 1 호가 snapshot, mock-first)
  tests/             # pytest suite
```

## Setup
```bash
cd trading-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then paste your free Alpaca paper keys
```

## The workflows
```bash
# 1) Live monitor + alerts (US market hours; ~22:30–05:00 Korea time)
python scripts/run_monitor.py

# 2) Download training data (saved as CSV in data/)
python scripts/download_data.py --period 3y --interval 1d

# 3) Train the model with walk-forward validation
python scripts/train_model.py --tp 0.04 --sl 0.02 --horizon 10

# 4) Backtest the trained model on one symbol (costs included)  -- IN-SAMPLE, optimistic
python scripts/run_backtest.py --symbol AAPL --threshold 0.55

# 5) TRUE out-of-sample backtest + buy-and-hold benchmark (the honest number)
python scripts/run_backtest_oos.py --symbols AAPL MSFT NVDA --threshold 0.55

# 6) LIVE PAPER TRADING on Alpaca (US) — confirm keys, then run during market hours
python scripts/alpaca_smoke_test.py                 # auth + balance + AAPL quote/trade
python scripts/run_paper_trader.py --threshold 0.55 # paper-only; idle when market closed
```

## Live paper trading (Alpaca, US)
`scripts/run_paper_trader.py` runs the trained model live against your Alpaca
**paper** account. Each new daily bar during US market hours it computes features
for the watchlist (AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, AMD, NFLX, SPY),
runs the model, applies the **risk layer** (sizing, stops/targets, daily-loss
kill switch), places **paper** market orders, and logs every decision — symbol,
action, proba, size, stop/target, reason — to `data/paper_trades.csv`.

- **Paper only.** The `TradingClient` is built with `paper=True` and there is no
  live switch; it can never route to a real account.
- **Idle off-hours / reconnects.** It checks the Alpaca clock and stays idle when
  the market is closed, acts once per new bar per symbol, and retries with backoff
  on transient errors. Risk equity is sized off the actual paper-account equity.
- Decision logic (`tagent/paper.py::decide_long`) is pure and the Alpaca clients
  are injected, so the decision/risk/order wiring is unit-tested with fakes (no
  network) in `tests/test_paper.py`.

```bash
python scripts/alpaca_smoke_test.py            # 1) confirm keys (REST, any hours)
python scripts/train_model.py                  # 2) ensure a model exists (US 3y)
python scripts/run_paper_trader.py             # 3) run during US market hours
python scripts/run_paper_trader.py --once      #    or: one forced pass now (wiring check)
```

> Set `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (paper keys) in `.env`. US market
> hours are 09:30–16:00 ET (≈22:30–05:00 KST). Free Alpaca data is delayed and
> daily bars run through the prior session — fine for a daily-bar model.

## Local web dashboard — live-analysis view
`scripts/run_dashboard.py` serves a clean white page at **http://localhost:8000**
that **visualizes the agent's pipeline on live data**, with **two independent
search boxes** (US stocks via Alpaca, crypto via Binance — separate markets, each
box drives only its own panel). Each panel shows:

- a **live candlestick chart** (SVG candles, blue = up / red = down, + volume) —
  US: 1-min Alpaca bars (poll ~5 s); crypto: 1-min Binance klines (poll ~1.5 s);
- a **pipeline strip**: `① API data → ② Features → ③ Model → ④ Decision`, with the
  live values flowing through, so a non-technical viewer sees how it works;
- an **agent-analysis panel** updated each poll: candle type, RSI, momentum,
  (crypto) order-book imbalance/absorption/spoof, the **model probability**, and
  the **BUY/HOLD/SELL** decision;
- US: the BUY/HOLD/SELL signals table + paper-account summary; crypto: the
  10-level ladder (asks red, bids green) + vanished-level tags + gauges.

The US decision comes from the **real ML model** (daily technical features); the
crypto decision comes from the **`OrderBookMemory`** microstructure (imbalance /
absorption / spoof). BUY=green, SELL=red, HOLD=grey. Searching a coin that isn't
subscribed yet calls `CryptoFeed.subscribe_symbol()` to start it.

### Agents per market (consistent) + the signal scorecard
Each market runs the same kind of agents, and the **scorecard** forward-tests
every one in its own **$10 000 paper book per (source, symbol)**, net of costs:

| market | agents (scorecard sources) |
|--------|----------------------------|
| US     | `us-ml` (LightGBM on daily klines) · `us-ta` (TA chart breakouts) |
| Crypto | `crypto-ml` (LightGBM per coin) · `crypto-ob` (order-book) · `crypto-ta` (TA breakouts) · `funding-carry` (the real edge) |

The prediction agents are scored on directional hit-rate. **`funding-carry`** is
different: a **delta-neutral** position (long spot + short perp on BTC/ETH) that
accrues the live 8h funding minus costs, **held continuously with hysteresis**
(enter when funding clears the hurdle, exit only on sustained-negative funding).
It marks its own $10k-per-coin book and — per the deepened study — sits
**flat-to-slightly-positive** (or flat when funding is negative, as it is now)
while the predictors churn and bleed fees. This is the one real net-of-cost edge.

Train the crypto ML models with `python scripts/train_crypto_ml.py` (per-coin
`models/crypto_<SYM>.pkl`, kline tech features + order-book micro where a
recording overlaps + triple-barrier labels). The `crypto-ml` agent loads them
live and is scored alongside the others.

> **No US order-book agent — and why.** Free Alpaca market data is **top-of-book
> only** (best bid/ask, no depth), so the `crypto-ob`-style microstructure agent
> (depth imbalance / absorption / spoofing) **cannot** be built for US equities on
> the free tier. A US order-book agent needs a **paid Level-2 depth feed** (e.g.
> **Polygon.io** stocks, or a direct exchange/SIP depth feed). Until then US runs
> ML + TA only; crypto (free Binance depth) additionally runs the order-book agent.

```bash
python scripts/run_dashboard.py            # http://localhost:8000
python scripts/run_dashboard.py --port 8080 --symbols btc eth sol
```

JSON endpoints (built by `tagent/dashboard.py::create_app` from injected sources,
so they're unit-tested with fakes — no Alpaca, no network):
`GET /candles?symbol=&market=` (us → Alpaca 1-min, crypto → Binance klines),
`GET /analysis?symbol=&market=` (features + model prob + decision),
`GET /signals?symbols=`, `GET /account`, `GET /orderbook?symbol=`,
`GET /memory?symbol=`. The crypto panel needs no key; if Alpaca keys / a model are
missing the US panel shows errors while crypto keeps working.

> **In-sample vs out-of-sample.** `run_backtest.py` scores a model trained on *all*
> history over that *same* history — it has effectively seen the answers, so its
> equity curve is over-optimistic and must not be trusted. `run_backtest_oos.py`
> trades each bar using a model trained only on *earlier* folds
> (`tagent.ml.train.walk_forward_predict`), so no bar is predicted by a model that
> saw it. Expect the OOS numbers to be far worse — and to compare them against
> buy-and-hold. In our run the in-sample AAPL backtest showed **+504%**, while the
> leak-free OOS result was **+19.6%** and *lost* to AAPL buy-and-hold (**+54.8%**).

## Run the tests
```bash
pip install pytest pandas numpy
pytest            # state, strategy, risk, features, labels, backtest, monitor, orderbook, short-selling, krx, kiwoom, flows, cross-sectional, paper, crypto, dashboard, krx-signals, models, microstructure, funding
```
> The pure-logic modules are covered by tests. The live feed (Alpaca) and the
> LightGBM trainer need your keys / heavy libraries installed to run end-to-end.

## How to read the results (honest expectations)
- A walk-forward **accuracy near 0.50** or **AUC near 0.50** means little or no
  edge — that is the *normal* result, not a failure of the code.
- A backtest must include costs (it does here). A great-looking curve still needs
  out-of-sample checks and **paper trading** before any real money.
- An **in-sample** backtest (model scored on data it trained on) routinely prints
  spectacular, fictional returns. Always cross-check with `run_backtest_oos.py`,
  and beat **buy-and-hold** out of sample after costs — otherwise there is no edge.
- The risk layer — not prediction accuracy — is what keeps losses survivable.

## Level-2 order-book depth memory
`tagent/orderbook.py` adds an `OrderBookMemory` per symbol. Each time a new
`OrderBookSnapshot` (10 ask levels ascending, 10 bid levels descending) arrives,
it is diffed against the previous one and every price level that **left the
visible top-10** is classified:

| reason       | meaning                                                                 |
|--------------|-------------------------------------------------------------------------|
| `filled`     | trades printed at/through that price since the last snapshot (or price advanced past it) — **consumed** liquidity |
| `cancelled`  | the resting size was pulled with **no** trade at that price, while it was still inside the live window — **possible spoof** |
| `scrolled`   | it only fell out because price moved and the window shifted over it — neither hit nor pulled |

The memory keeps the last **10 vanished levels per side** (`vanished_asks`,
`vanished_bids` deques) and exposes derived features via `snapshot()`:

- `filled_qty` / `cancelled_qty` / `scrolled_qty`
- `absorption_ratio = filled / (filled + cancelled)` — near 1.0 = book absorbing flow
- `spoof_ratio = cancelled / (filled + cancelled + scrolled)` — high = liquidity pulled
- `depth_imbalance = (bid_depth − ask_depth) / (bid_depth + ask_depth)` ∈ [−1, 1]

`Monitor` holds one memory per symbol, updates it on `on_orderbook` events
(passing the trades that printed since the previous snapshot so fills can be told
from cancels), and exposes `monitor.orderbook_snapshot(symbol)` for the strategy
and future agents. Tick handling is unchanged. When a feed supplies no trade data,
classification falls back to the documented price-movement heuristic.

> **Note:** Alpaca's free IEX equity feed does not stream Level-2 depth, so the
> default `AlpacaFeed` emits no order-book events today — the plumbing is in place
> for a crypto/L2-capable feed (or Kiwoom) to call `_emit_orderbook(...)`.

## Korean short-selling (공매도) features
`tagent/data/short_selling.py` loads KR short-selling data into a tidy frame
indexed by **date** with canonical columns: `short_volume`, `volume`,
`short_ratio` (= `short_volume / volume`, computed if absent), `short_value`, and
`short_balance` (공매도잔고, when available). It resolves common KRX/Korean
headers (e.g. `일자`, `공매도`, `거래량`, `공매도잔고`) automatically.

```python
from tagent.data.short_selling import load_short_selling
short = load_short_selling("005930")            # reads data/005930_short.csv
# or plug in a live source later:
short = load_short_selling("005930", fetch=my_krx_api_fn)
```

> **This data is DAILY / end-of-day and delayed.** KRX publishes figures for
> trading day *D* the following business day (balances lag further). The loader
> never pretends otherwise, and the feature layer enforces it.

`tagent/features_short.py` aligns those daily metrics to your price bars with
**no lookahead**: every short series is shifted by 1 day (EOD/delay) and then
as-of joined so a bar dated *T* only ever sees short data dated **before** *T*.
It produces `short_ratio`, `short_ratio_change`, `short_balance_change`, and a
rolling z-score `short_ratio_z`.

These merge **optionally** into the training set — the existing pipeline is
unchanged unless you ask for them:

```python
from tagent.ml.train import build_dataset
X, y = build_dataset(history, short_data={"005930": short})   # adds 4 short cols
```
or from the CLI: `python scripts/train_model.py --with-short` (uses
`data/<SYMBOL>_short.csv` where present, silently skips symbols without one).

## Real Korean-market data via pykrx (free, no Kiwoom key)
`tagent/data/krx_source.py` wires the layer above to **real KRX data** through
[`pykrx`](https://github.com/sharebook-kr/pykrx) — no broker account needed:

```bash
python scripts/download_krx.py --years 3      # default: 005930, 000660, 035420
python scripts/run_short_study.py --tickers 005930 000660 035420
```

- `get_krx_history(ticker, start, end)` returns the **same** OHLCV schema as
  `data/historical.py` (open/high/low/close/volume, `timestamp` index) and caches
  to `data/<TICKER>_1d.csv`, so `load_history("005930")` works afterwards.
- `krx_short_fetch(start, end)` returns a `fetch(symbol)` callable matching
  `load_short_selling`'s `fetch=` interface, mapping pykrx's
  `get_shorting_volume_by_date` + `get_shorting_balance_by_date` (Korean columns
  공매도 / 매수 / 공매도잔고 / 공매도금액) into the canonical short schema.

pykrx is imported lazily and results cache to CSV (so re-runs are offline).

> **Heads-up on credentials.** KRX **OHLCV is open access** and works out of the
> box. The **공매도 (short-selling) endpoints now require a *free* KRX website
> account**, which pykrx reads from the `KRX_ID` / `KRX_PW` environment variables.
> Without them those endpoints return empty and `download_krx.py` skips short data
> for that symbol (OHLCV is still saved). **No account needed:** export the CSV
> manually from data.krx.co.kr and drop it at `data/<TICKER>_short.csv` — the
> loader reads cp949/EUC-KR and maps the Korean headers automatically (see below).

#### Manual KRX short CSVs (no key)
`load_short_selling` robustly ingests data.krx.co.kr's **공매도거래** export:
tries UTF-8 then **cp949/EUC-KR**, parses `일자` as the date index, and maps the
flattened Korean headers (`공매도 수량_거래량_전체` → `short_volume`,
`공매도 금액_거래대금_전체` → `short_value`, `공매도 수량_순보유잔고수량` →
`short_balance`). That export has **no total-volume column**, so `short_ratio` is
completed from the price file's (real KRX) volume via `complete_short_ratio()` —
`run_short_study.py` does this automatically.

## 수급 (foreign/institutional flow) features
`tagent/data/flows_source.py` fetches per-ticker DAILY investor net buying —
foreign (외국인) and institutional (기관) — via pykrx's
`get_market_trading_value_by_date` (columns `외국인합계`/`기관합계` →
`foreign_net`/`inst_net`; value by default, volume optional). It caches to
`data/<code>_flows.csv` and matches `load_flows`'s `fetch=` interface.

> **Same gate as 공매도.** This per-ticker 수급 endpoint requires a free KRX
> account. Set `KRX_ID` / `KRX_PW` in `.env` — pykrx auto-logs-in and the loaders
> export them to the environment (`config.has_krx_login()` /
> `krx_source.ensure_krx_login()`). If KRX **rejects** the login (it prints
> `KRX 로그인 실패: 자격 증명을 확인하세요`), pykrx returns empty and the fetch
> raises a clear, caught error — in that case download the CSV manually (below).

#### Manual KRX 수급 CSV (no key)
If the KRX login doesn't work, export per-ticker investor trading by hand:

1. **data.krx.co.kr** → 통계(Statistics) → 기본통계 → 주식 →
   **[12009] 투자자별 거래실적(개별종목)** (Trading by Investor — individual stock).
2. Set **종목** = the ticker (e.g. `005930`), **기간** = your date range,
   **거래대금** (value) or 거래량 (volume), and **조회구분 = 순매수** (net buying).
3. Click the **CSV download** (다운로드) button.
4. Save as `data/<TICKER>_flows.csv` (e.g. `data/005930_flows.csv`).

The loader reads cp949/EUC-KR, parses `일자` as the date index, and maps the
investor columns **`외국인합계`(또는 `외국인`) → `foreign_net`** and
**`기관합계`(또는 `기관`) → `inst_net`** automatically — other columns are ignored.
Then `run_combined_study.py` / `run_broad_study.py` pick up the +수급 arms.

`tagent/features_flows.py` builds **leak-free** features (1-day EOD shift + as-of
align, exactly like the short features): `foreign_net`, `inst_net`, their rolling
sums, day-over-day change, and rolling z-scores. `merge_flow_features()` is
non-mutating, like the short merge.

## More gated KR signals (대차 / 신용 / 프로그램 / investor-detail / short-balance)
`tagent/data/krx_signals.py` adds a registry of 5 more gated signals (same
authenticated pykrx path, `KRX_ID`/`KRX_PW`). **pykrx supports 2; the other 3
need a manual KRX CSV** — for those the live fetch raises `ManualCsvRequired` with
the exact menu path + columns rather than failing silently:

| signal (`key`) | what | source |
|---|---|---|
| `investor_detail` | 투자자 상세 순매수 (연기금/금융투자/보험/투신/사모) | **pykrx** `get_market_trading_value_by_date(detail=True)` |
| `short_balance` | 공매도 순보유잔고 (outstanding short) | **pykrx** `get_shorting_balance_by_date` (chunked) |
| `lending` | 대차잔고 (securities-lending) | **manual CSV** — KRX 대차거래 잔고추이 |
| `margin` | 신용거래융자 잔고 (retail leverage) | **manual CSV** — KRX 신용거래융자 잔고 |
| `program` | 프로그램 매매 순매수 | **manual CSV** — KRX 프로그램매매 추이 |

```bash
python scripts/download_krx_signals.py --tickers 005930 000660 035420
# REAL   investor_detail / short_balance  -> data/<code>_<signal>.csv
# MANUAL lending / margin / program       -> prints the KRX menu path + columns
```

Each loads into a date-indexed canonical frame (`load_signal(key, code)`), caches
to `data/<code>_<signal>.csv`, reads cp949/Korean manual exports, and feeds the
**generic leak-free** feature builder `tagent/features_signals.py`
(`make_signal_features` / non-mutating `merge_signal_features`): per value column
→ level, day-change, rolling z-score, 1-day-shifted + as-of aligned. Verified
live: `investor_detail` (587 rows → +15 features) and `short_balance` (483 rows →
+9) fetch real; lending/margin/program await a manual export.

### Does 공매도 / 수급 add out-of-sample edge?
Scripts run **honest**, leak-free comparisons where every arm trains on the *same*
held-out bars/folds (so any gap is the added features' doing, not a different
window): `run_short_study.py` (technicals vs +공매도), `run_combined_study.py`
(4 arms, per stock), `run_broad_study.py` (4 arms aggregated across a universe).

**The verdict comes from the broad run — 50 KOSPI/KOSDAQ names** (`download_krx_universe.py`
fetches OHLCV + 공매도 + 수급 via an authenticated KRX login). Short-selling-**ban**
windows (2020‑03→2021‑05, 2023‑11→2025‑03) are **excluded** (during bans 공매도≈0,
a regulatory artifact). 공매도 history starts 2022 (KRX caps that endpoint at
~2y/request and ~12s/call, so full-2016 × 50 names ≈ 3h); 수급/OHLCV use max
history. Threshold 0.55, costs included, ~225–590 OOS bars/name.

**Aggregate over 50 names (mean / median total return, mean Sharpe):**

| arm | mean ret | median ret | Sharpe | beats buy & hold | beats technicals |
|-----|---------:|-----------:|-------:|:---:|:---:|
| technicals     | +26.9% | +18.9% | 0.48 | 15/50 | — |
| + 공매도        | +24.5% | +19.8% | 0.48 | 14/50 | 19/50 |
| + 수급          | +25.0% | +20.0% | 0.51 | 15/50 | 28/50 |
| + both         | +28.3% | +14.7% | 0.57 | 13/50 | 13/50→25/50 |
| **buy & hold** | **+137.8%** | **+83.1%** | **0.91** | — | — |

**Honest bottom line (n=50, the statistically meaningful run):**
- **Nothing beats buy & hold.** Every arm averages ~+25–28% vs B&H **+138%**, with
  lower Sharpe, and beats B&H on only ~13–15 of 50 names. In the 2022→2026 KR bull,
  the mostly-flat ML strategy has **no edge over simply holding**.
- **공매도 does not help.** It beats technicals on **19/50** names (<half) and is
  slightly *worse* on mean return — indistinguishable from noise.
- **수급 is marginal at best.** Beats technicals on **28/50** (a bare 56%) and nudges
  Sharpe 0.48→0.51, but no better mean return. By a paired sign test 28/50 is **not
  significant** (two-sided p≈0.4) — within noise.
- **The combination doesn't stack.** Best mean (+28.3%) but **worst median** (+14.7%) —
  driven by a few outliers, not consistency.
- Per-name effects flip sign everywhere (e.g. 196170 short **+118%** vs 086520 short
  **−62%**); the ~50% "beats technicals" rates are exactly what noise looks like.

> Earlier 3-name runs *looked* like 공매도/수급 sometimes helped — but those
> conclusions flipped with more data and don't survive the 50-name test. **Treat
> small-N, single-regime results as noise.** Caveats here too: one bull regime,
> 공매도 only from 2022, balance feature skipped on the fetched names, fixed
> threshold. The robust takeaway: *don't expect this model — with or without
> 공매도/수급 — to beat buy-and-hold.*

### Cross-sectional long-short (`run_cross_sectional.py`)
The per-stock up/down model has no edge, so `tagent/cross_sectional.py` reframes
it **cross-sectionally**: each (day, stock) is labelled by its forward N-day
return ranked *within that day's universe* (top vs bottom tercile); **one** model
trades all stocks; features are cross-sectionally z-scored each day. CV is
**purged + embargoed** (a horizon+embargo gap between train and test, since the
label looks N days ahead). Each day: **long the top decile, short the bottom**
(dollar-neutral) — plus a long-only top-decile variant. Costs on turnover.

Real OOS run — 50 names, horizon 5, **718-day panel 2022-02→2026-06** (ban-excluded,
two regimes: 2022 bear + 2025–26 bull), 598 OOS days, decile legs:

| strategy / feature set | total ret | Sharpe | maxDD |
|---|--:|--:|--:|
| **long-short**, technicals | −22.2% | −0.19 | −53% |
| **long-short**, +공매도+수급 | −31.9% | −0.35 | −58% |
| long-only top-decile, technicals | +137.9% | 1.26 | −19% |
| long-only top-decile, +공매도+수급 | +190.5% | 1.60 | −17% |
| equal-weight index (daily reb.) | +154.3% | 1.66 | −18% |
| buy & hold the universe | +232.9% | 1.69 | −26% |

**Honest read:**
- **The market-neutral long-short fails.** Both feature sets lose money with
  *negative* Sharpe and ~−55% drawdowns — the model can't rank future winners vs
  losers; going neutral just stripped out the beta that was the only thing earning.
  Adding 공매도+수급 makes the long-short **worse**.
- **Long-only top-decile is the one bright spot, but weak.** With +공매도+수급 it
  returns +190% (Sharpe 1.60) — beating technicals-only (+138%, 1.26) *and* edging
  the equal-weight index on return — but its Sharpe only **ties** the index (1.60 vs
  1.66) and it still **loses to buy-and-hold** (+233%). It's a concentrated long
  that rode the bull, not market-neutral alpha.
- **So flows help only in the long-only cut, and hurt the long-short** — not a
  robust signal. Net: still **no convincing cross-sectional edge**, and nothing
  beats buy-and-hold. (One 50-name sample over two regimes; fixed horizon/decile.)

## Model robustness + the all-features study (`run_robust_study.py`)
A robustness pass to make the model honest, then a 50-name OOS re-run with **all**
features (technicals + 공매도 + 수급 + investor_detail + short_balance):

- **Purged + embargoed walk-forward CV** (`ml/train.purged_walk_forward_splits`):
  training samples whose label window (`horizon` bars ahead) overlaps the test
  fold are purged, plus an embargo gap — used in `walk_forward_train`/`_predict`
  (backward-compatible: no `horizon` → reproduces the old `gap` splits exactly).
- **Probability calibration** (isotonic/Platt via `CalibratedClassifierCV`) so a
  threshold is *meaningful* — a calibrated 0.55 really means ~55% probability.
- **Ensemble** (`ml/models.make_model`): regularized **LightGBM + XGBoost +
  CatBoost**, each calibrated, probabilities averaged. Model choice is
  configurable; missing libraries degrade gracefully.
- **Tighter LightGBM** (`num_leaves` 31→15, `max_depth` 4, `min_child_samples`
  60, L1/L2) — less overfit on noisy features.
- **Feature selection** (`select_features`, LightGBM gain): drop near-zero
  importance and **report which features survive**, by group.

**Does any signal earn its place?** Global selection on the pooled 50-name panel
(35,400 rows, 52→51 features kept) says **yes** — and the new short-balance signal
*dominates*: **`short_balance_ratio` is the #1 feature by gain**, with
`short_balance` (#3) and `short_balance_value` (#4) also top‑5; `inst_net_sum5`
(수급) is top‑10. All 공매도 / 수급 / investor_detail / short_balance feature groups
survive.

> **Calibration changes how you trade.** A well-calibrated model with a ~40% base
> rate rarely exceeds a *fixed* 0.55 → it sits flat. The honest way to act on
> calibrated probabilities is a **relative** threshold: go long the top-conviction
> bars per name (`--quantile`, default top 30%). That's what the study uses.

**Robust 50-name OOS result** (calibrated ensemble, purged CV, selected features,
top-30% conviction, 2022+ ban-excluded):

| arm | mean ret | median ret | Sharpe | beats B&H | beats technicals |
|-----|---------:|-----------:|-------:|:---:|:---:|
| technicals | +51.5% | +34.7% | 0.49 | 15/50 | — |
| **ALL features** (공매도+수급+investor_detail+short_balance) | +39.0% | +13.9% | 0.37 | 12/50 | **18/50** |
| **buy & hold** | **+154.9%** | **+83.6%** | — | — | — |

**Honest read — importance ≠ edge.** `short_balance_ratio` is the single most
*important* feature in-sample, yet adding all the signals **lowered** OOS return
(+39% vs +51.5%), median (+13.9% vs +34.7%) and Sharpe (0.37 vs 0.49); the
all-features arm beats technicals on only **18/50** (<half) → the extra features
add noise OOS, not edge. And **neither beats buy & hold** (both ~+40–51% vs +155%;
beat B&H on 12–15/50). The robustness pass (calibration + purged CV + ensemble +
regularization + selection) is doing its job: it stops the seductive in-sample
importance of 공매도/수급/investor/short-balance from masquerading as out-of-sample
alpha. Net: **no reliable signal edge, nothing beats buy-and-hold** — consistent
with every prior cut, now on the strongest model.

## Live Korean feed via Kiwoom (REST + WebSocket)
`tagent/feeds/kiwoom_feed.py` implements the existing `MarketDataFeed` interface
against the **Kiwoom REST API** (openapi.kiwoom.com) — no Kiwoom OCX/OpenAPI+ or
broker terminal needed, just a REST app key. It streams the real-time WebSocket
and normalizes messages into our existing types, so `Monitor` and
`OrderBookMemory` work **unchanged**:

| Kiwoom real-time | type | → normalized to |
|------------------|------|-----------------|
| 주식호가잔량 (10-level order book) | `0D` | `OrderBookSnapshot` (asks ↑, bids ↓) + `Quote` (best bid/ask) |
| 주식체결 (executions)             | `0B` | `Trade` |

### Setup (mock-first)
1. Issue a REST key at **openapi.kiwoom.com** and enable it for the **mock**
   (모의투자) server first.
2. Put credentials in `.env` (never commit them):
   ```ini
   KIWOOM_ENV=mock                # mock = simulation, live = production
   KIWOOM_APP_KEY=your_app_key
   KIWOOM_SECRET_KEY=your_secret_key
   ```
3. **Smoke-test the key before streaming** (authenticate + one REST 호가 snapshot;
   never prints the token/secret; works regardless of market hours):
   ```bash
   python scripts/kiwoom_smoke_test.py            # prints "token OK" then top bid/ask
   ```
4. Run the monitor — it **auto-selects Kiwoom** when `KIWOOM_APP_KEY` is set
   (default KR watchlist `005930, 000660, 035420`; override with `WATCHLIST`),
   else falls back to Alpaca:
   ```bash
   python scripts/run_monitor.py
   ```
   (WebSocket streaming is quiet outside KR market hours ≈ 09:00–15:30 KST.)

### How it works
- **Auth** (`kiwoom_auth.py`): `POST {base}/oauth2/token` with
  `{grant_type:"client_credentials", appkey, secretkey}`; caches the token and
  refreshes ~60 s before the response's `expires_dt`. Auth errors surface the
  server's `return_msg` (e.g. *env mismatch*, *IP/terminal not allowlisted*) and
  **never** echo the key or secret.
- **Stream** (`kiwoom_feed.py`): connects `wss://{api|mockapi}.kiwoom.com:10000/api/dostk/websocket`,
  sends `{"trnm":"LOGIN","token":…}`, registers the watchlist with
  `{"trnm":"REG",…,"type":["0D","0B"]}`, echoes `PING` keepalives, and reconnects
  with exponential backoff. Prices/quantities are parsed from Kiwoom's
  sign-prefixed Korean integer strings (`"+74100"` → `74100`).
- Base URLs: mock `https://mockapi.kiwoom.com`, live `https://api.kiwoom.com`.

> **Docs vs. assumed.** Token endpoint/fields, the WS URL + LOGIN/REG/PING/REAL
> message shapes, the `0D`/`0B` type codes, and the REST 호가 api-id (`ka10004`,
> `POST /api/dostk/mrkcond`) are taken from the official docs. The per-FID field
> numbers for the 10 호가 levels (ask px `41–50`, ask qty `61–70`, bid px `51–60`,
> bid qty `71–80`; trade px `10`, qty `15`) follow Kiwoom's standard real-time FID
> spec — the docs confirm `41` = best ask and `51` = best bid, and the contiguous
> ranges follow from those anchors.

## Free crypto order-book feed (Binance, no key)
`tagent/feeds/crypto_feed.py` implements the same `MarketDataFeed` interface
against Binance's **public** combined WebSocket (`wss://stream.binance.com:9443/stream`)
— no API key, crypto trades 24/7. It normalizes:

| Binance stream | → our type |
|----------------|-----------|
| `<sym>@depth10@100ms` (`bids`↓ / `asks`↑) | `OrderBookSnapshot` (10 levels) + `Quote` (best bid/ask) |
| `<sym>@trade` | `Trade` |

It keeps **one `OrderBookMemory` per coin**, so the vanished-level tagging +
absorption / spoof / depth-imbalance features run live per symbol
(`feed.book_features("BTC")`).

**Selectable symbols (for a dashboard search/switch).** Defaults BTCUSDT,
ETHUSDT; add/remove coins at **runtime** — the feed sends a SUBSCRIBE/UNSUBSCRIBE
control frame over the live socket:

```python
from tagent.feeds.crypto_feed import CryptoFeed, normalize_symbol
normalize_symbol("btc")        # -> "BTCUSDT"  (bare base gets default quote)
normalize_symbol("ethbtc")     # -> "ETHBTC"   (BTC recognized as a quote)
normalize_symbol("!!")         # -> ValueError (rejected cleanly)

feed = CryptoFeed(symbols=["btc", "eth"])
feed.subscribe_symbol("sol")   # add SOLUSDT live; spins up its OrderBookMemory
feed.unsubscribe_symbol("eth") # drop it live
```

Collect a microstructure dataset (best bid/ask + per-coin memory features) to CSV:

```bash
python scripts/record_orderbook.py                       # BTCUSDT, ETHUSDT
python scripts/record_orderbook.py --symbols btc eth sol doge
```

Reconnects with exponential backoff and shuts down gracefully (`feed.stop()`).
Message parsing is pure/synchronous, so it's unit-tested with canned Binance
payloads (no network) in `tests/test_crypto_feed.py`.

### Microstructure edge experiment — does the order book predict price?
One honest question, one experiment: do the order-book features predict the
**forward** mid-price move (5s / 30s / 60s)?

```bash
python scripts/record_orderbook.py                 # 24/7 -> data/microstructure_<sym>.csv (~1/sec)
python scripts/run_microstructure_study.py         # IC + strict OOS test + verdict
```

`record_orderbook.py` logs, ~once a second, `ts, mid, best_bid/ask + sizes,
depth_imbalance, absorption_ratio, spoof_ratio, filled/cancelled qty`.
`tagent/microstructure.py` then computes, per feature and horizon:
- **Information Coefficient** — Spearman rank-corr of feature(t) vs forward
  return(t→t+H);
- a **strict OOS test** — split the recording *chronologically* (train earlier,
  test later; no shuffle), learn only the relationship's sign on train, and on the
  held-out part report OOS IC + directional **hit rate**, with the typical move
  size compared to round-trip **costs** (a signal that can't clear the spread +
  fees is not an edge).

The maths is unit-tested on synthetic data (`tests/test_microstructure.py`):
forward-return calc, IC (=±1 for monotone, ~0 for noise), and the no-lookahead
split. `run_microstructure_study.py` prints a clear **edge / no-edge verdict** per
feature/horizon.

### Funding-rate cash-and-carry — a real (small) structural premium
A different kind of edge: not a forecast, a **structural premium**. Long spot +
short the perpetual = delta-neutral, and every 8h the short collects the **funding
rate** (perp longs pay shorts when the perp trades rich to spot).

```bash
python scripts/run_funding_study.py        # public Binance data, no key
```

`tagent/data/funding.py` pulls public Binance funding history (+ perp mark price)
and 8h spot (aligned at the funding instant — kline open, not close), cached to
`data/funding_<sym>.csv`. `tagent/funding_study.py` simulates the carry net of
realistic costs (fees on both legs, spread, basis), **holding through small wiggles
with hysteresis** and only exiting on *sustained* negative funding; decisions use
past funding only (no lookahead). Reports annualized net yield, vol, max drawdown,
worst funding-flips, per coin + equal-weight basket.

**Verdict (real, ~333 days, BTC/ETH/SOL/BNB/XRP):** the carry is a **real positive
net-of-cost premium** — basket **+1.78%/yr, vol 0.5%, Sharpe +3.68, maxDD −0.6%**
(BTC +3.5%, Sharpe 6.7). Honest caveats: (1) **yield is small and regime-dependent**
— funding is compressed now; in bull runs it can be many× this; (2) the **high
Sharpe is misleading** — vol is tiny but the left tail is fat (SOL funding spiked
to ~−30 bps/8h; a sharp rally can **liquidate** the short-perp leg); (3) it **only
works if held** — timing funding by toggling in/out churns and the round-trip costs
wipe out the premium (a naive churning rule returned ~−12%/yr; the same data held
with hysteresis returned +1.8%). The order book gives a *statistical* signal you
can't trade (above); funding gives a *tradable* premium that's just **small**.

## Extending it (next phases)
- **OpenBB**: add a loader in `tagent/data/` for news/fundamentals to enrich features.
- **TradingAgents**: add an LLM decision layer that consumes the same state/features.
- **NautilusTrader**: implement a `MarketDataFeed` subclass + use its engine for
  research-to-live parity. Only the feed swaps; state/strategy/risk stay the same.
- **Kiwoom**: ✅ done — `feeds/kiwoom_feed.py` streams KR 호가/체결 via the REST
  WebSocket against the same interface (see *Live Korean feed via Kiwoom* above).

## Disclaimer
Educational software, not financial advice. Trading risks real loss. Paper-trade first.
