// Render-level browser verification of the dashboard chart using jsdom: loads
// the real page, stubs fetch with synthetic data, drives REAL mouse/wheel/click
// events, and asserts the rendered DOM/SVG — crosshair price correctness,
// zoom/pan, pivot dots, breakout marker, alerts feed, valid coordinates, and a
// clean console.
// Setup:  npm install jsdom     Run:  node scripts/verify_chart_browser.mjs
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { JSDOM, VirtualConsole } from 'jsdom';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(root, 'tagent/static/dashboard.html'), 'utf8');

// ---- synthetic data with a confirmed breakout, pivots, lines, levels ----
const closes = [], vols = [];
for (let i = 0; i < 60; i++) { closes.push(100 + (i % 2 ? 0.6 : -0.6) + 0.05 * i); vols.push(1000 + (i % 4) * 40); }
closes[59] = 107.5; vols[59] = 6500;
const candles = closes.map((c, i) => ({ t: `2026-06-04T07:${String(i).padStart(2, '0')}:00+00:00`,
  o: i ? closes[i - 1] : c, h: c + 0.6, l: c - 0.6, c, v: vols[i] }));
const taObj = {
  symbol: 'BTCUSDT', market: 'crypto',
  lines: [{ type: 'upper', points: [[10, 104.2, candles[10].t], [59, 107.0, candles[59].t]],
            pivots: [[10, 104.2, candles[10].t], [30, 105.6, candles[30].t], [50, 106.7, candles[50].t]] },
          { type: 'lower', points: [[11, 100.3, candles[11].t], [59, 101.4, candles[59].t]],
            pivots: [[11, 100.3, candles[11].t], [31, 100.8, candles[31].t]] }],
  levels: [{ type: 'resistance', price: 106.9, touches: 3 }, { type: 'support', price: 100.3, touches: 2 }],
  pivots: [{ type: 'high', index: 30, price: 105.6, t: candles[30].t },
           { type: 'high', index: 50, price: 106.7, t: candles[50].t },
           { type: 'low', index: 31, price: 100.8, t: candles[31].t }],
  breakout: { direction: 'up', index: 59, price: 107.5, vol_x: 4.1, t: candles[59].t },
  candle_patterns: [
    { name: 'engulfing', direction: 'bullish', index: 55, price: 106.5, label: 'Bull Engulf', t: candles[55].t },
    { name: 'pin_bar', direction: 'bearish', index: 58, price: 106.9, label: 'Shooting Star', t: candles[58].t }],
  pattern: 'ascending channel', n_bars: 60, updated_at: '2026-06-04T07:59:00+00:00',
  reason: 'up breakout', explanation: 'drew trendlines; broke upper on a volume spike',
};
const alertsAll = [
  { ts: '2026-06-04T07:59:01+00:00', symbol: 'BTCUSDT', market: 'crypto', kind: 'breakout',
    text: 'BTCUSDT broke above the upper trendline at 07:59:00, price 107.50, 4.1x volume -> bullish',
    meta: { direction: 'up' } },
  { ts: '2026-06-04T07:58:00+00:00', symbol: 'AAPL', market: 'us', kind: 'breakout',
    text: 'AAPL broke below the lower trendline', meta: { direction: 'down' } }];
const alertsFor = (sym) => (!sym || sym === 'all') ? alertsAll : alertsAll.filter(a => a.symbol === sym);
const SYMS = ['AAPL', 'BTCUSDT'];
const scAll = { enabled: true, symbol: 'all', symbols: SYMS, horizon_min: 15, trade_pct: 1.0, sources: {
  'us-ml': { source: 'us-ml', label: 'US ML model', total: 2, correct: 1, wrong: 0, pending: 1,
    hit_rate: 100.0, equity_start: 10000, equity: 10995.0, pct_change: 9.95, realized_pnl: 0,
    unrealized_pnl: 1000, costs: 5, trades: 1, open_positions: 1,
    recent: [{ ts: 't', symbol: 'AAPL', direction: 'BUY', side: 1, status: 'pending', entry: 100, resolved: null }] },
  'crypto-ob': { source: 'crypto-ob', label: 'Crypto order-book', total: 1, correct: 0, wrong: 1, pending: 0,
    hit_rate: 0.0, equity_start: 10000, equity: 9100.0, pct_change: -9.0, realized_pnl: -890,
    unrealized_pnl: 0, costs: 10, trades: 1, open_positions: 0,
    recent: [{ ts: 't', symbol: 'BTCUSDT', direction: 'bearish', side: -1, status: 'wrong', entry: 100, resolved: 109 }] },
  'us-ta': { source: 'us-ta', label: 'US TA chart', total: 3, correct: 2, wrong: 1, pending: 0,
    hit_rate: 66.7, equity_start: 10000, equity: 10010.0, pct_change: 0.1, realized_pnl: 12,
    unrealized_pnl: 0, costs: 2, trades: 3, open_positions: 0, recent: [],
    by_pattern: { engulfing: { total: 2, correct: 2, wrong: 0, pending: 0, hit_rate: 100.0 },
                  pin_bar: { total: 1, correct: 0, wrong: 1, pending: 0, hit_rate: 0.0 } } },
  'crypto-ml': { source: 'crypto-ml', label: 'Crypto ML model', total: 0, correct: 0, wrong: 0, pending: 0,
    hit_rate: null, equity_start: 10000, equity: 10000.0, pct_change: 0.0, realized_pnl: 0,
    unrealized_pnl: 0, costs: 0, trades: 0, open_positions: 0, recent: [] },
  'crypto-ta': { source: 'crypto-ta', label: 'Crypto TA chart', total: 0, correct: 0, wrong: 0, pending: 0,
    hit_rate: null, equity_start: 10000, equity: 10000.0, pct_change: 0.0, realized_pnl: 0,
    unrealized_pnl: 0, costs: 0, trades: 0, open_positions: 0, recent: [] },
  'funding-carry': { source: 'funding-carry', label: 'Funding carry (BTC/ETH)', total: 6,
    correct: 0, wrong: 0, pending: 0, hit_rate: null, equity_start: 20000, equity: 20007.4,
    pct_change: 0.04, realized_pnl: 17.4, unrealized_pnl: 0, costs: 10, trades: 2, open_positions: 0,
    recent: [{ ts: 't', symbol: 'BTCUSDT', direction: 'carry', side: 1, status: 'carry', entry: 2.1, resolved: null }] } } };
// BTC-only view: just the crypto-ob card has activity; others are a fresh $10k account
const scBTC = { enabled: true, symbol: 'BTCUSDT', symbols: SYMS, horizon_min: 15, trade_pct: 1.0, sources: {
  'us-ml': { source: 'us-ml', label: 'US ML model', total: 0, correct: 0, wrong: 0, pending: 0,
    hit_rate: null, equity_start: 10000, equity: 10000.0, pct_change: 0.0, realized_pnl: 0,
    unrealized_pnl: 0, costs: 0, trades: 0, open_positions: 0, recent: [] },
  'us-ta': scAll.sources['us-ta'], 'crypto-ml': scAll.sources['crypto-ml'],
  'crypto-ob': scAll.sources['crypto-ob'], 'crypto-ta': scAll.sources['crypto-ta'],
  'funding-carry': scAll.sources['funding-carry'] } };
const scorecardFor = (sym) => (sym === 'BTCUSDT') ? scBTC : scAll;
const fundingLive = { enabled: true, cycle: 5, capital: 10000, equity: 10031.2, accrued: 41.2,
  costs: 10.0, pct_change: 0.31, ann_yield_pct: 6.78, n_held: 2,
  held: [{ coin: 'BTCUSDT', weight: 0.6, notional: 6000, funding_bps: 2.4 },
         { coin: 'ETHUSDT', weight: 0.4, notional: 4000, funding_bps: 1.5 }],
  margin: { min_margin_ratio: 0.33, headroom_pct: 32.5, leverage: 3.0, maint_margin_rate: 0.005 } };
// order book + memory with a real-size level (2.3753 BTC @ 63,748 -> ~$151,400) and a tiny one (~$6)
const book = { symbol: 'BTCUSDT', status: 'live', best_ask: 63748, best_bid: 63740,
  asks: [[63748, 2.3753]], bids: [[63740, 0.0001]], ts: '' };
const memory = { symbol: 'BTCUSDT', absorption_ratio: 0.9, spoof_ratio: 0.1, depth_imbalance: 0.2,
  vanished: [{ side: 'ask', price: 63748, qty: 2.3753, reason: 'filled' }] };

const symOf = (u) => { const m = u.match(/[?&]symbol=([^&]+)/); return m ? decodeURIComponent(m[1]) : 'all'; };
const fetchURLs = [];
const fetchStub = async (url) => {
  const u = String(url); fetchURLs.push(u);
  const j = u.includes('/candles') ? { symbol: 'BTCUSDT', market: 'crypto', interval: '1m', candles }
    : u.includes('/ta') ? taObj
    : u.includes('/scorecard') ? scorecardFor(symOf(u))
    : u.includes('/funding_live') ? fundingLive
    : u.includes('/alerts') ? { alerts: alertsFor(symOf(u)) }
    : u.includes('/analysis') ? { symbol: 'BTCUSDT', market: 'crypto', action: 'HOLD', confidence: 0.1,
        probability: null, rsi: 50, momentum: 0, imbalance: 0, absorption: 0, spoof: 0, candle: { type: 'doji' } }
    : u.includes('/signals') ? { signals: [] }
    : u.includes('/account') ? { equity: 0, cash: 0, day_pnl: 0, positions: [] }
    : u.includes('/orderbook') ? book
    : u.includes('/memory') ? memory
    : {};
  return { json: async () => j };
};

const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', (e) => errors.push('jsdomError: ' + (e.message || e)));
const dom = new JSDOM(html, { runScripts: 'dangerously', pretendToBeVisual: true,
  virtualConsole: vc, beforeParse(window) { window.fetch = fetchStub; } });
const { window } = dom;
window.addEventListener('error', (e) => errors.push('window.onerror: ' + (e.message || e.error)));

const tick = (ms = 0) => new Promise((r) => window.setTimeout(r, ms));
await new Promise((r) => (window.document.readyState === 'complete' ? r() : window.addEventListener('load', r)));
await tick(30);   // let init usTick()/refresh promises settle

// ---- assertions ----
let fails = 0;
const ok = (name, cond, extra = '') => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${name}${extra ? ' — ' + extra : ''}`); if (!cond) fails++; };
const $ = (id) => window.document.getElementById(id);
const svg = $('cxChart'), chart = window.__charts.cx;

chart.setData(candles, taObj);            // deterministic data into the real chart

// 1. candles + overlay render
let h = svg.innerHTML;
const nRect = (h.match(/<rect/g) || []).length, nDot = (h.match(/class="pivdot"/g) || []).length;
ok('candles rendered', nRect >= candles.length, `${nRect} rects`);
ok('pivot dots marked', nDot === taObj.pivots.length, `${nDot}/${taObj.pivots.length} dots`);
ok('breakout polygon present', /<polygon/.test(h));
ok('candlestick patterns drawn + labelled',
  (h.match(/class="cpat"/g) || []).length === taObj.candle_patterns.length &&
  /Bull Engulf/.test(h) && /Shooting Star/.test(h));

// 2. coordinate validity (browser "Expected length" check)
const GEO = /\b(x1|y1|x2|y2|cx|cy|width|height|x|y|r)=("[^"]*"|[^\s>]+)/g, NUM = /^-?\d+(\.\d+)?$/;
const scan = (markup) => { let m, bad = []; GEO.lastIndex = 0;
  while ((m = GEO.exec(markup)) !== null) { const v = m[2].replace(/^"|"$/g, ''); if (!NUM.test(v)) bad.push(`${m[1]}=${m[2]}`); }
  for (const pm of markup.matchAll(/points="([^"]*)"/g)) for (const t of pm[1].split(/[\s,]+/).filter(Boolean)) if (!NUM.test(t)) bad.push(`pt ${t}`);
  return bad; };
const badCoords = () => scan(svg.innerHTML);
// negative control: the validator MUST flag the original `y2=6/` bug (not a no-op)
ok('coord validator catches known bug', scan('<line x1=0 y1=6 x2=410 y2=6/>').length > 0);
ok('all SVG coordinates finite', badCoords().length === 0, badCoords().slice(0, 4).join(','));

// 3. crosshair + tooltip with CORRECT cursor price
const mm = (x, y) => svg.dispatchEvent(new window.MouseEvent('mousemove', { clientX: x, clientY: y, bubbles: true }));
mm(200, 60);
h = svg.innerHTML;
ok('crosshair drawn', (h.match(/class="crosshair"/g) || []).length === 2);
const tip = $('cxTip');
ok('tooltip visible', tip.style.display === 'block');
ok('tooltip shows OHLC', /O\s/.test(tip.textContent) && /H\s/.test(tip.textContent) && /Vol/.test(tip.textContent));
const shown = (tip.textContent.match(/Cursor price\s*([\d.,]+)/) || [])[1];
const expected = chart.priceAtY(60).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
ok('cursor price correct', shown === expected, `tooltip=${shown} expected=${expected}`);

// 4. line/level hover detail
const lastLine = taObj.lines[0], p = lastLine.points[1];
chart.simMove(chart.xAt(p[0]), chart.yAt(p[1]));
ok('trendline hover shows detail', /trendline · pivots/.test($('cxTip').textContent), $('cxTip').textContent.slice(0, 60));
chart.simMove(10, chart.yAt(taObj.levels[0].price));
ok('level hover shows detail', /Resistance .* touches/.test($('cxTip').textContent));
chart.simLeave();

// 5. zoom (wheel) shrinks the visible window
const before = chart.view().count;
svg.dispatchEvent(new window.WheelEvent('wheel', { deltaY: -120, clientX: 200, clientY: 60, bubbles: true, cancelable: true }));
const after = chart.view().count;
ok('wheel zoom shrinks window', after < before, `${before} -> ${after}`);
ok('coords valid after zoom', badCoords().length === 0);

// 6. pan (drag) shifts the window start
const startBefore = chart.view().start;
svg.dispatchEvent(new window.MouseEvent('mousedown', { clientX: 320, clientY: 60, bubbles: true }));
svg.dispatchEvent(new window.MouseEvent('mousemove', { clientX: 180, clientY: 60, bubbles: true }));
window.dispatchEvent(new window.MouseEvent('mouseup', { bubbles: true }));
ok('drag pans the window', chart.view().start !== startBefore, `start ${startBefore} -> ${chart.view().start}`);

// 7. reset button restores full view
$('cxReset').dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
ok('reset restores full range', chart.view().count === candles.length && chart.view().start === 0);

// 8. expand button -> fullscreen class
$('cxExpand').dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
ok('expand toggles fullscreen', $('cxWrap').classList.contains('expanded'));

// 9. alerts feed populated (real render path)
await window.refreshAlerts();
const feed = $('alerts');
ok('alerts feed has entry', /class="arow/.test(feed.innerHTML) && /broke above/.test(feed.textContent));
ok('alert shows time + symbol', /07:59:01/.test(feed.textContent) && /BTCUSDT/.test(feed.textContent));

// 10. TA stamp shows updated time + reason
window.taStamp($('cxTAstamp'), taObj);
ok('TA updated stamp', /TA updated 07:59:00 — up breakout/.test($('cxTAstamp').textContent));
ok('chart live stamp set', /updated \d\d:\d\d:\d\d/.test($('cxStamp').textContent));

// 11d. live funding-carry panel: held coins, per-coin funding, ann yield, margin
await window.refreshFundingLive();
const fl = $('fundlive');
ok('funding-live panel shows equity + annualized yield',
  /\$10,031/.test(fl.textContent) && /\+6\.78%/.test(fl.textContent));
ok('funding-live shows held coins + per-coin funding',
  /BTCUSDT/.test(fl.textContent) && /\+2\.40 bps\/8h/.test(fl.textContent));
ok('funding-live shows margin headroom', /headroom @ 3x/.test(fl.textContent));

// 11. signal scorecard panel: $10k -> $X (+/- %), hit-rate, recent outcomes
await window.refreshScorecard();
const cards = $('scards');
ok('scorecard renders per-source cards', (cards.innerHTML.match(/class="scard"/g) || []).length === 6);
ok('scorecard shows equity + % change', /\$10,995/.test(cards.textContent) && /\+9\.95%/.test(cards.textContent));
ok('scorecard shows hit-rate', /hit-rate\s*100%/.test(cards.textContent) && /hit-rate\s*0%/.test(cards.textContent));
ok('scorecard shows costs paid', /costs paid\s*-\$5/.test(cards.textContent));
ok('scorecard recent outcome badge', /class="badge pending"/.test(cards.innerHTML) && /class="badge wrong"/.test(cards.innerHTML));
ok('scorecard shows all 6 sources incl funding-carry', (cards.innerHTML.match(/class="scard"/g) || []).length === 6);
ok('scorecard shows "0 signals yet"', /crypto-ta: 0 signals yet/.test(cards.textContent));
ok('scorecard shows per-pattern breakdown', /by pattern/.test(cards.textContent) &&
  /engulfing/.test(cards.textContent) && /100%/.test(cards.textContent));
ok('funding-carry card shows CARRY + funding bps',
  /Funding carry/.test(cards.textContent) && /CARRY/.test(cards.textContent) && /bps\/8h funding/.test(cards.textContent));

// 11a. per-symbol filter: choosing BTCUSDT filters BOTH alerts + scorecard; "all" restores
await window.refreshAlerts();                       // default = all
ok('alerts feed (all) shows both symbols', /BTCUSDT/.test($('alerts').textContent) && /AAPL/.test($('alerts').textContent));
const symSel = $('scSymB');
ok('filter dropdown populated with symbols', [...symSel.options].some(o => o.value === 'BTCUSDT') && [...symSel.options].some(o => o.value === 'all'));
symSel.value = 'BTCUSDT';
symSel.dispatchEvent(new window.Event('change', { bubbles: true }));
await tick(30);
ok('symbol filter requests BTCUSDT for both',
  fetchURLs.some(u => u.includes('/alerts') && u.includes('symbol=BTCUSDT')) &&
  fetchURLs.some(u => u.includes('/scorecard') && u.includes('symbol=BTCUSDT')));
ok('alerts filtered to BTCUSDT only', /BTCUSDT/.test($('alerts').textContent) && !/AAPL/.test($('alerts').textContent));
ok('both filter selects synced to BTCUSDT', $('scSymA').value === 'BTCUSDT' && $('scSymB').value === 'BTCUSDT');
$('scSymA').value = 'all';
$('scSymA').dispatchEvent(new window.Event('change', { bubbles: true }));
await tick(30);
ok('"all" toggle restores both symbols', /AAPL/.test($('alerts').textContent) && /BTCUSDT/.test($('alerts').textContent));

// 11b. timeframe selector refetches candles + TA on the chosen interval
const tf = $('cxTf'); tf.value = '30m';
tf.dispatchEvent(new window.Event('change', { bubbles: true }));
await tick(30);
ok('timeframe change refetches at 30m',
  fetchURLs.some(u => u.includes('/candles') && u.includes('interval=30m')) &&
  fetchURLs.some(u => u.includes('/ta') && u.includes('interval=30m')));

// 11c. order-book $value column = price x size, with thousands separators
await window.cxBook();
await tick(10);
const ladder = $('ladder');
ok('order-book shows ≈ $value', /≈ \$151,400/.test(ladder.textContent), ladder.textContent.replace(/\s+/g, ' ').slice(0, 80));
ok('tiny size shows small $value', /≈ \$6\b/.test(ladder.textContent));
ok('vanished level shows ≈ $value', /≈ \$151,400/.test($('vlist').textContent));

// 12. console is error-free
ok('console error-free', errors.length === 0, errors.slice(0, 3).join(' | '));

console.log(`\n${fails ? fails + ' CHECK(S) FAILED' : 'ALL CHECKS PASSED'} (${errors.length} console errors)`);
dom.window.close();
process.exit(fails ? 1 : 0);
