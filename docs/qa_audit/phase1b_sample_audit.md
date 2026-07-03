# Phase 1B — Gate 2 human sample-audit (NO RETURNS)

_Classification + entry/exit RULE only. Returns are computed ONLY after this table is signed off._

## Disposition of all 1,884 ex-ante calls under the corrected classifier

- **Scored** (directional test): **944** — SCHEDULE=20 · ROTATION=8 · DIP-BUY=149 · BREAKOUT=5 · LONG-GENERIC=760 · BEARISH=2
- **Excluded but counted**: TRIM=72 · HOLD-WAIT-CASH=300 · NOISE=32 · EXPOST=26 · CANCELLED=2 · AMBIGUOUS=141 · UNCLASSIFIED=367

### What changed vs the original run (call-set delta)

- **+760 LONG-GENERIC** — plain longs the old scorer DROPPED as UNCLASSIFIED (§1.3); now entered next-open + symmetric triple-barrier.
- **SCHEDULE is date-gated** — the 374 boilerplate '스케줄 매매' misfires are routed to LONG-GENERIC/excluded; only 20 calls with a real dated event remain SCHEDULE.
- **AVOID split (§1.1)** → BEARISH=2 (scored, sign −1) / HOLD-WAIT-CASH=300 (excluded) / TRIM=72 (excluded).
- **Tier-1 removed**: NOISE=32 · EXPOST=26 · CANCELLED(negation)=2 (holdings-substring avoids no longer fire).
- **Tier-3 ambiguous excluded for review**: AMBIGUOUS=141 (direction/ticker).
- **Sign gate fixed (§1.2)**: PASS ⇔ mean net>0 AND t≥+deflated_bar for EVERY segment; t≤−bar ⇒ WRONG-SIGNED. Default barriers SYMMETRIC ±8% (never −5/+8).

## Sample (15–20 calls) — verify class + rules; NO returns yet

| # | source | video [mm:ss] | ticker | corrected class | scored | entry rule | exit rule | quote (verbatim) |
|--:|---|---|---|---|:--:|---|---|---|
| 1 | LONG-GENERIC (newly included, §1.3) | `-GKk7z7oArQ` 03:22 | 포스코(005490) | **LONG-GENERIC** | ✅ | enter NEXT session open (≤10d window) | triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap | 지금 외국인들이 조금 매도가를 때렸습니다. 그동안에 27일 날짜부터 11일 날짜까지 매수 타점을 연일 잡고 들어왔었고 9일 날짜 하루 빼놓고 전부 다 타점 잡고 들어왔었죠. |
| 2 | LONG-GENERIC (newly included, §1.3) | `-GKk7z7oArQ` 25:42 | 전진건설로봇(079900) | **LONG-GENERIC** | ✅ | enter NEXT session open (≤10d window) | triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap | 실질적으로 로봇 관련되어 있는 주 등에서 아직까지 그 밸류로 못 따라간 종목들도 있습니다 전진건설 로봇 같은 경우에도 제가 이렇게 들어가가지고 그걸 먹고 나온 겁니다 그래서 여러분들께서도 이렇게 스케줄 매매를 하셔야 됩니다 |
| 3 | LONG-GENERIC (newly included, §1.3) | `02YBJgsqqi8` 00:26 | 포스코(005490) | **LONG-GENERIC** | ✅ | enter NEXT session open (≤10d window) | triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap | 그렇기 때문에 지금 이게 계속적으로 올려칠 수 있을지 수급 상태를 먼저 좀 보시는 게 좋을 것 같은데 지금 12월 달 때도 주가를 보시면 기관들 쪽에서 연일 매수 타점을 잡고 들어왔고요. 여기에 맞춰서 금투 쪽에서 따라 들어왔습니다. |
| 4 | LONG-GENERIC (newly included, §1.3) | `02YBJgsqqi8` 08:31 | 포스코(005490) | **LONG-GENERIC** | ✅ | enter NEXT session open (≤10d window) | triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap | 그런다고 해도 이거 잡고 들어가야 된다고 보고 있어요 왜 그러냐면 원래 포스트홀딩스테이 주가는 여러분들 리튬 부분으로 움직여요 |
| 5 | LONG-GENERIC (newly included, §1.3) | `03bygzBKE10` 07:51 | 대한전선(001440) | **LONG-GENERIC** | ✅ | enter NEXT session open (≤10d window) | triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap | 미래세증권이에요 여기 보시면 미래세증권이 여기 48억 원짜리를 매수한 사체 건명금액으로 해가지고 이걸 갖다가 370억에 사갔다는 거죠 |
| 6 | LONG-GENERIC (newly included, §1.3) | `061Ytt81JH8` 25:48 | 전진건설로봇(079900) | **LONG-GENERIC** | ✅ | enter NEXT session open (≤10d window) | triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap | 실질적으로 로봇 관련되어 있는 주 등에서 아직까지 그 밸류로 못 따라간 종목들도 있습니다 전진건설 로봇 같은 경우에도 제가 이렇게 들어가가지고 그걸 먹고 나온 겁니다 그래서 여러분들께서도 이렇게 스케줄 매매를 하셔야 됩니다 |
| 7 | powered play | `061Ytt81JH8` 01:34 | 에코프로(086520) | **DIP-BUY** | ✅ | enter AT stated support if a low touches it ±0.5% within 10d (else NON-TRIGGERED) | triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap | 그러니까 이거는 억지로 주가를 못 오르게 좀 누르고 있는 거예요 그래가지고 본인들이 저가 매수에서 최대한적으로 물량을 다 모은 다음에 한 방에 치고 올리려고 하는 모멘텀이 나왔을 때 이런 수급들이 움직이거든요 |
| 8 | powered play | `0IOc5U8_4jM` 00:37 | 네이버(035420) | **DIP-BUY** | ✅ | enter AT stated support if a low touches it ±0.5% within 10d (else NON-TRIGGERED) | triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap | 공매도 평균가도 21만 4천 177원 그리고 21만 3천 483원까지 올라와 있는 상태입니다 지금 21만 5천원 밑으로 또 빠지면은 기관들 쪽에서 싹쓸이 매수를 해가고 있는데 증권사별로 조금 나눠가지고 보시면은 |
| 9 | powered play | `0VkNtt3epUw` 03:43 | 삼천당제약(000250) | **DIP-BUY** | ✅ | enter AT stated support if a low touches it ±0.5% within 10d (else NON-TRIGGERED) | triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap | 지금 4만원 2천원대 기준점으로 해서 주가가 밑으로 빠지면 공매도 비중 줄이고 또 올라치면은 공매도 비중 늘리면서 지금 여기를 갖다가 매집선으로 가져간다는 겁니다 이럴 때는요 여러분들 주가가 이렇게 빠지지 않아요 그러니까 지금 공매도 수급들도 여기 4만 2천원대 밑에 물량을 모으기 위해서 비중 조절을 해가면서 공매도를 갖 |
| 10 | powered play | `1BCC9yNTJrU` 06:02 | 대우건설(047040) | **DIP-BUY** | ✅ | enter AT stated support if a low touches it ±0.5% within 10d (else NON-TRIGGERED) | triple-barrier: spoken target/stop else SYMMETRIC ±8%, else 20d cap | 상단 찍었다가 단계차익매도 상단 찍었다가 단계차익매도 하면서 매집선 형성시켰다가 올려치는 이런 모멘텀이 나오지 않을까라는 생각을 좀 하고 있는 상태입니다 |
| 11 | BEARISH (AVOID split, §1.1) | `6x1WdVqGYpQ` 14:31 | 네이버(035420) | **BEARISH** | ✅ | SHORT at NEXT session open (≤10d window), sign −1 | SYMMETRIC ±8% triple-barrier (sign −1), else 20d cap | 17% 이것도 문제에요 지금 대주주 부분에 대한 요건을 15% 에서 20% 제한을 하겠다라고 하는건데 15% 로 정해버리면 그러면 여러분들 네이버 같은 경우 2% 지분율을 팔아야 되거든요 그러니까 법률적인 부분으로 인해 가지고 국가기관이 일반 사기업의 대주주의 지분율을 갖다가 조정을 해버리는 거에요 그리고 지금 그러면 송 |
| 12 | BEARISH (AVOID split, §1.1) | `M9krAJRtZCE` 07:46 | 네이버(035420) | **BEARISH** | ✅ | SHORT at NEXT session open (≤10d window), sign −1 | SYMMETRIC ±8% triple-barrier (sign −1), else 20d cap | 67.45%예요 그럼 얘네는 얼마 팔아야 됩니까 19% 20% 제한한다고 하면 |
| 13 | Tier-3 ambiguous (review) | `-GKk7z7oArQ` 02:01 | 포스코(005490) | **AMBIGUOUS** | — | —  (excluded: ticker_resolution — surfaced for review, excluded) | — | 그래가지고 해당 보도 이후에 우리나라 증시만 좀 발작 처리가 일어나면서 다 내리치는 모멘텀이 좀 나왔었는데 반도체 주가 급등하면서 단기 조정 가능성에 대해서 우려가 커진 영향에 따라가지고 리스크 회피로 좀 판단이 나옵니다 그래가지고 이 부분 때문에 삼성전자하고 SK하이닉스가 눌리니까 다른 섹터들의 기업들까지 전체적으로 다 |
| 14 | Tier-3 ambiguous (review) | `0HCFPQtVry8` 21:11 | HLB(028300) | **AMBIGUOUS** | — | —  (excluded: direction_mismatch — surfaced for review, excluded) | — | 이 부분 때문에 타점 잡고 들어가시 한 다음에 저희가 매도과를 때리고 나온 겁니다 근데 여러분들 최근에 이게 불성실지연공시 5점 |
| 15 | Tier-3 ambiguous (review) | `1BCC9yNTJrU` 16:40 | 대우건설(047040) | **AMBIGUOUS** | — | —  (excluded: direction_mismatch — surfaced for review, excluded) | — | 그래서 제가 매수과랑 매도과랑 실시간 대응 부분에 대한 거를 |
| 16 | Tier-3 ambiguous (review) | `1FLZhvgwUWU` 19:43 | 삼성전자(005930) | **AMBIGUOUS** | — | —  (excluded: direction_mismatch — surfaced for review, excluded) | — | 이 지급 위에서 현재 보유한 25조원 규모의 이 자사주에 대한 거가 들어가고요 |
| 17 | excluded (Tier-1 filter) | `02YBJgsqqi8` 02:04 | 포스코(005490) | **NOISE** | — | —  (excluded: market-structure commentary (사이드카/반대매매/서킷브레이커/동시호가)) | — | 두 번의 서킷 브레이크하고 매도 사이드카 부분 때문에 폭락 처리가 나오면서 33만원대에서 35만원 사이로 매집선 가져갔다가 |
| 18 | excluded (Tier-1 filter) | `03bygzBKE10` 04:30 | 대한전선(001440) | **HOLD-WAIT-CASH** | — | —  (excluded: wait/cash, no directional bet — excluded, counted) | — | 눈치게임을 하면서 가져간다고 말씀드릴 수가 있겠고 지금 그런다고 하면 외국인 포지션은 계속적으로 관망세로 가져가신 게 맞고요 특히나가 지금 6월 5일 날짜 때 이때 블록들이 진행되면서 가져왔던 물량 아닙니까 |
| 19 | excluded (Tier-1 filter) | `061Ytt81JH8` 06:10 | 에코프로(086520) | **HOLD-WAIT-CASH** | — | —  (excluded: wait/cash, no directional bet — excluded, counted) | — | ESS를 생산하는 T&R 홀딩그룹의 장태런 회장은 |
| 20 | excluded (Tier-1 filter) | `061Ytt81JH8` 11:59 | 에코프로(086520) | **TRIM** | — | —  (excluded: profit-take/reduce of a long (익절/처분/비중축소) — excluded, ) | — | 지금 상장을 시키겠다는 건 손절치고 나갔겠다는 거냐 이게 말이 안된다 |

**STOP — awaiting human sign-off.** No returns/verdict computed until this sample is approved.
