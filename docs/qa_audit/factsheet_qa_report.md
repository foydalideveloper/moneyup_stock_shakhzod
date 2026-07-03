# Fact-sheet QA Audit

_generated 2026-06-30 10:49 · read-only · deterministic (no LLM in any count) · price ground truth: pykrx_

## Step 0 — Inventory & coverage

- Sheets: **508** · ex-ante calls: **1884** · ex-post: **1652**
- Persisted per-field: ex-ante 507/508 · ex-post 504 · primary_numbers 508 · watchlist 508 · timeline 508 · VLM 508 · frames>0 508 · publish 507
- **Full transcript persisted: 0/508** (excerpt-only: 507). Frame images: not persisted (media auto-deleted).
- **Auditable:** the SCORED call layer (quotes+direction) fully; primary/watchlist numbers vs real prices; CONFLICT tags; number plausibility. **Not auditable from disk:** per-frame OCR tokens, full transcript (garble runs on stored quotes only), frame images.

## Track A — SCORING-CRITICAL (call-rule replay across ALL calls)

Total flags: **1116** across **474** sheets (of 1884 ex-ante calls).

| category | count | % of ex-ante |
|---|--:|--:|
| long_generic_dropped | 428 | 22.7% |
| schedule_no_dated_event | 374 | 19.9% |
| direction_mismatch | 84 | 4.5% |
| ticker_resolution | 60 | 3.2% |
| trim_as_short | 58 | 3.1% |
| holdings_false_positive | 45 | 2.4% |
| market_structure_noise | 33 | 1.8% |
| expost_as_exante | 28 | 1.5% |
| avoid_as_short | 4 | 0.2% |
| negation_missed | 2 | 0.1% |

**Flagged examples (first 25):**

- `-GKk7z7oArQ` [02:01] 005490 **ticker_resolution** (stored=avoid→corrected=None): quote names 삼성전자(005930) but call ticker=005490 (resolved to primary/fallback)  
  > 그래가지고 해당 보도 이후에 우리나라 증시만 좀 발작 처리가 일어나면서 다 내리치는 모멘텀이 좀 나왔었는데 반도체 주가 급등하면서 단기 조정 가능성에 대해서 우려가 커진 영향에 따라가지고 리스크 회피로 좀 판단이 나옵니다 그래가지고 이 부분 때문에 삼성전자하고 SK하이닉스가 눌리니까 다
- `-GKk7z7oArQ` [03:22] 005490 **long_generic_dropped** (stored=long→corrected=long-generic): §1.3 plain-long is UNCLASSIFIED → dropped from scoring; should be LONG-GENERIC  
  > 지금 외국인들이 조금 매도가를 때렸습니다. 그동안에 27일 날짜부터 11일 날짜까지 매수 타점을 연일 잡고 들어왔었고 9일 날짜 하루 빼놓고 전부 다 타점 잡고 들어왔었죠.
- `-GKk7z7oArQ` [25:42] 079900 **schedule_no_dated_event** (stored=long→corrected=None): classified SCHEDULE but no concrete dated event in quote  
  > 실질적으로 로봇 관련되어 있는 주 등에서 아직까지 그 밸류로 못 따라간 종목들도 있습니다 전진건설 로봇 같은 경우에도 제가 이렇게 들어가가지고 그걸 먹고 나온 겁니다 그래서 여러분들께서도 이렇게 스케줄 매매를 하셔야 됩니다
- `02YBJgsqqi8` [00:26] 005490 **long_generic_dropped** (stored=long→corrected=long-generic): §1.3 plain-long is UNCLASSIFIED → dropped from scoring; should be LONG-GENERIC  
  > 그렇기 때문에 지금 이게 계속적으로 올려칠 수 있을지 수급 상태를 먼저 좀 보시는 게 좋을 것 같은데 지금 12월 달 때도 주가를 보시면 기관들 쪽에서 연일 매수 타점을 잡고 들어왔고요. 여기에 맞춰서 금투 쪽에서 따라 들어왔습니다.
- `02YBJgsqqi8` [02:04] 005490 **market_structure_noise** (stored=short→corrected=none): quote is market-structure (사이드카/반대매매/서킷브레이커/동시호가), not a call  
  > 두 번의 서킷 브레이크하고 매도 사이드카 부분 때문에 폭락 처리가 나오면서 33만원대에서 35만원 사이로 매집선 가져갔다가
- `02YBJgsqqi8` [08:31] 005490 **holdings_false_positive** (stored=avoid→corrected=none): avoid fired only via 홀딩 inside 홀딩스 (company name)  
  > 그런다고 해도 이거 잡고 들어가야 된다고 보고 있어요 왜 그러냐면 원래 포스트홀딩스테이 주가는 여러분들 리튬 부분으로 움직여요
- `03bygzBKE10` [07:51] 001440 **long_generic_dropped** (stored=long→corrected=long-generic): §1.3 plain-long is UNCLASSIFIED → dropped from scoring; should be LONG-GENERIC  
  > 미래세증권이에요 여기 보시면 미래세증권이 여기 48억 원짜리를 매수한 사체 건명금액으로 해가지고 이걸 갖다가 370억에 사갔다는 거죠
- `061Ytt81JH8` [21:41] 030530 **holdings_false_positive** (stored=avoid→corrected=none): avoid fired only via 홀딩 inside 홀딩스 (company name)  
  > 상대적으로 반도체 섹터 기업들은 굉장히 세게 올라가는데 지금 뭐 코리아 서키트 14, 원익홀딩스 13% SK하이닉스 7% 올라가면서 완전히 반도체 수급에 대한 쏠림 현상이 일어나거든요
- `061Ytt81JH8` [25:48] 079900 **schedule_no_dated_event** (stored=long→corrected=None): classified SCHEDULE but no concrete dated event in quote  
  > 실질적으로 로봇 관련되어 있는 주 등에서 아직까지 그 밸류로 못 따라간 종목들도 있습니다 전진건설 로봇 같은 경우에도 제가 이렇게 들어가가지고 그걸 먹고 나온 겁니다 그래서 여러분들께서도 이렇게 스케줄 매매를 하셔야 됩니다
- `0HCFPQtVry8` [04:09] 028300 **long_generic_dropped** (stored=long→corrected=long-generic): §1.3 plain-long is UNCLASSIFIED → dropped from scoring; should be LONG-GENERIC  
  > 이게 무슨 뜻이냐면 지금 주주배정이지 않습니까? 그래가지고 만약에 50%가 청약이 들어와가지고 배정처리가 들어가요. 그럼 나머지 50%가 남은 거 아닙니까?
- `0HCFPQtVry8` [21:11] 028300 **direction_mismatch** (stored=short→corrected=long): stored=short vs corrected=long  
  > 이 부분 때문에 타점 잡고 들어가시 한 다음에 저희가 매도과를 때리고 나온 겁니다 근데 여러분들 최근에 이게 불성실지연공시 5점
- `0HCFPQtVry8` [27:20] 079900 **schedule_no_dated_event** (stored=long→corrected=None): classified SCHEDULE but no concrete dated event in quote  
  > 실질적으로 로봇 관련되어 있는 주 등에서 아직까지 그 밸류로 못 따라간 종목들도 있습니다 전진건설 로봇 같은 경우에도 제가 이렇게 들어가가지고 그걸 먹고 나온 겁니다 그래서 여러분들께서도 이렇게 스케줄 매매를 하셔야 됩니다
- `0IOc5U8_4jM` [00:53] 008670 **long_generic_dropped** (stored=long→corrected=long-generic): §1.3 plain-long is UNCLASSIFIED → dropped from scoring; should be LONG-GENERIC  
  > 지금 네이버 여기 교보증권 같은 경우에도 사거래 연속으로 싹쓸이 매수를 하고 있고요 그 다음에 신한투자증권 오늘 매수 타점 잡고 들어왔습니다 보시면 한국투자증권은 오늘 매도가를 때렸는데
- `0IOc5U8_4jM` [25:59] 079900 **schedule_no_dated_event** (stored=long→corrected=None): classified SCHEDULE but no concrete dated event in quote  
  > 실질적으로 로봇 관련되어 있는 주 등에서 아직까지 그 밸류로 못 따라간 종목들도 있습니다 전진건설 로봇 같은 경우에도 제가 이렇게 들어가가지고 그걸 먹고 나온 겁니다 그래서 여러분들께서도 이렇게 스케줄 매매를 하셔야 됩니다
- `0VkNtt3epUw` [25:47] 079900 **schedule_no_dated_event** (stored=long→corrected=None): classified SCHEDULE but no concrete dated event in quote  
  > 실질적으로 로봇 관련되어 있는 주 등에서 아직까지 그 밸류로 못 따라간 종목들도 있습니다 전진건설 로봇 같은 경우에도 제가 이렇게 들어가가지고 그걸 먹고 나온 겁니다 그래서 여러분들께서도 이렇게 스케줄 매매를 하셔야 됩니다
- `12rzf9bdn3Y` [00:59] 001440 **long_generic_dropped** (stored=long→corrected=long-generic): §1.3 plain-long is UNCLASSIFIED → dropped from scoring; should be LONG-GENERIC  
  > 지금 대한전선 같은 경우는 그래도 수급 처리가 조금씩 들어오고 있긴 한데 지금 보시면은 프로그램 매수제도 최근에 5거래 연속으로 들어왔고요 거기에 맞춰가지고 공매도 수급도 조금 떨어집니다
- `12rzf9bdn3Y` [13:18] 001440 **trim_as_short** (stored=short→corrected=trim): profit-take/reduce (차익실현/비중축소/익절) labeled as bearish short  
  > 왜 그러냐면 지금 아시겠지만 지금 저게 주당 처분당가가 38,962원에 거래가 됐어요 그러니까 지금 저 블록딜 부분에 대한 거를 공시부터 쭉 보시면
- `18Il9mcYs60` [00:40] 032820 **long_generic_dropped** (stored=long→corrected=long-generic): §1.3 plain-long is UNCLASSIFIED → dropped from scoring; should be LONG-GENERIC  
  > 거기에 맞춰서 지금 기관수급이 셉니다 지금 금요일 날짜에 기관수급이 싹쓸이 매수를 해갔고요 오히려 외국인들은 단기차임에도 출혈을 시켰습니다
- `18Il9mcYs60` [26:59] 079900 **schedule_no_dated_event** (stored=long→corrected=None): classified SCHEDULE but no concrete dated event in quote  
  > 많이 올랐겠지만 실질적으로 로봇 관련돼 있는 주 등에서 아직까지 그 밸류로 못 따라간 종목들도 있습니다 전진건설 로봇 같은 경우에도 제가 이렇게 들어가 가지고 그걸 먹고 나온 겁니다 그래서 여러분들께서도 이렇게 스케줄 매매를 하셔야 됩니다 그리고 정확하게 타점 부분을 잡아 가지고 그 일
- `18Il9mcYs60` [27:32] 196170 **schedule_no_dated_event** (stored=long→corrected=None): classified SCHEDULE but no concrete dated event in quote  
  > 그리고 제가 이제 이거를 그냥 종목만 보내드리는 게 아니라 이렇게 리포트로 해가지고 종목 리스트 10종목 그래서 최근에는 뭐 2차전지 부분 당연히 턴어라운드 나오기 때문에 들어가야 되는 거고요 바이오 섹터 같은 경우에도 알테오젠 부분까지 해가지고 이 코스피 이전 상장 종목도 있고 거기에
- `1BCC9yNTJrU` [16:40] 047040 **direction_mismatch** (stored=short→corrected=long): stored=short vs corrected=long  
  > 그래서 제가 매수과랑 매도과랑 실시간 대응 부분에 대한 거를
- `1BCC9yNTJrU` [22:45] 079900 **schedule_no_dated_event** (stored=long→corrected=None): classified SCHEDULE but no concrete dated event in quote  
  > 실질적으로 로봇 관련되어 있는 주 등에서 아직까지 그 밸류로 못 따라간 종목들도 있습니다 전진건설 로봇 같은 경우에도 제가 이렇게 들어가가지고 그걸 먹고 나온 겁니다 그래서 여러분들께서도 이렇게 스케줄 매매를 하셔야 됩니다
- `1FLZhvgwUWU` [00:33] 005930 **long_generic_dropped** (stored=long→corrected=long-generic): §1.3 plain-long is UNCLASSIFIED → dropped from scoring; should be LONG-GENERIC  
  > 계속적인 반복적 패턴 지금 보시면 삼성전자와 하이닉스 쪽으로 수급에 대한 쏠림 현상이 들어가면 지수가 상승 처리가 나오고요
- `1FLZhvgwUWU` [01:07] 005930 **market_structure_noise** (stored=short→corrected=none): quote is market-structure (사이드카/반대매매/서킷브레이커/동시호가), not a call  
  > 지금 프로그램 매도세를 장 초반부터 던지면서 완전히 누르는 모멘텀이 나오지만 결과적으로 매도사이드카가 오늘 걸리나요? 좀 아직까지 확인이 안 되고 있는데
- `1FLZhvgwUWU` [19:43] 005930 **direction_mismatch** (stored=avoid→corrected=long): stored=avoid vs corrected=long  
  > 이 지급 위에서 현재 보유한 25조원 규모의 이 자사주에 대한 거가 들어가고요

## Track B — EXTRACTION QUALITY (vs real prices / dictionary / plausibility)

- Primary 현재가 — OCR consistent w/ reality: **40.2%** · stated: **88.2%** (n=507 checked)
- Primary 거래량 OCR within 2× of real: **18.0%**
- CONFLICT tags: 980 total · false (OCR≈stated→AGREE): 1 · OCR-wrong-side: 244 · stated-wrong: 9 · **false/auto-resolvable: 25.9%**
- Watchlist price accuracy: **72.1%** (18284/18715 rows had a real price)
- Transcript garble candidates: **13.4%** of sheets (72 candidates; detection-only, on stored quotes)
- Implausible primary numbers (field-aware guard): **401**

## Step 3 — Degraded-sheet list (candidates for TARGETED re-extract; NOT re-extracted here)

1 sheets flagged structurally degraded:

- `videoplayback_10to11_silent` — 0 transcript segs, 0 ex-ante calls, no publish date

## Self-check (5 hand-verifications vs raw data)

- SC1 086520 CONFLICT 3B2Oxc16lp4 pub=2026-06-23 OCR=13927 stated=100900 real=103000 → OCR wrong
- SC2 086520 CONFLICT 7SG2no8PFbE pub=2024-06-26 OCR=17789 stated=89319 real=87801 → OCR wrong
- SC3 HLB(028300) watchlist -GKk7z7oArQ OCR=750 real[46000,48700] → MISALIGNED/flagged
- SC4 clean call 061Ytt81JH8 086520 dir=long mmss=01:34 → 0 Track-A flags (PASS)
- SC5 holdings_false_positive 02YBJgsqqi8 005490 stored=avoid → flagged

## VERDICT

### (A) Is the SCORED call data clean enough for the Phase 1B re-run? (GATES 1B)

Track-A flags fall in **three tiers** with very different fixes:

| tier | what it means | fix | count | % of calls |
|---|---|---|--:|--:|
| **1. Extraction errors** | the *stored call is wrong* | post-hoc FILTER on stored calls (no re-extraction) | 170 | 9.0% |
| **2. Scorer-rule gaps** | the call is fine; `phase1b.classify()` mishandles it | one-line `classify()` amendments (§1.3 LONG-GENERIC + SCHEDULE-needs-date) | 802 | 42.6% |
| **3. Advisory / ambiguous** | genuinely unclear direction or quote-context ticker | manual review (medium confidence) | 144 | 7.6% |

- **Tier 1 (data errors): 170 calls (9.0%)** — concentrated in a few PRECISE, deterministically-fixable patterns (홀딩스 substring firing avoid; market-structure noise; profit-take/disposal mislabeled short; ex-post recaps filed as ex-ante). These are real but **auto-correctable by a post-hoc filter on the stored `exante_calls` — the 508 sheets do NOT need re-extraction.**
- **Tier 2 (scorer gaps): 802 calls (42.6%)** — DOMINATED by `long_generic_dropped` (428: plain-long calls the scorer drops as UNCLASSIFIED — §1.3 says score them as LONG-GENERIC) and `schedule_no_dated_event` (374: boilerplate '스케줄 매매' firing the SCHEDULE play with no real date). The sheet data is fine; `phase1b.classify()` must adopt the amendment before 1B or the primary test silently loses ~428 long signals.

🟡 **CONDITIONAL GO** — the scored call DATA is clean enough (Tier-1 errors are a small, precise, filterable set), BUT a trustworthy 1B re-run REQUIRES first: (a) apply the `classify()` amendments (§1.3 LONG-GENERIC + SCHEDULE-needs-concrete-date), and (b) run the Tier-1 post-hoc call filter. Both are code-level fixes on the existing data — **no re-extraction of the 508 sheets is needed** (only 1 structurally-degraded sheet(s), listed above).

### (B) Descriptive-quality fix list (shared OCR/Whisper step — does NOT block 1B)

- Primary 현재가 OCR consistent w/ real KRX price: **40.2%** · stated **88.2%** · watchlist **72.1%** (price source: pykrx). Lever: OCR row/column mapping.
- CONFLICT tags **25.9%** false/auto-resolvable (1 are OCR≈stated mis-tagged AGREE; 244 have OCR on the wrong side per real price) → auto-resolve CONFLICTs against the live price.
- 401 implausible primary numbers + garble candidates 13.4% → add the field-aware magnitude guard to extraction + a Korean-finance spell-normalizer. **None of (B) biases the 1B verdict.**
