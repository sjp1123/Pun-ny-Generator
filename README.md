# pun-recoverability — 아이디어 A 1단계 (영어 v0)

"복원 가능성 지표가 기존 지표 위에 설명력을 더하는가"를 공개 데이터와 학습 없는 모델로 검증하는 파이프라인.
설계 문서: `아이디어 A 1단계 설계: 영어 v0 파이프라인` (2026-11-02).

## 구조

```
data/            내려받은 원본 (pungen, gppl, w2v). 손대지 않음
cache/           후보표, LM 점수(parquet 조각), Kao 코사인
src/
  data.py        로더, SemEval↔UKP 대응, pun 위치 매핑, He 주석 검수
  g2p.py         M1  CMU 사전 + g2p_en(사전 밖 단어)
  phon.py        M2  panphon 가중 자질 편집 거리, ARPAbet→IPA, 어휘 축 벡터화 DP
  candidates.py  M3  {w_p} ∪ 발음 거리 상위 K개
  lmscore.py     M4  후보 문장 로그 확률(전역·국소·He 창), 접두부 KV 캐시 공유
  listener.py    M5  π, D, F, F*, T̂, A, L, rank
  baselines.py   M6  S_ratio, unusualness, Kao 모호성·뚜렷함, 발음 거리, 문장 로그 확률
  e2b_positions.py   E2b용 내용어 위치
experiments/
  precheck_recall.py   사전 점검 재측정 (단순 vs 자질 거리 재현율)
  build_cache.py       후보표 + LM 캐시 생성 (재실행하면 이어서)
  build_kao.py         word2vec 코사인 캐시
  e0_contamination.py  E0 infini-gram 오염 점검
  e1_recovery.py       E1 숨은 단어 찾기 (H1a)
  e2_detect.py         E2a/E2b 판별과 위치
  e3_funniness.py      E3 웃김 예측 (H1b, H1c, H2)
  ablation.py          ablation, 모델 크기 비교
  hand_check.py        잘 알려진 말장난 10개의 π
tests/           단위 테스트
results/         표(md), 수치(json), 항목별 csv
```

## 실행

Windows에서는 `python -X utf8`로 실행해야 panphon 자질표가 읽힌다.

```bash
pip install transformers accelerate pyarrow scikit-learn wordfreq gensim spacy cmudict requests pytest
pip install --no-deps panphon g2p_en      # editdistance 빌드 실패 회피 (panphon.distance는 안 씀)
pip install unicodecsv munkres regex pyyaml inflect distance
pip install en_core_web_sm-3.8.0-py3-none-any.whl
```

```bash
python -X utf8 experiments/precheck_recall.py
python -X utf8 experiments/build_cache.py --model pythia-160m --dataset semeval he
python -X utf8 experiments/build_kao.py
python -X utf8 experiments/e0_contamination.py
python -X utf8 experiments/e0_contamination.py --analyze --model pythia-160m
python -X utf8 experiments/e1_recovery.py --model pythia-160m
python -X utf8 experiments/e2_detect.py --model pythia-160m --part a
python -X utf8 experiments/e3_funniness.py --model pythia-160m
python -X utf8 experiments/ablation.py --model pythia-160m --models pythia-160m pythia-410m pythia-1.4b
python -X utf8 experiments/hand_check.py --model pythia-160m
python -X utf8 -m pytest tests -q
```

주 결과(확인 검정 H1a~c)는 `--model pythia-1.4b`로 다시 돌린다. GPU(16GB, fp16)에서
`build_cache.py --model pythia-1.4b --device cuda`. E2b는 `--dataset e2b`(위치 약 1.4만 개)라 GPU 권장.

## 설계 문서와 다르게 구현한 점 (이유)

1. **후보 순위는 단순 음소 편집 거리로.** 설계는 자질 거리(M2) 정규화 순위였지만, 다시 재보니
   재현율@100이 단순 편집 거리 85.5% vs panphon 자질 거리 61.1%였다(`results/precheck/recall.json`).
   panphon은 삽입·삭제 비용(분절음당 7.25)이 비슷한 소리 치환(0.25~2)보다 훨씬 커서
   alleged→ledge, orifice→office 같은 길이 변화형 숨은 단어를 뒤로 민다. 그래서 후보 생성은 단순 거리,
   발음 항 d는 자질 거리로 나눴다. 자질 거리 순위는 `candidates(rank_by="feature")`로 남겨 뒀다.
2. **자질 거리 단위.** d를 panphon 삽입 비용으로 나눠 삽입·삭제 하나가 1이 되게 했다. γ 격자(0~5, 0.25)가
   의미 있는 범위를 덮게 하려는 것이고, 순위·π에는 영향이 없다(γ가 흡수).
3. **S_ratio 창.** "He의 식 그대로"를 따라 He 코드의 창 `[i−2, i+2)`(앞 2단어, pun, 뒤 1단어)를 쓴다(ctx `he_local`).
   A 지표의 국소 문맥은 설계대로 앞뒤 2단어.
4. **Kao의 r.** 관련도 변환 `exp(r·z(cos))`의 r은 {0.5, 1, 2, 4} 중 학습 겹에서 기준선 회귀의 학습 ρ가
   가장 높은 값으로 고른다. z의 평균·표준편차도 학습 겹에서만 맞춘다.
5. **w_a 표제어 문제 (기록, 주 결과는 고치지 않음).** SemEval alter_word는 표제어인 경우가 많다
   (psalms→arm인데 문맥에 맞는 건 arms, soled→sell인데 sold). 160M E1 오답 660개 중 95개가 이 경우였다.
   설계대로 주 결과는 그대로(`--wa-mode strict`) 두고, 표제어가 같은 후보 중 log q가 가장 큰 것을 w_a로 보는
   `--wa-mode lemma`를 따로 보고한다(`e1_lemma.*`, `e3_lemma.*`). lemma 모드는 A 지표에만 적용되고
   기준선(S_ratio, Kao)은 원래 w_a를 쓴다.
6. **He 주석 검수 결과.** 설계 문서의 "tom barked가 pun, tom bached가 depun으로 뒤바뀜"은 사실과 반대다.
   SemEval·UKP 원문은 "Tom barked"(숨은 단어 bach)이고, He pun 33개는 모두 SemEval 원문·pun 단어와 일치했다.
   숨은 단어 철자만 5개가 다르다(bached/bach, selld/sell, noses/nose, arms/arm, tiles/tile). `results/he_pun_review.csv`.
7. **E2b는 CPU에서 K = 20.** 위치 11,495개 × 후보 101개는 CPU로 12시간 넘게 걸려 K를 줄였다. GPU에서는 `E2B=1 E2B_K=100`.
8. **CMU 사전 판.** NLTK 자원 서버 접속이 막혀 pip `cmudict` 패키지(같은 CMUdict)를 쓴다.
   어휘 크기는 설계 수치와 같은 26,926개.

---

# 1단계 보완 실험 (실험 1~4, 2026-11-12)

지침: `docs/EXPERIMENTS.md`. 사전 등록: `PREREG.md` (태그 `prereg-v1`). 고정 규칙 코드: `src/rules.py`.
모든 실행은 `results/<model>/<실험>/`에 `config.json`(설정·시드 20261112·git 커밋), `metrics.json`, `table.md`를 남긴다.
아래 판정은 모두 **dev 점검** 또는 **참고용(160M)**이다. 확인 검정은 1.4B·test에서만 한다.

```bash
python -X utf8 experiments/build_v1_cache.py --model pythia-160m --dataset semeval he   # left, C_LM 캐시
python -X utf8 experiments/v1_exp1_lemma.py      # 1-B
python -X utf8 experiments/v1_exp1_delta.py      # 1-C
python -X utf8 experiments/v1_exp2_noise.py      # 2 (LM 없음)
python -X utf8 experiments/v1_exp3_reading.py    # 3
python -X utf8 experiments/v1_exp4_phon.py       # 4-A, 4-B
python -X utf8 experiments/v1_confirm.py --model pythia-1.4b   # prereg-v1 이후에만
```

## 실험 1. 분석 규칙 고정

- **목적**: 1.4B 결과를 보기 전에 분할·채점·검정 규칙을 전부 고정한다.
- **설정**: dev 219 / test 879 (`splits/`, SHA-256은 PREREG). 표제어 도구 spaCy 3.8.18 + en_core_web_sm 3.8.0. 160M, dev.
- **결과 표**

  | 항목 | 값 (dev, 160M) |
  |---|---|
  | lemma(w_p) = lemma(w_a) 제외 | 214개 중 1개 (het_145) |
  | E1 acc@1 strict / 표제어 최댓값 / 표제어 합 | 0.465 / 0.559 / 0.559 |
  | strict와 표제어 합 판정이 갈린 항목 | 20개 (모두 strict 오답 → 합 정답) |
  | 손 점검 도구 오류 | 0 / 20 |
  | 1-C δ = ρ(M1_full) − ρ(M1_pair0) | 0.012 [−0.069, 0.096] (구현 확인만) |

- **판정**: dev 점검 — 표제어 합 채점과 δ 검정이 동작한다. 도구 오류가 기준(2개) 이하라 spaCy 유지.
- **다음 결정**: PREREG 커밋·태그 후 1.4B로 H1a~c 확인 검정.

## 실험 2. 웃김 점수의 잡음 상한

- **목적**: 사람 웃김 점수의 신뢰도로 어떤 예측이든 도달할 수 있는 ρ의 상한을 구한다.
- **설정**: `results.tsv` 판단 141,050개, 평가자 1,059명. Bradley–Terry(choix ILSR, α = 0.01, X = 0.5승), 반분 200회, 스피어만-브라운 보정.
- **결과 표**: `results/no-lm/exp2_noise/table.md` (ρ_max: 말장난 1,074 0.884 (평가자) / 0.895 (판단), 말장난 test 0.885, 전체 4,030 0.924. 점수 검증: item_scores.txt와 스피어만 0.994~0.996)
- **판정**: dev 점검 아님(LM 없는 데이터 분석). 말장난끼리 ρ_max 0.884 ≥ 0.4 — E3로 작은 차이를 가릴 여지가 있다. 160M v0의 E3 ρ 0.30~0.36은 상한의 34~41%. 말장난끼리는 전체보다 r_full이 0.071 낮다(예상대로). 이 실행의 config는 dirty 판정 수정 전 기록이라 dirty로 남았지만 코드는 fee814f 이후 바뀌지 않았다.
- **다음 결정**: 모든 E3 표에 상대 성능 ρ/ρ_max를 함께 적는다.

## 실험 3. 읽는 순간 vs 다 읽은 뒤

- **목적**: 임의의 국소 창(앞뒤 2단어)을 읽기 순서에 맞춘 두 시점(π_←: 왼쪽 문맥, π_→: 문장 전체)으로 바꾼다.
- **설정**: 기존 후보 C_phon에 왼쪽 문맥 점수(ctx `left`) 추가. γ는 dev(1.5). 160M, dev.
- **결과 표**

  | 점검 | 결과 |
  |---|---|
  | pun 뒤 단어 수 (1,098개) | 0개 629 / 1개 216 / 2개 110 / 3개 이상 143. 왼쪽에 영숫자 없음 9 (완전히 빈 7 + 따옴표만 2) |
  | 포화: D > 0.99 비율 | 국소 0.667, 전역 0.587, 읽는 순간 0.479. S_← 중앙값 4.44 (5~95%: 0.16~12.64) |
  | 구현 확인 | pun 뒤에 문자가 없는 문장 2개에서 π_← = π_→ (총변동거리 ~1e−15) |
  | 마지막 단어 문장의 π_← vs π_→ | 총변동거리 중앙값 0.154 — 뒤따르는 문장부호 확률 P(. │ 문맥, T)에서만 온다 |
  | He 6유형 S_← | pun 5.59, retrieved_aw_alter 6.95 > depun 2.78, retrieved_pw 1.44, retrieved_aw 0.66 (예측 방향) |

- **판정**: dev 점검 통과(구현·방향). 성능으로 채택 여부를 정하지 않았다.
- **다음 결정**: H3a·H3b로 등록. U와 뒤 단어 2개 이상 부분집합은 탐색.

## 실험 4. 발음 항 개선과 발음 기여 분리

- **목적**: (4-A) 약한 발음 항을 구간 가중치로 보강, (4-B) 후보 필터가 발음 항의 일을 대신했는지 분리.
- **설정**: 단순 음소 편집 거리, 목표 dev Σ log π_→(w_a 묶음). C_LM = 왼쪽 문맥 첫 토큰 확률 상위 100개. 160M, dev.
- **결과 표 (4-A)**

  | 발음 항 | 모수 | 5겹 검증 로그 우도 | acc@1 | 완전 동음 | 유사 동음 |
  |---|---|---|---|---|---|
  | γ = 0 | — | −537.8 | 0.432 | 0.581 | 0.371 |
  | 단일 γ | 1.5 | −466.3 | 0.559 | 0.774 | 0.470 |
  | 구간 (자유 = 단조) | 0.53, 2.78, 5.31 | −472.8 | 0.563 | 0.774 | 0.477 |
  | 발음만 | — | — | 0.437 | 0.887 | 0.252 |

- **결과 표 (4-B)**: E1 acc@1 / 후보 재현율 / E3 ρ(log F_→)

  | 후보 집합 | γ = 0 | 단일 γ | 구간 가중치 |
  |---|---|---|---|
  | C_phon (101) | 0.432 / 0.854 / 0.122 | 0.559 / 0.854 / 0.142 | 0.563 / 0.854 / 0.137 |
  | C_LM (101) | 0.113 / 0.258 / 0.092 | 0.249 / 0.258 / 0.089 | 0.249 / 0.258 / 0.139 |
  | C_union (200) | 0.183 / 0.873 / 0.126 | 0.577 / 0.873 / 0.164 | 0.549 / 0.873 / 0.153 |

- **판정**: dev 점검. 4-A — 구간 가중치는 단일 γ보다 낫지 않다(검증 우도는 더 낮음, 과적합 기미). 완전 동음 격차는
  v0의 0.28에서 0.113으로 줄었다(표제어 합·단순 거리 덕). 4-B — C_union에서 γ = 0이면 E1이 0.183으로 무너지고
  γ > 0에서 0.577로 회복: **발음 정보는 필수이고, 지금까지는 후보 필터가 그 일을 대신했다.** C_LM 재현율 0.258은
  언어 모델만으로는 숨은 단어를 떠올리지 못한다는 증거다.
- **다음 결정**: H4(C_union에서 단일 γ − γ = 0)를 등록. 구간 가중치는 H4a 탐색으로만.
- **참고**: C_LM은 첫 토큰만으로 줄 세우므로 같은 첫 토큰을 공유하는 단어(bar, barry, barcelona…)가 함께 들어온다.
