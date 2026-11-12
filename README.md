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
