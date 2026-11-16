# E5 v2 파일럿 2 보고서 (A = pythia-1.4b)

**결정**: C1~C4를 모두 넘은 모델이 없다. prereg-e5 태그를 달지 않고 실패를 기록한 뒤 설계로 돌아간다.

## 통과 기준 (C1~C4)

| 생성기 | C1 채우기 | C2 관문 통과 칸 ≥ 0.5 | C3 전부 통과 칸 < 0.5 | C4 판정기 성공 ≥ 0.20 | 모두 |
|---|---|---|---|---|---|
| olmo-2-1124-7b-instruct | 실패 0/20 | 통과 0.500 | 통과 0.100 | 실패 0.034 | 실패 |
| olmo-2-0325-32b-instruct-4bit | 실패 19/20 | 통과 0.900 | 통과 0.050 | 실패 0.076 | 실패 |

## olmo-2-1124-7b-instruct

칸 20개, underfilled 20, 생성 실패 5. 칸당 S1 호출: 최소 30, 중앙값 30.0, 최대 30. 칸별 후보 수 [1, 0, 8, 0, 9, 0, 5, 2, 9, 2, 2, 1, 1, 0, 3, 3, 2, 0, 9, 1]

| 단계 | 형식 | n | 실패율 | 사유별 비율 |
|---|---|---|---|---|
| S1 | oneliner | 300 | 0.647 | {'wa_position': 0.62, 'wa_count': 0.043, 'shape': 0.02, 'truncated': 0.013} |
| S1 | riddle | 300 | 0.407 | {'truncated': 0.31, 'wa_position': 0.09, 'wa_count': 0.08, 'length': 0.01} |
| S1 | all | 600 | 0.527 | {'wa_position': 0.355, 'truncated': 0.162, 'wa_count': 0.062, 'shape': 0.01, 'length': 0.005} |
| S3_calls | oneliner | 210 | 0.957 | {'shape': 0.714, 'edit_width': 0.162, 'suffix_changed': 0.09, 'wp_count': 0.043, 'wa_leak': 0.005} |
| S3_calls | riddle | 327 | 0.850 | {'edit_width': 0.535, 'suffix_changed': 0.263, 'wp_count': 0.107, 'shape': 0.043, 'wa_leak': 0.015} |
| S3_calls | all | 537 | 0.892 | {'edit_width': 0.389, 'shape': 0.305, 'suffix_changed': 0.196, 'wp_count': 0.082, 'wa_leak': 0.011} |
| final | oneliner | 9 | 0.000 | {} |
| final | riddle | 49 | 0.000 | {} |
| final | all | 58 | 0.000 | {} |

S3 시도 단위: {'n': 284, 'fail': 0.795774647887324, 'retried': 0.8908450704225352}

| 형식 | n | 관문 ③ | 관문 ② | w_a 1위 ③ | w_a 1위 ② |
|---|---|---|---|---|---|
| oneliner | 9 | 0.444 | 0.111 | 1.000 | 1.000 |
| riddle | 49 | 0.306 | 0.449 | 0.306 | 0.449 |
| all | 58 | 0.328 | 0.397 | 0.414 | 0.534 |

**D1** A_→(③) − A_→(②): 평균 -0.0078 (칸 묶음 95% -0.1617, 0.1377), 중앙값 0.0067, 오른 비율 0.672 (n = 58)
**D2** 판정기 성공률: 관문 통과 0.053 (n = 19), 관문 실패 0.026 (n = 39). P 후보 전체 0.034, w_p 되풀이 0.086
힌트 단어 사용률 0.569, topic_link 분위 {'0.1': 0.12711316496133804, '0.25': 0.17077115178108215, '0.5': 0.3439507633447647, '0.75': 0.4650455340743065, '0.9': 0.5117534399032593}
B1 규칙 통과 0.500 (n = 20). B0 규칙 통과 0.350, 중복률 0.150, 사유 {'pair_not_in_cmu': 6, 'pair_malformed': 3, 'pair_too_far': 3, 'wp_count': 1}

점수 상위 3개:
- (flee→family, 점수 11.40, 관문 X, 판정 X) After being deported unexpectedly, she finally sat down with her flee.
- (flee→family, 점수 10.83, 관문 X, 판정 X) After being deported unexpectedly, they gathered around the table, enjoying each other's company as a flee.
- (whipping→missing, 점수 10.04, 관문 X, 판정 X) After enduring countless misadventures, the key was frustratingly whipping.

점수 하위 3개:
- (bat→butt, 점수 -6.27, 관문 X, 판정 X) What did the fielder use to catch speedy balls? The bat.
- (bat→butt, 점수 -6.45, 관문 X, 판정 X) What does the fielder catch with both hands? The bat.
- (bat→butt, 점수 -6.68, 관문 X, 판정 X) What did the fielder use for support during the match? A bat.

## olmo-2-0325-32b-instruct-4bit

칸 20개, underfilled 1, 생성 실패 0. 칸당 S1 호출: 최소 10, 중앙값 15.0, 최대 30. 칸별 후보 수 [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8]

| 단계 | 형식 | n | 실패율 | 사유별 비율 |
|---|---|---|---|---|
| S1 | oneliner | 167 | 0.234 | {'wa_position': 0.234} |
| S1 | riddle | 167 | 0.114 | {'wa_position': 0.078, 'wa_count': 0.06, 'truncated': 0.012} |
| S1 | all | 334 | 0.174 | {'wa_position': 0.156, 'wa_count': 0.03, 'truncated': 0.006} |
| S3_calls | oneliner | 180 | 0.456 | {'edit_width': 0.383, 'suffix_changed': 0.067, 'wp_count': 0.056, 'wa_leak': 0.011} |
| S3_calls | riddle | 219 | 0.543 | {'edit_width': 0.397, 'suffix_changed': 0.078, 'wp_count': 0.078, 'shape': 0.068, 'wa_leak': 0.037} |
| S3_calls | all | 399 | 0.504 | {'edit_width': 0.391, 'suffix_changed': 0.073, 'wp_count': 0.068, 'shape': 0.038, 'wa_leak': 0.025} |
| final | oneliner | 98 | 0.000 | {} |
| final | riddle | 100 | 0.000 | {} |
| final | all | 198 | 0.000 | {} |

S3 시도 단위: {'n': 276, 'fail': 0.2826086956521739, 'retried': 0.44565217391304346}

| 형식 | n | 관문 ③ | 관문 ② | w_a 1위 ③ | w_a 1위 ② |
|---|---|---|---|---|---|
| oneliner | 98 | 0.388 | 0.286 | 0.582 | 0.724 |
| riddle | 100 | 0.500 | 0.450 | 0.550 | 0.620 |
| all | 198 | 0.444 | 0.369 | 0.566 | 0.672 |

**D1** A_→(③) − A_→(②): 평균 0.0338 (칸 묶음 95% -0.0435, 0.0927), 중앙값 0.0034, 오른 비율 0.788 (n = 198)
**D2** 판정기 성공률: 관문 통과 0.023 (n = 88), 관문 실패 0.118 (n = 110). P 후보 전체 0.076, w_p 되풀이 0.121
힌트 단어 사용률 0.616, topic_link 분위 {'0.1': 0.11299471706151965, '0.25': 0.17888221517205238, '0.5': 0.39264437556266785, '0.75': 0.5124511122703552, '0.9': 0.6065598130226135}
B1 규칙 통과 0.850 (n = 20). B0 규칙 통과 0.450, 중복률 0.000, 사유 {'pair_malformed': 4, 'pair_too_far': 3, 'pair_same_lemma': 3, 'wp_count': 1}

점수 상위 3개:
- (whipping→missing, 점수 13.83, 관문 X, 판정 X) After a thorough search, the important document was still whipping.
- (whipping→missing, 점수 12.38, 관문 X, 판정 X) Why isn't the project completed? A piece is whipping.
- (flee→family, 점수 11.85, 관문 X, 판정 X) After narrowly escaping danger, she returned home to her flee.

점수 하위 3개:
- (bat→butt, 점수 -5.64, 관문 X, 판정 X) Near the fielder at the bat.
- (bat→butt, 점수 -5.96, 관문 X, 판정 X) What's the thick end of the shovel called on a baseball field? The bat.
- (scatter→sector, 점수 -6.02, 관문 X, 판정 X) After slicing the pizza, fragments flew everywhere and a scatter.
