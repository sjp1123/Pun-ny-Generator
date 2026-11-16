# E5 v2 pilot3 보고서 (A = pythia-1.4b)

**결정**: C1~C4를 모두 넘은 모델이 없다. prereg-e5 태그를 달지 않고 실패를 기록한 뒤 설계로 돌아간다.

## 통과 기준 (C1~C4)

| 생성기 | C1 채우기 | C2 관문 통과 칸 ≥ 0.5 | C3 전부 통과 칸 < 0.5 | C4 판정기 성공 ≥ 0.20 | 모두 |
|---|---|---|---|---|---|
| olmo-2-1124-7b-instruct | 실패 6/20 | 통과 0.550 | 통과 0.050 | 통과 0.282 | 실패 |
| olmo-2-0325-32b-instruct-4bit | 실패 19/20 | 통과 1.000 | 통과 0.050 | 통과 0.203 | 실패 |

## olmo-2-1124-7b-instruct

칸 20개, underfilled 14, 생성 실패 3. 칸당 S1 호출: 최소 15, 중앙값 30.0, 최대 30. 칸별 후보 수 [2, 5, 5, 2, 10, 2, 6, 10, 10, 0, 10, 0, 9, 0, 2, 2, 6, 2, 10, 10]

| 단계 | 형식 | n | 실패율 | 사유별 비율 |
|---|---|---|---|---|
| S1 | oneliner | 287 | 0.634 | {'wa_position': 0.61, 'wa_count': 0.146, 'shape': 0.017, 'truncated': 0.007} |
| S1 | riddle | 263 | 0.106 | {'wa_position': 0.095, 'wa_count': 0.057, 'truncated': 0.004} |
| S1 | all | 550 | 0.382 | {'wa_position': 0.364, 'wa_count': 0.104, 'shape': 0.009, 'truncated': 0.005} |
| S3_calls | oneliner | 187 | 0.824 | {'suffix_changed': 0.422, 'edit_width': 0.369, 'wp_count': 0.193, 'truncated': 0.043, 'wa_leak': 0.016, 'shape': 0.005} |
| S3_calls | riddle | 434 | 0.839 | {'edit_width': 0.664, 'suffix_changed': 0.101, 'shape': 0.071, 'wp_count': 0.062, 'wa_leak': 0.025} |
| S3_calls | all | 621 | 0.834 | {'edit_width': 0.575, 'suffix_changed': 0.198, 'wp_count': 0.101, 'shape': 0.052, 'wa_leak': 0.023, 'truncated': 0.013} |
| final | oneliner | 33 | 0.000 | {} |
| final | riddle | 70 | 0.000 | {} |
| final | all | 103 | 0.000 | {} |

S3 시도 단위: {'n': 340, 'fail': 0.6970588235294117, 'retried': 0.8264705882352941}

| 형식 | n | 관문 ③ | 관문 ② | w_a 1위 ③ | w_a 1위 ② |
|---|---|---|---|---|---|
| oneliner | 33 | 0.424 | 0.485 | 0.727 | 0.909 |
| riddle | 70 | 0.271 | 0.500 | 0.371 | 0.671 |
| all | 103 | 0.320 | 0.495 | 0.485 | 0.748 |

**D1** A_→(③) − A_→(②): 평균 0.0873 (칸 묶음 95% 0.0009, 0.1643), 중앙값 0.0084, 오른 비율 0.718 (n = 103)
**D2** 판정기 성공률: 관문 통과 0.182 (n = 33), 관문 실패 0.329 (n = 70). P 후보 전체 0.282, w_p 되풀이 0.097
힌트 단어 사용률 0.524, topic_link 분위 {'0.1': 0.1295958161354065, '0.25': 0.2567385137081146, '0.5': 0.41291260719299316, '0.75': 0.5953681468963623, '0.9': 0.6561347246170044}
B1 규칙 통과 0.700 (n = 20). B0 규칙 통과 0.300, 중복률 0.100, 사유 {'pair_malformed': 7, 'pair_not_in_cmu': 5, 'pair_too_far': 2}

점수 상위 3개:
- (rein→rain, 점수 18.75, 관문 X, 판정 O) The weather forecast predicts thunderstorms with heavy rein this evening.
- (knight→night, 점수 13.18, 관문 X, 판정 O) After dining with a nobleman, they strolled outside to enjoy the cool knight air.
- (steep→sleep, 점수 12.69, 관문 X, 판정 O) Climbing the treacherous path, wearing pajamas, she soon drifted off to steep.

점수 하위 3개:
- (mark→arc, 점수 -6.55, 관문 X, 판정 X) What happens when achievements are numerous? A mark.
- (mark→arc, 점수 -8.76, 관문 X, 판정 X) What occurs when a milestone is achieved? A mark.
- (mist→midst, 점수 -9.51, 관문 X, 판정 X) What was the dessert's hidden feature? Chocolate mist.

## olmo-2-0325-32b-instruct-4bit

칸 20개, underfilled 1, 생성 실패 0. 칸당 S1 호출: 최소 10, 중앙값 14.5, 최대 30. 칸별 후보 수 [10, 10, 10, 10, 10, 10, 10, 10, 10, 7, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

| 단계 | 형식 | n | 실패율 | 사유별 비율 |
|---|---|---|---|---|
| S1 | oneliner | 162 | 0.222 | {'wa_position': 0.222, 'wa_count': 0.049} |
| S1 | riddle | 149 | 0.121 | {'wa_position': 0.101, 'wa_count': 0.067, 'truncated': 0.007} |
| S1 | all | 311 | 0.174 | {'wa_position': 0.164, 'wa_count': 0.058, 'truncated': 0.003} |
| S3_calls | oneliner | 182 | 0.467 | {'edit_width': 0.346, 'wp_count': 0.093, 'suffix_changed': 0.055, 'shape': 0.022} |
| S3_calls | riddle | 184 | 0.457 | {'edit_width': 0.375, 'wp_count': 0.065, 'suffix_changed': 0.06, 'wa_leak': 0.06, 'shape': 0.049} |
| S3_calls | all | 366 | 0.462 | {'edit_width': 0.361, 'wp_count': 0.079, 'suffix_changed': 0.057, 'shape': 0.036, 'wa_leak': 0.03} |
| final | oneliner | 97 | 0.000 | {} |
| final | riddle | 100 | 0.000 | {} |
| final | all | 197 | 0.000 | {} |

S3 시도 단위: {'n': 257, 'fail': 0.23346303501945526, 'retried': 0.42412451361867703}

| 형식 | n | 관문 ③ | 관문 ② | w_a 1위 ③ | w_a 1위 ② |
|---|---|---|---|---|---|
| oneliner | 97 | 0.340 | 0.175 | 0.784 | 0.814 |
| riddle | 100 | 0.580 | 0.470 | 0.680 | 0.780 |
| all | 197 | 0.462 | 0.325 | 0.731 | 0.797 |

**D1** A_→(③) − A_→(②): 평균 0.0403 (칸 묶음 95% -0.0187, 0.1069), 중앙값 0.0006, 오른 비율 0.756 (n = 197)
**D2** 판정기 성공률: 관문 통과 0.231 (n = 91), 관문 실패 0.179 (n = 106). P 후보 전체 0.203, w_p 되풀이 0.081
힌트 단어 사용률 0.721, topic_link 분위 {'0.1': 0.14282728135585784, '0.25': 0.31763245165348053, '0.5': 0.5362533926963806, '0.75': 0.6278131604194641, '0.9': 0.6569241285324097}
B1 규칙 통과 0.850 (n = 20). B0 규칙 통과 0.300, 중복률 0.000, 사유 {'pair_too_far': 6, 'pair_same_lemma': 5, 'pair_malformed': 2, 'pair_not_in_cmu': 1}

점수 상위 3개:
- (knight→night, 점수 14.51, 관문 X, 판정 O) After a long day as a nobleman, all he wanted was to spend the knight in peace.
- (lad→lap, 점수 13.58, 관문 X, 판정 O) The young boy fell asleep while resting on her mother's lad.
- (lad→lap, 점수 13.04, 관문 X, 판정 X) The young boy fell asleep with her book on her lad.

점수 하위 3개:
- (mark→arc, 점수 -5.89, 관문 X, 판정 X) What milestone forms in the sky after a lightning strike? A mark.
- (mist→midst, 점수 -5.94, 관문 X, 판정 X) Amidst the fog, they sought mist.
- (mark→arc, 점수 -7.68, 관문 X, 판정 X) Reached a milestone and created a mark.
