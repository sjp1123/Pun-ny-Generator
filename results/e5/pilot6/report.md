# E5 v2 pilot6 보고서 (A = pythia-1.4b)

**결정**: C1~C4와 S3 확인(최종 − ②) 중 넘지 못한 것이 있다. 6.13대로 태그를 달지 않고 기록한다(자동 대체 규칙 없음).

## 통과 기준 (C1~C4)

| 생성기 | C1 채우기 | C2 관문 통과 칸 ≥ 0.5 | C3 전부 통과 칸 < 0.5 | C4 판정기 성공 ≥ 0.20 | 모두 |
|---|---|---|---|---|---|
| olmo-2-0325-32b-instruct-4bit | 통과 20/20 | 통과 1.000 | 통과 0.000 | 통과 0.535 | 실패 |

C4는 phi-4 성립 판정(judge2_phi-4-4bit, 모든 P 후보 ③의 yes 비율). S3 확인 (③ − ② 성립 yes ≥ 0): 실패 -0.020 [-0.080, 0.035]

## olmo-2-0325-32b-instruct-4bit

칸 20개, underfilled 0, 생성 실패 0. 칸당 S1 호출: 최소 12, 중앙값 19.5, 최대 56. 칸별 후보 수 [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

| 단계 | 형식 | n | 실패율 | 사유별 비율 |
|---|---|---|---|---|
| S1 | oneliner | 232 | 0.362 | {'wa_position': 0.289, 'wa_count': 0.078, 'pos_mismatch': 0.017, 'shape': 0.009, 'truncated': 0.009} |
| S1 | riddle | 211 | 0.166 | {'wa_position': 0.114, 'wa_count': 0.071, 'pos_mismatch': 0.043} |
| S1 | all | 443 | 0.269 | {'wa_position': 0.205, 'wa_count': 0.074, 'pos_mismatch': 0.029, 'shape': 0.005, 'truncated': 0.005} |
| S3_calls | oneliner | 205 | 0.454 | {'edit_width': 0.366, 'suffix_changed': 0.068, 'wp_count': 0.059, 'wa_leak': 0.005} |
| S3_calls | riddle | 274 | 0.631 | {'edit_width': 0.405, 'wp_count': 0.215, 'suffix_changed': 0.201, 'wa_leak': 0.182, 'shape': 0.007} |
| S3_calls | all | 479 | 0.555 | {'edit_width': 0.388, 'wp_count': 0.148, 'suffix_changed': 0.144, 'wa_leak': 0.106, 'shape': 0.004} |
| S4_calls | oneliner | 133 | 0.248 | {'suffix_changed': 0.226, 'shape': 0.03, 'wp_count': 0.015, 'wa_leak': 0.008} |
| S4_calls | riddle | 106 | 0.057 | {'suffix_changed': 0.057} |
| S4_calls | all | 239 | 0.163 | {'suffix_changed': 0.151, 'shape': 0.017, 'wp_count': 0.008, 'wa_leak': 0.004} |
| final | oneliner | 100 | 0.000 | {} |
| final | riddle | 100 | 0.000 | {} |
| final | all | 200 | 0.000 | {} |

S3 시도 단위: {'n': 324, 'fail': 0.3425925925925926, 'retried': 0.4783950617283951}
S4 시도 단위: {'n': 213, 'fail': 0.06103286384976526, 'retried': 0.12206572769953052, 'unchanged': 0.0}

| 형식 | n | 관문 ③ | 관문 ② | w_a 1위 ③ | w_a 1위 ② |
|---|---|---|---|---|---|
| oneliner | 100 | 0.530 | 0.500 | 0.630 | 0.860 |
| riddle | 100 | 0.590 | 0.510 | 0.660 | 0.760 |
| all | 200 | 0.560 | 0.505 | 0.645 | 0.810 |

**D1** A_→(③) − A_→(②): 평균 0.1010 (칸 묶음 95% 0.0757, 0.1259), 중앙값 0.0133, 오른 비율 0.800 (n = 200)
힌트 단어 사용률 0.660, topic_link 분위 {'0.1': 0.1798245206475258, '0.25': 0.3080756366252899, '0.5': 0.42908911406993866, '0.75': 0.5522120594978333, '0.9': 0.6746124029159546}
B1 규칙 통과 0.800 (n = 20). B0 규칙 통과 0.450, 중복률 0.000, 사유 {'pair_too_far': 5, 'pair_same_lemma': 2, 'pair_malformed': 2, 'wa_leak': 1, 'pair_not_in_cmu': 1}

점수 상위 3개:
- (scent→send, 점수 13.99, 관문 X) When I finished crafting the perfume, the email automatically scent.
- (haul→hall, 점수 12.62, 관문 X) After a long drive, I finally found the room I needed down the haul.
- (scent→send, 점수 11.67, 관문 X) Where do letters go to be sweet? To the post office to scent.

점수 하위 3개:
- (saving→shaving, 점수 -2.38, 관문 X) After slashing expenses, he ended up smooth from saving.
- (bidding→building, 점수 -2.95, 관문 X) After a tense auction, they were thrilled to see a bidding.
- (soap→sew, 점수 -6.54, 관문 X) Before leaving, she realized her dress was dirty, so she soap.
