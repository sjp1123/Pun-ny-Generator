# E5 v2 pilot7 보고서 (A = pythia-1.4b)

**결정**: C1~C4와 S3 확인(③ − ②)을 모두 넘었다. 본 실행 구성(v4 쌍, S1 품사 검사, S3, S4 없음)을 확정하고 태그 준비로 간다.

## 통과 기준 (C1~C4)

| 생성기 | C1 채우기 | C2 관문 통과 칸 ≥ 0.5 | C3 전부 통과 칸 < 0.5 | C4 판정기 성공 ≥ 0.20 | 모두 |
|---|---|---|---|---|---|
| olmo-2-0325-32b-instruct-4bit | 통과 20/20 | 통과 1.000 | 통과 0.100 | 통과 0.540 | 통과 |

C4는 phi-4 성립 판정(judge2_phi-4-4bit, 모든 P 후보 ③의 yes 비율). S3 확인 (③ − ② 성립 yes ≥ 0): 통과 0.035 [-0.030, 0.100]

## olmo-2-0325-32b-instruct-4bit

칸 20개, underfilled 0, 생성 실패 0. 칸당 S1 호출: 최소 10, 중앙값 17.0, 최대 57. 칸별 후보 수 [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

| 단계 | 형식 | n | 실패율 | 사유별 비율 |
|---|---|---|---|---|
| S1 | oneliner | 207 | 0.319 | {'wa_position': 0.295, 'wa_count': 0.048, 'pos_mismatch': 0.005} |
| S1 | riddle | 198 | 0.207 | {'wa_position': 0.126, 'wa_count': 0.106, 'pos_mismatch': 0.04} |
| S1 | all | 405 | 0.264 | {'wa_position': 0.212, 'wa_count': 0.077, 'pos_mismatch': 0.022} |
| S3_calls | oneliner | 201 | 0.502 | {'edit_width': 0.448, 'suffix_changed': 0.04, 'wp_count': 0.03, 'wa_leak': 0.01} |
| S3_calls | riddle | 245 | 0.592 | {'edit_width': 0.441, 'wp_count': 0.131, 'suffix_changed': 0.127, 'wa_leak': 0.11, 'shape': 0.024} |
| S3_calls | all | 446 | 0.552 | {'edit_width': 0.444, 'suffix_changed': 0.087, 'wp_count': 0.085, 'wa_leak': 0.065, 'shape': 0.013} |
| final | oneliner | 100 | 0.000 | {} |
| final | riddle | 100 | 0.000 | {} |
| final | all | 200 | 0.000 | {} |

S3 시도 단위: {'n': 298, 'fail': 0.3288590604026846, 'retried': 0.4966442953020134}

| 형식 | n | 관문 ③ | 관문 ② | w_a 1위 ③ | w_a 1위 ② |
|---|---|---|---|---|---|
| oneliner | 100 | 0.560 | 0.490 | 0.630 | 0.800 |
| riddle | 100 | 0.570 | 0.460 | 0.640 | 0.670 |
| all | 200 | 0.565 | 0.475 | 0.635 | 0.735 |

**D1** A_→(③) − A_→(②): 평균 0.1175 (칸 묶음 95% 0.0814, 0.1547), 중앙값 0.0159, 오른 비율 0.870 (n = 200)
힌트 단어 사용률 0.660, topic_link 분위 {'0.1': 0.13363650739192962, '0.25': 0.2538336515426636, '0.5': 0.4175456464290619, '0.75': 0.5522120594978333, '0.9': 0.6865871548652649}
B1 규칙 통과 0.800 (n = 20). B0 규칙 통과 0.450, 중복률 0.000, 사유 {'pair_malformed': 3, 'pair_not_in_cmu': 3, 'pair_too_far': 3, 'pair_same_lemma': 2, 'wp_count': 1}

점수 상위 3개:
- (prey→pray, 점수 12.25, 관문 X) What do we do in church to avoid predators on Sunday? Sing and prey.
- (scent→send, 점수 10.71, 관문 O) What's the final action after composing a note? Mail it. Scent.
- (scent→send, 점수 10.50, 관문 O) After considering various options, they finally decided to scent the package.

점수 하위 3개:
- (soap→sew, 점수 -4.40, 관문 X) What do you add to water to clean a shirt? Mend soap.
- (soap→sew, 점수 -6.07, 관문 X) What cleanses a dirty shirt? A soap.
- (saving→shaving, 점수 -7.90, 관문 X) After realizing he was wasting money, he needed saving.
