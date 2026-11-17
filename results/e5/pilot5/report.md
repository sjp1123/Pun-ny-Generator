# E5 v2 pilot5 보고서 (A = pythia-1.4b)

**결정**: C1~C4와 S3 확인 중 넘지 못한 것이 있다. 본 실행 S3는 파일럿 4의 S3(바꾸기 허용, e5/v2/prompts)로 돌아간다.

## 통과 기준 (C1~C4)

| 생성기 | C1 채우기 | C2 관문 통과 칸 ≥ 0.5 | C3 전부 통과 칸 < 0.5 | C4 판정기 성공 ≥ 0.20 | 모두 |
|---|---|---|---|---|---|
| olmo-2-0325-32b-instruct-4bit | 실패 17/20 | 통과 0.950 | 통과 0.050 | 통과 0.770 | 실패 |

C4는 phi-4 성립 판정(judge2_phi-4-4bit, 모든 P 후보 ③의 yes 비율). S3 확인 (③ − ② 성립 yes ≥ 0): 통과 0.055 [-0.026, 0.146]

## olmo-2-0325-32b-instruct-4bit

칸 20개, underfilled 3, 생성 실패 0. 칸당 S1 호출: 최소 10, 중앙값 17.0, 최대 60. 칸별 후보 수 [10, 5, 10, 10, 10, 10, 10, 10, 10, 3, 10, 5, 10, 10, 10, 10, 10, 10, 10, 10]

| 단계 | 형식 | n | 실패율 | 사유별 비율 |
|---|---|---|---|---|
| S1 | oneliner | 346 | 0.208 | {'wa_position': 0.205, 'wa_count': 0.052, 'shape': 0.003} |
| S1 | riddle | 147 | 0.088 | {'wa_position': 0.082, 'wa_count': 0.054, 'truncated': 0.007} |
| S1 | all | 493 | 0.172 | {'wa_position': 0.168, 'wa_count': 0.053, 'truncated': 0.002, 'shape': 0.002} |
| S3_calls | oneliner | 498 | 0.833 | {'removed': 0.618, 'edit_width': 0.556, 'wp_count': 0.048, 'suffix_changed': 0.026, 'wa_leak': 0.016, 'shape': 0.004} |
| S3_calls | riddle | 190 | 0.463 | {'edit_width': 0.305, 'removed': 0.189, 'wa_leak': 0.058, 'wp_count': 0.053, 'suffix_changed': 0.016, 'shape': 0.005} |
| S3_calls | all | 688 | 0.731 | {'removed': 0.5, 'edit_width': 0.487, 'wp_count': 0.049, 'wa_leak': 0.028, 'suffix_changed': 0.023, 'shape': 0.004} |
| final | oneliner | 83 | 0.000 | {} |
| final | riddle | 102 | 0.020 | {'duplicate': 0.02} |
| final | all | 185 | 0.011 | {'duplicate': 0.011} |

S3 시도 단위: {'n': 408, 'fail': 0.5465686274509803, 'retried': 0.6862745098039216}

| 형식 | n | 관문 ③ | 관문 ② | w_a 1위 ③ | w_a 1위 ② |
|---|---|---|---|---|---|
| oneliner | 83 | 0.398 | 0.217 | 0.855 | 0.831 |
| riddle | 100 | 0.640 | 0.520 | 0.780 | 0.790 |
| all | 183 | 0.530 | 0.383 | 0.814 | 0.809 |

**D1** A_→(③) − A_→(②): 평균 0.0230 (칸 묶음 95% -0.0333, 0.0766), 중앙값 0.0006, 오른 비율 0.781 (n = 183)
힌트 단어 사용률 0.814, topic_link 분위 {'0.1': 0.35376386642456054, '0.25': 0.4553948640823364, '0.5': 0.5937854051589966, '0.75': 0.6434536576271057, '0.9': 0.6689988851547243}
B1 규칙 통과 0.850 (n = 20). B0 규칙 통과 0.300, 중복률 0.000, 사유 {'pair_too_far': 7, 'pair_not_in_cmu': 3, 'truncated': 1, 'pair_malformed': 1, 'wp_count': 1, 'wa_leak': 1}

점수 상위 3개:
- (knight→night, 점수 16.09, 관문 X) In the nobleman's castle, during the day she worked, but at knight she relaxed.
- (lad→lap, 점수 12.82, 관문 X) At the schoolyard, the kids played on the grass, laughing as they each took turns sitting on someone's lad.
- (lad→lap, 점수 12.36, 관문 X) The young fella's baby giggled as she sat on her mother's lad.

점수 하위 3개:
- (mark→arc, 점수 -5.05, 관문 X) What happens when you plug in a frayed cord and record the event? A mark.
- (mark→arc, 점수 -6.98, 관문 X) What milestone happens when the switch closes? A mark sparks.
- (mark→arc, 점수 -8.31, 관문 X) What spark makes a bridge in the air commemorating? A mark.
