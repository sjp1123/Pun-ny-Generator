# E5 v2 pilot4 보고서 (A = pythia-1.4b)

**결정**: olmo-2-0325-32b-instruct-4bit를 본 실행 생성기로 정한다 (C1~C4 모두 통과).

## 통과 기준 (C1~C4)

| 생성기 | C1 채우기 | C2 관문 통과 칸 ≥ 0.5 | C3 전부 통과 칸 < 0.5 | C4 판정기 성공 ≥ 0.20 | 모두 |
|---|---|---|---|---|---|
| olmo-2-0325-32b-instruct-4bit | 통과 20/20 | 통과 0.950 | 통과 0.050 | 통과 0.260 | 통과 |

## olmo-2-0325-32b-instruct-4bit

칸 20개, underfilled 0, 생성 실패 0. 칸당 S1 호출: 최소 10, 중앙값 13.5, 최대 27. 칸별 후보 수 [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

| 단계 | 형식 | n | 실패율 | 사유별 비율 |
|---|---|---|---|---|
| S1 | oneliner | 166 | 0.247 | {'wa_position': 0.241, 'wa_count': 0.096, 'shape': 0.006, 'truncated': 0.006} |
| S1 | riddle | 138 | 0.036 | {'wa_count': 0.036, 'wa_position': 0.029} |
| S1 | all | 304 | 0.151 | {'wa_position': 0.145, 'wa_count': 0.069, 'shape': 0.003, 'truncated': 0.003} |
| S3_calls | oneliner | 173 | 0.416 | {'edit_width': 0.329, 'wp_count': 0.11, 'suffix_changed': 0.023, 'shape': 0.012, 'wa_leak': 0.006} |
| S3_calls | riddle | 191 | 0.476 | {'edit_width': 0.351, 'wp_count': 0.11, 'suffix_changed': 0.099, 'wa_leak': 0.042, 'shape': 0.021} |
| S3_calls | all | 364 | 0.448 | {'edit_width': 0.341, 'wp_count': 0.11, 'suffix_changed': 0.063, 'wa_leak': 0.025, 'shape': 0.016} |
| final | oneliner | 101 | 0.010 | {'duplicate': 0.01} |
| final | riddle | 100 | 0.000 | {} |
| final | all | 201 | 0.005 | {'duplicate': 0.005} |

S3 시도 단위: {'n': 258, 'fail': 0.22093023255813954, 'retried': 0.4108527131782946}

| 형식 | n | 관문 ③ | 관문 ② | w_a 1위 ③ | w_a 1위 ② |
|---|---|---|---|---|---|
| oneliner | 100 | 0.430 | 0.310 | 0.800 | 0.860 |
| riddle | 100 | 0.580 | 0.540 | 0.730 | 0.800 |
| all | 200 | 0.505 | 0.425 | 0.765 | 0.830 |

**D1** A_→(③) − A_→(②): 평균 0.0538 (칸 묶음 95% -0.0061, 0.1218), 중앙값 0.0005, 오른 비율 0.785 (n = 200)
**D2** 판정기 성공률: 관문 통과 0.228 (n = 101), 관문 실패 0.293 (n = 99). P 후보 전체 0.260, w_p 되풀이 0.050
힌트 단어 사용률 0.730, topic_link 분위 {'0.1': 0.11581448167562486, '0.25': 0.4018334150314331, '0.5': 0.5067477822303772, '0.75': 0.6287425756454468, '0.9': 0.6707365751266481}
B1 규칙 통과 0.900 (n = 20). B0 규칙 통과 0.400, 중복률 0.000, 사유 {'pair_same_lemma': 3, 'pair_not_in_cmu': 3, 'pair_malformed': 2, 'wa_leak': 2, 'pair_too_far': 2, 'wp_count': 1}

점수 상위 3개:
- (rein→rain, 점수 19.37, 관문 X, 판정 O) After a sweltering summer day, all she yearned for was to hear the gentle patter of rein.
- (lad→lap, 점수 14.91, 관문 X, 판정 O) The young boy happily read the storybook while sitting on his mother's lad.
- (lad→lap, 점수 14.36, 관문 X, 판정 O) Where does a child often sit for a story with grandma? On grandma's lad.

점수 하위 3개:
- (mark→arc, 점수 -8.49, 관문 X, 판정 X) What's the spark in the air when it's really close to reaching a milestone? A mark.
- (mist→midst, 점수 -9.08, 관문 X, 판정 X) Amidst the haze of the celebration, she found herself mist.
- (mist→midst, 점수 -10.88, 관문 X, 판정 X) What do you find when searching through clutter under a cloudy sky? A needle mist.
