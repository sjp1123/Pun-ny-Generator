# 결과 정리 — pythia-160m

판정 지위: **개발용(주 모델 pythia-1.4b 아님 — 판정은 참고용)**

## 가설별 판정

| 가설 | 핵심 수치 (95% CI) | 판정 |
|---|---|---|
| H1a 전체 > LM만, 발음만 (E1 acc@1) | 전체−LM 0.041 [0.029, 0.055]; 전체−발음 0.065 [0.034, 0.096] | 지지 |
| H1b A 지표가 기준선 위에 Δρ (E3) | Δρ 0.021 [-0.006, 0.048] | 지지 안 됨 |
| H1c {w_p,w_a}·γ=0이면 효과 사라짐 | Δρ 0.012 [-0.008, 0.032] | 판정 불가(H1b 미지지) |
| (표제어 일치 w_a) H1a | 전체−LM 0.065 [0.049, 0.082]; 전체−발음 0.080 [0.048, 0.115] | 지지 |
| (표제어 일치 w_a) H1b | Δρ 0.039 [0.008, 0.069] | 지지 |
| (표제어 일치 w_a) H1c | Δρ 0.016 [-0.006, 0.039] | 지지 |
| H2 (탐색) 역 U자 | 2차항 -0.001 [-0.006, 0.003], 꼭짓점 -24.21 | 역 U자 아님 (탐색) |

## 사전 점검 재측정 (후보 재현율@100)

단순 편집 거리 0.855, panphon 자질 거리 0.611, 자질 거리(삽입·삭제 ×0.25) 0.808 (n=908)

## E0 오염 점검

count_olmo: 474/1098 (43.2%) 1회 이상, count_pile: 51/1098 (4.6%) 1회 이상
- count_olmo: log D_local 중앙값 본 집단 -0.00 vs 안 본 집단 -0.00, Mann-Whitney p = 0.0315
- count_pile: log D_local 중앙값 본 집단 -0.00 vs 안 본 집단 -0.00, Mann-Whitney p = 0.381

## E1

후보 재현율 0.808 (완전 동음 0.924, 유사 동음 0.763), γ = [0.75, 1.0, 1.0, 0.75, 0.75]

| 부분집합 | LM만 | 발음만 | 전체 |
|---|---|---|---|
| all (n=1074) | 0.344 | 0.319 | 0.385 |
| exact (n=302) | 0.483 | 0.854 | 0.570 |
| near (n=772) | 0.289 | 0.110 | 0.312 |

오류 50개 자동 분류: {'후보 누락': 18, '기타(손 분류 필요: 데이터 오류 등)': 18, '빈도 편향(자동 추정)': 8, '부분 단어': 6}

## E2a (He 6유형)

| 유형 | He z | log D_local | log F_local | A_global | S_ratio |
|---|---|---|---|---|---|
| pun | +1.13 | -0.01 | -5.76 | 0.143 | 0.59 |
| depun | +0.05 | -0.90 | -10.39 | 0.083 | -0.60 |
| retrieved_aw_alter | -0.22 | -0.01 | -3.62 | 0.062 | 1.95 |
| retrieved_aw | -0.24 | -0.71 | -11.01 | 0.035 | -0.96 |
| retrieved_pw_alter | -0.39 | -0.02 | -5.43 | 0.043 | -0.04 |
| retrieved_pw | -0.44 | -0.43 | -8.34 | 0.029 | -0.71 |

## E2b

- pun_vs_nonpun_joke: logD_local 0.743, logFstar_local 0.638, A_star_global 0.570, logFstar_global 0.690, L_star_global 0.756, combo 0.563
- pun_vs_nonjoke: logD_local 0.774, logFstar_local 0.665, A_star_global 0.631, logFstar_global 0.707, L_star_global 0.808, combo 0.621
- 위치 맞히기: logD_local 0.658, logFstar_local 0.238, A_star_global 0.123, logFstar_global 0.337, L_star_global 0.691, combo 0.156, chance 0.202

## E3 단일 특징 ρ

S_ratio +0.075, unusualness -0.175, ambiguity -0.022, distinctiveness +0.301, phon_dist -0.166, sent_lp +0.180, logF_local +0.196, logD_local +0.056, A_global -0.135, logF_global +0.263, A_local -0.102, logD_global +0.120, L_global +0.239, L_local +0.175

## Ablation (E3 Δρ)

- full (H1b): 0.021 [-0.006, 0.048]
- gamma0: 0.022 [-0.006, 0.051]
- pair, fit γ: 0.010 [-0.007, 0.027]
- pair, γ=0 (H1c): 0.012 [-0.008, 0.032]
- local only: 0.012 [-0.010, 0.033]
- global only: 0.022 [-0.005, 0.049]
- wa_lemma: full (H1b): 0.039 [0.008, 0.069]
- wa_lemma: pair, γ=0 (H1c): 0.016 [-0.006, 0.039]
- {w_p,w_a}·γ=0의 L_global vs He S_global ρ = 1.0000
