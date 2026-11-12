# 1-B 표제어 합 채점 dev 점검 — pythia-160m (dev 점검)

dev 한 단어 항목 214개 중 lemma(w_p)=lemma(w_a)로 제외 1개: ['het_145']

γ (dev, 단순 편집 거리) = 1.5

| 채점 | acc@1 | acc@5 | MRR |
|---|---|---|---|
| strict | 0.465 | 0.596 | 0.525 |
| max | 0.559 | 0.667 | 0.604 |
| sum | 0.559 | 0.667 | 0.605 |

strict와 표제어 합의 판정이 갈린 항목 20개 중 20개를 손 점검했다 (hand_check_20.csv).

- **het_28** soled→sell (목표 sell; strict False, 합 True)
  - 상위 후보: sold(sell) 0.918, filled(fill) 0.033, owned(own) 0.014
  - 상위 묶음: sell 0.919, fill 0.033, own 0.014 / M: sold 0.918, sell 0.000
- **het_24** psalms→arm (목표 arm; strict False, 합 True)
  - 상위 후보: arms(arm) 0.989, palms(palm) 0.006, sails(sail) 0.002
  - 상위 묶음: arm 0.989, palm 0.007, sail 0.002 / M: arms 0.989
- **het_133** taut→teach (목표 teach; strict False, 합 True)
  - 상위 후보: taught(teach) 0.777, tight(tight) 0.135, watched(watch) 0.023
  - 상위 묶음: teach 0.777, tight 0.135, watch 0.023 / M: taught 0.777
- **het_287** garden→guard (목표 guard; strict False, 합 True)
  - 상위 후보: guarding(guard) 0.536, given(give) 0.249, run(run) 0.101
  - 상위 묶음: guard 0.558, give 0.249, run 0.101 / M: guarding 0.536, guarded 0.011, guard 0.010, guards 0.002
- **het_221** roles→roll (목표 roll; strict False, 합 True)
  - 상위 후보: rolls(roll) 0.066, rules(rule) 0.058, goals(goal) 0.004
  - 상위 묶음: roll 0.066, rule 0.058, goal 0.004 / M: rolls 0.066, roll 0.000, rolled 0.000, rolling 0.000
- **het_265** loess→low (목표 low; strict False, 합 True)
  - 상위 후보: lowest(low) 0.624, most(most) 0.288, earliest(early) 0.027
  - 상위 묶음: low 0.624, most 0.288, early 0.027 / M: lowest 0.624, low 0.001
- **het_234** patience→patient (목표 patient; strict False, 합 True)
  - 상위 후보: patients(patient) 0.212, patient(patient) 0.013, passion(passion) 0.010
  - 상위 묶음: patient 0.225, passion 0.011, license 0.003 / M: patients 0.212, patient 0.013
- **het_282** marx→mark (목표 mark; strict False, 합 True)
  - 상위 후보: marks(mark) 0.333, marcus(marcus) 0.173, parker(parker) 0.166
  - 상위 묶음: mark 0.335, marcus 0.173, parker 0.166 / M: marks 0.333, marked 0.002, mark 0.000, marking 0.000
- **het_13** dyed→die (목표 die; strict False, 합 True)
  - 상위 후보: died(die) 0.761, cried(cry) 0.148, lied(lie) 0.056
  - 상위 묶음: die 0.765, cry 0.148, lie 0.056 / M: died 0.761, die 0.003, dying 0.001, dies 0.000
- **het_239** pi→pie (목표 pie; strict False, 합 True)
  - 상위 후보: pies(pie) 0.423, pie(pie) 0.403, buy(buy) 0.032
  - 상위 묶음: pie 0.826, buy 0.032, guy 0.029 / M: pies 0.423, pie 0.403, pied 0.000
- **het_59** toad→tow (목표 tow; strict False, 합 True)
  - 상위 후보: towed(tow) 0.759, lowered(lower) 0.088, turned(turn) 0.058
  - 상위 묶음: tow 0.759, lower 0.088, turn 0.058 / M: towed 0.759, towing 0.000, tow 0.000
- **het_194** haulways→hallway (목표 hallway; strict False, 합 True)
  - 상위 후보: hallways(hallway) 0.482, hallway(hallway) 0.238, walls(wall) 0.170
  - 상위 묶음: hallway 0.720, wall 0.170, hall 0.057 / M: hallways 0.482, hallway 0.238
- **het_335** breaths→breast (목표 breast; strict False, 합 True)
  - 상위 후보: breasts(breast) 0.738, dress(dress) 0.103, breast(breast) 0.068
  - 상위 묶음: breast 0.807, dress 0.103, breath 0.058 / M: breasts 0.738, breast 0.068, breasted 0.000
- **het_12** hz→hurt (목표 hurt; strict False, 합 True)
  - 상위 후보: hurts(hurt) 0.872, burns(burn) 0.083, hit(hit) 0.008
  - 상위 묶음: hurt 0.874, burn 0.083, hit 0.010 / M: hurts 0.872, hurt 0.002, hurting 0.000
- **het_86** clothed→close (목표 close; strict False, 합 True)
  - 상위 후보: closed(close) 0.821, close(close) 0.070, clothing(clothing) 0.025
  - 상위 묶음: close 0.894, clothing 0.025, clothe 0.025 / M: closed 0.821, close 0.070, closer 0.002, closes 0.000, closest 0.000
- **het_94** bubbled→double (목표 double; strict False, 합 True)
  - 상위 후보: doubled(double) 0.701, double(double) 0.254, doubles(double) 0.013
  - 상위 묶음: double 0.968, bottle 0.007, smuggle 0.006 / M: doubled 0.701, double 0.254, doubles 0.013, doubling 0.000
- **het_240** herd→hear (목표 hear; strict False, 합 True)
  - 상위 후보: heard(hear) 0.997, hurt(hurt) 0.001, had(have) 0.000
  - 상위 묶음: hear 0.997, hurt 0.001, have 0.000 / M: heard 0.997
- **het_126** roomers→rumor (목표 rumor; strict False, 합 True)
  - 상위 후보: rumors(rumor) 0.772, rumours(rumour) 0.124, rumor(rumor) 0.071
  - 상위 묶음: rumor 0.843, rumour 0.157, number 0.000 / M: rumors 0.772, rumor 0.071, rumored 0.000
- **het_303** stank→sink (목표 sink; strict False, 합 True)
  - 상위 후보: sank(sink) 0.973, sunk(sink) 0.027, stuck(stick) 0.001
  - 상위 묶음: sink 0.999, stick 0.001, stab 0.000 / M: sank 0.973, sunk 0.027, sink 0.000
- **het_224** dyed→die (목표 die; strict False, 합 True)
  - 상위 후보: died(die) 0.965, arrived(arrive) 0.027, dies(die) 0.004
  - 상위 묶음: die 0.970, arrive 0.027, cry 0.001 / M: died 0.965, dies 0.004, die 0.001, dying 0.000

## 손 점검 판정 (2026-11-12)

20개 모두 M(w_a) 구성원이 올바른 활용형이었다. 도구 오류 0개(기준: 2개 초과면 교체) → spaCy 유지, 예외 목록 없음.
참고: rumours/rumor처럼 철자 변이는 다른 표제어로 남는다(보수적). 20개 모두 strict 오답 → 표제어 합 정답 방향이었다.
