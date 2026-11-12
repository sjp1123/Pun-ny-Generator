# E2 판별과 위치 — pythia-160m

γ = 0.75 (SemEval E1 목적함수로 맞춤)

## E2b ROC-AUC (문장 2407, 위치 11495)

- pun_vs_nonpun_joke: logD_local=0.743 [0.715, 0.768], logFstar_local=0.638 [0.611, 0.666], A_star_global=0.570 [0.539, 0.600], logFstar_global=0.690 [0.663, 0.716], L_star_global=0.756 [0.728, 0.782], combo=0.563 [0.533, 0.592]
- pun_vs_nonjoke: logD_local=0.774 [0.751, 0.796], logFstar_local=0.665 [0.639, 0.689], A_star_global=0.631 [0.604, 0.658], logFstar_global=0.707 [0.682, 0.731], L_star_global=0.808 [0.786, 0.829], combo=0.621 [0.594, 0.649]
- 위치 맞히기 (n=1095): logD_local=0.658, logFstar_local=0.238, A_star_global=0.123, logFstar_global=0.337, L_star_global=0.691, combo=0.156, chance=0.202