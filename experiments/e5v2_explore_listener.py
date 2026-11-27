"""E5 탐색 (본 실행 결과를 본 뒤, 판정에 쓰지 않음): 청자 고르기(P2)가 무작위(P1)를 넘지 못한 이유.

규칙 통과 P 후보 791개 전체의 phi-4 판정, 청자 점수(scores.csv), 연구자 (c)(answers_cells.csv)로
여지(무작위 vs 최선), 관문 효과, 칸 안 AUC, 웃김 분포를 본다.
사용: python -X utf8 experiments/e5v2_explore_listener.py . > results/e5/main/olmo-2-0325-32b-instruct-4bit/A_pythia-1.4b/explore/listener_signal.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1])
G = ROOT / "results/e5/main/olmo-2-0325-32b-instruct-4bit"
A = G / "A_pythia-1.4b"
cand = pd.read_csv(G / "candidates.csv")
J = pd.read_csv(A / "judge2_phi-4-4bit/items.csv").set_index("cand_id")
X = pd.read_csv(A / "scores.csv").drop_duplicates("cand_id").set_index("cand_id")
S = pd.read_csv(A / "selections.csv")

P = cand[(cand.source == "P") & cand.rule_ok.astype(bool)].copy()
P["works"] = P.cand_id.map(J.works).map(lambda v: 1.0 if str(v) == "True" else 0.0 if str(v) == "False" else np.nan)
P["funny"] = P.cand_id.map(J.funny).astype(float)
for c in ["gate", "score", "A_g", "S_ratio", "sent_lp", "unusualness", "logF_g", "S_left"]:
    P[c] = P.cand_id.map(X[c])
P["gate"] = P.gate.fillna(False).astype(bool)
P["len"] = P.text.str.split().str.len()
print(f"rule-passing P candidates {len(P)} in {P.cell.nunique()} cells; phi-4 yes {P.works.mean():.3f}")

# 1. headroom: random pick vs best possible pick, per cell
g = P.groupby("cell").works.agg(["size", "mean", "max", "min"])
print(f"pool size per cell: median {g['size'].median():.0f}, min {g['size'].min()}, max {g['size'].max()}")
print(f"expected random pick (mean of cell rates) {g['mean'].mean():.3f}")
print(f"best possible pick (cell has >=1 yes)     {g['max'].mean():.3f}")
print(f"cells all-yes {(g['min'] == 1).mean():.3f}, all-no {(g['max'] == 0).mean():.3f}, mixed {((g['max'] == 1) & (g['min'] == 0)).mean():.3f}")

# 2. gate
gp = P.groupby("gate").works.agg(["size", "mean"])
print("gate pass/fail phi-4 yes:", gp.to_dict("index"))
gw = P.groupby(["cell", "gate"]).works.mean().unstack()
both = gw.dropna()
print(f"within-cell gate effect (cells with both): {(both[True] - both[False]).mean():+.3f} over {len(both)} cells")
gf = P.groupby("cell").gate.mean()
print(f"gate pass share per cell: mean {gf.mean():.3f}, cells with 0 pass {(gf == 0).sum()}, cells all pass {(gf == 1).sum()}")


# 3. within-cell AUC of each score for phi-4 works (mixed cells only)
def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, float)
    ok = ~(np.isnan(s) | np.isnan(y))
    s, y = s[ok], y[ok]
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    return ((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


print("within-cell AUC for phi-4 works (0.5 = no signal):")
for c in ["score", "A_g", "S_ratio", "sent_lp", "unusualness", "logF_g", "S_left", "len"]:
    v = [auc(d[c], d.works) for _, d in P.groupby("cell")]
    v = [x for x in v if not np.isnan(x)]
    print(f"  {c:12s} {np.mean(v):.3f}  (cells {len(v)})")
gpass = P[P.gate]
v = [auc(d["score"], d.works) for _, d in gpass.groupby("cell")]
v = [x for x in v if not np.isnan(x)]
print(f"  score among gate-passing only {np.mean(v):.3f} (cells {len(v)})")

# 4. funniness spread
fs = P.groupby("cell").funny.agg(["std", "min", "max"])
print(f"phi-4 funny 1-5 overall dist {P.funny.value_counts().sort_index().to_dict()}; within-cell sd mean {fs['std'].mean():.2f}")
v = [auc(d["score"], (d.funny >= 4).astype(float)) for _, d in P.groupby("cell")]
v = [x for x in v if not np.isnan(x)]
print(f"  score AUC for funny>=4 within cell {np.mean(v):.3f} (cells {len(v)})")

# 5. human (c) on rated P sentences, within cell
H = ROOT / "results/e5/main/olmo-2-0325-32b-instruct-4bit/A_pythia-1.4b/human"
K = pd.read_csv(H / "key.csv")
C = pd.read_csv(H / "answers_cells.csv")
KC = K[(K.sheet == "cells") & (K.source == "P")].merge(C, left_on="no", right_on="번호")
KC["c"] = KC["(c) 성공한 말장난"].astype(str).str.upper().eq("Y").astype(float)
KC["score"] = KC.cand_id.map(X.score)
KC["gate"] = KC.cand_id.map(X.gate).fillna(False).astype(bool)
KC["phi"] = KC.cand_id.map(P.set_index("cand_id").works)
v = [auc(d["score"], d.c) for _, d in KC.groupby("cell")]
v = [x for x in v if not np.isnan(x)]
print(f"human: rated P sentences {len(KC)}; within-cell score AUC for human (c) {np.mean(v):.3f} (mixed cells {len(v)})")
print(f"human (c) gate pass {KC[KC.gate].c.mean():.3f} (n {KC.gate.sum()}), gate fail {KC[~KC.gate].c.mean():.3f} (n {(~KC.gate).sum()})")
hc = KC.groupby("cell").c.agg(["max", "min"])
print(f"human: cells where every rated P sentence works {(hc['min'] == 1).mean():.3f}, mixed {((hc['max'] == 1) & (hc['min'] == 0)).mean():.3f}")

# 6. P2's gate status
s2 = S[S.condition == "P2"]
print(f"P2 picks with gate_fail (no candidate passed) {int(s2.gate_fail.fillna(False).astype(bool).sum())}/{len(s2)}")

# 7. which half of the pun does each listener term track? (rated P sentences pooled, researcher's (a) and (b))
KC["a"] = KC["(a) w_p 뜻으로 말이 됨"].astype(str).str.upper().eq("Y").astype(float)
KC["b"] = KC["(b) w_a로 읽어도 말이 됨"].astype(str).str.upper().eq("Y").astype(float)
print(f"human failure types on rated P sentences (n {len(KC)}): w_p reading fails {(KC.a == 0).mean():.3f}, "
      f"w_a reading fails {(KC.b == 0).mean():.3f}, both readings ok but not a pun {((KC.a == 1) & (KC.b == 1) & (KC.c == 0)).mean():.3f}")
for c in ["S_left", "A_g", "logF_g", "score"]:
    KC[c] = KC.cand_id.map(X[c])
    print(f"  {c:7s} pooled AUC for (a) w_p reading ok {auc(KC[c], KC.a):.3f}, for (b) w_a reading ok {auc(KC[c], KC.b):.3f}")
