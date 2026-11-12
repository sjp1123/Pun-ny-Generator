"""E5 관문 모수 (파이프라인 6~7단계에 쓰는 고정값). 생성물은 보지 않고 기존 데이터로만 정한다.

γ        : SemEval dev에서 Σ log π_lex(w_a)가 최대인 값 (PREREG의 fit_gamma, 후보 C, 전역 문맥)
A_→ 문턱 : He 주석에서 진짜 말장난(pun, 33개)과 오타형(retrieved_aw_alter: w_a 문장에 w_p만 바꿔 넣은 것)을
           A_→(전역 문맥, 표제어 합, 후보 C+w_a)로 가를 때 유든 J(민감도 + 특이도 − 1)가 최대인 값.
           J가 같으면 낮은(너그러운) 문턱. 관문은 "A_→ ≥ 문턱"이고 T̂_→(표제어 묶음 1위)가 w_a 표제어여야 한다.
참고로 He 6유형과 SemEval dev 말장난이 관문을 통과하는 비율도 남긴다.

사용: python -X utf8 experiments/e5_gate.py --model pythia-1.4b
필요: cache/lm/<model>/semeval_K100, he_K100 (build_cache.py --dataset semeval he)
출력: results/<model>/e5_gate/
"""
import argparse

from v1_common import *  # noqa: F403
from data import load_he_annotation
from rules import RunDir, fit_gamma

HE_POS, HE_NEG = "pun", "retrieved_aw_alter"


def youden(pos: np.ndarray, neg: np.ndarray):
    best = None
    for t in np.unique(np.r_[pos, neg]):
        tpr, fpr = float((pos >= t).mean()), float((neg >= t).mean())
        j = tpr - fpr
        if best is None or j > best["J"] + 1e-12:
            best = dict(threshold=float(t), J=j, tpr=tpr, fpr=fpr)
    return best


def he_metrics(model: str, gamma: float) -> pd.DataFrame:
    items, _ = load_he_annotation()
    typ = {it.id: it.meta["type"] for it in items}
    cand = add_lemma_cols(pd.read_parquet(CACHE / "candidates_he_K100.parquet"))
    cand = cand[~cand.same_lemma]
    m = Listener(scored(cand, lm_cache(model, "he_K100"), "global"), "C+wa").metrics(("gamma", gamma))
    m["type"] = m.index.map(typ)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-1.4b")
    a = ap.parse_args()
    D = Data(a.model, "dev")
    gamma = fit_gamma(D.listener("global", "C"))
    he = he_metrics(a.model, gamma)
    best = youden(he[he.type == HE_POS].A.to_numpy(float), he[he.type == HE_NEG].A.to_numpy(float))
    thr = best["threshold"]
    gate = lambda m: (m["rank"] == 1) & (m.A >= thr)
    he["gate"] = gate(he)
    sd = D.listener("global", "C+wa").metrics(("gamma", gamma))
    res = dict(gamma=gamma, A_threshold=thr, youden=best, n_pos=int((he.type == HE_POS).sum()),
               n_neg=int((he.type == HE_NEG).sum()),
               he_by_type={t: dict(n=int(len(g)), gate_pass=float(g.gate.mean()), A_mean=float(g.A.mean()),
                                   rank1=float((g["rank"] == 1).mean())) for t, g in he.groupby("type")},
               semeval_dev=dict(n=int(len(sd)), gate_pass=float(gate(sd).mean()), rank1=float((sd["rank"] == 1).mean()),
                                A_ge_threshold=float((sd.A >= thr).mean())))
    R = RunDir(a.model, "e5_gate", dict(gamma_fit="SemEval dev, 후보 C, 전역", threshold_fit=f"He {HE_POS} vs {HE_NEG}, 유든 J"))
    R.metrics(res)
    L = [f"# E5 관문 모수 — {a.model}", "", f"γ = {gamma} (SemEval dev), A_→ 문턱 = {thr:.4f} "
         f"(유든 J = {best['J']:.3f}, 민감도 {best['tpr']:.3f}, 오타형 통과 {best['fpr']:.3f})", "",
         "| 집단 | n | 관문 통과 | T̂ = w_a | A 평균 |", "|---|---|---|---|---|"]
    for t, v in res["he_by_type"].items():
        L.append(f"| He {t} | {v['n']} | {v['gate_pass']:.3f} | {v['rank1']:.3f} | {v['A_mean']:.3f} |")
    s = res["semeval_dev"]
    L.append(f"| SemEval dev 말장난 | {s['n']} | {s['gate_pass']:.3f} | {s['rank1']:.3f} | — |")
    R.table("table.md", L)


if __name__ == "__main__":
    main()
