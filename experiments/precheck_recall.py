"""사전 점검 재측정: 후보 재현율을 단순 편집 거리와 조음 자질 거리로 비교한다.

대상: SemEval 이형 말장난 중 두 단어 모두 CMU 수록, 숨은 단어가 한 단어인 항목.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from candidates import table, vocab
from data import RESULTS, semeval_items
from g2p import in_cmu, prons

KINDS = ["unit", "feature", "feature@0.5", "feature@0.25"]
KS = [10, 50, 100, 200, 500]


def main():
    words, zipf = vocab()
    widx = {w: i for i, w in enumerate(words)}
    tab = table()
    items = [i for i in semeval_items(require_single_word=True) if in_cmu(i.w_p) and in_cmu(i.w_a)]
    rows = []
    for it in items:
        r = dict(id=it.id, w_p=it.w_p, w_a=it.w_a, in_vocab=it.w_a in widx)
        for kind in KINDS:
            d, dn = tab.distances(prons(it.w_p), kind)
            dn = dn.copy()
            if it.w_p in widx:
                dn[widx[it.w_p]] = np.inf
                d = d.copy(); d[widx[it.w_p]] = np.inf
            if r["in_vocab"]:
                a = widx[it.w_a]
                order = np.lexsort((-zipf, dn))
                r[f"rank_{kind}"] = int(np.where(order == a)[0][0]) + 1
                r[f"d_{kind}"] = float(d[a])
                if kind == "unit":
                    for rad in (1, 2, 3):
                        r[f"n_radius{rad}"] = int((d <= rad).sum())
            else:
                r[f"rank_{kind}"] = np.inf
                r[f"d_{kind}"] = np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    out = dict(n=len(df), vocab=len(words), in_vocab=float(df.in_vocab.mean()))
    for kind in KINDS:
        out[kind] = {f"recall@{k}": float((df[f"rank_{kind}"] <= k).mean()) for k in KS}
    u = df[df.in_vocab]
    out["unit_distance_hist"] = {str(k): int(v) for k, v in
                                 u.d_unit.clip(upper=3).value_counts().sort_index().items()}
    for rad in (1, 2, 3):
        hit = (df.d_unit <= rad)
        out[f"radius{rad}"] = dict(recall=float(hit.mean()),
                                   median_candidates=float(df[f"n_radius{rad}"].median()),
                                   p90_candidates=float(df[f"n_radius{rad}"].quantile(0.9)))
    (RESULTS / "precheck").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "precheck" / "recall_items.csv", index=False)
    (RESULTS / "precheck" / "recall.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
