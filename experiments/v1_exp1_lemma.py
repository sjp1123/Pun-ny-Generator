"""실험 1-B: 표제어 합 채점의 dev 점검.

- lemma(w_p) = lemma(w_a)인 제외 항목 수
- dev E1 acc@1: strict / 표제어 최댓값 / 표제어 합 (단일 γ, 단순 편집 거리, dev에서 맞춤)
- strict와 표제어 합의 판정이 갈리는 dev 항목 20개(무작위) → 손 점검표
"""
import argparse

from v1_common import *  # noqa: F403
from rules import RunDir, e1_scores, fit_gamma, lemmas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m")
    a = ap.parse_args()
    D = Data(a.model, "dev")
    g = D.listener("global", restrict="C")
    gam = fit_gamma(g)
    res = dict(n_dev_single=len(D.items), excluded_same_lemma=D.excluded, n_used=len(D.keys), gamma=gam)
    ms = {m: g.metrics(("gamma", gam), mode=m) for m in ("strict", "max", "sum")}
    res["E1_dev"] = {m: e1_scores(v["rank"]) for m, v in ms.items()}

    hit = {m: ms[m]["rank"].fillna(99) <= 1 for m in ms}
    diff = hit["strict"] != hit["sum"]
    keys = np.array(ms["sum"].index[diff])
    rng = np.random.default_rng(SEED)
    pick = rng.choice(keys, size=min(20, len(keys)), replace=False)
    res["n_disagree"] = int(diff.sum())

    df = g.df
    lp = pd.Series(g.logpi(("gamma", gam)), index=df.index)
    rows = []
    for k in pick:
        sub = df[(df.key == k) & ~df.is_wp].assign(pi=np.exp(lp[df.key == k][~df[df.key == k].is_wp]))
        grp = sub.groupby("lemma").pi.sum().sort_values(ascending=False)
        top = sub.sort_values("pi", ascending=False).head(3)
        members = sub[sub.in_M].sort_values("pi", ascending=False)
        it = D.items[k]
        rows.append(dict(
            key=k, text=it.text, w_p=it.w_p, w_a=it.w_a, target_lemma=sub.target_lemma.iloc[0],
            strict_hit=bool(hit["strict"][k]), sum_hit=bool(hit["sum"][k]),
            top_candidates=", ".join(f"{r.cand}({r.lemma}) {r.pi:.3f}" for r in top.itertuples()),
            top_groups=", ".join(f"{l} {p:.3f}" for l, p in grp.head(3).items()),
            M_members=", ".join(f"{r.cand} {r.pi:.3f}" for r in members.itertuples()),
            tool_error="", note=""))
    R = RunDir(a.model, "exp1B_lemma", dict(split="dev", gamma=gam, dist="d_unit", restrict="C"))
    pd.DataFrame(rows).to_csv(R.dir / "hand_check_20.csv", index=False, encoding="utf-8-sig")
    R.metrics(res)
    L = [f"# 1-B 표제어 합 채점 dev 점검 — {a.model} (dev 점검)", "",
         f"dev 한 단어 항목 {len(D.items)}개 중 lemma(w_p)=lemma(w_a)로 제외 {len(D.excluded)}개: {D.excluded}", "",
         f"γ (dev, 단순 편집 거리) = {gam}", "", "| 채점 | acc@1 | acc@5 | MRR |", "|---|---|---|---|"]
    for m, v in res["E1_dev"].items():
        L.append(f"| {m} | {v['acc1']:.3f} | {v['acc5']:.3f} | {v['mrr']:.3f} |")
    L += ["", f"strict와 표제어 합의 판정이 갈린 항목 {res['n_disagree']}개 중 20개를 손 점검했다 (hand_check_20.csv).", ""]
    for r in rows:
        L.append(f"- **{r['key']}** {r['w_p']}→{r['w_a']} (목표 {r['target_lemma']}; strict {r['strict_hit']}, 합 {r['sum_hit']})")
        L.append(f"  - 상위 후보: {r['top_candidates']}")
        L.append(f"  - 상위 묶음: {r['top_groups']} / M: {r['M_members']}")
    R.table("table.md", L)


if __name__ == "__main__":
    main()
