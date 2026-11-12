"""E5 고르기 (파이프라인 7단계)와 자동 지표.

P1~P4는 같은 칸의 같은 후보 10개 중 규칙을 통과한 것에서 고른다(통과한 것이 없으면 위치를 찾은 후보, 그것도 없으면 전부).
  P1 무작위 1개 (칸마다 정해진 시드)
  P2 관문 통과 후보 중 점수(log F_→ + S_←) 1위. 통과가 없으면 점수 1위 + "관문 실패" 표시
  P3 같은 생성기가 고른 것 (e5_generate.py의 picks.csv)
  P4 He S_ratio 1위 (선택 조건)
B1, B0은 생성한 그대로. 사람 말장난(H)은 형식이 같은 칸에 붙인다(e5.human_for_cells).

자동 지표: 규칙 통과율·탈락 이유, A 관문 통과율(조건별), "관문 실패" 비율, 후보 다양성(칸 안 distinct-2,
문장 틀 반복률 = 첫 세 단어가 같은 칸의 다른 후보와 겹치는 비율). A 관문 통과율은 P2를 A로 골랐으므로
P2에 유리하게 기울어 있다(성공 판정은 사람·다른 계열 LLM으로 한다).

사용: python -X utf8 experiments/e5_select.py --stage pilot --gen olmo-2-1124-7b-instruct --model pythia-1.4b
출력: results/e5/<stage>/<gen>/A_<model>/ selections.csv, metrics.json, table.md
"""
import argparse
import re

from v1_common import *  # noqa: F403
from e5 import CONDITIONS, cells, human_for_cells, stable_seed, stage_dir, write_config
from rules import dump


def distinct2(texts) -> float:
    grams = [g for t in texts for g in zip(*(lambda w: (w, w[1:]))(re.findall(r"[a-z']+", t.lower())))]
    return len(set(grams)) / len(grams) if grams else np.nan


def template_rate(texts) -> float:
    heads = [" ".join(re.findall(r"[a-z']+", t.lower())[:3]) for t in texts]
    s = pd.Series(heads)
    return float(s.duplicated(keep=False).mean()) if len(s) else np.nan


def select(stage, C, cand, X, picks) -> pd.DataFrame:
    X = X.reindex(cand.cand_id)
    cand = cand.assign(gate=X.gate.to_numpy(), score=X.score.to_numpy(), S_ratio=X.S_ratio.to_numpy())
    pk = picks.set_index("cell").cand_id.to_dict() if len(picks) else {}
    hum = human_for_cells(C)
    rows = []
    for c in C:
        P = cand[(cand.cell == c["cell"]) & (cand.source == "P")]
        pool, kind = P[P.rule_ok], "rule_ok"
        if pool.empty:
            pool, kind = P[P.start.notna()], "located"
        if pool.empty:
            pool, kind = P, "all"
        base = dict(stage=stage, cell=c["cell"], pair_id=c["pair_id"], format=c["format"], w_p=c["w_p"], w_a=c["w_a"],
                    pool=kind, n_pool=len(pool), n_cand=len(P))
        if len(pool):
            rng = np.random.default_rng(stable_seed(stage, "P1", c["cell"]))
            rows.append(dict(base, condition="P1", cand_id=pool.cand_id.iloc[int(rng.integers(len(pool)))]))
            g = pool[pool.gate.fillna(False).astype(bool)]
            sc = (g if len(g) else pool).dropna(subset=["score"])
            if len(sc):
                rows.append(dict(base, condition="P2", cand_id=sc.loc[sc.score.idxmax(), "cand_id"], gate_fail=not len(g)))
            else:
                rows.append(dict(base, condition="P2", cand_id=rows[-1]["cand_id"], gate_fail=True, note="점수 없음: P1과 같음"))
            if isinstance(pk.get(c["cell"]), str):
                rows.append(dict(base, condition="P3", cand_id=pk[c["cell"]]))
            sr = pool.dropna(subset=["S_ratio"])
            if len(sr):
                rows.append(dict(base, condition="P4", cand_id=sr.loc[sr.S_ratio.idxmax(), "cand_id"]))
        b1 = cand[(cand.cell == c["cell"]) & (cand.source == "B1")]
        if len(b1):
            rows.append(dict(base, condition="B1", cand_id=b1.cand_id.iloc[0]))
        if c["cell"] in hum:
            rows.append(dict(base, condition="H", cand_id=f"H#{hum[c['cell']]['id']}"))
    for r in cand[cand.source == "B0"].itertuples():
        rows.append(dict(stage=stage, cell=None, pair_id=None, format=None, w_p=r.w_p, w_a=r.w_a, condition="B0",
                         cand_id=r.cand_id))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot", "main"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--model", default="pythia-1.4b")
    a = ap.parse_args()
    gdir = stage_dir(a.stage) / a.gen
    adir = gdir / f"A_{a.model}"
    cand = pd.read_csv(gdir / "candidates.csv")
    from e5_score import human_rows
    cand = pd.concat([cand, human_rows()], ignore_index=True)
    X = pd.read_csv(adir / "scores.csv").set_index("cand_id")
    picks = pd.read_csv(gdir / "picks.csv") if (gdir / "picks.csv").exists() else pd.DataFrame()
    present = set(cand.cell.dropna())
    C = [c for c in cells(a.stage) if c["cell"] in present]
    S = select(a.stage, C, cand, X, picks)
    info = cand.set_index("cand_id")
    S["text"] = S.cand_id.map(info.text)
    S["pun_word"] = S.cand_id.map(info.pun_word)
    S["rule_ok"] = S.cand_id.map(info.rule_ok)
    for col in ("gate", "score", "A_g", "rank_g", "S_left", "logF_g", "S_ratio"):
        S[col] = S.cand_id.map(X[col])
    S.to_csv(adir / "selections.csv", index=False)
    write_config(adir, dict(stage=a.stage, gen=a.gen, model=a.model, step="select", n_cells=len(C)))

    # ---- 자동 지표
    res = {}
    P = cand[cand.source == "P"]
    res["candidates"] = dict(
        n_cells=len(C), n_cand=len(P), cand_per_cell=float(P.groupby("cell").size().reindex([c["cell"] for c in C]).fillna(0).mean()),
        rule_pass=float(P.rule_ok.mean()), reasons=P.reasons.dropna().str.split(",").explode().value_counts().to_dict(),
        gate_pass_all=float(X.gate.reindex(P.cand_id).fillna(False).astype(bool).mean()),
        gate_pass_rule_ok=float(X.gate.reindex(P[P.rule_ok].cand_id).fillna(False).astype(bool).mean()),
        distinct2=float(np.nanmean([distinct2(g.text) for _, g in P.groupby("cell")])),
        template_rate=float(np.nanmean([template_rate(g.text) for _, g in P.groupby("cell")])))
    for src in ("B1", "B0"):
        b = cand[cand.source == src]
        res[src] = dict(n=len(b), rule_pass=float(b.rule_ok.mean()) if len(b) else np.nan,
                        reasons=b.reasons.dropna().str.split(",").explode().value_counts().to_dict())
    by = {}
    for (cond, fmt), g in S.groupby([S.condition, S.format.fillna("-")]):
        by.setdefault(cond, {})[fmt] = dict(n=len(g), rule_pass=float(g.rule_ok.astype(float).mean()),
                                            gate_pass=float(g.gate.fillna(False).astype(bool).mean()),
                                            scored=float(g.score.notna().mean()))
    for cond, g in S.groupby("condition"):
        by[cond]["all"] = dict(n=len(g), rule_pass=float(g.rule_ok.astype(float).mean()),
                               gate_pass=float(g.gate.fillna(False).astype(bool).mean()), scored=float(g.score.notna().mean()))
    res["by_condition"] = by
    p2 = S[S.condition == "P2"]
    res["P2_gate_fail"] = float(p2.gate_fail.astype(float).mean()) if len(p2) else np.nan
    same = S.pivot_table(index="cell", columns="condition", values="cand_id", aggfunc="first")
    res["same_pick"] = {f"{x}={y}": float((same[x] == same[y]).mean()) for x, y in
                        (("P1", "P2"), ("P2", "P3"), ("P2", "P4"), ("P1", "P3")) if x in same and y in same}
    dump(adir / "metrics.json", res)
    c = res["candidates"]
    L = [f"# E5 자동 지표 — {a.stage}, 생성기 {a.gen}, A = {a.model}", "",
         f"칸 {c['n_cells']}개, 후보 {c['n_cand']}개 (칸당 {c['cand_per_cell']:.1f}). 규칙 통과 {c['rule_pass']:.3f}, "
         f"관문 통과(전체 {c['gate_pass_all']:.3f}, 규칙 통과 후보 중 {c['gate_pass_rule_ok']:.3f}). "
         f"distinct-2 {c['distinct2']:.3f}, 문장 틀 반복 {c['template_rate']:.3f}", "",
         f"탈락 이유(P 후보): {c['reasons']}  /  B1: {res['B1']['reasons']}  /  B0: {res['B0']['reasons']}", "",
         f"P2 관문 실패 비율 {res['P2_gate_fail']:.3f}.  같은 후보를 고른 비율: "
         + ", ".join(f"{k} {v:.2f}" for k, v in res["same_pick"].items()), "",
         "| 조건 | 형식 | n | 규칙 통과 | A 관문 통과 |", "|---|---|---|---|---|"]
    for cond in CONDITIONS + ["H"]:
        for fmt, v in by.get(cond, {}).items():
            L.append(f"| {cond} | {fmt} | {v['n']} | {v['rule_pass']:.3f} | {v['gate_pass']:.3f} |")
    R_lines = L
    (adir / "table.md").write_text("\n".join(R_lines) + "\n", encoding="utf-8")
    print("\n".join(R_lines))


if __name__ == "__main__":
    main()
