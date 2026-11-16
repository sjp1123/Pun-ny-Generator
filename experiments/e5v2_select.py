"""E5 v2 고르기 (docs/E5_V2_DESIGN.md 6절). P1~P4는 모두 같은 규칙 통과 후보 집합에서 고른다(대체 규칙 없음).

  P1  고정 시드 무작위 1개
  P2  관문 통과 후보 중 점수(log F_→ + S_←) 1위. 통과가 없으면 점수 1위 + gate_fail
  P3  같은 생성기가 고른 것 (e5v2_generate.py의 picks.csv, 보여 준 순서는 고정 시드로 섞음)
  P4  (탐색) He S_ratio 1위, 동점은 고정 시드 무작위. 후보가 모두 −1이면 undefined
  후보가 0개인 칸: P1~P4 모두 generation_failed (cand_id 없음)
  B1, B0은 생성한 그대로(규칙 통과 여부 rule_ok를 함께 적음). H는 형식이 같은 칸에 붙인다(자기 쌍 그대로).

사용: python -X utf8 experiments/e5v2_select.py --stage pilot4 --gen olmo-2-0325-32b-instruct-4bit --model pythia-1.4b
출력: results/e5/<stage>/<gen>/A_<model>/ selections.csv, selections.jsonl, metrics.json, table.md
"""
import argparse

from v1_common import *  # noqa: F403
from e5 import human_for_cells, stable_seed, stage_dir, write_config
from e5v2 import cells_v2
from rules import dump

P_CONDS = ["P1", "P2", "P3", "P4"]


def pick_max_random(vals: pd.Series, seed: int):
    """최댓값 후보 중 고정 시드로 하나 (동점 처리)."""
    top = vals[vals == vals.max()].index.tolist()
    return top[int(np.random.default_rng(seed).integers(len(top)))]


def select_v2(stage, C, cand, X, picks) -> pd.DataFrame:
    X = X.reindex(cand.cand_id)
    cand = cand.assign(gate=X.gate.to_numpy(), score=X.score.to_numpy(), S_ratio=X.S_ratio.to_numpy())
    pk = picks.set_index("cell").cand_id.to_dict() if len(picks) else {}
    hum = human_for_cells(C)
    rows = []
    for c in C:
        pool = cand[(cand.cell == c["cell"]) & (cand.source == "P") & cand.rule_ok.astype(bool)]
        base = dict(stage=stage, cell=c["cell"], pair_id=c["pair_id"], format=c["format"], w_p=c["w_p"], w_a=c["w_a"],
                    n_pool=len(pool), underfilled=len(pool) < 10)
        if pool.empty:
            rows += [dict(base, condition=k, cand_id=None, generation_failed=True) for k in P_CONDS]
        else:
            rng = np.random.default_rng(stable_seed(stage, "P1", c["cell"]))
            rows.append(dict(base, condition="P1", cand_id=pool.cand_id.iloc[int(rng.integers(len(pool)))]))
            g = pool[pool.gate.fillna(False).astype(bool)]
            sc = (g if len(g) else pool).dropna(subset=["score"])
            rows.append(dict(base, condition="P2", gate_fail=not len(g),
                             cand_id=sc.loc[sc.score.idxmax(), "cand_id"] if len(sc) else None,
                             note=None if len(sc) else "점수 없음"))
            p3 = pk.get(c["cell"])
            rows.append(dict(base, condition="P3", cand_id=p3 if isinstance(p3, str) else None,
                             note=None if isinstance(p3, str) else "고르기 응답 없음"))
            sr = pool.set_index("cand_id").S_ratio.dropna()
            if len(sr) and (sr > -1).any():
                rows.append(dict(base, condition="P4", undefined=False,
                                 cand_id=pick_max_random(sr, stable_seed(stage, "P4", c["cell"]))))
            else:
                rows.append(dict(base, condition="P4", cand_id=None, undefined=True))
        b1 = cand[(cand.cell == c["cell"]) & (cand.source == "B1")]
        if len(b1):
            rows.append(dict(base, condition="B1", cand_id=b1.cand_id.iloc[0]))
        if c["cell"] in hum:
            h = hum[c["cell"]]
            rows.append(dict(base, condition="H", cand_id=f"H#{h['id']}", w_p=h["w_p"], w_a=h["w_a"], n_pool=None,
                             underfilled=None))
    for r in cand[cand.source == "B0"].itertuples():
        rows.append(dict(stage=stage, cell=None, pair_id=None, format=None, w_p=r.w_p, w_a=r.w_a, condition="B0",
                         cand_id=r.cand_id))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot2", "pilot3", "pilot4", "main"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--model", default="pythia-1.4b")
    a = ap.parse_args()
    gdir = stage_dir(a.stage) / a.gen
    adir = gdir / f"A_{a.model}"
    from e5_score import human_rows
    cand = pd.concat([pd.read_csv(gdir / "candidates.csv"), human_rows()], ignore_index=True)
    X = pd.read_csv(adir / "scores.csv").set_index("cand_id")
    picks = pd.read_csv(gdir / "picks.csv") if (gdir / "picks.csv").exists() else pd.DataFrame()
    cells_done = set(pd.read_csv(gdir / "cells.csv").cell)
    C = [c for c in cells_v2(a.stage) if c["cell"] in cells_done]
    S = select_v2(a.stage, C, cand, X, picks)
    info = cand.set_index("cand_id")
    for col in ("text", "pun_word", "rule_ok", "reasons"):
        S[col] = S.cand_id.map(info[col])
    for col in ("gate", "score", "A_g", "rank_g", "S_left", "logF_g", "S_ratio"):
        S[col] = S.cand_id.map(X[col])
    S.to_csv(adir / "selections.csv", index=False)
    S.to_json(adir / "selections.jsonl", orient="records", lines=True, force_ascii=False)
    write_config(adir, dict(stage=a.stage, gen=a.gen, model=a.model, step="select_v2", n_cells=len(C)))

    res = {}
    for cond, g in S.groupby("condition"):
        has = g.cand_id.notna()
        res[cond] = dict(n=len(g), n_selected=int(has.sum()), rule_pass=float(g[has].rule_ok.astype(float).mean()),
                         gate_pass=float(g[has].gate.fillna(False).astype(bool).mean()))
    p = S[S.condition.isin(P_CONDS)]
    res["generation_failed_cells"] = int(p[p.condition == "P1"].get("generation_failed", pd.Series(dtype=bool))
                                         .fillna(False).astype(bool).sum())
    res["P2_gate_fail"] = float(S[S.condition == "P2"].gate_fail.fillna(False).astype(float).mean())
    res["P4_undefined"] = int(S[S.condition == "P4"].undefined.fillna(False).astype(bool).sum())
    dump(adir / "metrics.json", res)
    L = [f"# E5 v2 고르기 — {a.stage}, 생성기 {a.gen}, A = {a.model}", "",
         f"칸 {len(C)}개, 생성 실패 칸 {res['generation_failed_cells']}, P2 gate_fail {res['P2_gate_fail']:.3f}, "
         f"P4 undefined {res['P4_undefined']}칸", "", "| 조건 | n | 고름 | 규칙 통과 | 관문 통과 |", "|---|---|---|---|---|"]
    for cond in ["B0", "B1", "P1", "P2", "P3", "P4", "H"]:
        if cond in res:
            v = res[cond]
            L.append(f"| {cond} | {v['n']} | {v['n_selected']} | {v['rule_pass']:.3f} | {v['gate_pass']:.3f} |")
    (adir / "table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
