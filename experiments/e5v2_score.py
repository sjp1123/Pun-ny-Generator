"""E5 v2 A 채점 (docs/E5_V2_DESIGN.md 6절). 후보마다 ③(최종)과 ②(S3 전, 진단 D1용)를 같은 방식으로 채점한다.

- 청자 Pythia-1.4B 베이스, 후보 C+wa(말장난 단어의 발음 상위 100 + w_a), γ와 A_→ 문턱은 파일럿 1의 e5_gate 값
  (γ = 1.0, 문턱 0.00033). 둘 다 C+wa로 맞춘 값이라 후보 집합도 C+wa를 쓴다(PREREG_E5 6절).
- 표제어 합, π_→(문장 전체), π_←(왼쪽 문맥만). 관문 = T̂_→ = w_a 표제어 ∧ A_→ ≥ 문턱.
- 점수 = log F_→ + S_←, S_← = −log π_←(w_p) (e5_score.py의 S_left와 같은 정의).
- B1·B0·사람 말장난(H)도 채점한다(위치를 찾은 것만).

사용: python -X utf8 experiments/e5v2_score.py --stage pilot4 --gen olmo-2-0325-32b-instruct-4bit --model pythia-1.4b --device cuda
출력: results/e5/<stage>/<gen>/A_<model>/ scores.csv, candidates.jsonl, config.json
"""
import argparse
import json

from v1_common import *  # noqa: F403
from data import RESULTS
from e5 import stage_dir, write_config
from e5_score import K, human_rows, items_to_score, score_items, top_lemmas

D2 = "@2"  # ② 항목 id 꼬리표


def scoring_frame(cand: pd.DataFrame) -> pd.DataFrame:
    """③·B1·B0·H는 그대로, P 후보의 ②는 id에 @2를 붙여 한 표로."""
    base = cand[["cand_id", "source", "text", "start", "end", "w_p", "w_a"]]
    p = cand[cand.source == "P"]
    two = pd.DataFrame(dict(cand_id=p.cand_id + D2, source="P@2", text=p.text2, start=p.start2, end=p.end2,
                            w_p=p.w_p, w_a=p.w_a))
    return pd.concat([base, two], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot2", "pilot3", "pilot4", "pilot5", "pilot6", "main"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    gdir = stage_dir(a.stage) / a.gen
    gate = json.loads((RESULTS / a.model / "e5_gate" / "metrics.json").read_text(encoding="utf-8"))
    gamma, thr = gate["gamma"], gate["A_threshold"]
    cand = pd.read_csv(gdir / "candidates.csv")
    H = human_rows()[["cand_id", "source", "text", "start", "end", "w_p", "w_a"]]
    frame = pd.concat([scoring_frame(cand), H], ignore_index=True)
    its = items_to_score(frame)
    out = gdir / f"A_{a.model}"
    write_config(out, dict(stage=a.stage, gen=a.gen, model=a.model, K=K, candidate_set="C+wa", gamma=gamma,
                           A_threshold=thr, gate_source=f"results/{a.model}/e5_gate", n_scorable=len(its),
                           score="logF_g + S_left, S_left = -log pi_left(w_p)"))
    X, c2, lm = score_items(its, a.model, f"e5v2_{a.stage}_{a.gen}", out, a.device, gamma, thr)
    X.to_csv(out / "scores.csv")

    # candidates.jsonl: P 후보마다 ②·③의 π 상위 5개, T̂_→, A_→, F_→, S_←, L, 관문
    top = {ctx: top_lemmas(c2, lm, ctx, gamma) for ctx in ("global", "left")}
    lines = []
    for r in cand[cand.source == "P"].itertuples():
        rec = dict(cand_id=r.cand_id, cell=r.cell, pair_id=r.pair_id, format=r.format, w_p=r.w_p, w_a=r.w_a,
                   text1=r.text1, text2=r.text2, text3=r.text, n_edit=r.n_edit, edited=r.edited, hint_used=r.hint_used)
        for tag, cid in (("s3", r.cand_id), ("s2", r.cand_id + D2)):
            x = X.loc[cid] if cid in X.index else None
            g = top["global"].get(cid, [])
            rec[tag] = None if x is None else dict(
                pi_global_top5=g, pi_left_top5=top["left"].get(cid, []),
                T_hat=next((lem for lem, _, wp in g if not wp), None), rank_wa=x.rank_g, A=x.A_g, logF=x.logF_g,
                S_left=x.S_left, L=x.L_g, S_ratio=x.S_ratio, gate=bool(x.gate), score=x.score)
        lines.append(json.dumps(rec, ensure_ascii=False, default=float))
    (out / "candidates.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    src = frame.set_index("cand_id").source.reindex(X.index)
    print(f"채점 {len(X)}개.  관문 통과율(출처별):")
    print(X.groupby(src).gate.agg(["size", "mean"]).to_string())


if __name__ == "__main__":
    main()
