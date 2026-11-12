"""E5 A 채점 (파이프라인 6단계). 위치를 찾은 생성물마다 π_→(전역), π_←(읽는 순간), 국소 문맥 점수와 He S_ratio.

- 청자가 듣는 단어는 문장에 쓰인 형태(pun_word)다. 후보 C = 그 단어의 발음 상위 100개 + w_a(강제 포함).
- γ와 A_→ 문턱은 e5_gate.py 결과(results/<A>/e5_gate)를 그대로 쓴다. 표제어 합 규칙은 PREREG와 같다.
- 관문 = T̂_→(표제어 묶음 1위)가 w_a 표제어 ∧ A_→ ≥ 문턱. 고르기 점수 = log F_→ + S_← (1:1 고정).
- 사람 말장난(상한선, SemEval 40개)도 같은 방식으로 채점한다(탐색 비교용).
채점할 수 없는 것: 위치 못 찾음, w_p/w_a가 한 단어(영문자)가 아님, pun_word와 w_a의 표제어가 같음.

사용: python -X utf8 experiments/e5_score.py --stage pilot --gen olmo-2-1124-7b-instruct --model pythia-1.4b --device cuda
출력: results/e5/<stage>/<gen>/A_<model>/ scores.csv, config.json  (LM 점수 캐시: cache/lm/<model>/e5_<stage>_<gen>)
"""
import argparse
import json

from v1_common import *  # noqa: F403
from baselines import lm_baselines
from candidates import candidates
from data import RESULTS, Item, match_case
from e5 import find_word, load_samples, stage_dir, write_config
from lmscore import LMScorer, Progress, score_item, set_threads, split_at
from rules import lemmas

K = 100


def items_to_score(cand: pd.DataFrame) -> list[Item]:
    df = cand[cand.start.notna() & cand.w_p.fillna("").str.isalpha() & cand.w_a.fillna("").str.isalpha()]
    out = []
    for r in df.itertuples():
        s, e = int(r.start), int(r.end)
        out.append(Item(id=r.cand_id, text=r.text, start=s, end=e, w_p=r.text[s:e].lower(), w_a=r.w_a.lower()))
    lem = lemmas({w for it in out for w in (it.w_p, it.w_a)})
    return [it for it in out if lem[it.w_p] != lem[it.w_a]]


def human_rows() -> pd.DataFrame:
    rows = []
    for h in load_samples()["human"]:
        s, e = find_word(h["text"], h["pun_word"])
        rows.append(dict(cand_id=f"H#{h['id']}", source="H", text=h["text"], pun_word=h["pun_word"], w_p=h["w_p"],
                         w_a=h["w_a"], format=h["format"], start=s, end=e, rule_ok=True))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot", "main"])
    ap.add_argument("--gen", required=True, help="생성기 폴더 이름 (results/e5/<stage>/ 아래)")
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    gdir = stage_dir(a.stage) / a.gen
    gate = json.loads((RESULTS / a.model / "e5_gate" / "metrics.json").read_text(encoding="utf-8"))
    gamma, thr = gate["gamma"], gate["A_threshold"]
    cand = pd.concat([pd.read_csv(gdir / "candidates.csv"), human_rows()], ignore_index=True)
    its = items_to_score(cand)
    out = gdir / f"A_{a.model}"
    write_config(out, dict(stage=a.stage, gen=a.gen, model=a.model, K=K, gamma=gamma, A_threshold=thr,
                           gate_source=f"results/{a.model}/e5_gate", n_scorable=len(its)))

    # ---- 후보표와 LM 점수 (이어서 만들기)
    ct_path = out / "cand_table.parquet"
    ct = pd.read_parquet(ct_path) if ct_path.exists() else pd.DataFrame()
    have = set(ct.key) if len(ct) else set()
    new = []
    for it in its:
        if it.id not in have:
            t = candidates(it.w_p, K, it.w_a)
            t.insert(0, "key", it.id)
            new.append(t)
    if new:
        ct = pd.concat([ct] + new, ignore_index=True)
        ct.to_parquet(ct_path, index=False)
    by_key = {k: list(g.cand) for k, g in ct.groupby("key", sort=False)}
    wr = CacheWriter(CACHE / "lm" / a.model / f"e5_{a.stage}_{a.gen}")
    todo = [it for it in its if it.id not in wr.done]
    if todo:
        set_threads()
        sc = LMScorer(a.model, a.device)
        prog = Progress(len(todo), f"{a.model} E5 {a.stage}")
        for it in todo:
            cs = by_key[it.id]
            rows = [dict(ctx=c, cand=w, logprob=lp, ntok=n) for c, w, lp, n in score_item(sc, it, cs)]
            prefix, lead = split_at(it)
            res, _ = sc.continuation(prefix, [lead + match_case(it.surface, w) for w in cs])
            rows += [dict(ctx="left", cand=w, logprob=lp, ntok=n) for w, (lp, n) in zip(cs, res)]
            wr.add(it.id, rows)
            prog.step()
        wr.flush()

    # ---- 지표
    lm = lm_cache(a.model, f"e5_{a.stage}_{a.gen}")
    c2 = add_lemma_cols(ct[ct.key.isin([it.id for it in its])])
    c2 = c2[~c2.same_lemma]
    spec = ("gamma", gamma)
    m = {ctx: Listener(scored(c2, lm, ctx), "C+wa").metrics(spec) for ctx in ("global", "left", "local")}
    X = pd.DataFrame(dict(rank_g=m["global"]["rank"], A_g=m["global"].A, logF_g=m["global"].logF,
                          logpi_wa_g=m["global"].logpi_wa, logD_g=m["global"].logD, L_g=m["global"].L,
                          S_left=m["left"].S, logF_left=m["left"].logF, logF_local=m["local"].logF,
                          logD_local=m["local"].logD))
    X = X.join(lm_baselines(its, c2, lm)[["S_ratio", "sent_lp"]])
    X["gate"] = (X.rank_g == 1) & (X.A_g >= thr)
    X["score"] = X.logF_g + X.S_left
    X.index.name = "cand_id"
    X.to_csv(out / "scores.csv")
    src = cand.set_index("cand_id").source.reindex(X.index)
    print(f"채점 {len(X)}개 / 대상 {len(cand)}개.  관문 통과율(출처별):")
    print(X.groupby(src).gate.agg(["size", "mean"]).to_string())


if __name__ == "__main__":
    main()
