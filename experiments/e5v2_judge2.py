"""E5 v2 LLM 판정 2판: 사람 자료로 검증한 판정기와 질문만 쓴다 (docs/JUDGE_VALIDATION.md, PREREG_E5 6.9).

e5v2_judge.py(Qwen2.5-7B: 성립·w_a 맞히기·웃김 짝 비교)는 세 질문 모두 검증에서 실패했다. 바뀐 점:
  판정기  microsoft/phi-4 (NF4 4비트). 생성기(OLMo)와 계열이 다르고 성립·웃김 점수 검증을 통과했다
  성립    쌍(w_p, w_a)을 알려 주고 "성공한 말장난인가" yes/no — 검증한 프롬프트 e5/judge_val/prompts/works_pair.txt 그대로
          대상: 칸의 모든 P 후보의 ③(완성본)과 ②(단어만 바꿈, S3의 효과 진단) + 고른 B1·B0·H
  웃김    1~5점 — funny_point.txt 그대로. 대상: 모든 P 후보의 ③ + 고른 B1·B0·H. 짝 비교와 BT는 쓰지 않는다
  w_a 맞히기(옛 "성공")는 묻지 않는다
모두 탐색 지표다. 검증이 보인 것은 He et al.의 기계 생성 문장에서 사람 한 명 수준의 성립 판정(균형 정확도 0.67)과
사람 말장난에서의 웃김 순위(ρ 0.33)뿐이고, E5 생성물에서의 정확도는 사람 라벨 없이는 모른다.

사용: python -X utf8 experiments/e5v2_judge2.py --stage pilot4 --gen olmo-2-0325-32b-instruct-4bit
      (CPU 시험) --backend ollama --judge qwen3.5:4b --think false --no-4bit --limit 2
출력: results/e5/<stage>/<gen>/A_<model>/judge2_<판정기>/ calls.jsonl, items.csv, by_condition.csv, by_pair.csv, metrics.json, table.md
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from e5 import stage_dir, write_config
from e5_gen import Generator, tag_of
from judge_validate import PROMPTS, Judge, one_to_five, vprompt, yes_no
from rules import dump

CONDS = ["B0", "B1", "P1", "P2", "P3", "P4", "H"]
SEED, B_BOOT = 20261117, 2000


def ok_str(x) -> bool:
    return isinstance(x, str) and bool(x.strip())


def targets(cand: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """판정할 문장: 모든 P 후보의 ③과 ②, 고른 B1·B0·H. 쌍이나 문장이 없으면(B0의 잘못된 쌍) 판정하지 않는다."""
    picked = S[S.condition.isin(["B1", "B0", "H"])].cand_id.dropna()
    ids = list(dict.fromkeys(list(cand[cand.source == "P"].index) + picked.tolist()))
    rows = []
    for cid in ids:
        r = cand.loc[cid]
        judgeable = ok_str(r.text) and ok_str(r.pun_word) and ok_str(r.w_a) and str(r.pun_word).lower() != str(r.w_a).lower()
        rows.append(dict(cand_id=cid, source=r.source, cell=r.get("cell"), pair_id=r.get("pair_id"), judgeable=judgeable,
                         text=r.text, pun_word=r.pun_word, w_a=r.w_a,
                         text2=r.get("text2") if r.source == "P" else None,
                         pun_word2=r.get("pun_word2") if r.source == "P" else None))
    return pd.DataFrame(rows)


def judge_items(J: Judge, T: pd.DataFrame) -> pd.DataFrame:
    works, works2, funny = [], [], []
    for r in T.itertuples():
        if not r.judgeable:
            works.append(None), works2.append(None), funny.append(one_to_five(
                J.ask(f"funny|{r.cand_id}", vprompt("funny_point", text=r.text), max_new=6)) if ok_str(r.text) else None)
            continue
        works.append(yes_no(J.ask(f"works|{r.cand_id}", vprompt("works_pair", w_p=r.pun_word, w_a=r.w_a, text=r.text))))
        works2.append(yes_no(J.ask(f"works2|{r.cand_id}", vprompt("works_pair", w_p=r.pun_word2, w_a=r.w_a, text=r.text2)))
                      if ok_str(r.text2) and ok_str(r.pun_word2) else None)
        funny.append(one_to_five(J.ask(f"funny|{r.cand_id}", vprompt("funny_point", text=r.text), max_new=6)))
    return T.assign(works=works, works2=works2, funny=funny)[["cand_id", "source", "cell", "pair_id", "judgeable",
                                                              "works", "works2", "funny"]]


def cluster_ci(d: pd.DataFrame, col: str, cluster="pair_id") -> dict:
    """d[col]의 평균과 쌍 묶음 재표집 95% 구간."""
    d = d.dropna(subset=[col])
    if not len(d):
        return dict(est=np.nan, lo=np.nan, hi=np.nan, n=0)
    rng = np.random.default_rng(SEED)
    groups = [g[col].to_numpy(float) for _, g in d.groupby(cluster)]
    reps = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(B_BOOT)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return dict(est=float(d[col].astype(float).mean()), lo=float(lo), hi=float(hi), n=len(d))


def summarize(S: pd.DataFrame, I: pd.DataFrame, gate: pd.Series) -> dict:
    """조건별 성립·웃김, 칸 짝 차이(P2 − P1 등), 모든 P 후보의 관문별·②③ 비교, 쌍별 표."""
    it = I.set_index("cand_id")
    M = S.dropna(subset=["cand_id"])[["condition", "cell", "pair_id", "cand_id"]].copy()
    M["works"] = [it.works.get(c) for c in M.cand_id]
    M["works"] = M.works.map(lambda x: 0.0 if x is None or pd.isna(x) else float(x))  # 판정 못 한 것(쌍 없음)은 성립 아님
    M["funny"] = [it.funny.get(c) for c in M.cand_id]
    M["judged"] = [bool(it.judgeable.get(c, False)) for c in M.cand_id]
    res = dict(by_condition={}, diffs={}, all_P={})
    for cond in CONDS:
        g = M[M.condition == cond]
        if len(g):
            res["by_condition"][cond] = dict(n=len(g), judged=int(g.judged.sum()), works=float(g.works.mean()),
                                             funny=float(g.funny.mean()) if g.funny.notna().any() else None)
    wide = {c: M[M.condition == c].set_index("cell") for c in CONDS}
    for a, b in (("P2", "P1"), ("P2", "P3"), ("P1", "B1"), ("P2", "B1")):
        if not len(wide[a]) or not len(wide[b]):
            continue
        j = wide[a][["pair_id", "works", "funny"]].join(wide[b][["works", "funny"]], rsuffix="_b", how="inner")
        j["d_works"], j["d_funny"] = j.works - j.works_b, j.funny - j.funny_b
        res["diffs"][f"{a}-{b}"] = dict(works=cluster_ci(j, "d_works"), funny=cluster_ci(j, "d_funny"))
    P = I[(I.source == "P") & I.judgeable].copy()
    P["gate"] = P.cand_id.map(gate)
    for c in ("works", "works2", "funny"):
        P[c] = P[c].map(lambda x: np.nan if x is None or pd.isna(x) else float(x)).astype(float)  # True/False → 1.0/0.0
    P["d_s3"] = P.works - P.works2
    res["all_P"] = dict(n=len(P), works3=cluster_ci(P, "works"), works2=cluster_ci(P, "works2"), s3_effect=cluster_ci(P, "d_s3"),
                        funny=cluster_ci(P, "funny"),
                        works_gate_pass=cluster_ci(P[P.gate == True], "works"), works_gate_fail=cluster_ci(P[P.gate == False], "works"),  # noqa: E712
                        funny_gate_pass=cluster_ci(P[P.gate == True], "funny"), funny_gate_fail=cluster_ci(P[P.gate == False], "funny"),  # noqa: E712
                        unparsed=float(P.works.isna().mean()))
    bp = P.groupby("pair_id").agg(n=("works", "size"), gate=("gate", "mean"), works3=("works", "mean"),
                                  works2=("works2", "mean"), funny=("funny", "mean"))
    res["pairs_with_any_works"] = int((bp.works3 > 0).sum())
    res["pairs_all_works"] = int((bp.works3 == 1).sum())
    return res, M, bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot2", "pilot3", "pilot4", "pilot5", "main"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    ap.add_argument("--judge", default="microsoft/phi-4")
    ap.add_argument("--no-4bit", action="store_true", help="4비트로 올리지 않음 (검증은 NF4 4비트로 했다)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--think", default=None, choices=["true", "false"])
    ap.add_argument("--limit", type=int, default=None, help="시험용: 앞의 칸 몇 개만 (폴더에 -limitN)")
    a = ap.parse_args()
    gdir = stage_dir(a.stage) / a.gen
    adir = gdir / f"A_{a.model}"
    tag = tag_of(a.judge) + ("" if a.no_4bit else "-4bit") + (f"-limit{a.limit}" if a.limit else "")
    out = adir / f"judge2_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(adir / "selections.csv")
    from e5_score import human_rows
    cand = pd.concat([pd.read_csv(gdir / "candidates.csv"), human_rows()], ignore_index=True).set_index("cand_id")
    if a.limit:
        cells = list(dict.fromkeys(S.cell.dropna()))[: a.limit]
        S = S[S.cell.isin(cells)]
        cand = cand[cand.cell.isin(cells) | cand.index.isin(S.cand_id)]
    J = Judge(lambda: Generator(a.backend, a.judge, a.device, think=None if a.think is None else a.think == "true",
                                load_4bit=not a.no_4bit), out / "calls.jsonl")
    I = judge_items(J, targets(cand, S))
    I.to_csv(out / "items.csv", index=False)
    gate = pd.read_csv(adir / "scores.csv").set_index("cand_id").gate
    res, M, bp = summarize(S, I, gate)
    M.to_csv(out / "by_condition.csv", index=False)
    bp.round(3).to_csv(out / "by_pair.csv")
    sha = lambda n: hashlib.sha256((PROMPTS / f"{n}.txt").read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n")).hexdigest()
    info = J.gen.info if J.gen is not None else dict(backend=a.backend, model=a.judge, note="모두 기록에서 읽음")
    write_config(out, dict(stage=a.stage, gen=a.gen, model=a.model, judge=info, temperature=0.0, limit=a.limit,
                           prompts_sha256=dict(works_pair=sha("works_pair"), funny_point=sha("funny_point")),
                           validation="docs/JUDGE_VALIDATION.md (results/judge_val/phi-4-4bit)", calls_from_cache=J.gen is None))
    dump(out / "metrics.json", res)
    f = lambda v: "—" if v["n"] == 0 else f"{v['est']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"
    A = res["all_P"]
    L = [f"# E5 v2 LLM 판정 2판 (탐색) — {a.stage}, {a.gen}, 판정기 {a.judge}{'' if a.no_4bit else ' (NF4 4비트)'}", "",
         "성립 = 쌍을 알려 주고 \"성공한 말장난인가\"에 yes. 웃김 = 1~5점. 판정 못 한 문장(쌍이 없는 B0)은 성립 아님으로 센다.",
         "[ ]는 쌍 묶음 재표집 95% 구간.", "",
         "| 조건 | n | 판정함 | 성립 yes | 웃김 평균 |", "|---|---|---|---|---|"]
    for cond, v in res["by_condition"].items():
        L.append(f"| {cond} | {v['n']} | {v['judged']} | {v['works']:.3f} | {'—' if v['funny'] is None else format(v['funny'], '.2f')} |")
    L += ["", "| 칸 짝 차이 | 성립 | 웃김 |", "|---|---|---|"]
    L += [f"| {k} | {f(v['works'])} | {f(v['funny'])} |" for k, v in res["diffs"].items()]
    L += ["", f"## 모든 P 후보 (n = {A['n']})", "",
          f"- 성립 yes: ③ 완성본 {f(A['works3'])}, ② 단어만 바꿈 {f(A['works2'])}, 차이(S3의 효과) {f(A['s3_effect'])}",
          f"- 관문 통과 후보의 성립 {f(A['works_gate_pass'])} (n = {A['works_gate_pass']['n']}), "
          f"관문 실패 {f(A['works_gate_fail'])} (n = {A['works_gate_fail']['n']})",
          f"- 웃김 평균 {f(A['funny'])}, 관문 통과 {f(A['funny_gate_pass'])}, 관문 실패 {f(A['funny_gate_fail'])}",
          f"- 성립 후보가 1개 이상인 쌍 {res['pairs_with_any_works']}/{len(bp)}, 후보 전부가 성립인 쌍 {res['pairs_all_works']}/{len(bp)}. "
          f"못 읽음 {A['unparsed']:.3f}", "",
          "| 쌍 | n | 관문 통과 | 성립 ③ | 성립 ② | 웃김 |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {int(r.n)} | {r.gate:.2f} | {r.works3:.2f} | {r.works2:.2f} | {r.funny:.2f} |" for k, r in bp.iterrows()]
    (out / "table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
