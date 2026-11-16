"""E5 v2 파일럿 2·3 보고서와 통과 기준 C1~C4 판정 (docs/E5_V2_DESIGN.md 8절).

모델(생성기 폴더)마다:
  단계별 실패율(모델 × 형식): S1 모양·w_a 위치·횟수·길이·잘림 / S3 끝 보존·수정 폭·w_a 누출·w_p 횟수 / 최종 규칙
  칸당 S1 호출 수 분포, 관문 통과율(②·③)
  D1  같은 후보의 ②와 ③의 A_→ 짝 비교 (S3가 A_→를 올리는가)
  D2  관문 통과 vs 실패 후보의 판정기 성공률 (A가 가짜 신호인가)
  topic_link(바뀐 단어와 w_p의 word2vec 코사인 최댓값, word2vec이 있을 때만) 분포, 힌트 단어 사용률
  점수 상위 3개·하위 3개 예시, B0 중복률
  C1 채우기: 모든 칸에서 S1 상한(MAX_S1) 안에 후보 10개
  C2 관문 통과 존재: 관문 통과 후보가 1개 이상인 칸 ≥ 50%
  C3 고를 여지: 후보가 모두 관문을 통과한 칸 < 50%
  C4 판정기 확인: 규칙 통과 후보 전체의 판정기 성공률 ≥ 0.20
결정 규칙: --gens 순서(7B, 32B)대로 처음으로 C1~C4를 모두 넘은 모델을 본 실행 생성기로. 없으면 태그를 달지 않는다.

사용: python -X utf8 experiments/e5v2_report.py --gens olmo-2-1124-7b-instruct olmo-2-0325-32b-instruct
출력: results/e5/<stage>/report.md, report.json (stage 기본 pilot4)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from e5 import stage_dir
from e5_gen import tag_of
from rules import dump

CRIT = dict(C2=0.5, C3=0.5, C4=0.20)
B, SEED_BOOT = 5000, 20261112


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def rate_table(df: pd.DataFrame, col: str, by="format") -> dict:
    """사유 목록 열(col)에서 사유별 비율 (행 수 기준), 형식별."""
    out = {}
    for fmt, g in list(df.groupby(by)) + [("all", df)]:
        n = len(g)
        ex = g[col].explode().dropna()
        out[fmt] = dict(n=n, fail=float(g[col].map(bool).mean()) if n else None,
                        reasons={k: round(v / n, 3) for k, v in ex.value_counts().items()} if n else {})
    return out


def cell_boot_mean(d: pd.Series, cells: pd.Series) -> list:
    rng = np.random.default_rng(SEED_BOOT)
    groups = [d[cells == c].to_numpy() for c in cells.unique()]
    if not groups:
        return [None, None]
    m = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]).mean() for _ in range(B)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def topic_links(J: list[dict]):
    from baselines import W2V
    if not Path(W2V).exists():
        return None
    from gensim.models import KeyedVectors
    kv = KeyedVectors.load_word2vec_format(str(W2V), binary=True, limit=1_000_000)
    out = {}
    for r in J:
        ws = [w for w in str(r.get("edited") or "").split() if w in kv.key_to_index]
        out[r["cand_id"]] = max((float(kv.similarity(w, r["w_p"])) for w in ws), default=np.nan) \
            if r["w_p"] in kv.key_to_index else np.nan
    return out


def one_model(stage: str, gen: str, model: str, judge_tag: str, w2v: bool) -> dict:
    gdir = stage_dir(stage) / gen
    adir = gdir / f"A_{model}"
    A = pd.DataFrame(read_jsonl(gdir / "attempts.jsonl"))
    cells = pd.read_csv(gdir / "cells.csv")
    cand = pd.read_csv(gdir / "candidates.csv")
    X = pd.read_csv(adir / "scores.csv").set_index("cand_id") if (adir / "scores.csv").exists() else pd.DataFrame()
    J = read_jsonl(adir / "candidates.jsonl")
    js = adir / f"judge_{judge_tag}" / "success.csv"
    succ = pd.read_csv(js).set_index("cand_id") if js.exists() else pd.DataFrame()
    R = dict(gen=gen)

    # 단계별 실패율
    s1 = A.assign(r=A.s1_reasons)
    R["S1"] = rate_table(s1, "r")
    s1ok = A[A.s1_ok]
    calls = [dict(format=r.format, r=t["reasons"]) for r in s1ok.itertuples() for t in r.s3_tries]
    R["S3_calls"] = rate_table(pd.DataFrame(calls, columns=["format", "r"]), "r")
    R["S3_attempts"] = dict(n=len(s1ok), fail=float((~s1ok.s3_ok.astype(bool)).mean()) if len(s1ok) else None,
                            retried=float(s1ok.s3_retried.astype(bool).mean()) if len(s1ok) else None)
    s3ok = A[A["s3_ok"].eq(True)] if "s3_ok" in A else A.iloc[:0]
    R["final"] = rate_table(s3ok.assign(r=s3ok.final_reasons), "r") if len(s3ok) else {}
    R["s1_calls_per_cell"] = dict(min=int(cells.n_s1_calls.min()), median=float(cells.n_s1_calls.median()),
                                  max=int(cells.n_s1_calls.max()), values=cells.n_s1_calls.tolist())
    R["cells"] = dict(n=len(cells), underfilled=int(cells.underfilled.sum()), failed=int(cells.generation_failed.sum()),
                      n_cand=cells.n_cand.tolist())

    # 관문, D1
    P = cand[cand.source == "P"].copy()
    if len(X):
        P["gate3"] = P.cand_id.map(X.gate).fillna(False).astype(bool)
        P["gate2"] = (P.cand_id + "@2").map(X.gate).fillna(False).astype(bool)
        P["A3"], P["A2"] = P.cand_id.map(X.A_g), (P.cand_id + "@2").map(X.A_g)
        P["rank3"], P["rank2"] = P.cand_id.map(X.rank_g), (P.cand_id + "@2").map(X.rank_g)
        P["score"] = P.cand_id.map(X.score)
        R["gate"] = {f: dict(n=len(g), s3=float(g.gate3.mean()), s2=float(g.gate2.mean()),
                             rank1_s3=float((g.rank3 == 1).mean()), rank1_s2=float((g.rank2 == 1).mean()))
                     for f, g in list(P.groupby("format")) + [("all", P)]}
        d = (P.A3 - P.A2).dropna()
        R["D1"] = dict(n=len(d), mean_diff=float(d.mean()) if len(d) else None,
                       ci95_cell_boot=cell_boot_mean(d, P.loc[d.index, "cell"]),
                       median_diff=float(d.median()) if len(d) else None, share_up=float((d > 0).mean()) if len(d) else None)
    # D2, C4
    if len(succ):
        P["judge"] = P.cand_id.map(succ.success)
        jp = P.dropna(subset=["judge"])
        R["judge_all_P"] = dict(n=len(jp), success=float(jp.judge.astype(float).mean()) if len(jp) else None,
                                echo=float(P.cand_id.map(succ.echo).dropna().astype(float).mean()) if len(jp) else None)
        if "gate3" in P:
            R["D2"] = {k: dict(n=int(len(g)), success=float(g.judge.astype(float).mean()) if len(g) else None)
                       for k, g in (("gate_pass", jp[jp.gate3]), ("gate_fail", jp[~jp.gate3]))}
    # topic_link, 힌트
    R["hint_used"] = float(P.hint_used.astype(bool).mean()) if len(P) else None
    if w2v and J:
        tl = topic_links(J)
        if tl is not None:
            v = pd.Series(tl).dropna()
            R["topic_link"] = dict(n=len(v), quantiles={str(q): float(v.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})
    # 예시
    if "score" in P:
        cols = ["cand_id", "text", "w_p", "w_a", "score", "gate3"] + (["judge"] if "judge" in P else [])
        Ps = P.dropna(subset=["score"]).sort_values("score", ascending=False)
        R["top3"] = Ps[cols].head(3).to_dict("records")
        R["bottom3"] = Ps[cols].tail(3).to_dict("records")
    # B0
    B0 = cand[cand.source == "B0"]
    from e5v2 import norm_dup
    nd = B0.text.fillna("").map(norm_dup)
    R["B0"] = dict(n=len(B0), rule_pass=float(B0.rule_ok.astype(bool).mean()) if len(B0) else None,
                   reasons=B0.reasons.fillna("").str.split(",").explode().replace("", np.nan).dropna().value_counts().to_dict(),
                   duplicate_rate=float(nd.duplicated(keep=False).mean()) if len(B0) else None)
    B1 = cand[cand.source == "B1"]
    R["B1"] = dict(n=len(B1), rule_pass=float(B1.rule_ok.astype(bool).mean()) if len(B1) else None)

    # C1~C4
    by_cell = P.groupby("cell")
    g_any = by_cell.gate3.any().reindex(cells.cell).fillna(False) if "gate3" in P else None
    g_all = by_cell.gate3.all().reindex(cells.cell).fillna(False) if "gate3" in P else None
    C = dict(
        C1=dict(value=int((cells.n_cand >= 10).sum()), of=len(cells), passed=bool((cells.n_cand >= 10).all())),
        C2=dict(value=float(g_any.mean()) if g_any is not None else None,
                passed=bool(g_any is not None and g_any.mean() >= CRIT["C2"])),
        C3=dict(value=float(g_all.mean()) if g_all is not None else None,
                passed=bool(g_all is not None and g_all.mean() < CRIT["C3"])),
        C4=dict(value=R.get("judge_all_P", {}).get("success"),
                passed=bool((R.get("judge_all_P", {}).get("success") or 0) >= CRIT["C4"])),
    )
    C["all"] = all(v["passed"] for v in C.values())
    R["criteria"] = C
    return R


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def markdown(stage, model, res: dict, decision: str) -> str:
    L = [f"# E5 v2 {stage} 보고서 (A = {model})", "", f"**결정**: {decision}", "",
         "## 통과 기준 (C1~C4)", "", "| 생성기 | C1 채우기 | C2 관문 통과 칸 ≥ 0.5 | C3 전부 통과 칸 < 0.5 | C4 판정기 성공 ≥ 0.20 | 모두 |",
         "|---|---|---|---|---|---|"]
    for g, R in res.items():
        C = R["criteria"]
        mark = lambda k: ("통과 " if C[k]["passed"] else "실패 ") + (
            f"{C[k]['value']}/{C[k]['of']}" if k == "C1" else fmt(C[k]["value"]))
        L.append(f"| {g} | {mark('C1')} | {mark('C2')} | {mark('C3')} | {mark('C4')} | {'통과' if C['all'] else '실패'} |")
    for g, R in res.items():
        L += ["", f"## {g}", "", f"칸 {R['cells']['n']}개, underfilled {R['cells']['underfilled']}, 생성 실패 {R['cells']['failed']}. "
              f"칸당 S1 호출: 최소 {R['s1_calls_per_cell']['min']}, 중앙값 {R['s1_calls_per_cell']['median']}, "
              f"최대 {R['s1_calls_per_cell']['max']}. 칸별 후보 수 {R['cells']['n_cand']}", "",
              "| 단계 | 형식 | n | 실패율 | 사유별 비율 |", "|---|---|---|---|---|"]
        for stg in ("S1", "S3_calls", "final"):
            for f, v in R.get(stg, {}).items():
                L.append(f"| {stg} | {f} | {v['n']} | {fmt(v['fail'])} | {v['reasons']} |")
        L += ["", f"S3 시도 단위: {R['S3_attempts']}"]
        if "gate" in R:
            L += ["", "| 형식 | n | 관문 ③ | 관문 ② | w_a 1위 ③ | w_a 1위 ② |", "|---|---|---|---|---|---|"]
            for f, v in R["gate"].items():
                L.append(f"| {f} | {v['n']} | {fmt(v['s3'])} | {fmt(v['s2'])} | {fmt(v['rank1_s3'])} | {fmt(v['rank1_s2'])} |")
            d = R["D1"]
            L += ["", f"**D1** A_→(③) − A_→(②): 평균 {fmt(d['mean_diff'], 4)} (칸 묶음 95% {fmt(d['ci95_cell_boot'][0], 4)}, "
                      f"{fmt(d['ci95_cell_boot'][1], 4)}), 중앙값 {fmt(d['median_diff'], 4)}, 오른 비율 {fmt(d['share_up'])} (n = {d['n']})"]
        if "D2" in R:
            d = R["D2"]
            L += [f"**D2** 판정기 성공률: 관문 통과 {fmt(d['gate_pass']['success'])} (n = {d['gate_pass']['n']}), "
                  f"관문 실패 {fmt(d['gate_fail']['success'])} (n = {d['gate_fail']['n']}). "
                  f"P 후보 전체 {fmt(R['judge_all_P']['success'])}, w_p 되풀이 {fmt(R['judge_all_P']['echo'])}"]
        L += [f"힌트 단어 사용률 {fmt(R['hint_used'])}" + (f", topic_link 분위 {R['topic_link']['quantiles']}" if "topic_link" in R else
                                                    " (topic_link: word2vec이 없어 계산 안 함)")]
        L += [f"B1 규칙 통과 {fmt(R['B1']['rule_pass'])} (n = {R['B1']['n']}). B0 규칙 통과 {fmt(R['B0']['rule_pass'])}, "
              f"중복률 {fmt(R['B0']['duplicate_rate'])}, 사유 {R['B0']['reasons']}"]
        for k in ("top3", "bottom3"):
            if k in R:
                L += ["", f"점수 {'상위' if k == 'top3' else '하위'} 3개:"]
                L += [f"- ({r['w_p']}→{r['w_a']}, 점수 {fmt(r['score'], 2)}, 관문 {'O' if r['gate3'] else 'X'}"
                      + (f", 판정 {'O' if r.get('judge') else 'X'}" if 'judge' in r else "") + f") {r['text']}" for r in R[k]]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot4", choices=["pilot2", "pilot3", "pilot4"])
    ap.add_argument("--gens", nargs="+", required=True, help="결정 규칙 순서대로 (7B 먼저)")
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--no-w2v", action="store_true")
    a = ap.parse_args()
    res = {g: one_model(a.stage, g, a.model, tag_of(a.judge), not a.no_w2v) for g in a.gens}
    chosen = next((g for g in a.gens if res[g]["criteria"]["all"]), None)
    decision = (f"{chosen}를 본 실행 생성기로 정한다 (C1~C4 모두 통과)." if chosen else
                "C1~C4를 모두 넘은 모델이 없다. prereg-e5 태그를 달지 않고 실패를 기록한 뒤 설계로 돌아간다.")
    out = stage_dir(a.stage)
    dump(out / "report.json", dict(decision=decision, chosen=chosen, order=a.gens, criteria=CRIT, models=res))
    md = markdown(a.stage, a.model, res, decision)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
