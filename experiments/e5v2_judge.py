"""E5 v2 LLM 판정 (탐색 지표, docs/E5_V2_DESIGN.md 7절). 판정기 Qwen2.5-7B-Instruct, 온도 0.

성공: "w_p가 어떤 다른 단어처럼 들리나 (w_p와 다른 한 단어로)" → 첫 단어. w_p와 같은 표제어면 실패(되풀이),
      w_a와 같은 표제어면 성공. 이어서 "말장난으로 성립하나 (yes/no)"를 따로 묻는다.
      대상: 칸의 모든 P 후보(파일럿 2 기준 C4·진단 D2) + 고른 B1·B0·H.
웃김: 같은 칸 안 조건(B1, P1~P4, H)끼리 두 순서로 묻는다. 두 번 다 같은 쪽 → 그쪽 승리, 엇갈리면 무승부.
      Bradley–Terry(choix ILSR, α = 0.01)에는 승리 = 2:0, 무승부 = 1:1로 넣는다(순서별 결과를 하나씩 넣는 것과 같다).
판정 호출은 judge.jsonl에 쌓이고, 다 기록돼 있으면 판정 모델을 올리지 않는다.

사용: python -X utf8 experiments/e5v2_judge.py --stage pilot4 --gen olmo-2-0325-32b-instruct-4bit --model pythia-1.4b
출력: results/e5/<stage>/<gen>/A_<model>/judge_<판정기>/ judge.jsonl, success.csv, pairs.csv, bt.csv, metrics.json, table.md
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from e5 import extract_json, stage_dir, write_config
from e5_gen import Generator, tag_of
from e5_judge import JUDGE_SYSTEM, Calls, bt_scores
from e5v2 import judge_outcome, render, words
from rules import dump

PAIR_CONDS = ["B1", "P1", "P2", "P3", "P4", "H"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot2", "pilot3", "pilot4", "main"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--think", default=None, choices=["true", "false"], help="ollama 추론형 판정기의 생각 모드 (CPU 시험용)")
    ap.add_argument("--no-funny", action="store_true", help="웃김 비교를 건너뜀 (파일럿 2 기준에는 필요 없음)")
    a = ap.parse_args()
    gdir = stage_dir(a.stage) / a.gen
    adir = gdir / f"A_{a.model}"
    out = adir / f"judge_{tag_of(a.judge)}"
    out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(adir / "selections.csv")
    from e5_score import human_rows
    cand = pd.concat([pd.read_csv(gdir / "candidates.csv"), human_rows()], ignore_index=True).set_index("cand_id")
    make_gen = lambda: Generator(a.backend, a.judge, a.device, think=None if a.think is None else a.think == "true")
    calls = Calls(make_gen, out / "judge.jsonl")

    def ask(cid, prompt, system=None, max_new=20):
        if cid not in calls.rec:
            if calls._gen is None:
                calls._gen = calls.make_gen()
            raw = calls._gen.chat(prompt, seed=0, temperature=0.0, max_new_tokens=max_new, system=system)
            r = dict(call_id=cid, prompt=prompt, raw=raw, parsed=extract_json(raw))
            calls.rec[cid] = r
            with calls.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return calls.rec[cid]["raw"]

    # ---- 성공
    ids = list(dict.fromkeys(list(cand[cand.source == "P"].index) +
                             S[S.condition.isin(["B1", "B0", "H"])].cand_id.dropna().tolist()))
    rows = []
    for cid in ids:
        r = cand.loc[cid]
        text, pw = r.text, r.pun_word
        if not isinstance(text, str) or not text or not isinstance(pw, str) or not pw:
            continue
        w_a = r.w_a
        succ = ask(f"succ|{cid}", render("judge_success", w_p=pw, text=text))
        works = ask(f"works|{cid}", render("judge_works", w_p=pw, text=text))
        o = judge_outcome(succ, pw, str(w_a))
        wk = words(works)
        rows.append(dict(cand_id=cid, source=r.source, cell=r.get("cell"), pun_word=pw, w_a=w_a, response=o["answer"],
                         echo=o["echo"], success=o["success"],
                         works=(wk[0].lower() == "yes") if wk else None))
    succ = pd.DataFrame(rows)
    succ.to_csv(out / "success.csv", index=False)

    # ---- 웃김
    prs, bt_rows = [], []
    if not a.no_funny:
        for cell, g in S[S.condition.isin(PAIR_CONDS) & S.cell.notna()].dropna(subset=["cand_id"]).groupby("cell"):
            g = g.dropna(subset=["text"])
            cids = list(dict.fromkeys(g.cand_id))
            txt = g.drop_duplicates("cand_id").set_index("cand_id").text
            wins = []
            for x, y in itertools.combinations(cids, 2):
                res = []
                for first, second in ((x, y), (y, x)):
                    raw = ask(f"pair|{first}|{second}", render("judge_funny", a=txt[first], b=txt[second]),
                              system=JUDGE_SYSTEM, max_new=60)
                    p = extract_json(raw)
                    ch = str((p or {}).get("funnier", "")).strip().upper()[:1] if isinstance(p, dict) else ""
                    res.append(first if ch == "A" else second if ch == "B" else None)
                outcome = res[0] if res[0] is not None and res[0] == res[1] else ("tie" if None not in res else "unparsed")
                prs.append(dict(cell=cell, x=x, y=y, first_order=res[0], second_order=res[1], outcome=outcome))
                ix, iy = cids.index(x), cids.index(y)
                if outcome == x:
                    wins += [(ix, iy)] * 2
                elif outcome == y:
                    wins += [(iy, ix)] * 2
                elif outcome == "tie":
                    wins += [(ix, iy), (iy, ix)]
            sc = bt_scores(len(cids), wins)
            for cid, s in zip(cids, sc):
                for cond in g[g.cand_id == cid].condition:
                    bt_rows.append(dict(cell=cell, condition=cond, cand_id=cid, bt=float(s)))
    pd.DataFrame(prs).to_csv(out / "pairs.csv", index=False)
    BT = pd.DataFrame(bt_rows, columns=["cell", "condition", "cand_id", "bt"])
    BT.to_csv(out / "bt.csv", index=False)
    info = calls._gen.info if calls._gen is not None else dict(backend=a.backend, model=a.judge, note="모두 기록에서 읽음")
    write_config(out, dict(stage=a.stage, gen=a.gen, model=a.model, judge=info, temperature=0.0,
                           calls_from_cache=calls._gen is None, funny=not a.no_funny))

    # ---- 요약
    res = {}
    M = S.dropna(subset=["cand_id"])[["condition", "cand_id"]].merge(succ[["cand_id", "success", "works", "echo"]],
                                                                     on="cand_id", how="left")
    for cond, g in M.groupby("condition"):
        b = BT[BT.condition == cond].bt
        res[cond] = dict(n=len(g), success=float(g.success.astype(float).mean()), echo=float(g.echo.astype(float).mean()),
                         works=float(g.works.astype(float).mean()), bt_mean=float(b.mean()) if len(b) else None)
    P = succ[succ.source == "P"]
    res["_all_P_candidates"] = dict(n=len(P), success=float(P.success.mean()) if len(P) else None,
                                    echo=float(P.echo.mean()) if len(P) else None)
    pp = pd.DataFrame(prs)
    if len(pp):
        res["_pairs"] = dict(n=len(pp), tie=float((pp.outcome == "tie").mean()),
                             unparsed=float((pp.outcome == "unparsed").mean()),
                             first_wins=float(pd.concat([(pp.first_order == pp.x), (pp.second_order == pp.y)]).mean()))
    dump(out / "metrics.json", res)
    L = [f"# E5 v2 LLM 판정 (탐색) — 판정기 {a.judge}", "",
         f"P 후보 전체 {res['_all_P_candidates']['n']}개: 성공 {res['_all_P_candidates']['success']}, "
         f"w_p 되풀이 {res['_all_P_candidates']['echo']}", "",
         "| 조건 | n | 성공 | 되풀이 | 성립 yes | 웃김 BT 평균 |", "|---|---|---|---|---|---|"]
    for cond in ["B0", "B1", "P1", "P2", "P3", "P4", "H"]:
        if cond in res:
            v = res[cond]
            bt = "—" if v["bt_mean"] is None else f"{v['bt_mean']:.3f}"
            L.append(f"| {cond} | {v['n']} | {v['success']:.3f} | {v['echo']:.3f} | {v['works']:.3f} | {bt} |")
    if "_pairs" in res:
        L += ["", f"웃김 비교 {res['_pairs']['n']}쌍: 무승부 {res['_pairs']['tie']:.3f}, 못 읽음 {res['_pairs']['unparsed']:.3f}, "
                  f"먼저 보인 쪽 선택 {res['_pairs']['first_wins']:.3f}"]
    (out / "table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
