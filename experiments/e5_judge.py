"""E5 LLM 판정 (탐색 지표, H5d). 생성기와 다른 계열 LLM에게 사람 평가와 같은 질문을 한다. 온도 0.

성공: 문장과 말장난 단어를 보여주고 "어떤 다른 단어처럼 들리나" 자유 응답 → w_a와 표제어가 같으면 성공.
      이어서 "말장난으로 성립하나"(yes/no).
웃김: 같은 칸 안에서 조건끼리 두 개씩 "어느 쪽이 더 웃긴가". 위치 효과를 지우려고 A/B 순서를 바꿔 두 번 묻는다.
      칸마다 Bradley–Terry(choix ILSR, α = 0.01)로 점수를 매기고 조건별 평균을 낸다.
같은 후보를 두 조건이 골랐으면 한 번만 묻는다(비교에서는 같은 점수).

사용: python -X utf8 experiments/e5_judge.py --stage pilot --gen olmo-2-1124-7b-instruct --model pythia-1.4b \
          --backend hf --judge Qwen/Qwen2.5-7B-Instruct
출력: results/e5/<stage>/<gen>/A_<model>/judge_<판정기>/ calls.jsonl, success.csv, pairs.csv, metrics.json, table.md
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from e5 import extract_json, stable_seed, stage_dir, write_config
from e5_gen import Generator, tag_of
from rules import dump, lemmas

PAIR_CONDS = ["B1", "P1", "P2", "P3", "P4", "H"]
JUDGE_SYSTEM = "You are a careful reader of English jokes. Answer only in the requested JSON."


def prompt_success(text, pun_word):
    return (f'Here is a joke:\n"{text}"\n\nThe word "{pun_word}" in it is meant as a pun: it sounds like a different '
            "English word. Which word does it sound like? Also, does the joke work as a pun?\n\n"
            'Return only JSON: {"word": "...", "works": "yes" or "no"}')


def prompt_pair(a, b):
    return (f"Which of these two jokes is funnier?\n\nA: {a}\n\nB: {b}\n\n"
            'Return only JSON: {"funnier": "A" or "B"}')


class Calls:
    """판정 호출 기록. 기록에 없는 호출이 처음 나올 때만 판정 모델을 올린다(다 기록돼 있으면 GPU 없이 다시 계산)."""

    def __init__(self, make_gen, path):
        self.make_gen, self._gen, self.path, self.rec = make_gen, None, path, {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.rec[r["call_id"]] = r

    def ask(self, cid, prompt):
        if cid not in self.rec:
            if self._gen is None:
                self._gen = self.make_gen()
            raw = self._gen.chat(prompt, seed=stable_seed("judge", cid), temperature=0.0, max_new_tokens=60,
                                system=JUDGE_SYSTEM)
            r = dict(call_id=cid, prompt=prompt, raw=raw, parsed=extract_json(raw))
            self.rec[cid] = r
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return self.rec[cid]["parsed"]


def bt_scores(n, wins):
    import choix
    if not wins:
        return np.zeros(n)
    return choix.ilsr_pairwise(n, wins, alpha=0.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot", "main"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--think", default=None, choices=["true", "false"])
    a = ap.parse_args()
    adir = stage_dir(a.stage) / a.gen / f"A_{a.model}"
    out = adir / f"judge_{tag_of(a.judge)}"
    S = pd.read_csv(adir / "selections.csv")
    make_gen = lambda: Generator(a.backend, a.judge, a.device, think=None if a.think is None else a.think == "true")
    calls = Calls(make_gen, out / "calls.jsonl")

    # ---- 성공
    U = S.dropna(subset=["text"]).drop_duplicates("cand_id")
    U = U[U.text.str.len() > 0]
    rows = []
    for r in U.itertuples():
        p = calls.ask(f"succ|{r.cand_id}", prompt_success(r.text, r.pun_word)) or {}
        word = str(p.get("word", "") if isinstance(p, dict) else "").strip().lower().split(" ")[0] if p else ""
        rows.append(dict(cand_id=r.cand_id, response=word,
                         works=str(p.get("works", "")).lower().startswith("y") if isinstance(p, dict) else None))
    succ = pd.DataFrame(rows)
    wa = S.drop_duplicates("cand_id").set_index("cand_id").w_a
    lem = lemmas(set(succ.response) | set(wa.dropna().str.lower()))
    succ["w_a"] = succ.cand_id.map(wa)
    succ["success"] = [bool(r and isinstance(w, str) and lem.get(r) == lem.get(w.lower())) for r, w in zip(succ.response, succ.w_a)]
    succ.to_csv(out / "success.csv", index=False)

    # ---- 웃김 (칸 안 두 개씩, 순서 바꿔 두 번)
    prs, bt_rows = [], []
    for cell, g in S[S.condition.isin(PAIR_CONDS) & S.cell.notna()].groupby("cell"):
        g = g.dropna(subset=["text"])
        ids = list(dict.fromkeys(g.cand_id))
        txt = g.drop_duplicates("cand_id").set_index("cand_id").text
        wins = []
        for x, y in itertools.combinations(ids, 2):
            for first, second in ((x, y), (y, x)):
                p = calls.ask(f"pair|{first}|{second}", prompt_pair(txt[first], txt[second]))
                ch = str((p or {}).get("funnier", "")).strip().upper()[:1] if isinstance(p, dict) else ""
                win = first if ch == "A" else second if ch == "B" else None
                prs.append(dict(cell=cell, first=first, second=second, winner=win))
                if win:
                    wins.append((ids.index(win), ids.index(second if win == first else first)))
        sc = bt_scores(len(ids), wins)
        for cid, s in zip(ids, sc):
            for cond in g[g.cand_id == cid].condition:
                bt_rows.append(dict(cell=cell, condition=cond, cand_id=cid, bt=float(s)))
    pd.DataFrame(prs).to_csv(out / "pairs.csv", index=False)
    BT = pd.DataFrame(bt_rows)
    BT.to_csv(out / "bt.csv", index=False)

    # ---- 요약
    M = S[["condition", "cand_id", "format", "score", "gate"]].merge(succ[["cand_id", "success", "works"]], on="cand_id", how="left")
    res = {}
    for cond, g in M.groupby("condition"):
        b = BT[BT.condition == cond].bt
        res[cond] = dict(n=len(g), success=float(g.success.astype(float).mean()),
                         works=float(g.works.astype(float).mean()), bt_mean=float(b.mean()) if len(b) else None)
    pos = pd.DataFrame(prs)
    res["_position_bias_first_wins"] = float((pos.winner == pos["first"]).mean()) if len(pos) else None
    res["_unparsed_pairs"] = float(pos.winner.isna().mean()) if len(pos) else None
    from scipy.stats import spearmanr
    m2 = M.dropna(subset=["score"])
    res["_spearman_A_score_vs_judge_success"] = float(spearmanr(m2.score, m2.success.astype(float)).statistic) if len(m2) > 2 else None
    dump(out / "metrics.json", res)
    # 판정기 정보: 이번에 모델을 올렸으면 그 정보, 모두 기록에서 읽었으면 기록을 만든 실행의 정보
    old = out / "config.json"
    info = calls._gen.info if calls._gen is not None else (
        json.loads(old.read_text(encoding="utf-8")).get("judge") if old.exists() else dict(backend=a.backend, model=a.judge))
    write_config(out, dict(stage=a.stage, gen=a.gen, model=a.model, judge=info, temperature=0.0,
                           calls_from_cache=calls._gen is None))
    L = [f"# E5 LLM 판정 (탐색) — 판정기 {a.judge}", "", "| 조건 | n | 성공(w_a 맞힘) | 성립 yes | 웃김 BT 평균 |", "|---|---|---|---|---|"]
    for cond in ["B0", "B1", "P1", "P2", "P3", "P4", "H"]:
        if cond in res:
            v = res[cond]
            bt = "—" if v["bt_mean"] is None else f"{v['bt_mean']:.3f}"
            L.append(f"| {cond} | {v['n']} | {v['success']:.3f} | {v['works']:.3f} | {bt} |")
    L += ["", f"앞에 놓인 쪽이 이긴 비율 {res['_position_bias_first_wins']}, 못 읽은 비교 {res['_unparsed_pairs']}, "
              f"A 점수와 판정 성공의 스피어만 ρ {res['_spearman_A_score_vs_judge_success']}"]
    (out / "table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
