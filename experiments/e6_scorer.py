"""E6 채점기 (docs/E6_DESIGN.md 2절): 연구자의 E5 라벨로 LLM 판정 세 문항과 고친 청자 점수를 검증한다.

  split    E5 본 실행 40쌍을 층마다 반씩 나눠 개발 20쌍·확인 20쌍으로 한다. B0(쌍 없음)은 cand_id로 나눈다.
           시드 stable_seed("e6", "judge_split") → e6/judge_split.json (커밋해서 고정)
  inputs   판정할 문장 표: E5 규칙 통과 P 후보 791 + B1 80 + H 40 + B0 80. 쌍과 짧은 뜻풀이(평가지와 같음) → e6/judge_inputs.csv
  judge    [GPU] 판정기 하나로 네 문항을 묻는다. 온도 0, 문장마다 1회, 답은 첫 단어 yes/no
             a   w_p를 원래 뜻으로 읽으면 말이 되는가            e6/prompts/judge_a.txt
             b   w_a로 바꿔 읽으면 말이 되는가                    e6/prompts/judge_b.txt
             c0  성공한 말장난인가 (검증한 works_pair 그대로)    e5/judge_val/prompts/works_pair.txt
             c1  c0에 뜻풀이와 "두 뜻이 함께 떠오름"을 더함       e6/prompts/judge_c1.txt
           → results/e6/scorer/judge_<tag>/answers.csv, calls.jsonl, config.json
  dev      개발 절반에서 후보를 비교하고 하나를 고른다(고르는 규칙은 아래, 미리 고정) → results/e6/scorer/dev/
  confirm  고른 조합을 확인 절반에서 한 번 잰다. 2.4의 기준 세 개 → results/e6/scorer/confirm/

후보
  판정 합성  판정기 × {c0, c1, a∧c0, a∧c1, a∧b∧c0, a∧b∧c1, a∧b}
  청자 점수  L0 = log F_→ + S_← (E5), L1 = log F_→, L2 = log F_→ − S_←, L3 = min(z(log F_→), z(−S_←)) (z는 칸 안 표준화)
고르는 규칙 (개발 절반, P와 B1 문장)
  판정 합성: 연구자 (a) 아니오 문장의 yes 비율 ≤ 0.35인 것 중 연구자 (c)와의 균형 정확도가 가장 높은 것. 같으면 문항이 적은 것
  청자 점수: 평가한 P 문장 안에서 칸마다 판정 yes 우선, 그다음 점수 1위를 골랐을 때 연구자 (c) 성립이 가장 높은 것.
             같으면 (c)에 대한 AUC가 높은 것
확인 기준 (확인 절반, P와 B1 문장): 균형 정확도 ≥ 0.70, 같은 문장에서 phi-4 works_pair(E5 judge2)보다 0.05 이상 높음,
  연구자 (a) 아니오 문장의 yes 비율 ≤ 0.35

사용: python -X utf8 experiments/e6_scorer.py split
      python -X utf8 experiments/e6_scorer.py inputs
      python -X utf8 experiments/e6_scorer.py judge --judge microsoft/phi-4            (GPU, NF4 4비트)
      python -X utf8 experiments/e6_scorer.py judge --judge Qwen/Qwen2.5-14B-Instruct  (GPU, NF4 4비트)
      (CPU 시험) judge --backend ollama --judge qwen3.5:4b --think false --no-4bit --limit 5
      python -X utf8 experiments/e6_scorer.py dev --judges phi-4-4bit qwen2.5-14b-instruct-4bit
      python -X utf8 experiments/e6_scorer.py confirm
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from data import RESULTS
from e5 import stable_seed, write_config
from rules import dump

ROOT = Path(__file__).resolve().parents[1]
E6_DIR = ROOT / "e6"
PROMPTS = E6_DIR / "prompts"
WORKS_PAIR = ROOT / "e5" / "judge_val" / "prompts" / "works_pair.txt"
SPLIT = E6_DIR / "judge_split.json"
INPUTS = E6_DIR / "judge_inputs.csv"
OUT = RESULTS / "e6" / "scorer"
E5A = RESULTS / "e5" / "main" / "olmo-2-0325-32b-instruct-4bit" / "A_pythia-1.4b"
QUESTIONS = ("a", "b", "c0", "c1")
RULES = {"c0": ["c0"], "c1": ["c1"], "a&c0": ["a", "c0"], "a&c1": ["a", "c1"],
         "a&b&c0": ["a", "b", "c0"], "a&b&c1": ["a", "b", "c1"], "a&b": ["a", "b"]}
LISTENERS = ("L0", "L1", "L2", "L3")
A_NO_YES_MAX, BAL_MIN, BAL_GAIN = 0.35, 0.70, 0.05


# ---------------------------------------------------------------- 나누기와 라벨
def split():
    from e5v2 import load_samples_v4
    rng = np.random.default_rng(stable_seed("e6", "judge_split"))
    main = pd.DataFrame(load_samples_v4()["main"])
    order = []  # 층마다 섞어 이어 붙이고 번갈아 나눈다 (d0 14, sub 13, indel 13 → 20 / 20)
    for _, g in main.groupby("stratum"):
        ids = list(g.pair_id)
        order += [ids[i] for i in rng.permutation(len(ids))]
    dev = sorted(order[0::2])
    conf = sorted(set(main.pair_id) - set(dev))
    K = pd.read_csv(E5A / "human" / "key.csv")
    b0 = sorted(K[K.sheet == "b0"].cand_id)
    b0 = [b0[i] for i in rng.permutation(len(b0))]
    out = dict(seed='stable_seed("e6", "judge_split")', dev_pairs=dev, confirm_pairs=conf,
               dev_b0=sorted(b0[: len(b0) // 2]), confirm_b0=sorted(b0[len(b0) // 2:]),
               note="L0~L3 청자 점수는 나누기 전(2026-11-27)에 40쌍 전체에서 탐색 계산으로 한 번 보았다. LLM 판정은 나누기 전에 돌리지 않았다.")
    SPLIT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    st = main.set_index("pair_id").stratum
    print(f"개발 {len(dev)}쌍 {st[dev].value_counts().to_dict()}, 확인 {len(conf)}쌍 {st[conf].value_counts().to_dict()}, "
          f"B0 {len(out['dev_b0'])}/{len(out['confirm_b0'])} → {SPLIT}")


def labels() -> pd.DataFrame:
    """연구자 라벨(평가지 한 줄 = 한 행). H는 두 칸에 붙어 두 번 나온다. half = dev / confirm."""
    sp = json.loads(SPLIT.read_text(encoding="utf-8"))
    K = pd.read_csv(E5A / "human" / "key.csv")
    C = pd.read_csv(E5A / "human" / "answers_cells.csv")
    B = pd.read_csv(E5A / "human" / "answers_b0.csv")
    X = pd.concat([K[K.sheet == "cells"].merge(C, left_on="no", right_on="번호"),
                   K[K.sheet == "b0"].merge(B, left_on="no", right_on="번호")], ignore_index=True)
    yes = lambda s: s.astype(str).str.strip().str.upper().eq("Y").astype(float)
    X["h_a"], X["h_b"], X["h_c"] = yes(X["(a) w_p 뜻으로 말이 됨"]), yes(X["(b) w_a로 읽어도 말이 됨"]), yes(X["(c) 성공한 말장난"])
    dev = set(sp["dev_pairs"]) | set(sp["dev_b0"])
    X["half"] = [("dev" if (r.pair_id in dev or r.cand_id in dev) else "confirm") for r in X.itertuples()]
    return X[["sheet", "no", "block", "cand_id", "source", "cell", "pair_id", "format", "half", "h_a", "h_b", "h_c",
              "웃김 순위"]].rename(columns={"웃김 순위": "rank"})


# ---------------------------------------------------------------- 판정할 문장
def inputs():
    from e5_score import human_rows
    from e5v2 import cells_v2
    from e5v2_human import short_gloss
    gdir = E5A.parent
    cand = pd.concat([pd.read_csv(gdir / "candidates.csv"), human_rows()], ignore_index=True).drop_duplicates("cand_id")
    pairs = {c["pair_id"]: c for c in cells_v2("main")}
    K = pd.read_csv(E5A / "human" / "key.csv")
    keep = (cand.source == "P") & cand.rule_ok.astype(bool) | cand.cand_id.isin(K.cand_id)
    rows = []
    for r in cand[keep].itertuples():
        pr = pairs.get(r.pair_id) if r.source in ("P", "B1") else None
        wp = r.pun_word if isinstance(r.pun_word, str) and r.pun_word.strip() else r.w_p
        rows.append(dict(cand_id=r.cand_id, source=r.source, cell=r.cell if isinstance(r.cell, str) else None,
                         pair_id=r.pair_id if isinstance(r.pair_id, str) else None, text=r.text, w_p=wp, w_a=r.w_a,
                         g_p=pr["gloss_p"] if pr else (short_gloss(r.w_p) if isinstance(r.w_p, str) else ""),
                         g_a=pr["gloss_a"] if pr else (short_gloss(r.w_a) if isinstance(r.w_a, str) else "")))
    T = pd.DataFrame(rows)
    T.to_csv(INPUTS, index=False)
    print(f"판정할 문장 {len(T)}개 {T.source.value_counts().to_dict()} → {INPUTS}")


# ---------------------------------------------------------------- 판정
def render(q: str, r) -> str:
    t = (WORKS_PAIR if q == "c0" else PROMPTS / f"judge_{q}.txt").read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
    ok = lambda x: isinstance(x, str) and bool(x.strip())
    kw = dict(text=r.text, w_p=r.w_p, w_a=r.w_a, gp=f" (meaning: {r.g_p})" if ok(r.g_p) else "",
              ga=f" (meaning: {r.g_a})" if ok(r.g_a) else "")
    return re.sub(r"\{(\w+)\}", lambda m: str(kw[m.group(1)]) if m.group(1) in kw else m.group(0), t)


def judgeable(r) -> bool:
    ok = lambda x: isinstance(x, str) and bool(x.strip())
    return ok(r.text) and ok(r.w_p) and ok(r.w_a) and str(r.w_p).lower() != str(r.w_a).lower()


def prompt_sha() -> dict:
    files = {q: (WORKS_PAIR if q == "c0" else PROMPTS / f"judge_{q}.txt") for q in QUESTIONS}
    return {q: hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n")).hexdigest() for q, f in files.items()}


def judge(a):
    from e5_gen import Generator, tag_of
    from judge_validate import Judge, yes_no
    T = pd.read_csv(INPUTS)
    if a.limit:
        T = T.groupby("source").head(a.limit)
    tag = tag_of(a.judge) + ("" if a.no_4bit else "-4bit") + (f"-limit{a.limit}" if a.limit else "")
    out = OUT / f"judge_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    J = Judge(lambda: Generator(a.backend, a.judge, a.device, think=None if a.think is None else a.think == "true",
                                load_4bit=not a.no_4bit), out / "calls.jsonl")
    rows = []
    for i, r in enumerate(T.itertuples(), 1):
        for q in QUESTIONS:
            ans = yes_no(J.ask(f"{q}|{r.cand_id}", render(q, r))) if judgeable(r) else None
            rows.append(dict(cand_id=r.cand_id, q=q, ans=ans))
        if i % 100 == 0:
            print(f"  {i}/{len(T)}", flush=True)
    A = pd.DataFrame(rows)
    A.to_csv(out / "answers.csv", index=False)
    info = J.gen.info if J.gen is not None else dict(backend=a.backend, model=a.judge, note="모두 기록에서 읽음")
    write_config(out, dict(judge=info, temperature=0.0, limit=a.limit, prompts_sha256=prompt_sha(), n_sentences=len(T)))
    W = A.pivot(index="cand_id", columns="q", values="ans")
    print(f"{tag}: 문장 {len(T)}, 못 읽음 {W.isna().mean().round(3).to_dict()}, yes {W.astype(float).mean().round(3).to_dict()} → {out}")


# ---------------------------------------------------------------- 평가
def answers(tag: str) -> pd.DataFrame:
    A = pd.read_csv(OUT / f"judge_{tag}" / "answers.csv")
    A["ans"] = A.ans.map(lambda v: 1.0 if str(v) == "True" else 0.0 if str(v) == "False" else np.nan)
    return A.pivot(index="cand_id", columns="q", values="ans")


def composite(W: pd.DataFrame, rule: str) -> pd.Series:
    """문항 모두 yes면 1. 못 읽은 답은 아니오로 센다."""
    return W[RULES[rule]].fillna(0.0).min(axis=1)


def listener_scores() -> pd.DataFrame:
    X = pd.read_csv(E5A / "scores.csv").drop_duplicates("cand_id").set_index("cand_id")
    cand = pd.read_csv(E5A.parent / "candidates.csv").set_index("cand_id")
    P = X.join(cand[["cell", "source", "rule_ok"]], how="inner")
    P = P[(P.source == "P") & P.rule_ok.astype(bool)].copy()
    z = lambda s: (s - s.groupby(P.cell).transform("mean")) / s.groupby(P.cell).transform("std").replace(0, np.nan)
    return pd.DataFrame(dict(L0=P.logF_g + P.S_left, L1=P.logF_g, L2=P.logF_g - P.S_left,
                             L3=np.minimum(z(P.logF_g), z(-P.S_left)).fillna(z(P.logF_g))), index=P.index)


def bal_acc(p, t) -> float:
    p, t = np.asarray(p, float), np.asarray(t, float)
    ok = ~(np.isnan(p) | np.isnan(t))
    p, t = p[ok], t[ok]
    if not ((t == 1).any() and (t == 0).any()):
        return np.nan
    return float(((p[t == 1] == 1).mean() + (p[t == 0] == 0).mean()) / 2)


def kappa(p, t) -> float:
    p, t = np.asarray(p, float), np.asarray(t, float)
    ok = ~(np.isnan(p) | np.isnan(t))
    p, t = p[ok], t[ok]
    po = (p == t).mean()
    pe = p.mean() * t.mean() + (1 - p.mean()) * (1 - t.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def auc(s, y) -> float:
    s, y = np.asarray(s, float), np.asarray(y, float)
    ok = ~(np.isnan(s) | np.isnan(y))
    s, y = s[ok], y[ok]
    p, n = s[y == 1], s[y == 0]
    return np.nan if not len(p) or not len(n) else float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def phi4_works() -> pd.Series:
    J = pd.read_csv(E5A / "judge2_phi-4-4bit" / "items.csv").set_index("cand_id")
    return J.works.map(lambda v: 1.0 if str(v) == "True" else 0.0 if str(v) == "False" else np.nan)


def agreement(L: pd.DataFrame, pred: pd.Series) -> dict:
    """P와 B1 문장(기준), 그리고 전체·조건별 참고 값."""
    d = L.assign(p=L.cand_id.map(pred))
    pb = d[d.source.isin(["P", "B1"])]
    out = dict(n=int(pb.p.notna().sum()), bal_acc=bal_acc(pb.p, pb.h_c), kappa=kappa(pb.p, pb.h_c),
               yes=float(pb.p.mean()), human_yes=float(pb.h_c.mean()),
               yes_when_a_no=float(pb[pb.h_a == 0].p.mean()) if (pb.h_a == 0).any() else np.nan,
               yes_when_flat=float(pb[(pb.h_a == 1) & (pb.h_b == 1) & (pb.h_c == 0)].p.mean()),
               all_bal_acc=bal_acc(d.p, d.h_c))
    out["by_source"] = {s: dict(n=int(g.p.notna().sum()), yes=float(g.p.mean()), human_yes=float(g.h_c.mean()),
                                bal_acc=bal_acc(g.p, g.h_c)) for s, g in d.groupby("source")}
    return out


def pick_sim(L: pd.DataFrame, accept: pd.Series | None, score: pd.Series | None) -> dict:
    """평가한 P 문장 안에서 칸마다 하나를 고른다: 판정 yes 우선, 그다음 점수 1위. 연구자 (c)로 잰다."""
    P = L[L.source == "P"].drop_duplicates(["cell", "cand_id"]).copy()
    P["acc"] = P.cand_id.map(accept).fillna(0.0) if accept is not None else 0.0
    P["s"] = P.cand_id.map(score).fillna(-np.inf) if score is not None else 0.0
    P["tie"] = P.cand_id.map(lambda c: stable_seed("e6", "pick", c) / 2**32)
    top = P.sort_values(["acc", "s", "tie"], ascending=False).groupby("cell").head(1)
    g = P.groupby("cell").h_c
    return dict(cells=int(P.cell.nunique()), pick=float(top.h_c.mean()), random=float(g.mean().mean()), best=float(g.max().mean()))


def dev(a):
    L = labels()
    D = L[L.half == "dev"]
    S = listener_scores()
    od = OUT / "dev"
    od.mkdir(parents=True, exist_ok=True)
    res = dict(baseline_phi4_works_pair=agreement(D, phi4_works()), judges={}, listeners={})
    for tag in a.judges:
        W = answers(tag)
        res["judges"][tag] = {rule: agreement(D, composite(W, rule)) for rule in RULES}
    ok = [(t, r, v) for t, rr in res["judges"].items() for r, v in rr.items() if v["yes_when_a_no"] <= A_NO_YES_MAX]
    pool = ok or [(t, r, v) for t, rr in res["judges"].items() for r, v in rr.items()]
    num = lambda v: -1.0 if v is None or np.isnan(v) else round(float(v), 6)
    best = max(pool, key=lambda x: (num(x[2]["bal_acc"]), -len(RULES[x[1]])))
    acc = composite(answers(best[0]), best[1])
    for name in LISTENERS:
        s = D.cand_id.map(S[name])
        res["listeners"][name] = dict(auc_c=auc(s, D.h_c), auc_a=auc(s, D.h_a), auc_b=auc(s, D.h_b),
                                      pick_with_judge=pick_sim(D, acc, S[name]), pick_alone=pick_sim(D, None, S[name]))
    res["judge_only_pick"] = pick_sim(D, acc, None)
    lbest = max(LISTENERS, key=lambda n: (num(res["listeners"][n]["pick_with_judge"]["pick"]), num(res["listeners"][n]["auc_c"])))
    res["chosen"] = dict(judge=best[0], rule=best[1], listener=lbest, met_a_no_cap=bool(ok))
    dump(od / "metrics.json", res)
    lines = [f"# E6 채점기 개발 절반 ({len(D[D.source.isin(['P', 'B1'])])} P·B1 문장)", "",
             f"기준선 phi-4 works_pair: 균형 정확도 {res['baseline_phi4_works_pair']['bal_acc']:.3f}, "
             f"(a) 아니오에 yes {res['baseline_phi4_works_pair']['yes_when_a_no']:.2f}", "",
             "| 판정기 | 합성 | 균형 정확도 | κ | yes | (a) 아니오에 yes | 밋밋함에 yes |", "|---|---|---|---|---|---|---|"]
    for t, rr in res["judges"].items():
        for r, v in rr.items():
            lines.append(f"| {t} | {r} | {v['bal_acc']:.3f} | {v['kappa']:.3f} | {v['yes']:.2f} | {v['yes_when_a_no']:.2f} | {v['yes_when_flat']:.2f} |")
    lines += ["", "| 청자 | AUC (c) | AUC (a) | AUC (b) | 판정+점수 고르기 | 점수만 고르기 |", "|---|---|---|---|---|---|"]
    for n, v in res["listeners"].items():
        lines.append(f"| {n} | {v['auc_c']:.3f} | {v['auc_a']:.3f} | {v['auc_b']:.3f} | {v['pick_with_judge']['pick']:.3f} | {v['pick_alone']['pick']:.3f} |")
    j = res["judge_only_pick"]
    lines += ["", f"평가한 P 문장 안 고르기: 무작위 {j['random']:.3f}, 최선 {j['best']:.3f}, 판정만 {j['pick']:.3f} ({j['cells']}칸)", "",
              f"고름: {res['chosen']}"]
    (od / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_config(od, dict(judges=a.judges, rules=RULES, listeners=LISTENERS, a_no_yes_max=A_NO_YES_MAX))
    print("\n".join(lines))


def confirm(a):
    ch = json.loads((OUT / "dev" / "metrics.json").read_text(encoding="utf-8"))["chosen"]
    L = labels()
    C = L[L.half == "confirm"]
    oc = OUT / "confirm"
    if (oc / "metrics.json").exists() and not a.force:
        sys.exit(f"이미 쟀다: {oc}. 확인 절반은 한 번만 잰다.")
    oc.mkdir(parents=True, exist_ok=True)
    acc = composite(answers(ch["judge"]), ch["rule"])
    S = listener_scores()
    r = agreement(C, acc)
    base = agreement(C, phi4_works())
    crit = dict(bal_acc_min=r["bal_acc"] >= BAL_MIN, gain_over_phi4=r["bal_acc"] - base["bal_acc"] >= BAL_GAIN,
                a_no_yes_cap=r["yes_when_a_no"] <= A_NO_YES_MAX)
    res = dict(chosen=ch, scorer=r, baseline_phi4_works_pair=base, criteria=crit, passed=all(crit.values()),
               pick_judge_listener=pick_sim(C, acc, S[ch["listener"]]), pick_listener_L0=pick_sim(C, None, S["L0"]),
               pick_judge_only=pick_sim(C, acc, None))
    dump(oc / "metrics.json", res)
    write_config(oc, dict(chosen=ch))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "inputs", "judge", "dev", "confirm"])
    ap.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    ap.add_argument("--judge", default="microsoft/phi-4")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--think", default=None, choices=["true", "false"])
    ap.add_argument("--limit", type=int, default=None, help="시험용: 출처마다 앞의 몇 문장만 (폴더에 -limitN)")
    ap.add_argument("--judges", nargs="+", default=["phi-4-4bit", "qwen2.5-14b-instruct-4bit"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"split": split, "inputs": inputs, "judge": lambda: judge(a), "dev": lambda: dev(a), "confirm": lambda: confirm(a)}[a.cmd]()


if __name__ == "__main__":
    main()
