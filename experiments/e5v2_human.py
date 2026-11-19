"""E5 사람 평가 (PREREG_E5 3절): 연구자 1명, 블라인드. 평가지 만들기와 채점.

  calib  : 보정 평가지. He et al. 2019 MTurk 자료에서 60문장(사람 말장난 15 + 기계 생성 45, 시드 고정), (c) 성공한 말장난인가만 묻는다.
           results/e5/human_calib/ 에 만든다. 자료 문장이 들어 있으므로 평가지·열쇠는 깃에 올리지 않고(.gitignore),
           채점 결과는 항목 id와 답만 남긴다.
  make   : 본 평가지. 칸 묶음(B1, P1~P4, H — 칸마다 한 덩어리, 같은 문장은 한 번)과 B0 묶음. 조건 이름은 열쇠 파일에만 있다.
  retest : 재검사 평가지. 본 평가가 끝나고 하루 이상 지난 뒤, 칸 8개를 다시(번호와 순서를 새로).
  score  : 채점. 보정(있으면), H5a·H5b(확인), H5c·H5d·H5e(탐색), 재검사(있으면).

답 쓰는 법: (a)~(d)는 Y / N (예·아니오·O·X·1·0도 받는다). 웃김 순위는 칸 안에서 1이 가장 웃김, 동점은 같은 수. B0 웃김은 1~5.
평가지는 엑셀에서 열어 같은 이름으로 저장한다(CSV, UTF-8).

사용: python -X utf8 experiments/e5v2_human.py calib
      python -X utf8 experiments/e5v2_human.py make   --stage main --gen olmo-2-0325-32b-instruct-4bit
      python -X utf8 experiments/e5v2_human.py retest --stage main --gen olmo-2-0325-32b-instruct-4bit
      python -X utf8 experiments/e5v2_human.py score  --stage main --gen olmo-2-0325-32b-instruct-4bit
출력: results/e5/<stage>/<gen>/A_<model>/human/ cells_sheet.csv, b0_sheet.csv, key.csv, guide.txt, (retest_sheet.csv, retest_key.csv),
      metrics.json, table.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from data import RESULTS
from e5 import gloss, gloss_common, stable_seed, stage_dir
from rules import SEED, dump

B_BOOT = 5000
CELL_CONDS = ["B1", "P1", "P2", "P3", "P4", "H"]
N_CALIB = dict(gold=15, machine=45)
N_RETEST = 8
CALIB_DIR = RESULTS / "e5" / "human_calib"
YN_COLS = ["(a) w_p 뜻으로 말이 됨", "(b) w_a로 읽어도 말이 됨", "(c) 성공한 말장난", "(d) 들어 본 농담"]
YES = {"Y", "YES", "예", "O", "1", "TRUE", "ㅇ"}
NO = {"N", "NO", "아니오", "아니요", "X", "0", "FALSE"}

GUIDE = """E5 사람 평가 방법 (PREREG_E5 3절)

칸 묶음 (cells_sheet.csv)
- 같은 '블록' 번호의 문장들은 같은 칸(같은 단어 쌍, 같은 형식)에서 나왔다. 어느 문장이 어떤 방법으로 만들어졌는지는 숨겨져 있다.
- 문장 안 [[ ]] 표시가 말장난 단어(w_p)다. '쌍'은 w_p → w_a(소리가 비슷해 함께 떠올라야 하는 단어), 옆에 두 단어의 뜻이 있다.
  사람이 쓴 말장난이 섞여 있고, 그 문장은 자기 쌍을 보여 준다.
- 문장마다 Y 또는 N:
  (a) w_p 원래 뜻으로 이 문장에서 말이 되는가
  (b) w_a로 바꿔 읽어도 이 문장에서 말이 되는가
  (c) 성공한 말장난인가 (두 뜻이 함께 떠오르고 말장난으로 성립하는가) ← 가장 중요
  (d) 전에 들어 본 농담인가
- 블록마다 '웃김 순위': 블록 안 문장을 웃긴 순서로 1, 2, 3 … (1이 가장 웃김, 비슷하면 같은 수).

B0 묶음 (b0_sheet.csv)
- 모델에게 아무 말장난이나 하나 쓰라고 한 문장들. 쌍은 모델이 스스로 적은 것이다.
- 문장마다 (a)~(d)를 Y/N으로, '웃김'은 1~5(5가 가장 웃김).

공통
- key.csv, retest_key.csv는 평가가 다 끝나기 전에는 열지 않는다. results/e5/main/ 의 다른 파일도 열지 않는다.
- 엑셀에서 열고 같은 이름(CSV UTF-8)으로 저장한다. 여러 날에 나눠도 된다.
- 순서: 보정(human_calib) → 칸 묶음과 B0 묶음 → 하루 이상 지나서 재검사.
"""


def mark(text: str, s, e) -> str:
    if pd.isna(s) or pd.isna(e):
        return text
    s, e = int(s), int(e)
    return text[:s] + "[[" + text[s:e] + "]]" + text[e:]


def short_gloss(w: str) -> str:
    g = gloss_common(str(w).lower()) or gloss(str(w).lower())
    return "" if not g else g[1]


def yn(x):
    v = str(x).strip().upper()
    return 1.0 if v in YES else 0.0 if v in NO else np.nan


# ---------------------------------------------------------------- 평가지
def make(adir: Path, stage: str):
    from e5_score import human_rows
    from e5v2 import cells_v2
    hd = adir / "human"
    hd.mkdir(exist_ok=True)
    gdir = adir.parent
    S = pd.read_csv(adir / "selections.csv").dropna(subset=["cand_id"])
    cand = pd.concat([pd.read_csv(gdir / "candidates.csv"), human_rows()], ignore_index=True).drop_duplicates("cand_id")
    cand = cand.set_index("cand_id")
    pairs = {c["pair_id"]: c for c in cells_v2(stage)}
    rng = np.random.default_rng(stable_seed(stage, "human2"))
    cells = list(dict.fromkeys(S[S.condition.isin(CELL_CONDS)].cell.dropna()))
    cells = [cells[i] for i in rng.permutation(len(cells))]
    rows, key = [], []
    n = 0
    for b, cell in enumerate(cells, 1):
        g = S[(S.cell == cell) & S.condition.isin(CELL_CONDS)]
        conds = g.groupby("cand_id").condition.apply(lambda x: ",".join(sorted(x)))
        ids = [i for i in conds.index if isinstance(cand.loc[i].text, str) and cand.loc[i].text.strip()]
        ids = [ids[i] for i in rng.permutation(len(ids))]
        for cid in ids:
            r = cand.loc[cid]
            n += 1
            wp, wa = str(r.w_p), str(r.w_a)
            pr = pairs.get(r.get("pair_id")) if r.source != "H" else None
            gp = pr["gloss_p"] if pr else short_gloss(wp)
            ga = pr["gloss_a"] if pr else short_gloss(wa)
            rows.append({"블록": b, "번호": n, "쌍": f"{wp} → {wa}", "w_p 뜻": gp, "w_a 뜻": ga,
                         "문장": mark(r.text, r.start, r.end), **{c: "" for c in YN_COLS}, "웃김 순위": ""})
            key.append(dict(sheet="cells", no=n, block=b, cand_id=cid, conditions=conds[cid], cell=cell,
                            pair_id=cell.split(":")[0], format=cell.split(":")[1], source=r.source))
    B0 = cand[cand.source == "B0"].reset_index()
    B0 = B0.iloc[rng.permutation(len(B0))]
    brow = []
    for m, r in enumerate(B0.itertuples(), 1):
        brow.append({"번호": m, "쌍": f"{r.pun_word} → {r.w_a}", "w_p 뜻": short_gloss(r.w_p) if isinstance(r.w_p, str) else "",
                     "w_a 뜻": short_gloss(r.w_a) if isinstance(r.w_a, str) else "", "문장": mark(str(r.text), r.start, r.end),
                     **{c: "" for c in YN_COLS}, "웃김 (1~5)": ""})
        key.append(dict(sheet="b0", no=m, block=None, cand_id=r.cand_id, conditions="B0", cell=None, pair_id=None,
                        format=None, source="B0", pair_ok=bool(r.rule_ok) or not str(r.reasons).startswith("pair")))
    pd.DataFrame(rows).to_csv(hd / "cells_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(brow).to_csv(hd / "b0_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(hd / "key.csv", index=False)
    (hd / "guide.txt").write_text(GUIDE, encoding="utf-8")
    print(f"칸 묶음 {len(cells)}블록 {len(rows)}문장, B0 {len(brow)}문장 → {hd}")


def retest(adir: Path, stage: str):
    hd = adir / "human"
    K = pd.read_csv(hd / "key.csv")
    C = pd.read_csv(hd / "cells_sheet.csv", encoding="utf-8-sig")
    rng = np.random.default_rng(stable_seed(stage, "human2-retest"))
    blocks = sorted(K[K.sheet == "cells"].block.dropna().astype(int).unique())
    pick = sorted(rng.choice(blocks, min(N_RETEST, len(blocks)), replace=False))
    rows, key, n = [], [], 0
    for b2, b in enumerate([pick[i] for i in rng.permutation(len(pick))], 1):
        g = C[C["블록"] == b]
        for _, r in g.iloc[rng.permutation(len(g))].iterrows():
            n += 1
            rows.append({"블록": b2, "번호": n, "쌍": r["쌍"], "w_p 뜻": r["w_p 뜻"], "w_a 뜻": r["w_a 뜻"], "문장": r["문장"],
                         **{c: "" for c in YN_COLS}, "웃김 순위": ""})
            key.append(dict(no=n, block=b2, orig_no=int(r["번호"]), orig_block=int(b)))
    pd.DataFrame(rows).to_csv(hd / "retest_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(hd / "retest_key.csv", index=False)
    print(f"재검사 {len(pick)}블록 {n}문장 → {hd / 'retest_sheet.csv'}")


def calib():
    from judge_validate import he_items
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    H = he_items()
    rng = np.random.default_rng(stable_seed("human2-calib"))
    gold = H[H.method == 5]
    mach = H[H.method != 5]
    take = pd.concat([gold.iloc[rng.choice(len(gold), N_CALIB["gold"], replace=False)],
                      mach.iloc[rng.choice(len(mach), N_CALIB["machine"], replace=False)]])
    take = take.iloc[rng.permutation(len(take))].reset_index(drop=True)
    import re
    rows, key = [], []
    for i, r in take.iterrows():
        m = re.search(r"(?<![A-Za-z])" + re.escape(r.w_p) + r"(?![A-Za-z])", r.text, re.I)
        rows.append({"번호": i + 1, "쌍": f"{r.w_p} → {r.w_a}", "문장": mark(r.text, *(m.span() if m else (None, None))),
                     "(c) 성공한 말장난": ""})
        key.append(dict(no=i + 1, item=r["item"], method=int(r.method), human_pun=bool(r.human_pun)))
    pd.DataFrame(rows).to_csv(CALIB_DIR / "calib_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(CALIB_DIR / "calib_key.csv", index=False)
    print(f"보정 {len(rows)}문장 → {CALIB_DIR / 'calib_sheet.csv'}")


# ---------------------------------------------------------------- 채점
def boot_mean(groups: list[np.ndarray], rng) -> np.ndarray:
    return np.array([np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
                     for _ in range(B_BOOT)])


def ci(est, reps) -> dict:
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi))


def norm_rank(df: pd.DataFrame) -> pd.Series:
    """블록 안 웃김 순위(1 = 가장 웃김, 동점 허용) → 0~1 (1 = 가장 웃김)."""
    r = pd.to_numeric(df["웃김 순위"], errors="coerce")
    rr = r.groupby(df["블록"]).rank(method="average")
    n = r.groupby(df["블록"]).transform("count")
    return (1 - (rr - 1) / (n - 1)).where(n > 1)


def kappa(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def score_calib() -> dict | None:
    f = CALIB_DIR / "calib_sheet.csv"
    if not f.exists():
        return None
    S = pd.read_csv(f, encoding="utf-8-sig").merge(pd.read_csv(CALIB_DIR / "calib_key.csv"), left_on="번호", right_on="no")
    S["ans"] = S["(c) 성공한 말장난"].map(yn)
    if S.ans.isna().all():
        return None
    S[["item", "method", "human_pun", "ans"]].to_csv(CALIB_DIR / "calib_answers.csv", index=False)  # 문장 없이 id와 답만
    M = S[S.method != 5].dropna(subset=["ans"])
    bal = (M[M.human_pun].ans.mean() + (1 - M[~M.human_pun].ans).mean()) / 2
    return dict(n=int(S.ans.notna().sum()), machine_bal_acc=float(bal), machine_n=len(M),
                gold_yes=float(S[S.method == 5].ans.mean()), human_one_vs_two=0.631, judge_threshold=0.58)


def score(adir: Path) -> dict:
    hd = adir / "human"
    K = pd.read_csv(hd / "key.csv")
    C = pd.read_csv(hd / "cells_sheet.csv", encoding="utf-8-sig")
    B = pd.read_csv(hd / "b0_sheet.csv", encoding="utf-8-sig")
    for d in (C, B):
        for c in YN_COLS:
            d[c] = d[c].map(yn)
    C["norm_rank"] = norm_rank(C)
    KC = K[K.sheet == "cells"].merge(C, left_on="no", right_on="번호")
    KB = K[K.sheet == "b0"].merge(B, left_on="no", right_on="번호")
    KB["valid"] = KB["(c) 성공한 말장난"].where(KB.pair_ok.astype(bool), 0.0)
    KC["valid"] = KC["(c) 성공한 말장난"]
    long = KC.assign(condition=KC.conditions.str.split(",")).explode("condition")  # 문장 하나를 고른 조건마다
    W = long.pivot_table(index=["cell", "pair_id", "format"], columns="condition", values=["valid", "norm_rank"], aggfunc="first")
    rng = np.random.default_rng(SEED)
    res = dict(n_cells=int(KC.cell.nunique()), n_items=len(KC), n_b0=len(KB),
               missing=dict(cells_c=int(KC.valid.isna().sum()), b0_c=int(KB["(c) 성공한 말장난"].isna().sum()),
                            rank=int(KC.norm_rank.isna().sum())))
    res["by_condition"] = {c: dict(valid=float(W["valid"][c].mean()), funny_rank=float(W["norm_rank"][c].mean()))
                           for c in CELL_CONDS if c in W["valid"]}
    res["by_condition"]["B0"] = dict(valid=float(KB.valid.mean()), funny_1to5=float(pd.to_numeric(KB["웃김 (1~5)"], errors="coerce").mean()))
    pair_groups = lambda s: [g.to_numpy(float) for _, g in s.dropna().groupby(level="pair_id")]

    def paired(col, a, b):
        d = (W[col][a] - W[col][b]).dropna()
        return ci(d.mean(), boot_mean(pair_groups(d), rng))

    # H5a: P1(쌍 묶음) − B0(항목)
    p1 = W["valid"]["P1"].dropna()
    b0 = KB.valid.dropna().to_numpy(float)
    reps = boot_mean(pair_groups(p1), rng) - np.array([b0[rng.integers(0, len(b0), len(b0))].mean() for _ in range(B_BOOT)])
    res["H5a"] = ci(p1.mean() - b0.mean(), reps)
    res["H5a"]["passed"] = res["H5a"]["lo"] > 0
    res["H5b"] = paired("valid", "P2", "P1")
    res["H5b"]["passed"] = res["H5b"]["lo"] > 0
    res["H5c_explore"] = paired("norm_rank", "P2", "P1")
    res["H5d_explore"] = dict(P2_P3_valid=paired("valid", "P2", "P3"), P2_P3_funny=paired("norm_rank", "P2", "P3"))
    res["P1_B1"] = paired("valid", "P1", "B1")
    nb = KB[KB["(d) 들어 본 농담"] != 1].valid.dropna().to_numpy(float)
    res["H5a_known_removed"] = dict(est=float(p1.mean() - nb.mean()) if len(nb) else None, n_b0=len(nb))
    res["H5e_explore"] = {f: dict(valid=float(g.valid.mean()), funny_rank=float(g.norm_rank.mean()))
                          for f, g in KC.groupby("format")}
    res["both_meanings"] = float(((KC["(a) w_p 뜻으로 말이 됨"] == 1) & (KC["(b) w_a로 읽어도 말이 됨"] == 1)).mean())
    # 쌍 층
    try:
        from e5v2 import load_samples_v4
        strat = {p["pair_id"]: p["stratum"] for p in load_samples_v4()["main"] + load_samples_v4()["pilot6"]}
        P = KC[KC.source == "P"]
        res["by_stratum"] = {k: float(g.valid.mean()) for k, g in P.groupby(P.pair_id.map(strat))}
    except Exception:
        pass
    # phi-4와 사람 (c)
    jf = adir / "judge2_phi-4-4bit" / "items.csv"
    if jf.exists():
        J = pd.read_csv(jf).set_index("cand_id")
        x = KC.assign(phi=KC.cand_id.map(J.works).map(lambda v: np.nan if pd.isna(v) else float(str(v) in ("True", "1.0"))))
        x = x.dropna(subset=["phi", "valid"])
        res["phi4_vs_human"] = dict(n=len(x), agree=float((x.phi == x.valid).mean()), kappa=kappa(x.phi, x.valid),
                                    phi_yes=float(x.phi.mean()), human_yes=float(x.valid.mean()))
    # 재검사
    rf = hd / "retest_sheet.csv"
    if rf.exists():
        R = pd.read_csv(rf, encoding="utf-8-sig")
        for c in YN_COLS:
            R[c] = R[c].map(yn)
        if R["(c) 성공한 말장난"].notna().any():
            R["norm_rank"] = norm_rank(R)
            R = R.merge(pd.read_csv(hd / "retest_key.csv"), left_on="번호", right_on="no").merge(
                C[["번호", "(c) 성공한 말장난", "norm_rank"]], left_on="orig_no", right_on="번호", suffixes=("", "_1"))
            res["retest"] = dict(n=len(R), kappa_c=kappa(R["(c) 성공한 말장난"], R["(c) 성공한 말장난_1"]),
                                 rank_spearman=float(R.norm_rank.corr(R.norm_rank_1, method="spearman")))
    cal = score_calib()
    if cal:
        res["calibration"] = cal
    dump(hd / "metrics.json", res)
    f = lambda v: f"{v['est']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
    L = [f"# E5 사람 평가 (연구자 1명, 블라인드) — {adir.parent.parent.name}", "",
         f"칸 {res['n_cells']}개, 칸 묶음 문장 {res['n_items']}개, B0 {res['n_b0']}개. 빈 답: {res['missing']}", "",
         "| 가설 | 값 [95%, 쌍 묶음 재표집 5,000회] | 판정 |", "|---|---|---|",
         f"| H5a 성립(P1) − 성립(B0) | {f(res['H5a'])} | {'통과' if res['H5a']['passed'] else '실패'} |",
         f"| H5b 성립(P2) − 성립(P1) | {f(res['H5b'])} | {'통과' if res['H5b']['passed'] else '실패'} |",
         f"| H5c (탐색) 웃김 순위 P2 − P1 | {f(res['H5c_explore'])} | — |",
         f"| H5d (탐색) 성립 P2 − P3 | {f(res['H5d_explore']['P2_P3_valid'])} | — |",
         f"| (탐색) 성립 P1 − B1 | {f(res['P1_B1'])} | — |", "",
         "| 조건 | 성립 | 웃김 |", "|---|---|---|"]
    for c, v in res["by_condition"].items():
        L.append(f"| {c} | {v['valid']:.3f} | {v.get('funny_rank', v.get('funny_1to5')):.3f} |")
    L += ["", f"두 뜻 모두((a)·(b) 예) {res['both_meanings']:.3f}. 형식별 {res['H5e_explore']}. 쌍 층별 {res.get('by_stratum')}"]
    for k in ("phi4_vs_human", "retest", "calibration"):
        if k in res:
            L.append(f"{k}: {res[k]}")
    (hd / "table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "make", "retest", "score"])
    ap.add_argument("--stage", default="main")
    ap.add_argument("--gen", default="olmo-2-0325-32b-instruct-4bit")
    ap.add_argument("--model", default="pythia-1.4b")
    a = ap.parse_args()
    if a.cmd == "calib":
        return calib()
    adir = stage_dir(a.stage) / a.gen / f"A_{a.model}"
    {"make": lambda: make(adir, a.stage), "retest": lambda: retest(adir, a.stage), "score": lambda: score(adir)}[a.cmd]()


if __name__ == "__main__":
    main()
