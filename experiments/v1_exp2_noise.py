"""실험 2: 웃김 점수의 잡음 상한.

주 점수: Bradley–Terry (choix ILSR, α = 0.01). X(우열 없음)는 양쪽에 0.5승.
  choix는 가중치를 받지 않아서 승리는 2번, 무승부는 양방향 1번씩 넣고 α를 0.02로 두 배 한다
  (전이율과 정규화 항이 함께 두 배가 되어 α = 0.01, 0.5승과 같은 해).
보조 점수: 승률 (승 + 0.5·X) / 등장 횟수.
반분: 평가자 1,059명을 무작위 반씩(200회), 보조로 쌍마다 판단 5개를 2/3으로 나눔(200회).
보정: r_full = 2 r_half / (1 + r_half), ρ_max ≈ √r_full.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import choix

from data import UKP_DIR, norm, semeval_items
from rules import SEED, RunDir, split_ids

ALPHA = 0.01
N_REP = 200


def load_judgments():
    r = pd.read_csv(UKP_DIR / "results.tsv", sep="\t", quoting=3, dtype=str, keep_default_na=False)
    s = pd.read_csv(UKP_DIR / "item_scores.txt", sep="\t", header=None, names=["id", "score", "text"],
                    quoting=3, dtype={"text": str}, keep_default_na=False)
    k2i = {norm(t): i for i, t in zip(s.id, s.text)}
    J = pd.DataFrame(dict(worker=r["Worker ID"], a=r["Text A"].map(lambda t: k2i[norm(t)]),
                          b=r["Text B"].map(lambda t: k2i[norm(t)]), ans=r.Answer))
    return J, s.set_index("id").score


def bt(J: pd.DataFrame, n: int) -> np.ndarray:
    a, b, ans = J.a.to_numpy(), J.b.to_numpy(), J.ans.to_numpy()
    wa, wb, x = ans == "A", ans == "B", ans == "X"
    data = ([(i, j) for i, j in zip(a[wa], b[wa])] * 2 + [(j, i) for i, j in zip(a[wb], b[wb])] * 2
            + [(i, j) for i, j in zip(a[x], b[x])] + [(j, i) for i, j in zip(a[x], b[x])])
    return choix.ilsr_pairwise(n, data, alpha=2 * ALPHA)


def winrate(J: pd.DataFrame, n: int) -> np.ndarray:
    w = np.zeros(n)
    c = np.zeros(n)
    for col, win in (("a", "A"), ("b", "B")):
        idx = J[col].to_numpy()
        np.add.at(w, idx, (J.ans == win).to_numpy() + 0.5 * (J.ans == "X").to_numpy())
        np.add.at(c, idx, 1)
    return np.where(c > 0, w / np.maximum(c, 1), np.nan)


def rs(x, y, idx):
    return float(spearmanr(x[idx], y[idx]).statistic)


def sb(r):
    return 2 * r / (1 + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=N_REP)
    args = ap.parse_args()
    J, gold = load_judgments()
    n = len(gold)
    pun_all = sorted({it.meta["ukp_id"] for it in semeval_items(require_single_word=True)})
    test_ids = split_ids("test")
    pun_test = sorted({it.meta["ukp_id"] for it in semeval_items(require_single_word=True) if it.id in test_ids})
    sets = {"말장난 1,074": np.array(pun_all), "말장난 test": np.array(pun_test), "전체 4,030": np.arange(n)}
    g = gold.reindex(range(n)).to_numpy()

    full_bt, full_wr = bt(J, n), winrate(J, n)
    check = {name: dict(bt=rs(full_bt, g, idx), winrate=rs(full_wr, g, idx)) for name, idx in sets.items()}
    check["bt_vs_winrate_all"] = rs(full_bt, full_wr, np.arange(n))

    rng = np.random.default_rng(SEED)
    workers = J.worker.unique()
    pair = pd.Series([tuple(sorted(t)) for t in zip(J.a, J.b)], index=J.index)
    groups = pair.groupby(pair).indices
    res = {"평가자": {k: [] for k in sets}, "판단": {k: [] for k in sets}}
    res_wr = {"평가자": {k: [] for k in sets}, "판단": {k: [] for k in sets}}
    for rep in range(args.reps):
        half = set(rng.permutation(workers)[: len(workers) // 2])
        m = J.worker.isin(half).to_numpy()
        h1, h2 = J[m], J[~m]
        b1, b2 = bt(h1, n), bt(h2, n)
        w1, w2 = winrate(h1, n), winrate(h2, n)
        for k, idx in sets.items():
            res["평가자"][k].append(rs(b1, b2, idx))
            res_wr["평가자"][k].append(rs(w1, w2, idx[~np.isnan(w1[idx]) & ~np.isnan(w2[idx])]))
        # 판단 단위: 쌍마다 판단을 무작위로 섞어 앞 2개 / 나머지로 (3개 쪽은 무작위)
        sel = np.zeros(len(J), bool)
        for idx in groups.values():
            p = rng.permutation(idx)
            k2 = 2 if rng.random() < 0.5 else len(p) - 2
            sel[p[:k2]] = True
        d1, d2 = J[sel], J[~sel]
        c1, c2 = bt(d1, n), bt(d2, n)
        v1, v2 = winrate(d1, n), winrate(d2, n)
        for k, idx in sets.items():
            res["판단"][k].append(rs(c1, c2, idx))
            res_wr["판단"][k].append(rs(v1, v2, idx[~np.isnan(v1[idx]) & ~np.isnan(v2[idx])]))
        if rep % 20 == 0:
            print(f"{rep}/{args.reps}", flush=True)

    def summ(v):
        v = np.array(v)
        m = float(v.mean())
        return dict(r_half_mean=m, r_half_lo=float(np.percentile(v, 2.5)), r_half_hi=float(np.percentile(v, 97.5)),
                    r_full=sb(m), rho_max=float(np.sqrt(sb(m))))
    table = {split: {k: summ(v) for k, v in d.items()} for split, d in res.items()}
    table_wr = {split: {k: summ(v) for k, v in d.items()} for split, d in res_wr.items()}
    R = RunDir("no-lm", "exp2_noise", dict(alpha=ALPHA, ties="0.5승 (승 2회·무승부 양방향 1회, α×2)",
                                          reps=args.reps, n_workers=len(workers), n_judgments=len(J)))
    R.metrics(dict(score_check=check, bt=table, winrate=table_wr, sets={k: len(v) for k, v in sets.items()}))
    rows = [("말장난 1,074", "평가자"), ("말장난 test", "평가자"), ("말장난 1,074", "판단"), ("전체 4,030", "평가자"),
            ("말장난 test", "판단"), ("전체 4,030", "판단")]
    L = ["# 실험 2 웃김 점수의 잡음 상한", "",
         f"판단 {len(J):,}개, 평가자 {len(workers):,}명, 반복 {args.reps}회, 시드 {SEED}", "",
         "## 점수 검증 (전체 판단으로 계산한 점수 vs item_scores.txt, 스피어만)", "",
         "| 집합 | Bradley–Terry | 승률 |", "|---|---|---|"]
    for k in sets:
        L.append(f"| {k} | {check[k]['bt']:.3f} | {check[k]['winrate']:.3f} |")
    L += ["", f"BT vs 승률 (전체): {check['bt_vs_winrate_all']:.3f}", "",
          "## 잡음 상한 (Bradley–Terry)", "", "| 집합 | 분할 방식 | r_half 평균 (95% 구간) | r_full | ρ_max |", "|---|---|---|---|---|"]
    for k, sp in rows:
        t = table[sp][k]
        L.append(f"| {k} | {sp} | {t['r_half_mean']:.3f} ({t['r_half_lo']:.3f}, {t['r_half_hi']:.3f}) | {t['r_full']:.3f} | {t['rho_max']:.3f} |")
    L += ["", "## 보조: 승률 점수", "", "| 집합 | 분할 방식 | r_half 평균 | r_full | ρ_max |", "|---|---|---|---|---|"]
    for k, sp in rows:
        t = table_wr[sp][k]
        L.append(f"| {k} | {sp} | {t['r_half_mean']:.3f} | {t['r_full']:.3f} | {t['rho_max']:.3f} |")
    gap = table["평가자"]["전체 4,030"]["r_full"] - table["평가자"]["말장난 1,074"]["r_full"]
    L += ["", f"전체 4,030 − 말장난 1,074 r_full 차이 (평가자 분할): {gap:.3f}"]
    R.table("table.md", L)


if __name__ == "__main__":
    main()
