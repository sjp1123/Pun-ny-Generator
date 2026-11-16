"""E5 v2 탐색: 말장난 지표와 LLM 판정의 칸 안 순위 상관 (H5d의 미리보기, 가설 검정 아님).

지표 (P 후보 ③, 청자 Pythia-1.4B 등은 e5v2_score가 이미 계산):
  S_left (놀람, −log π_←(w_p)), logF_g (해소), A_g (모호성, 관문), score (P2 = logF_g + S_left),
  S_ratio (He), sent_lp (문장 로그 확률/토큰), unusualness (He, scores.csv에 있을 때만: 파일럿 4부터),
  Kao 모호성·뚜렷함 (word2vec GoogleNews, z·r는 확인 실험의 dev 적합값 results/pythia-1.4b/confirm/config.json)
판정:
  success (w_a가 들리는가, 칸의 모든 P 후보), funny_bt (웃김 BT, 고른 P 후보만 = 칸당 최대 4개)
칸 효과를 빼려고 지표·판정을 칸 안에서 순위로 바꾼 뒤(칸 평균 0으로 맞춘 순위) 모아서 Spearman ρ를 구한다.
95% 구간은 칸 묶음 재표집 2,000회. 칸 안에서 판정이 모두 같은 칸은 정보가 없어 상관에 기여하지 않는다.

사용: python -X utf8 experiments/e5v2_metrics.py --stage pilot3 --gen olmo-2-0325-32b-instruct-4bit
출력: results/e5/<stage>/<gen>/A_<model>/metrics_vs_judge.csv, .md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from baselines import kao_cosines, kao_features
from data import CACHE, Item
from e5 import stage_dir
from e5_gen import tag_of

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["score", "S_left", "logF_g", "A_g", "S_ratio", "sent_lp", "unusualness", "ambiguity", "distinctiveness"]
LABEL = dict(score="P2 점수 (logF + S_←)", S_left="S_← 놀람", logF_g="log F 해소", A_g="A 모호성 (관문)",
             S_ratio="He S_ratio", sent_lp="문장 로그 확률/토큰", unusualness="He unusualness",
             ambiguity="Kao 모호성", distinctiveness="Kao 뚜렷함")
N_BOOT = 2000


def centered_rank(df: pd.DataFrame, col: str) -> pd.Series:
    r = df.groupby("cell")[col].rank()
    return r - r.groupby(df.cell).transform("mean")


def rho_by_cell(df: pd.DataFrame, x: str, y: str, rng) -> dict:
    d = df[["cell", x, y]].dropna()
    d = d[d.groupby("cell")[y].transform("nunique") > 1]  # 칸 안에서 판정이 갈린 칸만
    if d.cell.nunique() < 3:
        return dict(rho=np.nan, lo=np.nan, hi=np.nan, n=len(d), cells=d.cell.nunique())
    rx, ry = centered_rank(d, x), centered_rank(d, y)
    f = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
    est = f(rx, ry)
    cells = d.cell.unique()
    idx = {c: np.nonzero(d.cell.to_numpy() == c)[0] for c in cells}
    bs = []
    for _ in range(N_BOOT):
        take = np.concatenate([idx[c] for c in rng.choice(cells, len(cells))])
        bs.append(f(rx.to_numpy()[take], ry.to_numpy()[take]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(rho=est, lo=float(lo), hi=float(hi), n=len(d), cells=len(cells))


def kao(cand: pd.DataFrame, tag: str) -> pd.DataFrame:
    cfg = json.loads((ROOT / "results/pythia-1.4b/confirm/config.json").read_text(encoding="utf-8"))
    P = next(v for v in _walk(cfg) if isinstance(v, dict) and "kao_r" in v)
    its = [Item(id=r.cand_id, text=r.text, start=int(r.start), end=int(r.end), w_p=r.w_p, w_a=r.w_a)
           for r in cand.itertuples()]
    cos = kao_cosines(its, CACHE / f"kao_cosines_e5_{tag}.pkl")  # 후보 번호가 생성기끼리 겹치므로 따로
    return kao_features(cos, [it.id for it in its], P["kao_mu"], P["kao_sd"], P["kao_r"])


def _walk(o):
    yield o
    if isinstance(o, dict):
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    a = ap.parse_args()
    gdir = stage_dir(a.stage) / a.gen
    adir = gdir / f"A_{a.model}"
    jdir = adir / f"judge_{tag_of(a.judge)}"
    cand = pd.read_csv(gdir / "candidates.csv")
    cand = cand[cand.source == "P"]
    sc = pd.read_csv(adir / "scores.csv").set_index("cand_id")
    D = cand[["cand_id", "cell", "w_p", "w_a", "text", "start", "end"]].set_index("cand_id")
    D = D.join(sc.reindex(columns=[m for m in METRICS if m in sc.columns])).join(kao(cand.reset_index(), f"{a.stage}_{a.gen}"))
    D["S_ratio"] = D.S_ratio.where(D.S_ratio >= 0)  # −1은 "정의 안 됨"
    succ = pd.read_csv(jdir / "success.csv").set_index("cand_id")
    D["success"] = succ.success.reindex(D.index).astype(float)
    bt = pd.read_csv(jdir / "bt.csv").drop_duplicates("cand_id").set_index("cand_id").bt
    D["funny_bt"] = bt.reindex(D.index)
    D = D.reset_index()
    rng = np.random.default_rng(0)
    rows = []
    for m in METRICS:
        if m not in D or D[m].notna().sum() == 0:
            continue
        for y in ("success", "funny_bt"):
            rows.append(dict(metric=m, judge=y, **rho_by_cell(D, m, y, rng)))
    R = pd.DataFrame(rows)
    R.to_csv(adir / "metrics_vs_judge.csv", index=False)
    D.drop(columns=["start", "end"]).to_csv(adir / "metrics_table.csv", index=False)
    fmt = lambda r: "—" if np.isnan(r.rho) else f"{r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] (n {r.n}, 칸 {r.cells})"
    L = [f"# 지표 vs LLM 판정 (탐색) — {a.stage}, {a.gen}", "",
         "칸 안 순위 상관(Spearman, 칸 평균 0으로 맞춘 순위). [ ]는 칸 묶음 재표집 95% 구간. 판정이 갈린 칸만 쓴다.",
         f"P 후보 {len(D)}개, 칸 {D.cell.nunique()}개. 웃김 BT는 고른 P 후보에만 있다(같은 후보를 여러 조건이 고르면 한 번).", "",
         "| 지표 | 판정기 성공 (w_a가 들림) | 판정기 웃김 BT |", "|---|---|---|"]
    for m in METRICS:
        g = R[R.metric == m].set_index("judge")
        if len(g):
            L.append(f"| {LABEL[m]} | {fmt(g.loc['success'])} | {fmt(g.loc['funny_bt'])} |")
    if "unusualness" not in sc.columns:
        L += ["", "unusualness는 이 결과의 scores.csv에 없다(파일럿 4부터 저장)."]
    (adir / "metrics_vs_judge.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
