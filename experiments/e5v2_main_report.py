"""E5 본 실행 분석 (PREREG_E5 7.1): phi-4 성립으로 H5a·H5b를 판정하고 보조 지표를 함께 보고한다.

입력 (모두 이미 있는 파일, 판정기와 생성기는 다시 돌리지 않는다)
  selections.csv                        조건별로 고른 문장 (e5v2_select.py)
  judge2_<판정기>/items.csv              phi-4 성립(works)·웃김(funny) (e5v2_judge2.py)
  candidates.csv + 사람 말장난            Kao 지표용 문장·위치·쌍
  novelty_<색인>.csv                     8단어 이상 일치 표시 (e5_novelty.py). main에서는 먼저 만들어 둬야 한다

판정 (3.4와 7.1)
  성립 = 고른 문장(③)에 phi-4가 works_pair yes. 판정하지 못한 문장(쌍이 유효하지 않은 B0, 응답을 못 읽음, 문장 없음)은 0
  H5a  성립(P1) − 성립(B0). P1은 쌍 묶음, B0은 항목 재표집(짝 없음). 95% 하한 > 0이면 지지
  H5b  칸별 성립(P2) − 성립(P1)의 평균, 쌍 묶음 재표집. 95% 하한 > 0이면 지지
  재표집 B = 5,000, 95% 백분위, 시드 20261112. 통계마다 같은 시드로 새 난수열을 만든다(계산 순서와 무관하게 재현)
민감도: 오염 표시 문장을 뺀 H5a(표시된 P1·B0 문장을 뺌)와 H5b(P1이나 P2가 표시된 칸을 뺌). 조회 실패는 표시 안 됨으로 본다
보조 (판정에 쓰지 않음): 조건별 성립, 칸 짝 차이(P2−P3, P1−B1, P2−B1, P4−P1),
  형식별 H5a·H5b(H5e, B0은 형식이 없어 형식별 H5a는 그 형식의 P1 vs B0 전체),
  Kao ambiguity·distinctiveness, distinct-2, template rate, overlap rate, phi-4 웃김(1~5)
  H(사람 말장난)는 40개가 각 2칸에 붙으므로 조건별 값은 고유 문장 40개로 센다. P4는 S_ratio가 정의되지 않은 칸을 뺀다.

B0 쌍 규칙 (7.2, --b0-rule)
  3.2     (기본, 7.2 정정) 3.2절과 사람 채점(e5v2_human.make의 pair_ok)과 같다. b0_pair_reasons에 걸린 B0
          (후보표 rule_ok가 거짓이고 reasons가 pair로 시작)은 성립 0, Kao 지표에서도 뺀다. 출력 analysis_7.2/
  judge2  7.1대로 실행한 규칙. e5v2_judge2.py가 판정하지 않은 B0(쌍이 비었거나 두 단어가 같음)만 0. 출력 analysis_7.1/

화면에는 결과 숫자를 출력하지 않는다(연구자가 블라인드 평가를 할지 정하기 전에 결과를 보지 않도록). 결과는 파일로만 남긴다.

사용: python -X utf8 experiments/e5v2_main_report.py --stage main --gen olmo-2-0325-32b-instruct-4bit
      (시험) --stage pilot7 --novelty <임의 표시 파일>
출력: results/e5/<stage>/<gen>/A_<model>/analysis_7.2/ (또는 analysis_7.1/) config.json, metrics.json, table.md, rows.csv, kao.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from e5 import stage_dir, write_config
from e5_score import human_rows
from e5_select import distinct2, template_rate
from rules import SEED, dump

B_BOOT = 5000
CONDS = ["B0", "B1", "P1", "P2", "P3", "P4", "H"]
ITEM_RESAMPLE = {"B0", "H"}  # 짝이 없는 조건은 항목 재표집, 나머지는 쌍 묶음
PAIRED = [("P2", "P3"), ("P1", "B1"), ("P2", "B1"), ("P4", "P1")]
PREREG = {"3.2": "7.2", "judge2": "7.1"}
assert SEED == 20261112


def b0_pair_ok(r) -> bool:
    """3.2절의 B0 쌍 유효성. e5v2_human.make의 pair_ok와 같은 식(생성 때 b0_pair_reasons를 reasons에 적었다)."""
    return bool(r.rule_ok) or not str(r.reasons).startswith("pair")


def to01(x) -> float:
    """phi-4 응답(True/False/None, 문자열 포함) → 1/0/NaN."""
    if x is True or x == "True":
        return 1.0
    if x is False or x == "False":
        return 0.0
    return np.nan


# ---------- 재표집 ----------

def reps_cluster(d: pd.DataFrame, col: str, rng, cluster="pair_id") -> np.ndarray:
    """쌍 묶음 재표집으로 d[col] 평균의 재표집 값 B개."""
    g = d.groupby(cluster)[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    idx = rng.integers(0, len(g), (B_BOOT, len(g)))
    return s[idx].sum(1) / c[idx].sum(1)


def reps_item(x: np.ndarray, rng) -> np.ndarray:
    x = np.asarray(x, float)
    return x[rng.integers(0, len(x), (B_BOOT, len(x)))].mean(1)


def ci(est: float, reps: np.ndarray, n: int, **extra) -> dict:
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi), n=int(n), **extra)


def mean_ci(d: pd.DataFrame, col: str, item: bool) -> dict:
    d = d.dropna(subset=[col])
    if not len(d):
        return dict(est=np.nan, lo=np.nan, hi=np.nan, n=0)
    rng = np.random.default_rng(SEED)
    reps = reps_item(d[col], rng) if item else reps_cluster(d, col, rng)
    return ci(d[col].mean(), reps, len(d))


def verdict(r: dict) -> str:
    return "—" if r["n"] == 0 or np.isnan(r["lo"]) else ("지지" if r["lo"] > 0 else "지지 안 됨")


# ---------- 가설 ----------

def h5a(M: pd.DataFrame) -> dict:
    """성립(P1) − 성립(B0). P1 쌍 묶음, B0 항목, 서로 독립."""
    p1, b0 = M[M.condition == "P1"], M[M.condition == "B0"]
    if not len(p1) or not len(b0):
        return dict(est=np.nan, lo=np.nan, hi=np.nan, n=0)
    rng = np.random.default_rng(SEED)
    r1 = reps_cluster(p1, "works", rng)
    r0 = reps_item(b0.works.to_numpy(), rng)
    r = ci(p1.works.mean() - b0.works.mean(), r1 - r0, len(p1) + len(b0), n_P1=len(p1), n_B0=len(b0),
           P1=float(p1.works.mean()), B0=float(b0.works.mean()))
    r["verdict"] = verdict(r)
    return r


def cell_diff(M: pd.DataFrame, a: str, b: str, col="works") -> pd.DataFrame:
    """칸마다 a − b (두 조건 모두 값이 있는 칸만)."""
    A = M[M.condition == a].set_index("cell")[["pair_id", "format", col, "cand_id"]]
    B = M[M.condition == b].set_index("cell")[[col, "cand_id"]]
    j = A.join(B, rsuffix="_b", how="inner").dropna(subset=[col, f"{col}_b"])
    j["d"] = j[col] - j[f"{col}_b"]
    j["same"] = j.cand_id == j.cand_id_b
    return j.reset_index()


def paired(M: pd.DataFrame, a: str, b: str, col="works") -> dict:
    j = cell_diff(M, a, b, col)
    if not len(j):
        return dict(est=np.nan, lo=np.nan, hi=np.nan, n=0)
    r = ci(j.d.mean(), reps_cluster(j, "d", np.random.default_rng(SEED)), len(j),
           pairs=int(j.pair_id.nunique()), same_candidate=int(j.same.sum()))
    r["verdict"] = verdict(r)
    return r


# ---------- 보조 지표 ----------

def kao_table(cand: pd.DataFrame, ids: list, tag: str) -> pd.DataFrame:
    """고른 문장의 Kao ambiguity·distinctiveness. 위치나 쌍이 없는 문장(유효하지 않은 B0 등)은 뺀다."""
    from data import CACHE, Item
    from e5v2_metrics import kao
    c = cand.loc[[i for i in ids if i in cand.index]].reset_index()
    ok = (c.start.notna() & c.end.notna() & c.w_p.astype(str).str.strip().ne("") & c.w_a.notna()
          & c.w_a.astype(str).str.strip().ne("") & (c.w_p.astype(str).str.lower() != c.w_a.astype(str).str.lower()))
    c = c[ok]
    if not len(c):
        return pd.DataFrame(columns=["ambiguity", "distinctiveness"])
    return kao(c, tag)


def diversity(M: pd.DataFrame) -> dict:
    out = {}
    for cond in CONDS:
        u = M[(M.condition == cond) & M.text.notna()].drop_duplicates("cand_id")
        if len(u):
            out[cond] = dict(n=len(u), distinct2=float(distinct2(u.text)), template_rate=float(template_rate(u.text)))
    return out


def by_condition(M: pd.DataFrame, col: str) -> dict:
    out = {}
    for cond in CONDS:
        g = M[M.condition == cond]
        if cond == "H":
            g = g.drop_duplicates("cand_id")
        if cond == "P4":
            g = g[g.cand_id.notna()]
        if len(g):
            out[cond] = mean_ci(g, col, item=cond in ITEM_RESAMPLE)
    return out


# ---------- 표 ----------

def f(r: dict, nd=3, sign=False) -> str:
    if not r or r.get("n", 0) == 0 or np.isnan(r["est"]):
        return "—"
    s = "+" if sign else ""
    return f"{r['est']:{s}.{nd}f} [{r['lo']:{s}.{nd}f}, {r['hi']:{s}.{nd}f}]"


def markdown(a, R: dict) -> str:
    rule = (f"B0 쌍 규칙 3.2(7.2 정정): b0_pair_reasons에 걸린 B0 {R['counts']['b0_pair_invalid']}개는 성립 0, Kao에서 뺌."
            if a.b0_rule == "3.2" else "B0 쌍 규칙 judge2(7.1대로 실행): e5v2_judge2.py가 판정하지 않은 B0만 0.")
    L = [f"# E5 본 실행 분석 (PREREG_E5 {PREREG[a.b0_rule]}) — {a.stage}, {a.gen}", "",
         "성립 = 고른 문장에 phi-4가 works_pair yes. 판정하지 못한 문장은 0. [ ]는 95% 재표집 구간"
         f"(B = {B_BOOT}, 시드 {SEED}). P 조건과 B1은 쌍 묶음, B0과 H는 항목 재표집.", "", rule, "",
         "## 가설 판정", "", "| 가설 | 비교 | 추정 [95%] | n | 판정 |", "|---|---|---|---|---|"]
    h = R["hypotheses"]
    L.append(f"| H5a 쌍 제공 | 성립(P1) − 성립(B0) | {f(h['H5a'], sign=True)} | P1 {h['H5a'].get('n_P1', 0)}, "
             f"B0 {h['H5a'].get('n_B0', 0)} | **{h['H5a'].get('verdict', '—')}** |")
    L.append(f"| H5b A 필터 | 칸별 성립(P2 − P1) | {f(h['H5b'], sign=True)} | {h['H5b']['n']}칸 "
             f"(같은 후보 {h['H5b'].get('same_candidate', 0)}) | **{h['H5b'].get('verdict', '—')}** |")
    s = R["sensitivity"]
    L += ["", "## 민감도: 오염 표시 문장을 뺀 버전", ""]
    if s.get("skipped"):
        L.append(f"계산하지 않음: {s['skipped']}")
    else:
        L += [f"오염 표시(OLMo-mix와 {s['flag_n']}단어 이상 일치) {s['flagged']}/{s['checked']}문장, 조회 실패가 섞인 문장 "
              f"{s['incomplete']}(표시 안 됨으로 봄).", "",
              "| 가설 | 추정 [95%] | n | 하한 > 0 |", "|---|---|---|---|",
              f"| H5a | {f(s['H5a'], sign=True)} | P1 {s['H5a'].get('n_P1', 0)}, B0 {s['H5a'].get('n_B0', 0)} | "
              f"{s['H5a'].get('verdict', '—')} |",
              f"| H5b | {f(s['H5b'], sign=True)} | {s['H5b']['n']}칸 | {s['H5b'].get('verdict', '—')} |"]
        if s.get("note"):
            L += ["", s["note"]]
    L += ["", "## 조건별 (보조)", "",
          "| 조건 | 문장 | 성립 | phi-4 웃김 (1~5) | Kao ambiguity | Kao distinctiveness | distinct-2 | template rate | overlap rate |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cond in CONDS:
        w = R["by_condition"]["works"].get(cond)
        if not w:
            continue
        dv = R["diversity"].get(cond, {})
        ov = R["overlap_rate"].get(cond)
        L.append(f"| {cond} | {w['n']} | {f(w)} | {f(R['by_condition']['funny'].get(cond, {}), 2)} | "
                 f"{f(R['by_condition']['kao_ambiguity'].get(cond, {}), 3)} | "
                 f"{f(R['by_condition']['kao_distinctiveness'].get(cond, {}), 3)} | "
                 f"{dv.get('distinct2', float('nan')):.3f} | {dv.get('template_rate', float('nan')):.3f} | "
                 f"{'—' if ov is None else format(ov, '.3f')} |")
    L += ["", "Kao ambiguity는 A와 같은 성질을 재므로 P2에 유리하게 기울어 있다. 판정에 쓰지 않는다.", "",
          "## 칸 짝 차이 (탐색)", "", "| 비교 | 성립 | 웃김 | Kao distinctiveness | 칸 |", "|---|---|---|---|---|"]
    for k, v in R["paired"].items():
        L.append(f"| {k} | {f(v['works'], sign=True)} | {f(v['funny'], 2, sign=True)} | "
                 f"{f(v['kao_distinctiveness'], 3, sign=True)} | {v['works']['n']} |")
    L += ["", "## 형식별 (H5e, 탐색)", "", "| 형식 | H5a | H5b |", "|---|---|---|"]
    for fm, v in R["by_format"].items():
        L.append(f"| {fm} | {f(v['H5a'], sign=True)} | {f(v['H5b'], sign=True)} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main")
    ap.add_argument("--gen", default="olmo-2-0325-32b-instruct-4bit")
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--judge-tag", default="phi-4-4bit")
    ap.add_argument("--index", default="olmo")
    ap.add_argument("--novelty", default=None, help="오염 표시 파일 (기본: A_<model>/novelty_<색인>.csv)")
    ap.add_argument("--no-kao", action="store_true", help="시험용: word2vec을 불러오지 않는다")
    ap.add_argument("--b0-rule", choices=list(PREREG), default="3.2", help="B0 쌍 규칙 (7.2). judge2는 7.1대로 실행한 규칙")
    a = ap.parse_args()

    gdir = stage_dir(a.stage) / a.gen
    adir = gdir / f"A_{a.model}"
    out = adir / f"analysis_{PREREG[a.b0_rule]}"
    S = pd.read_csv(adir / "selections.csv")
    I = pd.read_csv(adir / f"judge2_{a.judge_tag}" / "items.csv").drop_duplicates("cand_id").set_index("cand_id")
    cand = pd.concat([pd.read_csv(gdir / "candidates.csv"), human_rows()], ignore_index=True)
    cand = cand.drop_duplicates("cand_id").set_index("cand_id")

    # 조건 × 칸 표. 문장이 없거나 판정하지 못한 것은 성립 0 (P4의 정의 안 됨은 by_condition에서 뺀다)
    M = S[["condition", "cell", "pair_id", "format", "cand_id", "text"]].copy()
    M["judged"] = [bool(I.judgeable.get(c, False)) if pd.notna(c) else False for c in M.cand_id]
    M["works"] = [to01(I.works.get(c)) if pd.notna(c) else np.nan for c in M.cand_id]
    M["works"] = M.works.fillna(0.0)
    M["funny"] = [float(I.funny.get(c)) if pd.notna(c) and pd.notna(I.funny.get(c)) else np.nan for c in M.cand_id]
    M.loc[(M.condition == "P4") & M.cand_id.isna(), "works"] = np.nan  # 정의 안 됨: 0이 아니라 빠짐
    # B0 쌍 유효성 (7.2): 3.2 규칙이면 b0_pair_reasons에 걸린 B0은 성립 0, Kao에서도 뺀다
    b0_ok = {cid: b0_pair_ok(r) for cid, r in cand[cand.source == "B0"].iterrows()}
    M["pair_invalid"] = (M.condition == "B0") & M.cand_id.map(lambda c: not b0_ok.get(c, True))
    if a.b0_rule == "3.2":
        M.loc[M.pair_invalid, "works"] = 0.0

    # 오염 표시
    npath = Path(a.novelty) if a.novelty else adir / f"novelty_{a.index}.csv"
    if npath.exists():
        N = pd.read_csv(npath).drop_duplicates("cand_id").set_index("cand_id")
        M["flag"] = [bool(N.flag.get(c, False)) if pd.notna(c) else False for c in M.cand_id]
        M["checked"] = [c in N.index for c in M.cand_id]
    elif a.stage == "main":
        sys.exit(f"오염 표시 파일이 없다: {npath}. 먼저 e5_novelty.py --stage main을 돌린다.")
    else:
        N = None

    # Kao
    kao_rows = M[~M.pair_invalid] if a.b0_rule == "3.2" else M
    ids = list(dict.fromkeys(kao_rows.cand_id.dropna()))
    K = pd.DataFrame(columns=["ambiguity", "distinctiveness"]) if a.no_kao else kao_table(cand, ids, f"{a.stage}_{a.gen}")
    M["kao_ambiguity"] = M.cand_id.map(K["ambiguity"]) if len(K) else np.nan
    M["kao_distinctiveness"] = M.cand_id.map(K["distinctiveness"]) if len(K) else np.nan

    R = dict(hypotheses=dict(H5a=h5a(M), H5b=paired(M, "P2", "P1")))

    if N is None:
        R["sensitivity"] = dict(skipped=f"오염 표시 파일 없음 ({npath.name})")
    else:
        u = M.dropna(subset=["cand_id"]).drop_duplicates("cand_id")
        keep = M[~M.flag]
        bad_cells = set(M[M.condition.isin(["P1", "P2"]) & M.flag].cell)
        R["sensitivity"] = dict(flag_n=8, flagged=int(u.flag.sum()), checked=int(u.checked.sum()),
                                incomplete=int(N.incomplete.reindex(u.cand_id).fillna(False).astype(bool).sum()),
                                H5a=h5a(keep), H5b=paired(M[~M.cell.isin(bad_cells)], "P2", "P1"),
                                note=None if u.checked.all() else f"표시 파일에 없는 문장 {int((~u.checked).sum())}개(표시 안 됨으로 봄)")

    R["by_condition"] = {k: by_condition(M, c) for k, c in
                         (("works", "works"), ("funny", "funny"),
                          ("kao_ambiguity", "kao_ambiguity"), ("kao_distinctiveness", "kao_distinctiveness"))}
    R["diversity"] = diversity(M)
    R["overlap_rate"] = {}
    for cond in CONDS:
        u = M[(M.condition == cond) & M.cand_id.notna()].drop_duplicates("cand_id")
        R["overlap_rate"][cond] = float(u.flag.mean()) if N is not None and len(u) else None
    R["paired"] = {f"{x}-{y}": dict(works=paired(M, x, y), funny=paired(M, x, y, "funny"),
                                    kao_distinctiveness=paired(M, x, y, "kao_distinctiveness")) for x, y in PAIRED}
    R["paired"]["P2-P1"] = dict(works=R["hypotheses"]["H5b"], funny=paired(M, "P2", "P1", "funny"),
                                kao_distinctiveness=paired(M, "P2", "P1", "kao_distinctiveness"))
    # B0은 형식 없이 만들었으므로(칸·쌍·형식 없음) 형식별 H5a는 그 형식의 P1 vs B0 전체
    R["by_format"] = {fm: dict(H5a=h5a(M[(M["format"] == fm) | (M.condition == "B0")]),
                               H5b=paired(M[M["format"] == fm], "P2", "P1"))
                      for fm in sorted(M["format"].dropna().unique())}
    R["counts"] = dict(cells=int(M.cell.nunique()), pairs=int(M.pair_id.nunique()),
                       unjudged={c: int((~M[(M.condition == c)].judged).sum()) for c in CONDS},
                       p4_undefined=int(((M.condition == "P4") & M.cand_id.isna()).sum()),
                       b0_pair_invalid=int(M.pair_invalid.sum()), b0_rule=a.b0_rule)

    write_config(out, dict(stage=a.stage, gen=a.gen, model=a.model, judge_tag=a.judge_tag, novelty=str(npath) if N is not None else None,
                           boot=B_BOOT, prereg=f"PREREG_E5 {PREREG[a.b0_rule]}", b0_rule=a.b0_rule, kao=not a.no_kao))
    M.drop(columns=["text"]).to_csv(out / "rows.csv", index=False)
    if len(K):
        K.to_csv(out / "kao.csv")
    dump(out / "metrics.json", R)
    (out / "table.md").write_text(markdown(a, R), encoding="utf-8")
    print(f"썼다: {out}")


if __name__ == "__main__":
    main()
