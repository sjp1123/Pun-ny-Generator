"""E5 사람 평가: 평가지 내보내기(export)와 분석(analyze). 평가자에게는 조건 이름을 숨기고 순서를 섞는다.

평가 1 (성공): 문장과 말장난 단어를 보여주고 "이 단어가 어떤 다른 단어처럼 들리나요?"(자유 응답),
              이어서 "말장난으로 성립하나요?"(예/아니오). 응답이 w_a와 표제어가 같으면 성공.
              대상: B0, B1, P1~P4, 사람 말장난. 두 조건이 같은 후보를 골랐으면 한 번만 묻는다.
평가 2 (웃김): 같은 칸(쌍 × 형식) 안에서 B1, P1, P2, P3, 사람(H)끼리 두 개씩 "어느 쪽이 더 웃긴가"(--with-p4면 P4도).
              좌우는 무작위. 칸마다 Bradley–Terry(choix ILSR, α = 0.01)로 점수.

응답 파일 형식 (export가 빈 틀을 만든다):
  success_responses.csv : rater, uid, response, works (yes/no)
  pair_responses.csv    : rater, comp_id, choice (L/R)

판정 (사전 등록, PREREG_E5.md):
  H5a 성공률(P1) − 성공률(B0) 95% CI 하한 > 0. B0은 쌍이 없어 짝짓지 않는다(P1은 쌍 묶음, B0은 항목 재표집).
  H5b 성공률(P2) − 성공률(P1) 하한 > 0, 쌍 묶음 재표집(짝지은 방식).
  H5c 칸별 BT(P2) − BT(P1)의 평균 하한 > 0, 쌍 묶음 재표집.
  H5a~c는 수수께끼형·한 줄 상황형 80칸으로 판정한다(Tom Swifty 칸은 H5e 탐색에만).
  오염 표시(8단어 이상 일치) 생성물을 뺀 버전도 함께 보고한다.

사용: python -X utf8 experiments/e5_human.py export  --stage main --gen <생성기> --model pythia-1.4b
      python -X utf8 experiments/e5_human.py analyze --stage main --gen <생성기> --model pythia-1.4b
출력: results/e5/<stage>/<gen>/A_<model>/human/
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from e5 import stable_seed, stage_dir, write_config
from rules import B_BOOT, SEED, dump, lemmas

SUCCESS_CONDS = ["B0", "B1", "P1", "P2", "P3", "P4", "H"]
PAIR_CONDS = ["B1", "P1", "P2", "P3", "H"]
MAIN_FORMATS = ("riddle", "oneliner")


def uid(*parts) -> str:
    return f"{stable_seed('uid', *parts):08x}"[:8]


def export(adir: Path, with_p4: bool):
    hd = adir / "human"
    hd.mkdir(exist_ok=True)
    S = pd.read_csv(adir / "selections.csv")
    S = S[S.text.fillna("").str.len() > 0]
    U = S[S.condition.isin(SUCCESS_CONDS)].drop_duplicates("cand_id").copy()
    U["uid"] = [uid("s", c) for c in U.cand_id]
    rng = np.random.default_rng(stable_seed("human-success-order"))
    U = U.iloc[rng.permutation(len(U))]
    U[["uid", "text", "pun_word"]].to_csv(hd / "success_items.csv", index=False)
    U[["uid", "cand_id", "w_a"]].to_csv(hd / "success_key.csv", index=False)
    conds = PAIR_CONDS + (["P4"] if with_p4 else [])
    comps = []
    for cell, g in S[S.condition.isin(conds) & S.cell.notna()].groupby("cell"):
        ids = list(dict.fromkeys(g.cand_id))
        txt = g.drop_duplicates("cand_id").set_index("cand_id").text
        for x, y in itertools.combinations(ids, 2):
            r = np.random.default_rng(stable_seed("side", cell, x, y)).random() < 0.5
            left, right = (y, x) if r else (x, y)
            comps.append(dict(comp_id=uid("p", cell, x, y), cell=cell, left=left, right=right,
                              left_text=txt[left], right_text=txt[right]))
    C = pd.DataFrame(comps)
    C = C.iloc[np.random.default_rng(stable_seed("human-pair-order")).permutation(len(C))]
    C[["comp_id", "left_text", "right_text"]].to_csv(hd / "pair_items.csv", index=False)
    C[["comp_id", "cell", "left", "right"]].to_csv(hd / "pair_key.csv", index=False)
    pd.DataFrame(columns=["rater", "uid", "response", "works"]).to_csv(hd / "success_responses.csv", index=False)
    pd.DataFrame(columns=["rater", "comp_id", "choice"]).to_csv(hd / "pair_responses.csv", index=False)
    write_config(hd, dict(step="human export", with_p4=with_p4, n_success_items=len(U), n_pairs=len(C)))
    print(f"성공 평가 {len(U)}개, 비교 {len(C)}건 → {hd}  (평가자 3명이면 {3 * (len(U) + len(C))}건)")


def cluster_boot(stat, clusters: np.ndarray, B=B_BOOT, seed=SEED):
    """쌍(묶음) 단위 복원 추출. stat(선택된 묶음 배열) → 실수 배열."""
    uc = np.unique(clusters)
    rng = np.random.default_rng(seed)
    point = np.atleast_1d(stat(uc))
    reps = np.array([np.atleast_1d(stat(rng.choice(uc, len(uc)))) for _ in range(B)], float)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5], axis=0)
    return point, lo, hi


def analyze(adir: Path, index="olmo"):
    import choix
    hd = adir / "human"
    S = pd.read_csv(adir / "selections.csv")
    sk = pd.read_csv(hd / "success_key.csv")
    sr = pd.read_csv(hd / "success_responses.csv")
    pk = pd.read_csv(hd / "pair_key.csv")
    pr = pd.read_csv(hd / "pair_responses.csv")
    if sr.empty or pr.empty:
        raise SystemExit("응답 파일이 비어 있다")
    nov = adir / f"novelty_{index}.csv"
    flagged = set(pd.read_csv(nov).query("flag").cand_id) if nov.exists() else set()

    sr = sr.merge(sk, on="uid")
    resp = sr.response.fillna("").str.strip().str.lower().str.split().str[0].fillna("")
    lem = lemmas(set(resp) | set(sr.w_a.str.lower()))
    sr["success"] = [lem.get(r) == lem.get(w.lower()) and r != "" for r, w in zip(resp, sr.w_a)]
    sr["works_yes"] = sr.works.astype(str).str.lower().str.startswith("y")
    per = sr.groupby("cand_id").agg(success=("success", "mean"), works=("works_yes", "mean"), n_raters=("rater", "nunique"))

    pr = pr.merge(pk, on="comp_id")
    bt = []
    for cell, g in pr.groupby("cell"):
        ids = sorted(set(g.left) | set(g.right))
        ix = {c: i for i, c in enumerate(ids)}
        wins = [(ix[l], ix[r]) if ch == "L" else (ix[r], ix[l]) for l, r, ch in zip(g.left, g.right, g.choice.str.upper())
                if ch in ("L", "R")]
        sc = choix.ilsr_pairwise(len(ids), wins, alpha=0.01) if wins else np.zeros(len(ids))
        bt += [dict(cell=cell, cand_id=c, bt=float(s)) for c, s in zip(ids, sc)]
    BT = pd.DataFrame(bt)

    D = S.merge(per, left_on="cand_id", right_index=True, how="left").merge(BT, on=["cell", "cand_id"], how="left")
    D["flag"] = D.cand_id.isin(flagged)
    D.to_csv(hd / "analysis_items.csv", index=False)
    res = dict(n_raters=int(sr.rater.nunique()), flagged=len(flagged))
    for version, X in (("all", D), ("no_flagged", D[~D.flag])):
        res[version] = tests(X)
    dump(hd / "metrics.json", res)
    L = ["# E5 사람 평가 결과", "", f"평가자 {res['n_raters']}명, 오염 표시 {len(flagged)}개", ""]
    for version in ("all", "no_flagged"):
        r = res[version]
        L += [f"## {'전체' if version == 'all' else '오염 표시 뺀 버전'}", "", "| 가설 | 차이 (95% CI) | 판정 |", "|---|---|---|"]
        for h in ("H5a", "H5b", "H5c"):
            v = r[h]
            L.append(f"| {h} | {v['diff']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | {'지지' if v['supported'] else '지지 안 됨'} |")
        L += ["", "| 조건 | 형식 | n | 성공률 | 성립 yes | BT 평균 |", "|---|---|---|---|---|---|"]
        for k, v in r["by_condition_format"].items():
            c, f = k.split("|")
            L.append(f"| {c} | {f} | {v['n']} | {v['success']:.3f} | {v['works']:.3f} | {v['bt']:.3f} |")
        L.append("")
    (hd / "table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def tests(D: pd.DataFrame) -> dict:
    main = D[D.format.isin(MAIN_FORMATS)]
    wide = main.pivot_table(index=["pair_id", "cell"], columns="condition", values=["success", "bt"], aggfunc="first")
    pairs = wide.index.get_level_values("pair_id").to_numpy()
    b0 = D[D.condition == "B0"].success.dropna().to_numpy(float)
    rng = np.random.default_rng(SEED + 1)
    out = {}

    def mean_over(col, cond, sel):
        v = wide[(col, cond)][np.isin(pairs, sel)] if (col, cond) in wide else pd.Series(dtype=float)
        return float(np.nanmean(v)) if len(v) else np.nan

    # H5a: 짝 없음. P1은 쌍 묶음, B0은 항목을 따로 재표집
    p1_point = mean_over("success", "P1", np.unique(pairs))
    uc = np.unique(pairs)
    reps = []
    for _ in range(B_BOOT):
        sel = rng.choice(uc, len(uc))
        reps.append(mean_over("success", "P1", sel) - float(np.mean(rng.choice(b0, len(b0)))) if len(b0) else np.nan)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    d = p1_point - (float(b0.mean()) if len(b0) else np.nan)
    out["H5a"] = dict(diff=d, ci=[float(lo), float(hi)], supported=bool(lo > 0), P1=p1_point,
                      B0=float(b0.mean()) if len(b0) else None, n_B0=len(b0))
    for h, col in (("H5b", "success"), ("H5c", "bt")):
        diff = (wide[(col, "P2")] - wide[(col, "P1")]).to_numpy(float) if (col, "P2") in wide and (col, "P1") in wide else np.full(len(wide), np.nan)
        p, lo, hi = cluster_boot(lambda sel: np.nanmean(diff[np.isin(pairs, sel)]), pairs)
        out[h] = dict(diff=float(p[0]), ci=[float(lo[0]), float(hi[0])], supported=bool(lo[0] > 0))
    # 탐색: P1 vs B1(틀의 효과), P2 vs P3(H5d), 형식별(H5e)
    for name, (x, y) in {"P1_minus_B1": ("P1", "B1"), "P2_minus_P3": ("P2", "P3")}.items():
        for col in ("success", "bt"):
            if (col, x) in wide and (col, y) in wide:
                diff = (wide[(col, x)] - wide[(col, y)]).to_numpy(float)
                p, lo, hi = cluster_boot(lambda sel: np.nanmean(diff[np.isin(pairs, sel)]), pairs)
                out[f"{name}_{col}"] = dict(diff=float(p[0]), ci=[float(lo[0]), float(hi[0])])
    out["by_condition_format"] = {f"{c}|{f}": dict(n=int(len(g)), success=float(g.success.mean()), works=float(g.works.mean()),
                                                    bt=float(g.bt.mean()) if g.bt.notna().any() else float("nan"))
                                  for (c, f), g in D.assign(format=D.format.fillna("-")).groupby(["condition", "format"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["export", "analyze"])
    ap.add_argument("--stage", required=True, choices=["pilot", "main"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--with-p4", action="store_true")
    a = ap.parse_args()
    adir = stage_dir(a.stage) / a.gen / f"A_{a.model}"
    export(adir, a.with_p4) if a.step == "export" else analyze(adir)


if __name__ == "__main__":
    main()
