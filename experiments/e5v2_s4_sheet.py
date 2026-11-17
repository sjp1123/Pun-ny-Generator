"""S4 다듬기의 효과를 연구자가 블라인드로 본다 (PREREG_E5 6.13): ③(S4 전)과 ④(S4 뒤)를 A/B로 섞어 보여 주고 더 나은 말장난을 고른다.

make : 파일럿의 P 후보 중 S4가 문장을 바꾼 것(④ ≠ ③)에서 칸마다 최대 2개, 모두 N_MAX개까지(시드 고정) 뽑아
       평가지(s4_ab_sheet.csv, 엑셀에서 열림)와 답 열쇠(s4_ab_key.csv, 평가 전에 열지 않는다)를 만든다.
       A/B 중 어느 쪽이 ④인지는 문항마다 무작위.
score: 평가지의 '답' 열(A / B / = )을 읽어 ④를 고른 비율을 계산한다.
       판정 규칙(6.13): 답이 갈린 문항(= 제외)에서 ④를 고른 비율이 0.5를 넘으면 S4를 본 실행에 쓴다.

사용: python -X utf8 experiments/e5v2_s4_sheet.py make  --stage pilot6 --gen olmo-2-0325-32b-instruct-4bit
      python -X utf8 experiments/e5v2_s4_sheet.py score --stage pilot6 --gen olmo-2-0325-32b-instruct-4bit
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from e5 import stable_seed, stage_dir
from rules import dump

N_MAX, PER_CELL = 40, 2
GUIDE = """S4 A/B 평가 방법
- 문항마다 같은 쌍으로 만든 문장 두 개(A, B)가 있다. 어느 쪽이 나중에 다듬은 것인지는 알려 주지 않는다.
- '말장난 단어 → 숨은 단어'와 두 단어의 뜻을 보고, 더 나은 말장난을 고른다.
  기준: 두 뜻이 모두 문장에서 자연스럽게 서는가, 문장이 자연스러운가, 더 웃긴가 (이 순서로 무게를 둔다).
- '답' 열에 A, B, 또는 = (비슷함)을 적는다. s4_ab_key.csv는 다 끝나기 전에는 열지 않는다.
"""


def make(gdir: Path, out: Path, stage: str):
    c = pd.read_csv(gdir / "candidates.csv")
    P = c[(c.source == "P") & c.s4_used.astype(str).eq("True") & c.text3.notna()].copy()
    P = P[P.text.str.strip().str.lower() != P.text3.str.strip().str.lower()].sort_values("cand_id")
    rng = np.random.default_rng(stable_seed(stage, "s4_ab"))
    P = P.groupby("cell", group_keys=False).apply(lambda g: g.sample(min(PER_CELL, len(g)), random_state=int(rng.integers(1 << 31))))
    P = P.sample(min(N_MAX, len(P)), random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    from e5v2 import load_samples_v4
    gl = {p["pair_id"]: p for p in load_samples_v4()["pilot6" if stage == "pilot6" else "main"]}
    sheet, key = [], []
    for i, r in P.iterrows():
        four_is_a = bool(rng.integers(2))
        a, b = (r.text, r.text3) if four_is_a else (r.text3, r.text)
        g = gl.get(r.pair_id, {})
        sheet.append({"번호": i + 1, "쌍": f"{r.w_p} → {r.w_a}", "w_p 뜻": g.get("gloss_p", ""), "w_a 뜻": g.get("gloss_a", ""),
                      "A": a, "B": b, "답": ""})
        key.append(dict(no=i + 1, cand_id=r.cand_id, s4_side="A" if four_is_a else "B"))
    pd.DataFrame(sheet).to_csv(out / "s4_ab_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(out / "s4_ab_key.csv", index=False)
    (out / "s4_ab_guide.txt").write_text(GUIDE, encoding="utf-8")
    n_changed = int((c[(c.source == "P")].s4_unchanged.astype(str) == "False").sum())
    print(f"평가지 {len(sheet)}문항: {out / 's4_ab_sheet.csv'} (S4가 문장을 바꾼 P 후보 {n_changed}개 중)")


def score(out: Path) -> dict:
    S = pd.read_csv(out / "s4_ab_sheet.csv", encoding="utf-8-sig")
    K = pd.read_csv(out / "s4_ab_key.csv")
    D = S.merge(K, left_on="번호", right_on="no")
    ans = D["답"].astype(str).str.strip().str.upper().replace({"＝": "=", "NAN": ""})
    bad = D[~ans.isin(["A", "B", "="])]
    if len(bad):
        raise SystemExit(f"답이 A, B, = 가 아닌 문항: {bad['번호'].tolist()}")
    pick4 = (ans == D.s4_side)
    decided = ans != "="
    res = dict(n=len(D), same=int((~decided).sum()), decided=int(decided.sum()), prefer_s4=int((pick4 & decided).sum()),
               prefer_s3=int((~pick4 & decided).sum()))
    res["share_s4"] = res["prefer_s4"] / res["decided"] if res["decided"] else None
    res["keep_s4"] = bool(res["share_s4"] is not None and res["share_s4"] > 0.5)
    dump(out / "s4_ab_result.json", res)
    print(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--stage", default="pilot6")
    ap.add_argument("--gen", required=True)
    a = ap.parse_args()
    gdir = stage_dir(a.stage) / a.gen
    if a.cmd == "make":
        make(gdir, gdir, a.stage)
    else:
        score(gdir)


if __name__ == "__main__":
    main()
