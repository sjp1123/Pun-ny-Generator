"""K1 B 갈래 후보 (ko/DESIGN.md 3절): 국립국어원 「한국어 학습용 어휘 목록」(2003, 공공누리 제1유형)에서 같은 명사가
다른 동형어 번호로 두 번 이상 나오는 낱말(눈01 신체 / 눈04 내리는 눈). 뜻 표시는 목록의 '풀이' 열(원어 한자나 짧은 단서:
'동물', '~과 글')을 쓴다. 목록의 번호는 2003년 당시 표준국어대사전 판의 어깨번호라 지금 사전의 번호와 어긋난다
(말05가 지금 사전에서는 십이지의 '오', 이상05가 시인 이상) — 그래서 사전 뜻풀이를 번호로 붙이지 않는다.

입력: ko/k1/local/한국어 학습용 어휘 목록.xls (연구자가 내려받음, 깃에 올리지 않음)
출력: ko/k1/local/curate_b.csv (연구자 고르기: 말장난에 쓸 두 뜻의 번호를 '1,4'처럼, 쓸 수 없으면 비움) — 사전 뜻풀이가 들어 있어 로컬에만
      ko/k1/homograph_candidates.csv (낱말, 번호, 원어, 등급만) — 올림
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd


K1 = Path(__file__).resolve().parents[1] / "k1"


def main():
    V = pd.read_excel(K1 / "local" / "한국어 학습용 어휘 목록.xls", header=0)
    V.columns = ["rank", "word", "pos", "orig", "grade"]
    V["base"] = V.word.str.replace(r"\d+$", "", regex=True)
    V["no"] = V.word.str.extract(r"(\d+)$")[0]
    N = V[(V.pos == "명") & V["no"].notna()]
    H = N[N.groupby("base")["no"].transform("nunique") >= 2].copy()
    H["no_int"] = H["no"].astype(int)
    H = H.sort_values(["rank"])
    rows, cand = [], []
    for base, g in sorted(H.groupby("base"), key=lambda x: x[1]["rank"].min()):
        senses = []
        for r in g.sort_values("no_int").itertuples():
            orig = r.orig if isinstance(r.orig, str) else "(풀이 없음)"
            senses.append(f"[{r.no_int}] {orig} ({r.grade})")
            cand.append(dict(word=base, no=r.no_int, orig=orig, grade=r.grade, freq_rank=r.rank))
        rows.append({"번호": len(rows) + 1, "낱말": base, "뜻": " || ".join(senses), "쓸 뜻 두 개 (예: 1,4)": "", "메모": ""})
    pd.DataFrame(rows).to_csv(K1 / "local" / "curate_b.csv", index=False, encoding="utf-8-sig")
    C = pd.DataFrame(cand)
    C.to_csv(K1 / "homograph_candidates.csv", index=False)
    print(f"B 후보 {len(rows)}낱말 ({len(C)}뜻). 풀이 없는 뜻 {int((C.orig == '(풀이 없음)').sum())}개")


if __name__ == "__main__":
    main()
