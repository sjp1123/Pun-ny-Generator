"""K1 연구자 고르기 목록 (ko/DESIGN.md 4절). 사전은 어떤 뜻이 흔한지 알려 주지 않으므로 연구자가 고른다.

curate_d0.csv  → A 갈래 같은 소리 쌍(사전 확인을 통과한 것). '쓸만함'에 Y/N.
curate_b.csv   → B 갈래 동형어 후보. 빈도 상위 명사 중
                 - 한 음절이면서 Kiwi가 조사·어미·접사로도 분석하는 것(의, 은, 가 …)은 뺀다
                 - 드문 뜻(사람·땅 이름, 연도, 옛 제도, 종교 용어, 동식물 종 이름, 방언·옛말 등)을 뺀 명사 뜻이 어깨번호 2개 이상
                 빈도순으로 앞의 N개. '뜻 p', '뜻 a'에 두 뜻의 [번호]를 적는다(둘 다 흔하고 서로 다른 낱말일 때만). 아니면 비운다.
목록은 로컬(ko/k1/local/)에만 둔다(사전 뜻풀이가 들어 있음). 고른 결과는 낱말과 어깨번호만 남겨 올린다.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

K1 = Path(__file__).resolve().parents[1] / "k1"
N_SHOW = 300
RARE = re.compile(r"(\d{3,4}[~～]|\(\d{3,4}|조선 시대|고려 시대|신라|백제|옛날에|예전에|불교|천주교|기독교|북한|옛말|방언|사투리|성\(姓\)|성씨|"
                  r"사람의 이름|지명|산 이름|강 이름|수도|군의|읍의|면의|화학 원소|별의 이름|별자리|곤충|식물의 하나|풀의 하나|나무의 하나|"
                  r"물고기|새의 하나|의 일종|짐승|조개|광물|벌레|버섯|괘\(卦\)|육십사괘|팔괘|시경|악보|공척보|음을 나타내는)")


def good(i) -> bool:
    d = (i["definition"] or "").strip()
    return i["pos"] == "명사" and len(d) >= 6 and not d.startswith("→") and not RARE.search(d)


def main():
    cache = {}
    for line in (K1 / "local" / "stdict_cache.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        cache[r["q"]] = r["items"]
    A = pd.read_csv(K1 / "local" / "curate_d0.csv", encoding="utf-8-sig")
    A = A[A.keep].reset_index(drop=True)
    pd.DataFrame({"번호": range(1, len(A) + 1), "쌍": A.a + " / " + A.b, "발음": A.pron, "앞 낱말 뜻": A.a_def, "뒤 낱말 뜻": A.b_def,
                  "쓸만함 (Y/N)": ""}).to_csv(K1 / "local" / "curate_d0.csv", index=False, encoding="utf-8-sig")
    from kiwipiepy import Kiwi
    k = Kiwi()
    B = pd.read_csv(K1 / "local" / "curate_b.csv", encoding="utf-8-sig").sort_values("freq_rank")
    rows = []
    for r in B.itertuples():
        if len(r.word) == 1:
            tags = {t.tag[0] for res in k.analyze(r.word, top_n=5) for t in res[0]}
            if tags & {"J", "E", "X"}:
                continue
        ns = [i for i in cache[r.word] if good(i)]
        if len({i["sup_no"] for i in ns}) < 2:
            continue
        rows.append({"번호": len(rows) + 1, "낱말": r.word, "빈도 순위": r.freq_rank,
                     "뜻": " || ".join(f"[{i['sup_no']}] {i['definition'][:45]}" for i in ns[:7]), "뜻 p": "", "뜻 a": "", "메모": ""})
        if len(rows) == N_SHOW:
            break
    pd.DataFrame(rows).to_csv(K1 / "local" / "curate_b.csv", index=False, encoding="utf-8-sig")
    print(f"A 고르기 {len(A)}쌍, B 고르기 {len(rows)}낱말 (빈도 순위 {rows[0]['빈도 순위']}~{rows[-1]['빈도 순위']})")
    print([r["낱말"] for r in rows[:80]])


if __name__ == "__main__":
    main()
