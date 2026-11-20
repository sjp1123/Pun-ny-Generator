"""K1 사전 확인 (ko/DESIGN.md 4절): 표준국어대사전으로 A 갈래 d0 쌍을 거르고, B 갈래 동형어 후보를 모은다.

A 갈래 d0: 두 낱말 모두 사전에 명사 표제어가 있고, 그 뜻풀이가 '→ 다른 말'(비표준 표기·참고 항목)뿐이 아니어야 한다.
           한쪽이 다른 쪽을 가리키는 쌍(오랫만 → 오랜만)은 표기 변이라 뺀다.
B 갈래 후보: wordfreq 한국어 상위 명사(Kiwi NNG 한 형태소, 한 음절 포함) 중 빈도 상위 N_B개를 찾아, 명사 뜻(의존 명사·참고 항목 제외)이
           어깨번호가 다른 표제어로 2개 이상인 낱말. 어느 뜻이 흔한지는 사전에 없으므로 연구자가 고른다(curate_b.csv).
출력(로컬, 사전 뜻풀이가 들어 있어 깃에 올리지 않음): ko/k1/local/curate_d0.csv, ko/k1/local/curate_b.csv
요약(올림): ko/k1/k1_dict_summary.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import wordfreq

from stdict import Stdict

K1 = Path(__file__).resolve().parents[1] / "k1"
N_B = 3000


def noun_senses(items) -> list[dict]:
    """명사 표제어 중 참고 항목('→ …')이 아닌 것."""
    return [i for i in items if i["pos"] == "명사" and i["definition"] and not i["definition"].strip().startswith("→")]


def points_to(items, other: str) -> bool:
    return any(i["definition"] and i["definition"].strip().startswith("→") and other in i["definition"] for i in items)


def main():
    d = Stdict()
    D0 = pd.read_csv(K1 / "pairs_d0.csv", keep_default_na=False)
    rows = []
    for r in D0.itertuples():
        ia, ib = d.lookup(r.a), d.lookup(r.b)
        na, nb = noun_senses(ia), noun_senses(ib)
        variant = points_to(ia, r.b) or points_to(ib, r.a)
        rows.append(dict(a=r.a, b=r.b, pron=r.pron, a_nouns=len(na), b_nouns=len(nb), variant=variant,
                         keep=bool(na and nb and not variant),
                         a_def=" | ".join(i["definition"][:40] for i in na[:3]), b_def=" | ".join(i["definition"][:40] for i in nb[:3]),
                         usable=""))
    A = pd.DataFrame(rows)
    (K1 / "local").mkdir(exist_ok=True)
    A.to_csv(K1 / "local" / "curate_d0.csv", index=False, encoding="utf-8-sig")
    # B 갈래
    from kiwipiepy import Kiwi
    k = Kiwi()
    fd = wordfreq.get_frequency_dict("ko")
    pool = []
    for w, f in sorted(fd.items(), key=lambda x: -x[1]):
        if not re.fullmatch(r"[가-힣]+", w):
            continue
        an = k.analyze(w, top_n=1)[0][0]
        if len(an) == 1 and an[0].tag == "NNG" and an[0].form == w:
            pool.append((w, f))
        if len(pool) == N_B:
            break
    brows = []
    for n, (w, f) in enumerate(pool):
        ns = noun_senses(d.lookup(w))
        sups = {i["sup_no"] for i in ns}
        if len(sups) >= 2:
            brows.append(dict(word=w, freq_rank=n + 1, n_senses=len(sups),
                              senses=" || ".join(f"[{i['sup_no']}] {i['definition'][:50]}" for i in ns[:8]), pick_p="", pick_a=""))
        if n % 500 == 0:
            print(f"  B 후보 찾는 중 {n}/{len(pool)}  후보 {len(brows)}", flush=True)
    B = pd.DataFrame(brows)
    B.to_csv(K1 / "local" / "curate_b.csv", index=False, encoding="utf-8-sig")
    summ = dict(d0_pairs=len(A), d0_both_nouns=int((A.a_nouns.gt(0) & A.b_nouns.gt(0)).sum()), d0_variants=int(A.variant.sum()),
                d0_kept=int(A.keep.sum()), b_pool=len(pool), b_candidates=len(B),
                b_one_syllable=int((B.word.str.len() == 1).sum()) if len(B) else 0)
    (K1 / "k1_dict_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(summ)
    print(A[A.keep][["a", "b", "pron"]].to_string(index=False))
    print(A[~A.keep][["a", "b", "a_nouns", "b_nouns", "variant"]].to_string(index=False))


if __name__ == "__main__":
    main()
