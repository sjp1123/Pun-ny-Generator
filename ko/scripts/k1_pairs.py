"""K1 A 갈래 쌍 세기 (ko/DESIGN.md 4절): 명사 목록 → 표준 발음 → d0(철자 다름, 발음 같음), d1(자모 하나 차이).

명사: wordfreq 한국어 빈도 목록(형태소 단위, 약 3만 개)에서 한글로만 된 상위 3만 개 중, Kiwi가 한 형태소 일반명사(NNG)로 분석하고
      2음절 이상인 것. 한 음절은 조사·어미 조각(을, 의, 은)과 섞여 뺀다.
발음: ko/src/ko_g2p.py (g2pkk 규칙 + Kiwi 품사). 자모는 jamo.h2j.
d1:   발음의 자모열 편집 거리 1 (삭제 이웃으로 찾고 다시 확인). 치환(sub)과 더하기·빼기(indel).
아직 하지 않은 것: 소리 비슷함 문턱(K2 사람 말장난으로 정함), 뜻 거리, 사전으로 낱말 확인(표준국어대사전 API).

출력: ko/k1/nouns.csv(낱말, 빈도, 발음), ko/k1/pairs_d0.csv, ko/k1/pairs_d1.csv
"""
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import wordfreq
from jamo import h2j

from ko_g2p import pron

OUT = Path(__file__).resolve().parents[1] / "k1"
TOP = 30000


def nouns() -> pd.DataFrame:
    from kiwipiepy import Kiwi
    k = Kiwi()
    d = wordfreq.get_frequency_dict("ko")
    ws = [(w, f) for w, f in sorted(d.items(), key=lambda x: -x[1]) if re.fullmatch(r"[가-힣]+", w)][:TOP]
    rows = []
    for w, f in ws:
        if len(w) < 2:
            continue
        r = k.analyze(w, top_n=1)[0][0]
        if len(r) == 1 and r[0].tag == "NNG" and r[0].form == w:
            rows.append(dict(word=w, freq=f))
    return pd.DataFrame(rows)


def lev1(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


def main():
    OUT.mkdir(exist_ok=True)
    f = OUT / "nouns.csv"
    if f.exists():
        N = pd.read_csv(f, keep_default_na=False)
    else:
        N = nouns()
        N["pron"] = [pron(w) for w in N.word]
        N.to_csv(f, index=False)
    J = dict(zip(N.word, (h2j(p) for p in N.pron)))
    P = dict(zip(N.word, N.pron))
    g = collections.defaultdict(list)
    for w in N.word:
        g[P[w]].append(w)
    d0 = [dict(a=a, b=b, pron=p) for p, ws in g.items() if len(ws) > 1 for i, a in enumerate(ws) for b in ws[i + 1:]]
    keys = collections.defaultdict(set)
    for w in N.word:
        s = J[w]
        keys[s].add(w)
        for i in range(len(s)):
            keys[s[:i] + s[i + 1:]].add(w)
    d1 = set()
    for ws in keys.values():
        ws = sorted(ws)
        for i, a in enumerate(ws):
            for b in ws[i + 1:]:
                if lev1(J[a], J[b]):
                    d1.add((a, b))
    D1 = pd.DataFrame([dict(a=a, b=b, pron_a=P[a], pron_b=P[b], kind="sub" if len(J[a]) == len(J[b]) else "indel")
                       for a, b in sorted(d1)])
    pd.DataFrame(d0).to_csv(OUT / "pairs_d0.csv", index=False)
    D1.to_csv(OUT / "pairs_d1.csv", index=False)
    print(f"명사 {len(N)}개 (철자와 발음이 다른 것 {(N.word != N.pron).sum()}개). d0 {len(d0)}쌍, d1 {len(D1)}쌍 "
          f"(치환 {(D1.kind == 'sub').sum()}, 더하기·빼기 {(D1.kind == 'indel').sum()})")


if __name__ == "__main__":
    main()
