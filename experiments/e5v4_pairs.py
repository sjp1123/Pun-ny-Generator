"""E5 v4 쌍과 표본 (PREREG_E5 6.13): 파일럿 6 10쌍, 본 실행 40쌍, Tom Swifty 10쌍과 S3 힌트.

v3 쌍(e5/pairs_v3.csv: 거리 0~1, 흔한 뜻이 있는 단어, 나머지 쌍 규칙은 e5_pairs.py)에서 세 가지를 더 거른다.
파일럿 4 결과에서 본 문제(6.12 뒤 검토)를 고치는 규칙이다.
  1. 대표 발음만: CMUdict 첫 발음(없으면 신경망 g2p)으로 거리를 다시 잰다. 여러 발음 중 가장 가까운 것을 쓰면
     nice [N IY S](도시 Nice)처럼 다른 낱말의 발음으로 가짜 동음 쌍이 생긴다(nice → niece).
  2. 사람 말장난만큼 비슷한 소리: 대표 발음 기준
       d0     발음이 같음(철자는 다름)
       sub    음소 하나 치환, 조음 자질 거리 ≤ SUB_MAX. SUB_MAX = SemEval 이형 말장난의 거리 1 치환 쌍의 자질 거리 중앙값(1.25).
              우리 거리 1 쌍의 치환은 중앙값 3.5였다(steep/sleep 3.75, hill/hymn 3.12). 사람은 vault/fault, bidder/bitter처럼 가깝다
       indel  음소 하나 더하기·빼기 (사람 말장난 거리 1의 49%가 이 꼴: heir/aired, tone/toner)
     정규화 거리 ≤ 0.4는 대표 발음으로 다시 확인한다.
  3. 품사: w_a의 흔한 뜻(e5.gloss_common)의 품사를 w_p도 가져야 한다(WordNet에 그 품사 뜻이 있음). nice(형용사)는 niece(명사)
     자리에 못 들어간다. 두 방향이 다 되면 무작위, 한 방향만 되면 그 방향.
     w_p의 WordNet 품사 집합(pos_set_p)도 적어 두어 생성 때 S1 문장의 w_a 자리 품사를 검사한다(e5v2.pos_fits).
표본: 세 층(d0, sub, indel)에서 파일럿 6 4·3·3, 본 실행 14·13·13, Tom Swifty 4·3·3. 파일럿 1~5에 쓴 단어와 프롬프트 예시
단어는 뺀다. 세 묶음 사이에 같은 단어가 없다. 힌트는 e5v2_pairs.hints_for. 시드 stable_seed("e5v4", "samples").

출력: e5/pairs_v4.csv (거른 쌍 목록), e5/samples_v4.json (커밋해서 고정)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from baselines import W2V
from e5 import E5_DIR, gloss_common, load_samples, stable_seed, wordnet
from e5v2 import EXAMPLE_WORDS, SAMPLES_V4, load_samples_v2, load_samples_v3
from g2p import prons
from phon import pair_distance

PAIRS_V4 = E5_DIR / "pairs_v4.csv"
SUB_MAX = 1.25
NORM_MAX = 0.4
STRATA = ("d0", "sub", "indel")
SPLITS = (("pilot6", "e5s", (4, 3, 3)), ("main", "e5n", (14, 13, 13)), ("tom_swifty", "e5u", (4, 3, 3)))


def primary(word: str) -> tuple:
    return prons(word)[0]


def classify(a: str, b: str):
    """대표 발음으로 (층, 단순 거리, 자질 거리, 정규화 거리). 규칙에 안 맞으면 층 None."""
    pa, pb = primary(a), primary(b)
    if not pa or not pb:
        return None, None, None, None
    du = pair_distance([pa], [pb], "unit")[0]
    df = pair_distance([pa], [pb], "feature")[0]
    dn = du / max(len(pa), len(pb))
    if du == 0:
        return "d0", du, df, dn
    if du == 1 and dn <= NORM_MAX:
        if len(pa) == len(pb):
            return ("sub" if df <= SUB_MAX else None), du, df, dn
        return "indel", du, df, dn
    return None, du, df, dn


def wn_pos(word: str) -> set:
    out = set()
    for s in wordnet().synsets(word):
        if word in s.lemma_names() and s.pos() in "nvasr":
            out.add({"s": "a"}.get(s.pos(), s.pos()))
    return out


POS_LETTER = {"noun": "n", "verb": "v", "adjective": "a"}


def direction(a: str, b: str, rng):
    """(w_p, w_a) 또는 None. w_a 흔한 뜻의 품사를 w_p가 가져야 한다."""
    ok = []
    for wp, wa in ((a, b), (b, a)):
        g = gloss_common(wa)
        if g and POS_LETTER.get(g[0]) in wn_pos(wp):
            ok.append((wp, wa))
    if not ok:
        return None
    return ok[int(rng.integers(len(ok)))] if len(ok) == 2 else ok[0]


def main():
    from gensim.models import KeyedVectors
    from e5v2_pairs import hints_for
    P = pd.read_csv(E5_DIR / "pairs_v3.csv", keep_default_na=False)  # "null", "nan" 같은 낱말을 결측으로 읽지 않게
    rows = []
    for r in P.itertuples():
        st, du, df, dn = classify(r.a, r.b)
        rows.append(dict(a=r.a, b=r.b, cos=r.cos, stratum=st, d_unit=du, d_feat=df, d_unit_norm=dn))
    Q = pd.DataFrame(rows)
    print(f"v3 쌍 {len(Q)}개 → 대표 발음 기준 층: {Q.stratum.value_counts(dropna=False).to_dict()}", flush=True)
    Q = Q[Q.stratum.notna()].reset_index(drop=True)
    Q.to_csv(PAIRS_V4, index=False)

    banned = ({w for p in load_samples()["pilot"] for w in (p["w_p"], p["w_a"])}
              | {w for p in load_samples_v2()["pilot2"] for w in (p["w_p"], p["w_a"])}
              | {w for p in load_samples_v3()["pilot3"] for w in (p["w_p"], p["w_a"])} | set(EXAMPLE_WORDS))
    rng = np.random.default_rng(stable_seed("e5v4", "samples"))
    order = {s: iter(rng.permutation(np.nonzero(Q.stratum.values == s)[0])) for s in STRATA}
    taken, chosen = set(), {}
    for split, prefix, counts in SPLITS:
        picks = []
        for s, n in zip(STRATA, counts):
            k = 0
            for i in order[s]:
                a, b = Q.a.iat[i], Q.b.iat[i]
                if {a, b} & (banned | taken):
                    continue
                d = direction(a, b, rng)
                if d is None:
                    continue
                picks.append((i, d))
                taken |= {a, b}
                k += 1
                if k == n:
                    break
            if k < n:
                print(f"경고: {split} {s} 층 {k}/{n}개만", flush=True)
        chosen[split] = (prefix, picks)
    print("word2vec GoogleNews 불러오는 중", flush=True)
    kv = KeyedVectors.load_word2vec_format(str(W2V), binary=True, limit=1_000_000)
    out = dict(seed='stable_seed("e5v4", "samples")', rules=dict(pronunciation="CMUdict 첫 발음", sub_feature_max=SUB_MAX,
                                                                  norm_max=NORM_MAX, strata=list(STRATA),
                                                                  pos="w_a 흔한 뜻의 품사를 w_p가 가짐"),
               n_pairs=dict(all=len(Q), **{s: int((Q.stratum == s).sum()) for s in STRATA}),
               excluded_words=dict(pilots=len(banned) - len(EXAMPLE_WORDS), prompt_examples=sorted(EXAMPLE_WORDS)))
    for split, (prefix, picks) in chosen.items():
        rows = []
        for m, (i, (wp, wa)) in enumerate(picks):
            r = Q.iloc[i]
            gp, ga = gloss_common(wp), gloss_common(wa)
            rows.append(dict(pair_id=f"{prefix}{m:03d}", w_p=wp, w_a=wa, stratum=r.stratum, d_unit=float(r.d_unit),
                             d_feat=round(float(r.d_feat), 3), d_unit_norm=round(float(r.d_unit_norm), 3), cos=round(float(r.cos), 4),
                             pron_p=" ".join(primary(wp)), pron_a=" ".join(primary(wa)),
                             pos_p=gp[0], gloss_p=gp[1], pos_a=ga[0], gloss_a=ga[1], pos_set_p="".join(sorted(wn_pos(wp))),
                             hints=hints_for(kv, wp, wa)))
        out[split] = rows
    SAMPLES_V4.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for split, _, _ in SPLITS:
        print(split)
        for p in out[split]:
            print(f"  {p['pair_id']} {p['stratum']:5s} {p['w_p']:>12} -> {p['w_a']:<12} [{p['pron_p']}] / [{p['pron_a']}] "
                  f"feat={p['d_feat']:.2f} {p['pos_a']}  hints={','.join(p['hints'])}")
    short = {p["pair_id"]: p["hints"] for s, _, _ in SPLITS for p in out[s] if len(p["hints"]) < 5}
    print(f"힌트 5개 미만 {len(short)}쌍 {short}")


if __name__ == "__main__":
    main()
