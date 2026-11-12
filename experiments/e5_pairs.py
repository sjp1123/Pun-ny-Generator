"""E5 단어 쌍 목록과 표본 (docs/E5_DESIGN.md "단어 쌍 표본").

1. CMUdict 영문 소문자 단어 중 빈도(wordfreq zipf) 상위 2만 개
2. 품사: WordNet에 소문자 표제어 그대로 명사·동사·형용사 뜻이 있는 단어(고유명사·굴절형 제외), spaCy 불용어 제외
3. 음소 편집 거리(강세 뺀 단순 편집 거리, 발음이 여럿이면 가장 가까운 쌍) 1~2
4. 표제어가 다르고(한쪽 표제어가 다른 단어 안에 들어 있어도 뺀다: cover/uncover, miller/milling),
   word2vec GoogleNews 코사인 < 0.3
5. SemEval 이형 말장난의 (pun_word, alter_word) 쌍은 표제어 기준으로 뺀다(순서 무관)
[설계에서 바꾼 점]
- 정규화 거리(편집 수 ÷ 긴 쪽 음소 수) ≤ 0.4도 요구한다. 거리 2만 두면 짧은 단어는 통째로 다른 소리가 된다(hoe→aim).
- 3글자 이상, 모음 글자 포함(약어 mg·ab 제외), 비속어·비하어(e5.offensive, 표제어로도 검사) 제외,
  이름처럼 쓰이는 단어 제외(GoogleNews에서 대문자형 순위가 소문자형보다 8배 넘게 앞: carter, graham, kat).
  연결 단어에도 같은 거르기를 쓴다.
표본(시드 20261112): 파일럿 10쌍, 본 실행 40쌍, Tom Swifty 10쌍. 무작위 순서로 훑으며 이미 뽑힌 쌍과 단어가
겹치면 건너뛴다(같은 단어가 두 쌍에 나오지 않게). w_p/w_a 방향은 무작위.
한 줄 상황형의 연결 단어: w_p의 word2vec 이웃 중 w_a보다 w_p에 가까운 일반 단어 5개.
사람 말장난(상한선): SemEval 40개 = 수수께끼형 20 + 한 줄형 20 (Tom Swifty 제외).

출력: e5/pairs_all.csv, e5/samples.json (커밋해서 고정한다)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from baselines import W2V
from candidates import vocab
from data import load_semeval, semeval_items
from e5 import E5_DIR, SAID_RE, SAMPLES, format_ok, gloss, offensive
from g2p import prons
from phon import PronTable
from rules import SEED, lemmas

TOP_N = 20000
COS_MAX = 0.3
NORM_MAX = 0.4
NAME_RATIO = 8
N_PILOT, N_MAIN, N_TS, N_HUMAN = 10, 40, 10, 40


def name_like(w: str, kv) -> bool:
    """GoogleNews에서 대문자형(Carter)이 소문자형(carter)보다 8배 넘게 앞 순위(더 흔함)면 이름으로 본다."""
    lo, cap = kv.key_to_index.get(w), kv.key_to_index.get(w.capitalize())
    return lo is not None and cap is not None and lo > NAME_RATIO * cap


def plain_word(w: str, stop, kv) -> bool:
    return (len(w) >= 3 and any(c in "aeiouy" for c in w) and w not in stop and w in kv.key_to_index
            and gloss(w) is not None and not offensive(w) and not offensive(lemmas([w])[w])
            and not name_like(w, kv))


def eligible_words(kv) -> list[str]:
    from spacy.lang.en.stop_words import STOP_WORDS
    words, zipf = vocab()
    order = np.argsort(-zipf, kind="stable")[:TOP_N]
    return [words[i] for i in order if plain_word(words[i], STOP_WORDS, kv)]


def semeval_pairs() -> set:
    rows = [(r["pun_word"].lower(), r["alter_word"].strip().lower()) for r in load_semeval()]
    lem = lemmas({w for r in rows for w in r})
    return {frozenset((lem[a], lem[b])) for a, b in rows}


def all_pairs(kv) -> pd.DataFrame:
    words = eligible_words(kv)
    print(f"후보 단어 {len(words)}개 (상위 {TOP_N}개 중)", flush=True)
    tab = PronTable(words, prons)
    lem = lemmas(words)
    seen = semeval_pairs()
    rows = []
    for i, w in enumerate(words):
        d, dn = tab.distances(prons(w), "unit")
        for j in np.nonzero((d >= 1) & (d <= 2) & (dn <= NORM_MAX))[0]:
            if j <= i:
                continue
            v = words[j]
            if lem[w] == lem[v] or frozenset((lem[w], lem[v])) in seen or lem[w] in v or lem[v] in w:
                continue
            cos = float(kv.similarity(w, v))
            if cos >= COS_MAX:
                continue
            rows.append(dict(a=w, b=v, d_unit=float(d[j]), d_unit_norm=float(dn[j]), cos=cos))
        if i % 2000 == 0:
            print(f"  {i}/{len(words)}  쌍 {len(rows)}", flush=True)
    return pd.DataFrame(rows)


def link_words(kv, w_p: str, w_a: str, k=5) -> list[str]:
    from spacy.lang.en.stop_words import STOP_WORDS
    words = set(vocab()[0])
    lem = lemmas({w_p, w_a})
    out = []
    for w, s in kv.most_similar(w_p, topn=300):
        if not w.isalpha() or not w.islower() or w not in words or w_p in w or w_a in w or w in w_p:
            continue
        if not plain_word(w, STOP_WORDS, kv):
            continue
        lw = lemmas([w])[w]
        if lw in (lem[w_p], lem[w_a]) or kv.similarity(w, w_a) >= s:
            continue
        out.append(w)
        if len(out) == k:
            break
    return out


def human_puns(rng) -> list[dict]:
    import re
    quote = re.compile(r"[\"“”]|''|(?<![A-Za-z])'|'(?![A-Za-z])")
    tail_ly = re.compile(r"\b(said|says|asked|replied|\w+ed)\s+(\w+\s+)?\w+ly[.!?\"']*\s*$", re.I)
    swifty = lambda t: SAID_RE.search(t) or re.search(r"\bTom\b", t) or tail_ly.search(t)
    its = [it for it in semeval_items(require_single_word=True) if not swifty(it.text)]
    rid = [it for it in its if format_ok(it.text, "riddle") and not quote.search(it.text)]  # 대화체 제외
    one = [it for it in its if format_ok(it.text, "oneliner")]
    pick = lambda xs, n, f: [dict(id=it.id, text=it.text, pun_word=it.surface, w_p=it.w_p, w_a=it.w_a, format=f)
                             for it in (xs[i] for i in sorted(rng.choice(len(xs), n, replace=False)))]
    return pick(rid, N_HUMAN // 2, "riddle") + pick(one, N_HUMAN // 2, "oneliner")


def main():
    from gensim.models import KeyedVectors
    print("word2vec GoogleNews 불러오는 중", flush=True)
    kv = KeyedVectors.load_word2vec_format(str(W2V), binary=True, limit=1_000_000)
    E5_DIR.mkdir(exist_ok=True)
    P = all_pairs(kv)
    P.to_csv(E5_DIR / "pairs_all.csv", index=False)
    print(f"쌍 {len(P)}개 (거리 1: {(P.d_unit == 1).sum()}, 2: {(P.d_unit == 2).sum()})")

    rng = np.random.default_rng(SEED)
    idx, used = [], set()
    for i in rng.permutation(len(P)):
        a, b = P.a.iat[i], P.b.iat[i]
        if a in used or b in used:
            continue
        idx.append(i)
        used |= {a, b}
        if len(idx) == N_PILOT + N_MAIN + N_TS:
            break
    flip = rng.random(len(idx)) < 0.5
    sel = []
    for n, (i, f) in enumerate(zip(idx, flip)):
        r = P.iloc[i]
        wp, wa = (r.b, r.a) if f else (r.a, r.b)
        gp, ga = gloss(wp), gloss(wa)
        sel.append(dict(pair_id=f"e5p{n:03d}", w_p=wp, w_a=wa, d_unit=r.d_unit,
                        d_unit_norm=round(r.d_unit_norm, 3), cos=round(r.cos, 4),
                        pos_p=gp[0], gloss_p=gp[1], pos_a=ga[0], gloss_a=ga[1],
                        link_words=link_words(kv, wp, wa)))
    S = dict(seed=SEED, source=dict(top_n=TOP_N, cos_max=COS_MAX, d_unit=[1, 2], d_unit_norm_max=NORM_MAX,
                                     n_pairs=len(P), wordnet="3.0",
                                     w2v="GoogleNews-vectors-negative300 (상위 100만 단어)"),
             pilot=sel[:N_PILOT], main=sel[N_PILOT:N_PILOT + N_MAIN], tom_swifty=sel[N_PILOT + N_MAIN:],
             human=human_puns(rng))
    SAMPLES.write_text(json.dumps(S, indent=1, ensure_ascii=False), encoding="utf-8")
    for k in ("pilot", "main", "tom_swifty"):
        print(f"\n[{k}]")
        for p in S[k]:
            print(f"  {p['pair_id']}  {p['w_p']:>12} -> {p['w_a']:<12} d={p['d_unit']:.0f}/{p['d_unit_norm']:.2f} "
                  f"cos={p['cos']:.2f}  link={','.join(p['link_words'])}")


if __name__ == "__main__":
    main()
