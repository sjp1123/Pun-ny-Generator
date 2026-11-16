"""E5 v2 표본 (docs/E5_V2_DESIGN.md 8절): 파일럿 2 쌍 10개와 S3 힌트 단어.

파일럿 2 쌍: e5/pairs_all.csv(쌍 규칙은 v1과 같음, SemEval 쌍은 이미 빠져 있음)를 무작위 순서로 훑으며
  - e5/samples.json에 쓰인 단어(파일럿 1, 본 실행 40쌍, Tom Swifty)가 들어간 쌍은 건너뛴다
    → 파일럿 1·2와 본 실행 쌍이 겹치지 않는다(본 실행 40쌍에서 파일럿 쌍이 빠진 상태가 유지된다)
  - 프롬프트 예시 단어(bagel, beagle, bread, bed)가 들어간 쌍도 건너뛴다
  - 이미 뽑힌 쌍과 단어가 겹치면 건너뛴다. w_p/w_a 방향은 무작위.
힌트 단어(S3): w_p의 word2vec 이웃 중 소문자 영문 한 단어(CMUdict 어휘, 비속어 제외 — 오타·약어 빼기) 상위 20개에서
  - w_p의 굴절형(표제어가 같거나 한쪽이 다른 쪽을 품음)과 w_a 쪽 단어(표제어가 w_a와 같음)를 빼고
  - w_a와 코사인 0.3 이상인 단어를 뺀 뒤, 힌트끼리 표제어가 같으면 앞의 것만 두고 앞에서 5개.
  파일럿 2와 본 실행 40쌍 모두 만든다(GPU 머신에는 word2vec이 없다).
시드: stable_seed("e5v2", "pilot2").

출력: e5/samples_v2.json (커밋해서 고정)
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from baselines import W2V
from e5 import E5_DIR, gloss, load_samples, offensive, stable_seed
from e5v2 import EXAMPLE_WORDS, SAMPLES_V2
from rules import lemmas

N_PILOT2 = 10
N_HINT, HINT_POOL, HINT_COS_MAX = 5, 20, 0.3


@lru_cache(maxsize=1)
def _vocab() -> set:
    from candidates import vocab
    return set(vocab()[0])


def hints_for(kv, w_p: str, w_a: str) -> list[str]:
    lem = lemmas({w_p, w_a})
    pool, vocab_set = [], _vocab()
    for w, _ in kv.most_similar(w_p, topn=200):
        if w.isalpha() and w.islower() and w in vocab_set and not offensive(w):
            pool.append(w)
        if len(pool) == HINT_POOL:
            break
    lw = lemmas(pool)
    out, seen = [], set()
    for w in pool:
        if lw[w] in (lem[w_p], lem[w_a]) or w_p in w or w in w_p:
            continue
        if kv.similarity(w, w_a) >= HINT_COS_MAX or lw[w] in seen:  # 힌트끼리 같은 표제어(decal/decals)는 하나만
            continue
        seen.add(lw[w])
        out.append(w)
        if len(out) == N_HINT:
            break
    return out


def main():
    from gensim.models import KeyedVectors
    S = load_samples()
    used = {w for k in ("pilot", "main", "tom_swifty") for p in S[k] for w in (p["w_p"], p["w_a"])}
    banned = used | set(EXAMPLE_WORDS)
    P = pd.read_csv(E5_DIR / "pairs_all.csv")
    rng = np.random.default_rng(stable_seed("e5v2", "pilot2"))
    idx, taken = [], set()
    for i in rng.permutation(len(P)):
        a, b = P.a.iat[i], P.b.iat[i]
        if {a, b} & (banned | taken):
            continue
        idx.append(i)
        taken |= {a, b}
        if len(idx) == N_PILOT2:
            break
    flip = rng.random(len(idx)) < 0.5
    print("word2vec GoogleNews 불러오는 중", flush=True)
    kv = KeyedVectors.load_word2vec_format(str(W2V), binary=True, limit=1_000_000)
    pilot2 = []
    for n, (i, f) in enumerate(zip(idx, flip)):
        r = P.iloc[i]
        wp, wa = (r.b, r.a) if f else (r.a, r.b)
        gp, ga = gloss(wp), gloss(wa)
        pilot2.append(dict(pair_id=f"e5q{n:03d}", w_p=wp, w_a=wa, d_unit=float(r.d_unit),
                           d_unit_norm=round(float(r.d_unit_norm), 3), cos=round(float(r.cos), 4),
                           pos_p=gp[0], gloss_p=gp[1], pos_a=ga[0], gloss_a=ga[1], hints=hints_for(kv, wp, wa)))
    hints_main = {p["pair_id"]: hints_for(kv, p["w_p"], p["w_a"]) for p in S["main"]}
    out = dict(seed=f'stable_seed("e5v2", "pilot2")', n_pairs_all=len(P),
               excluded_words=dict(samples_v1=len(used), prompt_examples=sorted(EXAMPLE_WORDS)),
               hint_rule=dict(pool=HINT_POOL, n=N_HINT, cos_wa_max=HINT_COS_MAX,
                              w2v="GoogleNews-vectors-negative300 (상위 100만 단어)"),
               pilot2=pilot2, hints_main=hints_main)
    SAMPLES_V2.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for p in pilot2:
        print(f"  {p['pair_id']}  {p['w_p']:>12} -> {p['w_a']:<12} d={p['d_unit']:.0f}/{p['d_unit_norm']:.2f} "
              f"cos={p['cos']:.2f}  hints={','.join(p['hints'])}")
    short = {k: v for k, v in hints_main.items() if len(v) < N_HINT}
    print(f"본 실행 힌트: {len(hints_main)}쌍, 5개 미만 {len(short)}쌍 {short}")


if __name__ == "__main__":
    main()
