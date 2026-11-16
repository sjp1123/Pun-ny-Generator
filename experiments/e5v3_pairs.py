"""E5 v3 쌍과 표본 (파일럿 2 실패 뒤 쌍 재설계, PREREG_E5 6.5): 파일럿 3 10쌍, 본 실행 40쌍, Tom Swifty 10쌍과 S3 힌트.

파일럿 2에서 판정기 성공이 거의 한 쌍(heir→aired)에만 나왔고 쌍 10개 중 7개가 음소 거리 2였다(6.4).
바꾼 점 (나머지 쌍 규칙은 e5_pairs.py와 같다: 빈도 상위 2만, 명·동·형, 정규화 거리 ≤ 0.4, 표제어 다름,
코사인 < 0.3, SemEval 쌍 제외, 3글자 이상, 비속어·이름 제외):
  1. 음소 거리 0~1. 거리 0은 철자가 다른 동음이의어(heir/air)다. 거리 0 쌍에는 글자 포함 거르기를 쓰지 않는다.
  2. 두 단어 모두 SemCor 빈도가 있는 뜻이 있어야 하고, 뜻풀이는 가장 흔한 뜻(e5.gloss_common)을 쓴다.
표본: 거리 0과 1을 반씩(파일럿 3 5+5, 본 실행 20+20, Tom Swifty 5+5). 거리별로 무작위 순서로 훑으며
  파일럿 1(e5/samples.json의 pilot)·파일럿 2(e5/samples_v2.json)·프롬프트 예시 단어가 든 쌍과 이미 뽑힌 단어가 든 쌍은 건너뛴다.
  w_p/w_a 방향은 무작위. 힌트는 e5v2_pairs.hints_for(같은 규칙).
시드: stable_seed("e5v3", "samples").

출력: e5/pairs_v3.csv (쌍 목록), e5/samples_v3.json (커밋해서 고정)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from baselines import W2V
from e5 import E5_DIR, gloss_common, load_samples, stable_seed
from e5_pairs import all_pairs, eligible_words
from e5v2 import EXAMPLE_WORDS, SAMPLES_V3, load_samples_v2
from e5v2_pairs import hints_for

PAIRS_V3 = E5_DIR / "pairs_v3.csv"
D_RANGE = (0, 1)
SPLITS = (("pilot3", "e5r", 10), ("main", "e5m", 40), ("tom_swifty", "e5t", 10))


def main():
    from gensim.models import KeyedVectors
    print("word2vec GoogleNews 불러오는 중", flush=True)
    kv = KeyedVectors.load_word2vec_format(str(W2V), binary=True, limit=1_000_000)
    words = [w for w in eligible_words(kv) if gloss_common(w) is not None]
    print(f"흔한 뜻이 있는 단어 {len(words)}개", flush=True)
    P = all_pairs(kv, *D_RANGE, words=words)
    P.to_csv(PAIRS_V3, index=False)
    print(f"쌍 {len(P)}개 (거리 0: {(P.d_unit == 0).sum()}, 1: {(P.d_unit == 1).sum()})", flush=True)

    banned = ({w for p in load_samples()["pilot"] for w in (p["w_p"], p["w_a"])}
              | {w for p in load_samples_v2()["pilot2"] for w in (p["w_p"], p["w_a"])} | set(EXAMPLE_WORDS))
    rng = np.random.default_rng(stable_seed("e5v3", "samples"))
    order = {d: iter(rng.permutation(np.nonzero(P.d_unit.values == d)[0])) for d in D_RANGE}
    taken, out = set(), dict(seed='stable_seed("e5v3", "samples")', d_range=list(D_RANGE),
                             n_pairs=dict(all=len(P), **{f"d{d}": int((P.d_unit == d).sum()) for d in D_RANGE}),
                             excluded_words=dict(pilots=len(banned) - len(EXAMPLE_WORDS),
                                                 prompt_examples=sorted(EXAMPLE_WORDS)))
    for split, prefix, n in SPLITS:
        idx = []
        for d in D_RANGE:
            k = 0
            for i in order[d]:
                a, b = P.a.iat[i], P.b.iat[i]
                if {a, b} & (banned | taken):
                    continue
                idx.append(i)
                taken |= {a, b}
                k += 1
                if k == n // len(D_RANGE):
                    break
        flip = rng.random(len(idx)) < 0.5
        rows = []
        for m, (i, f) in enumerate(zip(idx, flip)):
            r = P.iloc[i]
            wp, wa = (r.b, r.a) if f else (r.a, r.b)
            gp, ga = gloss_common(wp), gloss_common(wa)
            rows.append(dict(pair_id=f"{prefix}{m:03d}", w_p=wp, w_a=wa, d_unit=float(r.d_unit),
                             d_unit_norm=round(float(r.d_unit_norm), 3), cos=round(float(r.cos), 4),
                             pos_p=gp[0], gloss_p=gp[1], pos_a=ga[0], gloss_a=ga[1], hints=hints_for(kv, wp, wa)))
        out[split] = rows
    SAMPLES_V3.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for split, _, _ in SPLITS:
        print(split)
        for p in out[split]:
            print(f"  {p['pair_id']}  {p['w_p']:>12} -> {p['w_a']:<12} d={p['d_unit']:.0f}/{p['d_unit_norm']:.2f} "
                  f"cos={p['cos']:.2f}  hints={','.join(p['hints'])}")
    short = {p["pair_id"]: p["hints"] for s, _, _ in SPLITS for p in out[s] if len(p["hints"]) < 5}
    print(f"힌트 5개 미만 {len(short)}쌍 {short}")


if __name__ == "__main__":
    main()
