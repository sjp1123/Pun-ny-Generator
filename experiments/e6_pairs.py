"""E6 쌍 (docs/E6_DESIGN.md 1절): 파일럿 10쌍, 시험 40쌍.

e5/pairs_v4.csv(대표 발음 기준 d0·sub·indel 층)에서 지금까지 쓴 단어를 모두 빼고 e5v4_pairs.py와 같은 규칙으로 뽑는다.
  뺄 단어: e5/samples*.json의 모든 묶음(파일럿 1~7, E5 본 실행, Tom Swifty, 사람 말장난)의 w_p·w_a, 프롬프트 예시 단어
  방향: w_a 흔한 뜻의 품사를 w_p가 가져야 한다(e5v4_pairs.direction)
  층 비율: d0·sub·indel = 파일럿 0·5·5, 시험 0·20·20. 두 묶음 사이에 같은 단어가 없다
    d0(발음이 같은 쌍)은 pairs_v4에 37쌍뿐이고 E5가 거의 다 썼다. 남은 2쌍도 품사 방향 규칙을 통과하지 못해 d0 없이 뽑는다
    (docs/E6_DESIGN.md 1절)
  시드: stable_seed("e6", "samples")
학습용 쌍(7절)은 여기서 뽑지 않는다. 나중에 뽑을 때 이 파일의 단어도 뺀다.

사용: python -X utf8 experiments/e6_pairs.py
출력: e6/samples.json (커밋해서 고정)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from baselines import W2V
from e5 import E5_DIR, gloss_common, stable_seed
from e5v2 import EXAMPLE_WORDS
from e5v4_pairs import NORM_MAX, PAIRS_V4, STRATA, SUB_MAX, direction, primary, wn_pos

ROOT = Path(__file__).resolve().parents[1]
E6_DIR = ROOT / "e6"
SAMPLES_E6 = E6_DIR / "samples.json"
SPLITS = (("pilot", "e6p", (0, 5, 5)), ("test", "e6t", (0, 20, 20)))


def used_words() -> set:
    out = set(EXAMPLE_WORDS)
    for f in sorted(E5_DIR.glob("samples*.json")):
        for v in json.loads(f.read_text(encoding="utf-8")).values():
            if isinstance(v, list):
                out |= {str(p[k]).lower() for p in v if isinstance(p, dict) for k in ("w_p", "w_a") if p.get(k)}
    return out


def main():
    from gensim.models import KeyedVectors
    from e5v2_pairs import hints_for
    Q = pd.read_csv(PAIRS_V4, keep_default_na=False)
    banned = used_words()
    rng = np.random.default_rng(stable_seed("e6", "samples"))
    order = {s: iter(rng.permutation(np.nonzero(Q.stratum.values == s)[0])) for s in STRATA}
    taken, chosen = set(), {}
    for split, prefix, counts in SPLITS:
        picks = []
        for s, n in zip(STRATA, counts):
            k = 0
            for i in order[s]:
                a, b = Q.a.iat[i], Q.b.iat[i]
                if {a.lower(), b.lower()} & (banned | taken):
                    continue
                d = direction(a, b, rng)
                if d is None:
                    continue
                picks.append((i, d))
                taken |= {a.lower(), b.lower()}
                k += 1
                if k == n:
                    break
            if k < n:
                print(f"경고: {split} {s} 층 {k}/{n}개만", flush=True)
        chosen[split] = (prefix, picks)
    print("word2vec GoogleNews 불러오는 중", flush=True)
    kv = KeyedVectors.load_word2vec_format(str(W2V), binary=True, limit=1_000_000)
    out = dict(seed='stable_seed("e6", "samples")', source="e5/pairs_v4.csv",
               rules=dict(pronunciation="CMUdict 첫 발음", sub_feature_max=SUB_MAX, norm_max=NORM_MAX, strata=list(STRATA),
                          pos="w_a 흔한 뜻의 품사를 w_p가 가짐"),
               n_pairs=dict(all=len(Q), **{s: int((Q.stratum == s).sum()) for s in STRATA}),
               excluded_words=len(banned))
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
    E6_DIR.mkdir(exist_ok=True)
    SAMPLES_E6.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for split, _, _ in SPLITS:
        print(split)
        for p in out[split]:
            print(f"  {p['pair_id']} {p['stratum']:5s} {p['w_p']:>12} -> {p['w_a']:<12} [{p['pron_p']}] / [{p['pron_a']}] "
                  f"feat={p['d_feat']:.2f} {p['pos_a']}  hints={','.join(p['hints'])}")
    short = {p["pair_id"]: p["hints"] for s, _, _ in SPLITS for p in out[s] if len(p["hints"]) < 5}
    print(f"뺀 단어 {len(banned)}개. 힌트 5개 미만 {len(short)}쌍 {short}")


if __name__ == "__main__":
    main()
