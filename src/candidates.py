"""M3 후보 생성: C(w_p) = {w_p} ∪ 발음 거리 상위 K개.

어휘는 CMU ∩ 빈도 zipf ≥ 3인 영문 소문자 단어. 순위는 정규화 거리, 같으면 빈도 높은 쪽이 앞.
E1에서는 w_a를 강제로 넣지 않고, E2·E3용으로는 w_a를 덧붙인 뒤 in_C 플래그를 남긴다.
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

from g2p import cmu, prons
from phon import PronTable, cost_tables, normalize, pair_distance, PIDX

K_DEFAULT = 100
ZIPF_MIN = 3.0
INDEL = float(cost_tables("feature")[1][PIDX["T"]])  # panphon 분절음 삽입 비용 (= Σ 자질 가중치)


@lru_cache(maxsize=1)
def vocab() -> tuple[list[str], np.ndarray]:
    words, zipf = [], []
    for w in cmu():
        if re.fullmatch(r"[a-z]+", w):
            z = zipf_frequency(w, "en")
            if z >= ZIPF_MIN:
                words.append(w)
                zipf.append(z)
    return words, np.array(zipf)


@lru_cache(maxsize=1)
def table() -> PronTable:
    words, _ = vocab()
    return PronTable(words, prons)


def candidates(w_p: str, K: int = K_DEFAULT, w_a: str | None = None,
               kinds=("feature", "unit"), rank_by="unit") -> pd.DataFrame:
    """후보표: cand, rank(w_p=0), d(자질 거리), d_norm, d_unit, d_unit_norm, zipf, in_C, is_wp, is_wa.

    자질 거리 d는 panphon 값을 분절음 하나의 삽입 비용으로 나눠, 삽입·삭제 하나가 1이 되게 맞췄다.
    순위는 기본으로 단순 편집 거리를 쓴다(사전 점검 재측정에서 재현율이 더 높았음, results/precheck).
    """
    words, zipf = vocab()
    tab = table()
    src = prons(w_p)
    dist = {k: tab.distances(src, k) for k in kinds}
    d_rank = dist[rank_by][1].copy()
    excl = [i for i, w in enumerate(words) if w == w_p]
    d_rank[excl] = np.inf
    order = np.lexsort((-zipf, d_rank))[:K]
    rows = [dict(cand=w_p, rank=0, d=0.0, d_norm=0.0, d_unit=0.0, d_unit_norm=0.0,
                 zipf=zipf_frequency(w_p, "en"), in_C=True)]
    for r, i in enumerate(order, 1):
        rows.append(dict(cand=words[i], rank=r, d=dist["feature"][0][i], d_norm=dist["feature"][1][i],
                         d_unit=dist["unit"][0][i], d_unit_norm=dist["unit"][1][i],
                         zipf=zipf[i], in_C=True))
    df = pd.DataFrame(rows)
    df[["d", "d_norm"]] /= INDEL
    if w_a is not None and w_a != w_p and w_a not in set(df.cand):
        pa = prons(w_a)
        d, la, lb = pair_distance(pa, src, "feature")
        du, ua, ub = pair_distance(pa, src, "unit")
        df.loc[len(df)] = dict(cand=w_a, rank=-1, d=d / INDEL, d_norm=normalize(d, la, lb) / INDEL, d_unit=du,
                               d_unit_norm=normalize(du, ua, ub), zipf=zipf_frequency(w_a, "en"),
                               in_C=False)
    df["is_wp"] = df.cand == w_p
    df["is_wa"] = (df.cand == w_a) & ~df.is_wp if w_a else False
    return df
