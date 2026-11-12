"""M6 기준선 (듣는 사람 모델과 같은 언어 모델·토큰화 사용).

S_ratio, unusualness: He et al. 2019 (pungen/pungen/scorer.py와 같은 식)
모호성·뚜렷함: Kao et al. 2016, 사람 관련도 대신 word2vec GoogleNews 코사인
발음 거리: d(w_p, w_a) 정규화 값 (Palmann & Miller 2025)
문장 로그 확률: ℓ_g(w_p) / 토큰 수
"""
import pickle
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from wordfreq import word_frequency

from data import CACHE, DATA

W2V = DATA / "w2v" / "GoogleNews-vectors-negative300.bin.gz"
CONTENT_TAGS = ("NN", "VB", "JJ")


def _pivot(lm: pd.DataFrame, ctx: str, which: pd.DataFrame) -> pd.Series:
    s = lm[lm.ctx == ctx].merge(which, on=["key", "cand"])
    return s.set_index("key").logprob


def lm_baselines(items, cand: pd.DataFrame, lm: pd.DataFrame) -> pd.DataFrame:
    """S_ratio, 국소·전역 놀람, unusualness, 문장 로그 확률."""
    wp = cand[cand.is_wp][["key", "cand"]]
    wa = cand[cand.is_wa][["key", "cand"]]
    g_p, g_a = _pivot(lm, "global", wp), _pivot(lm, "global", wa)
    l_p, l_a = _pivot(lm, "he_local", wp), _pivot(lm, "he_local", wa)
    ntok = lm[lm.ctx == "global"].merge(wp, on=["key", "cand"]).set_index("key").ntok
    df = pd.DataFrame(dict(S_global=g_a - g_p, S_local=l_a - l_p))
    df["S_ratio"] = np.where((df.S_global > 0) & (df.S_local > 0), df.S_local / df.S_global, -1.0)
    df["sent_lp"] = g_p / ntok
    by_id = {it.id: it for it in items}
    uni, nw = {}, {}
    for k in df.index:
        words = re.findall(r"[A-Za-z0-9']+", by_id[k].text.lower())
        uni[k] = sum(np.log(word_frequency(w, "en", minimum=1e-8)) for w in words)
        nw[k] = len(words)
    df["unusualness"] = -(g_p - pd.Series(uni)) / pd.Series(nw)
    return df


def phon_baselines(cand: pd.DataFrame) -> pd.DataFrame:
    wa = cand[cand.is_wa].set_index("key")
    return pd.DataFrame(dict(phon_dist=wa.d_norm, phon_dist_unit=wa.d_unit, phon_dist_raw=wa.d))


# ---------------------------------------------------------------- Kao
@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner", "parser"])


def content_words(text: str) -> list[str]:
    from spacy.lang.en.stop_words import STOP_WORDS
    return [t.text for t in _nlp()(text)
            if t.tag_.startswith(CONTENT_TAGS) and t.text.lower() not in STOP_WORDS and t.is_alpha]


def kao_cosines(items, path=None) -> dict:
    """항목별 (문맥 내용어, cos(w_i, w_p), cos(w_i, w_a), p(w_p), p(w_a)). 캐시에 저장한다."""
    path = path or CACHE / "kao_cosines.pkl"
    done = pickle.loads(path.read_bytes()) if path.exists() else {}
    todo = [it for it in items if it.id not in done]
    if todo:
        from gensim.models import KeyedVectors
        print("word2vec GoogleNews 불러오는 중 (몇 분 걸림)", flush=True)
        kv = KeyedVectors.load_word2vec_format(str(W2V), binary=True)

        def vec(w):
            for c in (w, w.lower(), w.capitalize()):
                if c in kv.key_to_index:
                    return kv[c] / np.linalg.norm(kv[c])
            return None

        for it in todo:
            words = content_words(it.text)
            vp, va = vec(it.w_p), vec(it.w_a or "")
            ws, cp, ca = [], [], []
            for w in words:
                v = vec(w)
                if v is None or vp is None or va is None:
                    continue
                ws.append(w)
                cp.append(float(v @ vp))
                ca.append(float(v @ va))
            done[it.id] = dict(words=ws, cos_p=np.array(cp), cos_a=np.array(ca),
                               ok=vp is not None and va is not None,
                               prior_p=word_frequency(it.w_p, "en", minimum=1e-8),
                               prior_a=word_frequency(it.w_a or "", "en", minimum=1e-8))
        path.write_bytes(pickle.dumps(done))
    return done


def kao_features(cos: dict, keys, mu: float, sd: float, r: float) -> pd.DataFrame:
    """모호성(뜻 사후분포 엔트로피)과 뚜렷함(초점 분포의 대칭 KL).

    p(w|m, f=1) = p(w)·g(w,m),  g(w,m) = exp(r·z(cos)) / Σ_m' p(m')·exp(r·z(cos'))
    (He의 GoodmanScoreCaculator와 같은 정규화). z는 학습 분할의 평균·표준편차로 맞춘다.
    """
    rows = {}
    for k in keys:
        c = cos[k]
        if not c["ok"] or len(c["cos_p"]) == 0:
            rows[k] = dict(ambiguity=np.nan, distinctiveness=np.nan)
            continue
        prior = np.array([c["prior_p"], c["prior_a"]])
        prior = prior / prior.sum()
        e = np.exp(r * (np.stack([c["cos_p"], c["cos_a"]]) - mu) / sd)  # (2, n)
        g = e / (prior @ e)
        logpost = np.log(prior) + np.log1p(g).sum(1)
        post = np.exp(logpost - np.logaddexp(*logpost))
        amb = -np.sum(post * np.log(np.clip(post, 1e-300, None)))
        q = g / (1 + g)  # p(f=1 | w, m)
        kl = lambda a, b: a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
        dist = float(np.sum(kl(q[0], q[1]) + kl(q[1], q[0])))
        rows[k] = dict(ambiguity=amb, distinctiveness=dist)
    return pd.DataFrame.from_dict(rows, orient="index")


def kao_fit_params(cos: dict, train_keys) -> tuple[float, float]:
    allc = np.concatenate([np.r_[cos[k]["cos_p"], cos[k]["cos_a"]] for k in train_keys if cos[k]["ok"]])
    return float(allc.mean()), float(allc.std())
