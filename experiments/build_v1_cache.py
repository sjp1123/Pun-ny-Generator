"""보완 실험용 LM 캐시.

실험 3: 기존 후보(C_phon)에 왼쪽 문맥 점수 ctx="left" — log P_LM(T | x_{1:i-1}) (후보 토큰만, 앞 공백 포함)
실험 4-B: C_LM = 왼쪽 문맥에서 첫 토큰 로그 확률로 어휘를 줄 세운 상위 100개 + w_p (+ w_a 강제)
          → 후보표 candidates_semevalLM_K100.parquet, 점수 ctx = global, left

사용: python -X utf8 experiments/build_v1_cache.py --model pythia-160m --dataset semeval he
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

from candidates import table, vocab
from data import CACHE, load_he_annotation, match_case, semeval_items
from g2p import prons
from lmscore import CacheWriter, LMScorer, Progress, set_threads, split_at
from phon import normalize, pair_distance

K_LM = 100


class FirstTokens(dict):
    """(앞 공백 문자열, 대문자 여부)별 어휘 단어의 첫 토큰 id. 처음 쓰일 때 만든다."""

    def __init__(self, scorer, words):
        super().__init__()
        self.scorer, self.words = scorer, words

    def __missing__(self, key):
        lead, cap = key
        forms = [lead + (w[:1].upper() + w[1:] if cap else w) for w in self.words]
        ids = self.scorer.tok(forms, add_special_tokens=False)["input_ids"]
        self[key] = np.array([i[0] for i in ids])
        return self[key]


def first_token_table(scorer, words):
    return FirstTokens(scorer, words)


def lm_candidates(it, logp_next, ftab, words, zipf, K=K_LM):
    prefix, lead = split_at(it)
    cap = it.surface[:1].isupper()
    s = logp_next.numpy()[ftab[(lead, cap)]].astype(float)
    s[[i for i, w in enumerate(words) if w == it.w_p]] = -np.inf
    order = np.lexsort((-zipf, -s))[:K]
    return [words[i] for i in order], s[order]


def lm_cand_table(it, lm_words, lm_scores):
    words, zipf = vocab()
    widx = {w: i for i, w in enumerate(words)}
    src = prons(it.w_p)
    du, dun = table().distances(src, "unit")
    df_, dfn = table().distances(src, "feature")
    rows = [dict(cand=it.w_p, rank=0, first_lp=np.nan, d=0.0, d_norm=0.0, d_unit=0.0, d_unit_norm=0.0,
                 zipf=zipf_frequency(it.w_p, "en"), in_C=True)]
    for r, (w, s) in enumerate(zip(lm_words, lm_scores), 1):
        i = widx[w]
        rows.append(dict(cand=w, rank=r, first_lp=s, d=df_[i], d_norm=dfn[i], d_unit=du[i], d_unit_norm=dun[i],
                         zipf=zipf[i], in_C=True))
    df = pd.DataFrame(rows)
    from candidates import INDEL
    df[["d", "d_norm"]] /= INDEL
    if it.w_a and it.w_a != it.w_p and it.w_a not in set(df.cand):
        pa = prons(it.w_a)
        d, la, lb = pair_distance(pa, src, "feature")
        u, ua, ub = pair_distance(pa, src, "unit")
        df.loc[len(df)] = dict(cand=it.w_a, rank=-1, first_lp=np.nan, d=d / INDEL, d_norm=normalize(d, la, lb) / INDEL,
                               d_unit=u, d_unit_norm=normalize(u, ua, ub), zipf=zipf_frequency(it.w_a, "en"), in_C=False)
    df["is_wp"] = df.cand == it.w_p
    df["is_wa"] = (df.cand == it.w_a) & ~df.is_wp
    df.insert(0, "key", it.id)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m")
    ap.add_argument("--dataset", nargs="+", default=["semeval", "he"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    set_threads()
    sc = LMScorer(args.model, args.device)
    words, zipf = vocab()
    ftab = first_token_table(sc, words)
    for ds in args.dataset:
        its = semeval_items() if ds == "semeval" else load_he_annotation()[0]
        cand = pd.read_parquet(CACHE / f"candidates_{ds}_K100.parquet")
        by_key = {k: list(g.cand) for k, g in cand.groupby("key", sort=False)}
        wl = CacheWriter(CACHE / "lm" / args.model / f"{ds}_K100_left")
        lm_dir = CACHE / "lm" / args.model / f"{ds}LM_K100"
        wg = CacheWriter(lm_dir) if ds == "semeval" else None
        ct_path = CACHE / f"candidates_{ds}LM_K100.parquet"
        ct_parts = [pd.read_parquet(ct_path)] if ct_path.exists() else []
        if wg is not None:  # 후보표에 없는 항목은 점수가 있어도 다시 만든다
            wg.done &= set(ct_parts[0].key) if ct_parts else set()
        todo = [it for it in its if it.id not in wl.done or (wg is not None and it.id not in wg.done)]
        prog = Progress(len(todo), f"{args.model} {ds} left/C_LM")
        for it in todo:
            prefix, lead = split_at(it)
            conts = [lead + match_case(it.surface, c) for c in by_key[it.id]]
            res, logp_next = sc.continuation(prefix, conts)
            if it.id not in wl.done:
                wl.add(it.id, [dict(ctx="left", cand=c, logprob=lp, ntok=n) for c, (lp, n) in zip(by_key[it.id], res)])
            if wg is not None and it.id not in wg.done:
                lw, ls = lm_candidates(it, logp_next, ftab, words, zipf)
                ct = lm_cand_table(it, lw, ls)
                ct_parts.append(ct)
                cs = list(ct.cand)
                lres, _ = sc.continuation(prefix, [lead + match_case(it.surface, c) for c in cs])
                tail = it.text[it.end:]
                gres = sc.score_shared_prefix(prefix, [lead + match_case(it.surface, c) + tail for c in cs])
                rows = [dict(ctx="left", cand=c, logprob=lp, ntok=n) for c, (lp, n) in zip(cs, lres)]
                rows += [dict(ctx="global", cand=c, logprob=lp, ntok=n) for c, (lp, n) in zip(cs, gres)]
                wg.add(it.id, rows)
                if len(ct_parts) % 50 == 0:
                    pd.concat(ct_parts, ignore_index=True).drop_duplicates(["key", "cand"]).to_parquet(ct_path, index=False)
            prog.step()
        wl.flush()
        if wg is not None:
            wg.flush()
            pd.concat(ct_parts, ignore_index=True).drop_duplicates(["key", "cand"]).to_parquet(ct_path, index=False)


if __name__ == "__main__":
    main()
