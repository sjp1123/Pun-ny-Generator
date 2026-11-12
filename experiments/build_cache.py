"""후보표와 LM 점수 캐시를 만든다. 실험 스크립트는 이 캐시만 읽는다.

사용: python -X utf8 experiments/build_cache.py --model pythia-160m --dataset semeval he
      python -X utf8 experiments/build_cache.py --model pythia-160m --dataset e2b --k 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import warnings

warnings.filterwarnings("ignore")
import pandas as pd

from candidates import K_DEFAULT, candidates
from data import CACHE, Item, load_he_annotation, semeval_items
from lmscore import CacheWriter, LMScorer, Progress, score_item, set_threads


def items_for(dataset: str) -> list[Item]:
    if dataset == "semeval":
        return semeval_items()
    if dataset == "he":
        return load_he_annotation()[0]
    if dataset == "e2b":
        from e2b_positions import e2b_items
        return e2b_items()
    raise ValueError(dataset)


def candidate_table(dataset: str, items: list[Item], K: int) -> pd.DataFrame:
    path = CACHE / f"candidates_{dataset}_K{K}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames, memo = [], {}
    prog = Progress(len(items), f"후보 {dataset}")
    for it in items:
        k = (it.w_p, it.w_a)
        if k not in memo:
            memo[k] = candidates(it.w_p, K=K, w_a=it.w_a)
        df = memo[k].copy()
        df.insert(0, "key", it.id)
        frames.append(df)
        prog.step()
    out = pd.concat(frames, ignore_index=True)
    CACHE.mkdir(exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m")
    ap.add_argument("--dataset", nargs="+", default=["semeval", "he"])
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--candidates-only", action="store_true")
    args = ap.parse_args()
    set_threads()
    scorer = None
    for ds in args.dataset:
        items = items_for(ds)
        cand = candidate_table(ds, items, args.k)
        if args.candidates_only:
            continue
        scorer = scorer or LMScorer(args.model, args.device, args.dtype)
        w = CacheWriter(CACHE / "lm" / args.model / f"{ds}_K{args.k}")
        by_key = {k: g for k, g in cand.groupby("key", sort=False)}
        todo = [it for it in items if it.id not in w.done]
        prog = Progress(len(todo), f"{args.model} {ds}")
        for it in todo:
            cands = list(by_key[it.id].cand)
            rows = [dict(ctx=c, cand=t, logprob=lp, ntok=n)
                    for c, t, lp, n in score_item(scorer, it, cands, he_window=ds != "e2b")]
            w.add(it.id, rows)
            prog.step()
        w.flush()


if __name__ == "__main__":
    main()
