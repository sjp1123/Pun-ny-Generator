"""E5 새로움·오염 검사 (자동 지표). 고른 생성물(조건별)과 사람 말장난을 infini-gram으로 조회한다.

최장 일치: 공백 단위 단어열에서 색인에 1회 이상 나오는 가장 긴 연속 구간의 단어 수(두 포인터, 단어열 길이의 약 2배 조회).
1~2단어 구간은 조회하지 않고 있다고 본다(수조 토큰 색인에서 사실상 항상 참이고, 짧은 조회가 가장 느리다).
조회가 실패하면 그 구간은 없는 것으로 보고 incomplete로 표시한다(다시 실행하면 실패한 조회만 다시 한다).
8단어 이상 일치하면 "오염 표시". 분석은 넣은 버전과 뺀 버전을 모두 보고한다(e5_human.py --analyze).
색인: 주 생성기(OLMo 2) 학습 자료 v4_olmo-mix-1124_llama. --index pile로 Pile도 볼 수 있다.

사용: python -X utf8 experiments/e5_novelty.py --stage main --gen olmo-2-1124-7b-instruct --model pythia-1.4b
출력: results/e5/<stage>/<gen>/A_<model>/novelty_<index>.csv, queries_<index>.jsonl(조회 캐시)
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from e0_contamination import INDEXES, count
from e5 import stage_dir

FLAG_N = 8
ASSUME = 2


def longest_match(words, cnt) -> tuple[int, bool]:
    """(최장 일치 단어 수, 조회 실패 여부). 반복 내내 words[i:j]는 색인에 있는 구간이다."""
    n = len(words)
    i, j = 0, min(ASSUME, n)
    best, failed = j, False
    while i < n:
        c = cnt(" ".join(words[i:j + 1])) if j < n else 0
        if c is None:
            failed, c = True, 0
        if c > 0:
            j += 1
            best = max(best, j - i)
        else:
            i += 1
            j = max(j, min(i + ASSUME, n))
    return best, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot", "main"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--index", default="olmo", choices=list(INDEXES))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    adir = stage_dir(a.stage) / a.gen / f"A_{a.model}"
    S = pd.read_csv(adir / "selections.csv").dropna(subset=["text"]).drop_duplicates("cand_id")
    qpath = adir / f"queries_{a.index}.jsonl"
    cache = {}
    if qpath.exists():
        for line in qpath.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["count"] is not None:
                cache[r["q"]] = r["count"]
    f = qpath.open("a", encoding="utf-8")
    lock = threading.Lock()

    def cnt(q):
        if q in cache:
            return cache[q]
        c = count(INDEXES[a.index], q)
        with lock:
            f.write(json.dumps(dict(q=q, count=c), ensure_ascii=False) + "\n")
            f.flush()
        if c is not None:
            cache[q] = c
        return c

    def one(r):
        best, failed = longest_match(r.text.split(), cnt)
        return dict(cand_id=r.cand_id, n_words=len(r.text.split()), longest=best, incomplete=failed)

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, S.itertuples()))
    f.close()
    N = pd.DataFrame(rows)
    N["flag"] = N.longest >= FLAG_N
    N.to_csv(adir / f"novelty_{a.index}.csv", index=False)
    M = S.merge(N, on="cand_id")
    print(f"색인 {INDEXES[a.index]}: {FLAG_N}단어 이상 일치 {int(N.flag.sum())}/{len(N)}, "
          f"조회 실패가 섞인 문장 {int(N.incomplete.sum())}")
    print(M.groupby("condition").agg(n=("flag", "size"), flagged=("flag", "mean"),
                                     longest_median=("longest", "median")).to_string())


if __name__ == "__main__":
    main()
