"""E0 학습 데이터 오염 점검 (탐색).

pun word를 가운데 둔 연속 8단어(문장이 더 짧으면 문장 전체)를 infini-gram API로
Pile-train 색인과 OLMo-mix-1124 색인에서 센다. 1회 이상이면 "본 적 있을 수 있음".
Pythia-deduped는 Pile에서 중복을 지운 데이터로 학습해서, 0회는 확실히 안 본 것이지만
1회 이상이라고 꼭 봤다는 뜻은 아니다.

사용: python -X utf8 experiments/e0_contamination.py            # 조회(캐시됨)
      python -X utf8 experiments/e0_contamination.py --model pythia-160m   # D_local 비교
"""
import json
import time

import requests

from common import *  # noqa: F403
from data import semeval_items

API = "https://api.infini-gram.io/"
INDEXES = {"pile": "v4_piletrain_llama", "olmo": "v4_olmo-mix-1124_llama"}
WINDOW = 8
WORKERS = 4


def query_text(it) -> str:
    spans, idx = it.ws_tokens()
    if len(spans) <= WINDOW:
        return it.text
    s = min(max(0, idx - WINDOW // 2 + 1), len(spans) - WINDOW)
    return it.text[spans[s][0] : spans[s + WINDOW - 1][1]]


def count(index, q, tries=5):
    for t in range(tries):
        try:
            r = requests.post(API, json=dict(index=index, query_type="count", query=q), timeout=60)
            j = r.json()
            if "count" in j:
                return int(j["count"])
            if "error" in j:
                print("API 오류:", j["error"], flush=True)
        except Exception as e:  # 네트워크 오류는 잠깐 쉬고 다시
            print("재시도:", e, flush=True)
        time.sleep(2 ** t)
    return None


def fetch():
    od = RESULTS / "e0"
    od.mkdir(parents=True, exist_ok=True)
    raw = od / "counts.jsonl"
    done = {}
    if raw.exists():
        for line in raw.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["count"] is not None:  # 실패한 조회는 다시 시도
                done[(r["key"], r["index"])] = r
    from concurrent.futures import ThreadPoolExecutor
    items = semeval_items()
    jobs = [(it.id, name, idx, query_text(it)) for it in items for name, idx in INDEXES.items()
            if (it.id, name) not in done]
    with raw.open("a", encoding="utf-8") as f, ThreadPoolExecutor(WORKERS) as ex:
        futs = [(j, ex.submit(count, j[2], j[3])) for j in jobs]
        for n, ((key, name, _, q), fu) in enumerate(futs):
            r = dict(key=key, index=name, query=q, count=fu.result())
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            done[(key, name)] = r
            if n % 200 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    df = pd.DataFrame(done.values()).pivot(index="key", columns="index", values="count")
    df.columns = [f"count_{c}" for c in df.columns]
    df["query"] = pd.DataFrame(done.values()).drop_duplicates("key").set_index("key")["query"]
    df.to_csv(od / "counts.csv")
    summary = {c: dict(n=int(df[c].notna().sum()), seen=int((df[c] > 0).sum()),
                       seen_rate=float((df[c] > 0).mean())) for c in df.filter(like="count_")}
    save_json(od / "summary.json", summary)
    print(summary)
    return df


def analyze(model, K):
    """본 집단과 안 본 집단의 D_local 분포 비교 (외웠다면 본 쪽 D가 낮아야 함)."""
    from scipy.stats import mannwhitneyu
    from listener import fit_gamma, listener, load_scores
    counts = pd.read_csv(RESULTS / "e0" / "counts.csv").set_index("key")
    items = {it.id: it for it in semeval_items(require_single_word=True)}
    cand, lm = load_cache(model, "semeval", K)
    cand = cand[cand.key.isin(items)]
    g = load_scores(cand, lm, "global")
    gam = fit_gamma(g, list(items), restrict="C")
    m = listener(load_scores(cand, lm, "local"), gam, restrict="C+wa")
    out = dict(model=model, gamma=gam)
    for c in counts.filter(like="count_"):
        seen = (counts[c].reindex(m.index) > 0)
        a, b = m.logD[seen], m.logD[~seen]
        out[c] = dict(n_seen=int(seen.sum()), n_unseen=int((~seen).sum()),
                      median_logD_seen=float(a.median()), median_logD_unseen=float(b.median()),
                      mannwhitney_p=float(mannwhitneyu(a, b).pvalue) if len(a) and len(b) else None)
    save_json(out_dir(model) / "e0_D_local.json", out)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = parser("E0 오염 점검")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.model, a.k)
    else:
        fetch()
