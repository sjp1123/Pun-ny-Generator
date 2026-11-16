"""LLM 판정기 검증: 이미 있는 사람 평가 자료로 판정기가 사람과 얼마나 맞는지 잰다 (docs/JUDGE_VALIDATION.md).

E5 파일럿 2~4에서 판정기(Qwen2.5-7B)의 세 질문이 모두 못 미더웠다. 판정기를 다시 쓰기 전에 사람 자료로 먼저 거른다.
  V1 성립   He et al. 2019 MTurk "Is it a pun?" — 148쌍 × 6방법(사람 말장난 + 기계 생성 5종) 882문장, 문장당 3명.
            판정기 yes/no vs 사람 다수결. 프롬프트 셋: works_cur(E5가 쓰던 것, w_a 모름),
            works_pair(쌍을 알려 줌 = 사람 과제와 같음), works_cot(쌍을 알려 주고 두 뜻이 각각 서는지 따져 본 뒤 답함)
  V2 웃김(점수)  같은 문장의 MTurk 웃김 1~5점(문장당 5명, 0 = 1로). 판정기 1~5점 vs 사람 평균
  V3 웃김(짝)   UKP 짝 비교 원자료(문장 쌍마다 5명) 중 말장난끼리이고 4명 이상이 같은 쪽을 고른 쌍 300개. E5의 웃김 프롬프트로 두 순서
  V4 w_a 맞히기  SemEval 이형 말장난 300개(음소 거리 0/1/2+ 층화). 농담을 준 경우 vs 단어만 준 경우 → 문맥 효과
  V5 대조 문장  E5 후보의 ①(w_a 원문)·②(단어만 바꿈)·③(완성) 100개씩 — 판정기가 ①도 말장난이라 하는가 (기술 통계)
합격 기준은 단순 규칙("사람이 쓴 문장이면 yes", "더 긴 쪽이 웃기다")이 넘지 못하는 곳에 건다: V1은 기계 생성 문장만의 균형 정확도,
V2는 사람 말장난만의 ρ, V3는 말장난끼리인 쌍.
결과 파일에는 자료의 문장을 넣지 않는다(항목 id와 판정기 답만). 문장은 data/에서 같은 시드로 다시 만든다.

사용: python -X utf8 experiments/judge_validate.py --judge Qwen/Qwen2.5-7B-Instruct [--load-4bit] [--tests V1 V2 V3 V4 V5]
      (CPU 시험) --backend ollama --judge qwen3.5:4b --think false --limit 6
출력: results/judge_val/<판정기>/ calls.jsonl, v1_works.csv, v2_funny_point.csv, v3_funny_pair.csv, v4_success.csv, v5_controls.csv,
      metrics.json, table.md, config.json
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from data import DATA, RESULTS, UKP_DIR, detokenize, semeval_items
from e5 import extract_json, write_config
from e5_gen import Generator, tag_of
from e5_judge import JUDGE_SYSTEM
from e5v2 import judge_outcome, render, unit_distance, words
from rules import dump

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "e5" / "judge_val" / "prompts"
MTURK = DATA / "pungen" / "MTurk_aggregate_results"
HE_METHODS = {0: "pku", 1: "retrieve", 2: "retrieve_repl", 3: "rule", 4: "neural", 5: "gold"}  # pungen postprocess.py
SEED = 20261117
N_V3, N_V4, N_V5 = 300, 300, 100
# 합격 기준 (docs/JUDGE_VALIDATION.md, 판정기를 돌리기 전에 고정). 단순 규칙이 못 넘는 곳에 건다:
#   "사람이 쓴 문장이면 yes"는 전체 균형 정확도 0.743, 전체 웃김 ρ 0.56을 얻고, "더 긴 쪽"은 짝 비교 0.715를 얻는다.
PASS = dict(v1a_machine_bal_acc=0.58, v1a_lower_min=0.50,  # 기계 생성만. 사람 한 명 vs 나머지 둘 = 0.631
            v1b_gold_yes=0.70,                              # 사람 말장난의 yes 비율. 사람 다수결 = 0.85
            v2_gold_rho=0.20, v2_lower_min=0.0,             # 사람 말장난만. 사람 한 명 vs 나머지 넷 = 0.243
            v3_acc=0.72, v3_tie_max=0.30)                   # 말장난끼리 쌍만. "더 긴 쪽" = 0.667~0.68
HUMAN = dict(v1_all=0.776, v1_machine=0.631, v1_gold_yes=0.845, v2_gold=0.243, v2_machine=0.067,
             base_v1_gold_only=0.743, base_v2_gold_only=0.56)  # 사람 자료에서 미리 계산한 값 (문서의 표와 같음)
B_BOOT = 2000


def h8(*parts) -> str:
    return hashlib.sha1("|||".join(parts).encode("utf-8")).hexdigest()[:10]


def vprompt(name: str, **kw) -> str:
    t = (PROMPTS / f"{name}.txt").read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
    return re.sub(r"\{(\w+)\}", lambda m: str(kw[m.group(1)]) if m.group(1) in kw else m.group(0), t)


# ---------------------------------------------------------------- 사람 자료
def he_mturk(fn: str) -> pd.DataFrame:
    """MTurk 원자료 → (pair, method, sent, ans, worker) 긴 표. order_info로 방법 번호를 되돌린다."""
    d = pd.read_csv(MTURK / fn)
    rows = []
    for _, r in d.iterrows():
        for i in (1, 2):
            order = str(r[f"Input.order_info_{i}"]).split("-")
            for j in range(6):
                rows.append(dict(pair=r[f"Input.Pun_alter_{i}"], w_p=r[f"Input.Pun_word_{i}"], method=int(order[j]),
                                 sent=r[f"Input.Sentence{i}_{j + 1}"], ans=int(r[f"Answer.Sentence{i}_{j + 1}"]),
                                 worker=r["WorkerId"]))
    R = pd.DataFrame(rows)
    R = R[R.sent.map(lambda s: isinstance(s, str) and bool(s.strip()))].reset_index(drop=True)  # 빈 문장(생성 실패) 제외
    R["w_a"] = [p[len(w) + 1:] for p, w in zip(R.pair, R.w_p)]
    R["item"] = [f"he|{h8(p, str(m), s)}" for p, m, s in zip(R.pair, R.method, R.sent)]
    return R


def he_items() -> pd.DataFrame:
    """문장 단위: 사람 성립 다수결(3명), 웃김 평균(5명). 말장난 단어가 문장에 없는 것은 뺀다."""
    S, F = he_mturk("pun_success_final.csv"), he_mturk("funniness_final.csv")
    key = ["item", "pair", "w_p", "w_a", "method", "sent"]
    a = S.assign(yes=S.ans == 1).groupby(key).yes.agg(n_succ="size", yes="sum").reset_index()
    a["human_pun"] = a.yes * 2 > a.n_succ
    f = F.assign(s=F.ans.clip(lower=1)).groupby("item").s.agg(n_fun="size", human_funny="mean")
    a = a.join(f, on="item")
    a["text"] = [detokenize(s.split()) for s in a.sent]
    has = [bool(re.search(r"(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])", t, re.I)) for w, t in zip(a.w_p, a.text)]
    return a[has].drop(columns="sent").sort_values("item").reset_index(drop=True)


def ukp_pairs(n: int) -> pd.DataFrame:
    """말장난끼리인 쌍 가운데 사람 5명 중 4명 이상이 같은 쪽을 고른 쌍에서 n개 (시드 고정).
    말장난 대 농담 아님 같은 쌍은 범주만 알아도 맞히므로 뺀다."""
    from data import load_ukp
    r = pd.read_csv(UKP_DIR / "results.tsv", sep="\t", quoting=3, encoding="utf-8")
    r.columns = ["worker", "ans", "a", "b"]
    g = r.groupby(["a", "b"]).ans.agg(n="size", va=lambda x: (x == "A").sum(), vb=lambda x: (x == "B").sum()).reset_index()
    u = load_ukp()
    pun = set(u[u.category.isin(["hetero_pun", "homo_pun"])].text)
    g = g[(g.n == 5) & (g[["va", "vb"]].max(axis=1) >= 4) & g.a.isin(pun) & g.b.isin(pun)].copy()
    g["len_a"], g["len_b"] = g.a.str.split().str.len(), g.b.str.split().str.len()
    g["item"] = [f"ukp|{h8(a, b)}" for a, b in zip(g.a, g.b)]
    g["human"] = np.where(g.va > g.vb, "a", "b")
    g = g.sort_values("item").reset_index(drop=True)
    return g.iloc[np.sort(np.random.default_rng(SEED).choice(len(g), min(n, len(g)), replace=False))].reset_index(drop=True)


def semeval_sample(n: int) -> pd.DataFrame:
    its = [it for it in semeval_items(require_single_word=True) if it.w_a.isalpha() and it.w_p.isalpha()]
    D = pd.DataFrame([dict(item=f"sem|{it.id}", text=it.text, pun_word=it.surface, w_a=it.w_a,
                           d=min(2, int(unit_distance(it.w_p, it.w_a)))) for it in its]).sort_values("item")
    rng = np.random.default_rng(SEED)
    parts = [g.iloc[np.sort(rng.choice(len(g), min(n // 3, len(g)), replace=False))] for _, g in D.groupby("d")]
    return pd.concat(parts).reset_index(drop=True)


def pilot_controls(gdir: Path, n: int) -> pd.DataFrame:
    c = pd.read_csv(gdir / "candidates.csv")
    c = c[(c.source == "P") & c.text1.notna() & c.text2.notna()].sort_values("cand_id").reset_index(drop=True)
    c = c.iloc[np.sort(np.random.default_rng(SEED).choice(len(c), min(n, len(c)), replace=False))]
    rows = []
    for r in c.itertuples():
        rows += [dict(item=f"ctl|{r.cand_id}|1", ver="1 w_a 원문", text=r.text1, word=r.w_a, other=r.w_p),
                 dict(item=f"ctl|{r.cand_id}|2", ver="2 단어만 바꿈", text=r.text2, word=r.w_p, other=r.w_a),
                 dict(item=f"ctl|{r.cand_id}|3", ver="3 완성", text=r.text, word=r.w_p, other=r.w_a)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 판정 호출
class Judge:
    """호출 기록(calls.jsonl)에는 id와 답만 남긴다. 기록에 다 있으면 모델을 올리지 않는다."""

    def __init__(self, make, path: Path):
        self.make, self.path, self.gen = make, path, None
        self.rec = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rec[r["call_id"]] = r["raw"]

    def ask(self, cid: str, prompt: str, system=None, max_new=12, keep=None) -> str:
        """keep: 긴 답(works_cot의 풀이)에서 기록에 남길 부분만 고른다. 풀이에 자료 문장이 섞일 수 있어서
        전체 답은 깃에 올리지 않는 calls_local.jsonl에만 둔다."""
        if cid not in self.rec:
            if self.gen is None:
                self.gen = self.make()
            raw = self.gen.chat(prompt, seed=0, temperature=0.0, max_new_tokens=max_new, system=system)
            if keep is not None:
                with self.path.with_name("calls_local.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(call_id=cid, raw=raw), ensure_ascii=False) + "\n")
                raw = keep(raw)
            self.rec[cid] = raw
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(call_id=cid, raw=raw), ensure_ascii=False) + "\n")
        return self.rec[cid]


def yes_no(raw: str):
    w = words(raw)
    return None if not w or w[0].lower() not in ("yes", "no") else w[0].lower() == "yes"


def answer_line(raw: str):
    """works_cot: 마지막 'Answer: yes/no'."""
    m = re.findall(r"answer\s*:\s*\**\s*(yes|no)", raw or "", re.I)
    return m[-1].lower() == "yes" if m else None


WORKS = {  # 성립 프롬프트: 이름 → (프롬프트, 답 읽기, 최대 토큰)
    "works_cur": (lambda w, o, t: render("judge_works", w_p=w, text=t), yes_no, 12, None),
    "works_pair": (lambda w, o, t: vprompt("works_pair", w_p=w, w_a=o, text=t), yes_no, 12, None),
    "works_cot": (lambda w, o, t: vprompt("works_cot", w_p=w, w_a=o, text=t), answer_line, 120,
                  lambda raw: {True: "Answer: yes", False: "Answer: no", None: "(no answer line)"}[answer_line(raw)]),
}


def ask_works(J, prefix: str, name: str, item: str, word: str, other: str, text: str):
    make, parse, mx, keep = WORKS[name]
    return parse(J.ask(f"{prefix}{name[6:]}|{item}", make(word, other, text), max_new=mx, keep=keep))


def one_to_five(raw: str):
    m = re.search(r"[1-5]", raw or "")
    return int(m.group(0)) if m else None


# ---------------------------------------------------------------- 통계
def cluster_boot(df: pd.DataFrame, cluster: str, stat, B=B_BOOT):
    """묶음(cluster) 재표집 95% 구간. stat(DataFrame) → 실수."""
    rng = np.random.default_rng(SEED)
    groups = [g for _, g in df.groupby(cluster)]
    reps = [stat(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(B)]
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return dict(est=float(stat(df)), lo=float(lo), hi=float(hi))


def bal_acc(d: pd.DataFrame, col: str) -> float:
    d = d.dropna(subset=[col])
    pos, neg = d[d.human_pun], d[~d.human_pun]
    if not len(pos) or not len(neg):
        return np.nan
    return float((pos[col].astype(bool).mean() + (~neg[col].astype(bool)).mean()) / 2)


def kappa(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def spearman(x, y) -> float:
    x, y = pd.Series(list(x), dtype=float), pd.Series(list(y), dtype=float)
    ok = x.notna() & y.notna()
    return float(x[ok].corr(y[ok], method="spearman")) if ok.sum() > 2 and x[ok].nunique() > 1 and y[ok].nunique() > 1 else np.nan


def ci(v: dict) -> str:
    return "—" if np.isnan(v["est"]) else f"{v['est']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"


# ---------------------------------------------------------------- 시험
def run_v1_v2(J: Judge, H: pd.DataFrame, tests, out: Path, res: dict, L: list):
    if "V1" in tests:
        for name in WORKS:
            H[name] = [ask_works(J, "v1", name, r.item, r.w_p, r.w_a, r.text) for r in H.itertuples()]
        H[["item", "pair", "method", "n_succ", "yes", "human_pun", *WORKS]].to_csv(out / "v1_works.csv", index=False)
        M = H[H.method != 5]
        res["V1"] = dict(n=len(H), n_machine=len(M), machine_human_yes=int(M.human_pun.sum()), human_yes=float(H.human_pun.mean()),
                         pass_threshold={k: v for k, v in PASS.items() if k.startswith("v1")}, human=HUMAN)
        L += ["## V1 성립 (He MTurk \"Is it a pun?\", 사람 다수결 대비)", "",
              f"문장 {len(H)}개(기계 생성 {len(M)}개 중 사람 yes {int(M.human_pun.sum())}개), 쌍 {H.pair.nunique()}개.",
              "합격: (a) 기계 생성 문장만의 균형 정확도 ≥ 0.58이고 95% 하한 > 0.50, 그리고 (b) 사람 말장난의 yes 비율 ≥ 0.70.",
              "사람 수준: 한 명이 나머지 둘(일치할 때)과 맞는 균형 정확도는 기계 생성만 0.631, 전체 0.776. "
              "\"사람이 쓴 문장이면 yes\" 규칙은 전체 0.743을 얻으므로 전체 값은 기준으로 쓰지 않는다.", "",
              "| 프롬프트 | (a) 기계 생성만 균형 정확도 [95%] | (b) 사람 말장난 yes | 전체 균형 정확도 [95%] | 전체 yes 비율 | κ | 못 읽음 | 합격 |",
              "|---|---|---|---|---|---|---|---|"]
        for col in WORKS:
            d = H.dropna(subset=[col])
            ba = cluster_boot(H, "pair", lambda x: bal_acc(x, col))
            bm = cluster_boot(M, "pair", lambda x: bal_acc(x, col))
            gy = float(d[d.method == 5][col].astype(bool).mean())
            ok = bool(bm["est"] >= PASS["v1a_machine_bal_acc"] and bm["lo"] > PASS["v1a_lower_min"] and gy >= PASS["v1b_gold_yes"])
            v = dict(yes=float(d[col].astype(bool).mean()), tpr=float(d[d.human_pun][col].astype(bool).mean()),
                     fpr=float(d[~d.human_pun][col].astype(bool).mean()), bal_acc=ba, bal_acc_machine=bm, gold_yes=gy,
                     machine_tpr=float(d[(d.method != 5) & d.human_pun][col].astype(bool).mean()),
                     machine_fpr=float(d[(d.method != 5) & ~d.human_pun][col].astype(bool).mean()),
                     kappa=kappa(d[col], d.human_pun), unparsed=float(H[col].isna().mean()),
                     by_method={HE_METHODS[m]: float(g[col].astype(bool).mean()) for m, g in d.groupby("method")},
                     passed=ok)
            res["V1"][col] = v
            L.append(f"| {col} | {ci(bm)} | {gy:.3f} | {ci(ba)} | {v['yes']:.3f} | {v['kappa']:.3f} | "
                     f"{v['unparsed']:.3f} | {'통과' if ok else '실패'} |")
        hm = {HE_METHODS[m]: float(g.human_pun.mean()) for m, g in H.groupby("method")}
        res["V1"]["human_by_method"] = hm
        L += ["", "방법별 yes 비율:", "", "| | " + " | ".join(hm) + " |", "|---|" + "---|" * len(hm),
              "| 사람 다수결 | " + " | ".join(f"{v:.2f}" for v in hm.values()) + " |"]
        for col in WORKS:
            L.append(f"| {col} | " + " | ".join(f"{res['V1'][col]['by_method'].get(k, np.nan):.2f}" for k in hm) + " |")
        L.append("")
    if "V2" in tests:
        F = H[H.human_funny.notna()].copy()
        F["judge_funny"] = [one_to_five(J.ask(f"v2|{r.item}", vprompt("funny_point", text=r.text), max_new=6)) for r in F.itertuples()]
        F[["item", "pair", "method", "n_fun", "human_funny", "judge_funny"]].to_csv(out / "v2_funny_point.csv", index=False)
        rho = lambda x: spearman(x.judge_funny, x.human_funny)
        allr = cluster_boot(F, "pair", rho)
        mach = cluster_boot(F[F.method != 5], "pair", rho)
        gold = cluster_boot(F[F.method == 5], "pair", rho)
        res["V2"] = dict(n=len(F), rho_all=allr, rho_machine=mach, rho_gold=gold, unparsed=float(F.judge_funny.isna().mean()),
                         judge_mean_by_method={HE_METHODS[m]: float(g.judge_funny.mean()) for m, g in F.groupby("method")},
                         human_mean_by_method={HE_METHODS[m]: float(g.human_funny.mean()) for m, g in F.groupby("method")},
                         pass_threshold=dict(gold_rho=PASS["v2_gold_rho"], lower_min=PASS["v2_lower_min"]),
                         passed=bool(gold["est"] >= PASS["v2_gold_rho"] and gold["lo"] > PASS["v2_lower_min"]))
        L += ["## V2 웃김 점수 (He MTurk 1~5점, 사람 평균 대비 Spearman ρ)", "",
              f"문장 {len(F)}개. 합격: 사람 말장난만의 ρ ≥ 0.20이고 95% 하한 > 0.",
              "사람 수준(한 명 vs 나머지 넷 평균): 사람 말장난만 0.243, 기계 생성만 0.067(사람끼리도 안 맞아 기준을 둘 수 없다). "
              "\"사람이 쓴 문장이면 높은 점수\" 규칙이 전체 ρ 0.56을 얻으므로 전체 값은 기준으로 쓰지 않는다.", "",
              "| 범위 | ρ [95%, 쌍 묶음] |", "|---|---|",
              f"| 사람 말장난만 (기준) | {ci(gold)} |", f"| 기계 생성만 | {ci(mach)} |", f"| 전체 | {ci(allr)} |", "",
              f"못 읽음 {res['V2']['unparsed']:.3f}. 합격: **{'통과' if res['V2']['passed'] else '실패'}**", "",
              "방법별 평균 점수: " + ", ".join(f"{k} 사람 {res['V2']['human_mean_by_method'][k]:.2f} / 판정기 {v:.2f}"
                                       for k, v in res["V2"]["judge_mean_by_method"].items()), ""]


def run_v3(J: Judge, out: Path, res: dict, L: list, limit):
    U = ukp_pairs(N_V3)[:limit]
    rows = []
    for r in U.itertuples():
        pick = []
        for tag, first, second in (("ab", r.a, r.b), ("ba", r.b, r.a)):
            p = extract_json(J.ask(f"v3|{r.item}|{tag}", render("judge_funny", a=first, b=second), system=JUDGE_SYSTEM, max_new=60))
            ch = str((p or {}).get("funnier", "")).strip().upper()[:1] if isinstance(p, dict) else ""
            pick.append(None if ch not in ("A", "B") else ("a" if (ch == "A") == (tag == "ab") else "b"))
        first_pos = [None if x is None else (x == "a") == (t == "ab") for x, t in zip(pick, ("ab", "ba"))]
        outcome = pick[0] if pick[0] is not None and pick[0] == pick[1] else ("tie" if None not in pick else "unparsed")
        longer = "a" if r.len_a > r.len_b else "b" if r.len_b > r.len_a else "same"
        rows.append(dict(item=r.item, human=r.human, votes=max(r.va, r.vb), longer=longer, pick_ab=pick[0], pick_ba=pick[1],
                         outcome=outcome, first1=first_pos[0], first2=first_pos[1]))
    V = pd.DataFrame(rows)
    V.to_csv(out / "v3_funny_pair.csv", index=False)
    score = lambda d: float(((d.outcome == d.human) + 0.5 * (d.outcome == "tie")).mean()) if len(d) else np.nan
    dec = V[V.outcome.isin(["a", "b"])]
    acc = float((dec.outcome == dec.human).mean()) if len(dec) else np.nan
    half = score(V)
    per_order = float(pd.concat([V.pick_ab == V.human, V.pick_ba == V.human]).mean())
    tie = float((V.outcome == "tie").mean())
    first = float(pd.concat([V.first1, V.first2]).dropna().astype(bool).mean())
    base_len = float(np.where(V.longer == "same", 0.5, V.longer == V.human).mean())
    short_win = V[(V.longer != "same") & (V.longer != V.human)]  # 사람이 더 짧은 쪽을 고른 쌍
    long_win = V[(V.longer != "same") & (V.longer == V.human)]
    res["V3"] = dict(n=len(V), decided=len(dec), acc_decided=acc, acc_tie_half=half, acc_per_order=per_order, tie=tie,
                     first_position=first, unparsed=float((V.outcome == "unparsed").mean()), baseline_longer=base_len,
                     acc_human_chose_shorter=score(short_win), n_human_chose_shorter=len(short_win),
                     acc_human_chose_longer=score(long_win), n_human_chose_longer=len(long_win),
                     pass_threshold=dict(acc=PASS["v3_acc"], tie_max=PASS["v3_tie_max"]),
                     passed=bool(half >= PASS["v3_acc"] and tie <= PASS["v3_tie_max"]))
    L += ["## V3 웃김 짝 비교 (UKP 원자료, 말장난끼리인 쌍, 사람 5명 중 4명 이상이 고른 쪽 대비)", "",
          f"쌍 {len(V)}개. 두 순서가 같은 답: {len(dec)}개, 엇갈림(무승부) {tie:.3f}, 먼저 보인 쪽 선택 {first:.3f}.",
          f"합격: 정확도(무승부 = 0.5) ≥ 0.72 그리고 무승부 ≤ 0.30. \"더 긴 쪽이 웃기다\" 규칙은 이 표본에서 {base_len:.3f}.", "",
          "| 지표 | 값 |", "|---|---|", f"| 정확도 (무승부 = 0.5, 기준) | {half:.3f} |", f"| 정확도 (답이 갈리지 않은 쌍만) | {acc:.3f} |",
          f"| 정확도 (순서별 답 하나씩) | {per_order:.3f} |",
          f"| 사람이 더 긴 쪽을 고른 쌍 (n = {len(long_win)}) | {score(long_win):.3f} |",
          f"| 사람이 더 짧은 쪽을 고른 쌍 (n = {len(short_win)}) | {score(short_win):.3f} |", "",
          f"합격: **{'통과' if res['V3']['passed'] else '실패'}**", ""]


def run_v4(J: Judge, out: Path, res: dict, L: list, limit):
    S = semeval_sample(N_V4)
    if limit:
        S = S.groupby("d").head(max(1, limit // 3))
    rows = []
    for r in S.itertuples():
        ctx = judge_outcome(J.ask(f"v4ctx|{r.item}", render("judge_success", w_p=r.pun_word, text=r.text)), r.pun_word, r.w_a)
        no = judge_outcome(J.ask(f"v4no|{r.pun_word.lower()}", vprompt("success_noctx", w_p=r.pun_word)), r.pun_word, r.w_a)
        rows.append(dict(item=r.item, d=r.d, with_joke=ctx["success"], no_context=no["success"], echo=ctx["echo"]))
    V = pd.DataFrame(rows)
    V.to_csv(out / "v4_success.csv", index=False)
    res["V4"] = dict(n=len(V), by_distance={})
    L += ["## V4 w_a 맞히기 (SemEval 사람 말장난, 농담을 준 경우 vs 단어만 준 경우)", "",
          "| 음소 거리 | n | 농담과 함께 | 단어만 | 문맥 효과 (차이) | w_p 되풀이 |", "|---|---|---|---|---|---|"]
    for name, g in [(f"{d}" if d < 2 else "2 이상", g) for d, g in V.groupby("d")] + [("전체", V)]:
        v = dict(n=len(g), with_joke=float(g.with_joke.mean()), no_context=float(g.no_context.mean()),
                 lift=float(g.with_joke.mean() - g.no_context.mean()), echo=float(g.echo.mean()))
        res["V4"]["by_distance"][name] = v
        L.append(f"| {name} | {v['n']} | {v['with_joke']:.3f} | {v['no_context']:.3f} | {v['lift']:+.3f} | {v['echo']:.3f} |")
    L.append("")


def run_v5(J: Judge, gdir: Path, out: Path, res: dict, L: list, limit):
    C = pilot_controls(gdir, N_V5)
    if limit:
        C = C[: limit * 3]
    for name in WORKS:
        C[name] = [ask_works(J, "v5", name, r.item, r.word, r.other, r.text) for r in C.itertuples()]
    C[["item", "ver", *WORKS]].to_csv(out / "v5_controls.csv", index=False)
    t = C.groupby("ver")[list(WORKS)].agg(lambda x: x.dropna().astype(bool).mean())
    res["V5"] = dict(n=len(C) // 3, source=str(gdir.relative_to(RESULTS)).replace("\\", "/"), yes_rate=t.to_dict())
    L += ["## V5 대조 문장 (E5 후보의 ①·②·③, yes 비율)", "",
          f"후보 {len(C) // 3}개 ({res['V5']['source']}). ①은 w_a가 그대로 쓰인 평범한 문장이라 yes가 낮아야 한다.", "",
          "| 문장 | " + " | ".join(WORKS) + " |", "|---|" + "---|" * len(WORKS)]
    L += [f"| {k} | " + " | ".join(f"{r[n]:.3f}" for n in WORKS) + " |" for k, r in t.iterrows()] + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--think", default=None, choices=["true", "false"])
    ap.add_argument("--tests", nargs="+", default=["V1", "V2", "V3", "V4", "V5"], choices=["V1", "V2", "V3", "V4", "V5"])
    ap.add_argument("--pilot", default="e5/pilot4/olmo-2-0325-32b-instruct-4bit", help="V5에 쓸 E5 생성 폴더 (results/ 아래)")
    ap.add_argument("--limit", type=int, default=None, help="시험용: 시험마다 앞의 몇 항목만 (결과 폴더에 -limitN)")
    a = ap.parse_args()
    tag = tag_of(a.judge) + ("-4bit" if a.load_4bit else "") + (f"-limit{a.limit}" if a.limit else "")
    out = RESULTS / "judge_val" / tag
    out.mkdir(parents=True, exist_ok=True)
    J = Judge(lambda: Generator(a.backend, a.judge, a.device, think=None if a.think is None else a.think == "true",
                                load_4bit=a.load_4bit), out / "calls.jsonl")
    res, L = {}, [f"# 판정기 검증 — {a.judge}{' (NF4 4비트)' if a.load_4bit else ''}", ""]
    if {"V1", "V2"} & set(a.tests):
        H = he_items()
        if a.limit:
            H = H.groupby("method").head(a.limit).reset_index(drop=True)
        run_v1_v2(J, H, a.tests, out, res, L)
    if "V3" in a.tests:
        run_v3(J, out, res, L, a.limit)
    if "V4" in a.tests:
        run_v4(J, out, res, L, a.limit)
    if "V5" in a.tests:
        run_v5(J, RESULTS / a.pilot, out, res, L, a.limit)
    info = J.gen.info if J.gen is not None else dict(backend=a.backend, model=a.judge, note="모두 기록에서 읽음")
    write_config(out, dict(judge=info, tests=a.tests, limit=a.limit, temperature=0.0, sample_seed=SEED, pass_thresholds=PASS,
                           n=dict(v3=N_V3, v4=N_V4, v5=N_V5), calls_from_cache=J.gen is None))
    v1 = res.get("V1", {})
    res["_pass"] = dict(works=v1.get("works_pair", {}).get("passed"), works_cot=v1.get("works_cot", {}).get("passed"),
                        works_cur=v1.get("works_cur", {}).get("passed"),
                        funny_point=res.get("V2", {}).get("passed"), funny_pair=res.get("V3", {}).get("passed"))
    yn = lambda v: "—" if v is None else ("통과" if v else "실패")
    L[2:2] = ["| 쓰임 | 성립 (쌍 알려 줌) | 성립 (쌍 + 따져 보기) | 성립 (E5 프롬프트) | 웃김 점수 | 웃김 짝 비교 |", "|---|---|---|---|---|---|",
              f"| 판정 | {yn(res['_pass']['works'])} | {yn(res['_pass']['works_cot'])} | {yn(res['_pass']['works_cur'])} | "
              f"{yn(res['_pass']['funny_point'])} | "
              f"{yn(res['_pass']['funny_pair'])} |", ""]
    dump(out / "metrics.json", res)
    (out / "table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
