"""1단계 보완 실험(docs/EXPERIMENTS.md)의 고정 분석 규칙.

- 분할: dev = semeval/hetero/dev.json, test = test.json (splits/*.txt, SHA-256은 PREREG.md)
- 숨은 단어 채점: 표제어 합  π_lex(w_a) = Σ_{T∈C, T≠w_p, lemma(T)=lemma(w_a)} π(T)
- 발음 항: 단순 음소 편집 거리 × 단일 γ (구간 가중치는 H4a용)
- 부트스트랩: 항목 복원 추출, B = 5,000, 95% 백분위, 비교는 짝지은 방식
"""
import hashlib
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from data import CACHE, RESULTS, ROOT, load_semeval

SEED = 20261112
B_BOOT = 5000
SPLITS = ROOT / "splits"
LEMMA_EXCEPTIONS = ROOT / "src" / "lemma_exceptions.json"
LEMMA_TOOL = "spaCy 3.8.18 + en_core_web_sm 3.8.0 (단어 하나씩, 문맥 없이)"
BINS = (0, 1, 2, 3)  # d = 0, 1, 2, 3 이상


# ---------------------------------------------------------------- 분할
def write_splits() -> dict:
    SPLITS.mkdir(exist_ok=True)
    out = {}
    for split in ("dev", "test"):
        ids = [r["id"] for r in load_semeval() if r["split"] == split]
        p = SPLITS / f"{split}_ids.txt"
        p.write_bytes(("\n".join(ids) + "\n").encode("utf-8"))
        out[split] = dict(n=len(ids), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return out


@lru_cache(maxsize=None)
def split_ids(split: str) -> frozenset:
    return frozenset((SPLITS / f"{split}_ids.txt").read_text(encoding="utf-8").split())


def split_sha(split: str) -> str:
    return hashlib.sha256((SPLITS / f"{split}_ids.txt").read_bytes()).hexdigest()


# ---------------------------------------------------------------- 표제어
@lru_cache(maxsize=1)
def _exceptions() -> dict:
    return json.loads(LEMMA_EXCEPTIONS.read_text(encoding="utf-8")) if LEMMA_EXCEPTIONS.exists() else {}


def lemmas(words) -> dict:
    """단어 → 표제어. spaCy 결과는 cache/lemmas.parquet에 두고, 예외 목록이 우선한다."""
    path = CACHE / "lemmas.parquet"
    lem = pd.read_parquet(path).set_index("word").lemma.to_dict() if path.exists() else {}
    todo = sorted({w for w in words if w not in lem})
    if todo:
        from baselines import _nlp
        for w, doc in zip(todo, _nlp().pipe(todo, batch_size=2000)):
            lem[w] = doc[0].lemma_.lower() if len(doc) else w
        pd.DataFrame(dict(word=list(lem), lemma=list(lem.values()))).to_parquet(path, index=False)
    exc = _exceptions()
    return {w: exc.get(w, lem[w]) for w in words}


def add_lemma_cols(cand: pd.DataFrame) -> pd.DataFrame:
    """lemma(후보), 항목별 목표 표제어 lemma(w_a), M(w_a) 표시 in_M, 제외 표시 same_lemma."""
    wa = cand[cand.is_wa].set_index("key").cand
    wp = cand[cand.is_wp].set_index("key").cand
    lem = lemmas(set(cand.cand) | set(wa) | set(wp))
    out = cand.copy()
    out["lemma"] = out.cand.map(lem)
    tgt = wa.map(lem)
    out["target_lemma"] = out.key.map(tgt)
    out["in_M"] = (out.lemma == out.target_lemma) & ~out.is_wp
    same = (wp.map(lem) == tgt)
    out["same_lemma"] = out.key.map(same).fillna(False).astype(bool)
    return out


# ---------------------------------------------------------------- 발음 항
def penalty(d: np.ndarray, spec) -> np.ndarray:
    """spec = ("gamma", γ) → γ·d,  ("bins", (w1, w2, w3+)) → w_{b(d)} (w0 = 0)."""
    kind, val = spec
    if kind == "gamma":
        return val * d
    w = np.r_[0.0, np.asarray(val, float)]
    return w[np.clip(np.floor(d), 0, 3).astype(int)]


# ---------------------------------------------------------------- 듣는 사람
def _seg(keys):
    starts = np.r_[0, np.nonzero(keys[1:] != keys[:-1])[0] + 1]
    gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(keys)]))
    return starts, gid


def _lse(x, starts, gid):
    m = np.maximum.reduceat(x, starts)
    safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return np.log(np.add.reduceat(np.exp(x - safe[gid]), starts)) + safe


def _entropy2(lp_a, lp_b):
    a = np.exp(lp_a - np.logaddexp(lp_a, lp_b))
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(a * np.log(a) + (1 - a) * np.log1p(-a))
    return np.where((a <= 0) | (a >= 1), 0.0, h)


class Listener:
    """한 문맥(ctx)의 후보표 + LM 점수를 정렬해 두고, 발음 항 spec만 바꿔 지표를 다시 계산한다.

    df 열: key, cand, logprob, d_unit, is_wp, is_wa, in_C, lemma, in_M (add_lemma_cols)
    restrict: "C"(원래 후보만, E1) / "C+wa"(w_a 강제 포함, E2·E3) / "pair"({w_p} ∪ M(w_a))
    """

    def __init__(self, df: pd.DataFrame, restrict: str = "C+wa", dist: str = "d_unit"):
        if restrict == "C":
            df = df[df.in_C]
        elif restrict == "pair":
            df = df[df.is_wp | df.in_M | df.is_wa]
        self.df = df.sort_values(["key"], kind="stable").reset_index(drop=True)
        self.keys = self.df.key.to_numpy()
        self.starts, self.gid = _seg(self.keys)
        self.lp = self.df.logprob.to_numpy(float)
        self.d = self.df[dist].to_numpy(float)
        self.is_wp = self.df.is_wp.to_numpy(bool)
        self.in_M = self.df.in_M.to_numpy(bool)
        self.is_wa = self.df.is_wa.to_numpy(bool)
        self.index = pd.Index(self.keys[self.starts], name="key")

    def logpi(self, spec, use_lm=True):
        logq = (self.lp if use_lm else 0.0) - penalty(self.d, spec)
        return logq - _lse(logq, self.starts, self.gid)[self.gid]

    def loglik_lex(self, spec) -> np.ndarray:
        """항목별 log π_lex(w_a) (M이 비면 NaN). 적합 목표 Σ log π_→(w_a 묶음)용."""
        lp = self.logpi(spec)
        out = _lse(np.where(self.in_M, lp, -np.inf), self.starts, self.gid)
        return np.where(np.isfinite(out), out, np.nan)

    def metrics(self, spec, mode: str = "sum", use_lm=True) -> pd.DataFrame:
        """mode: "sum"(표제어 합, 주), "max"(표제어 최댓값), "strict"(alter_word 그대로)."""
        lp = self.logpi(spec, use_lm)
        st, gid = self.starts, self.gid
        n = len(st)
        lp_wp = np.full(n, np.nan)
        lp_wp[gid[self.is_wp]] = lp[self.is_wp]
        logD = _lse(np.where(self.is_wp, -np.inf, lp), st, gid)
        if mode == "sum":
            lp_wa = _lse(np.where(self.in_M, lp, -np.inf), st, gid)
        elif mode == "max":
            lp_wa = np.maximum.reduceat(np.where(self.in_M, lp, -np.inf), st)
        else:
            lp_wa = np.maximum.reduceat(np.where(self.is_wa, lp, -np.inf), st)
        lp_wa = np.where(np.isfinite(lp_wa), lp_wa, np.nan)
        masked = np.where(self.is_wp, -np.inf, lp)
        mx = np.maximum.reduceat(masked, st)
        out = pd.DataFrame(dict(
            logpi_wp=lp_wp, S=-lp_wp, logpi_wa=lp_wa, logD=logD, logF=lp_wa - logD,
            A=_entropy2(lp_wp, lp_wa), L=lp_wa - lp_wp,
            logFstar=mx - logD, L_star=mx - lp_wp, A_star=_entropy2(lp_wp, mx),
        ), index=self.index)
        out["D"] = np.exp(logD)
        out["rank"] = self._rank(lp, mode)
        return out

    def _rank(self, lp, mode):
        """E1 순위. sum: 표제어 묶음(확률 합) 순위, max/strict: 후보 하나하나의 순위."""
        df = pd.DataFrame(dict(key=self.keys, lemma=self.df.lemma.to_numpy(), lp=lp,
                               wp=self.is_wp, M=self.in_M, wa=self.is_wa))
        df = df[~df.wp]
        if mode == "sum":
            g = df.groupby(["key", "lemma"], sort=False).agg(lp=("lp", lambda x: np.logaddexp.reduce(x.to_numpy())),
                                                              M=("M", "any")).reset_index()
        else:
            g = df.assign(M=df.M if mode == "max" else df.wa)
        tgt = g[g.M].groupby("key").lp.max()
        g["tgt"] = g.key.map(tgt)
        rank = (g.lp > g.tgt).groupby(g.key).sum() + 1.0
        rank = rank.where(g.groupby("key").tgt.first().notna())
        return rank.reindex(self.index).to_numpy(float)


def e1_scores(rank: np.ndarray) -> dict:
    r = np.asarray(rank, float)
    hit = lambda k: np.where(np.isnan(r), 0.0, (r <= k).astype(float))
    return dict(acc1=float(hit(1).mean()), acc5=float(hit(5).mean()),
                mrr=float(np.where(np.isnan(r), 0.0, 1.0 / np.where(np.isnan(r), 1.0, r)).mean()))


# ---------------------------------------------------------------- 적합
GAMMA_GRID = np.arange(0, 5.0001, 0.25)


def fit_gamma(lst: Listener, keys=None) -> float:
    """dev에서 Σ log π_lex(w_a)가 최대인 γ (0~5, 0.25 간격). M이 빈 항목은 뺀다."""
    best, best_ll = 0.0, -np.inf
    sel = None if keys is None else lst.index.isin(list(keys))
    for g in GAMMA_GRID:
        ll = lst.loglik_lex(("gamma", g))
        ll = ll if sel is None else ll[sel]
        s = np.nansum(ll)
        if s > best_ll:
            best, best_ll = float(g), s
    return best


def fit_bins(lst: Listener, keys=None, monotone=False, x0=(1.0, 2.0, 4.0)):
    """구간 가중치 (w1, w2, w3+) ∈ [0, 30]. 자유형은 L-BFGS-B, 단조형(w1≤w2≤w3+)은 SLSQP."""
    from scipy.optimize import minimize
    sel = None if keys is None else lst.index.isin(list(keys))

    def nll(w):
        ll = lst.loglik_lex(("bins", w))
        return -np.nansum(ll if sel is None else ll[sel])
    bounds = [(0.0, 30.0)] * 3
    if monotone:
        cons = [dict(type="ineq", fun=lambda w: w[1] - w[0]), dict(type="ineq", fun=lambda w: w[2] - w[1])]
        r = minimize(nll, np.array(x0), method="SLSQP", bounds=bounds, constraints=cons)
    else:
        r = minimize(nll, np.array(x0), method="L-BFGS-B", bounds=bounds)
    return tuple(float(v) for v in r.x), float(-r.fun)


# ---------------------------------------------------------------- 부트스트랩
def boot(stat, n: int, B: int = B_BOOT, seed: int = SEED):
    """항목 복원 추출 부트스트랩. stat(idx) → 실수 배열. (점추정, 하한, 상한)"""
    rng = np.random.default_rng(seed)
    point = np.atleast_1d(np.asarray(stat(np.arange(n)), float))
    reps = np.array([np.atleast_1d(stat(rng.integers(0, n, n))) for _ in range(B)], float)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5], axis=0)
    return point, lo, hi


# ---------------------------------------------------------------- 실행 기록
def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    # dirty: 추적 중인 파일이 커밋과 다를 때만 (새 결과 파일 같은 추적 안 된 파일은 세지 않음)
    return dict(commit=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain", "--untracked-files=no")))


class RunDir:
    """results/<model>/<실험>/ 에 config.json, metrics.json, 표(md)를 남긴다."""

    def __init__(self, model: str, exp: str, config: dict):
        self.dir = RESULTS / model / exp
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(config, model=model, experiment=exp, seed=SEED, bootstrap_B=B_BOOT,
                   git=git_state(), argv=sys.argv, lemma_tool=LEMMA_TOOL)
        dump(self.dir / "config.json", cfg)

    def metrics(self, obj):
        dump(self.dir / "metrics.json", obj)

    def table(self, name: str, lines):
        (self.dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))


def dump(path: Path, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv), encoding="utf-8")


def ci(p, lo, hi, nd=3):
    return f"{p:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
