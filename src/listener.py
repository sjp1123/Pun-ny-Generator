"""M5 듣는 사람 모델과 지표.

    log q_κ(T) = ℓ_κ(T) − γ·d(φ_T, φ_p),   π_κ = softmax_{T∈C} log q_κ

모든 계산은 로그 공간에서 한다. log D는 w_p를 뺀 후보의 log-sum-exp로 직접 구해
π(w_p)가 1에 가까워도 0으로 뭉개지지 않게 한다.
"""
import numpy as np
import pandas as pd

from candidates import INDEL  # noqa: F401  (거리 단위 설명용)

LN10 = np.log(10.0)


def load_scores(cand: pd.DataFrame, lm: pd.DataFrame, ctx: str) -> pd.DataFrame:
    """후보표와 LM 캐시를 한 문맥(ctx)에 대해 합친다. 후보표의 순서를 유지한다."""
    s = lm[lm.ctx == ctx][["key", "cand", "logprob", "ntok"]]
    out = cand.merge(s, on=["key", "cand"], how="inner", sort=False)
    return out


def _segments(keys: np.ndarray):
    starts = np.r_[0, np.nonzero(keys[1:] != keys[:-1])[0] + 1]
    gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(keys)]))
    return starts, gid


def _lse(x: np.ndarray, starts, gid):
    """구간별 log-sum-exp. -inf만 있는 구간은 -inf."""
    m = np.maximum.reduceat(x, starts)
    safe = np.where(np.isfinite(m), m, 0.0)
    s = np.add.reduceat(np.exp(x - safe[gid]), starts)
    with np.errstate(divide="ignore"):
        return np.log(s) + safe


def add_lemma_flags(cand: pd.DataFrame) -> pd.DataFrame:
    """wa_lemma: 후보의 표제어가 w_a의 표제어와 같은지. SemEval alter_word는 표제어인 경우가 많다
    (arm ↔ arms, sell ↔ sold). spaCy 결과는 cache/lemmas.parquet에 둔다."""
    from data import CACHE
    path = CACHE / "lemmas.parquet"
    lem = pd.read_parquet(path).set_index("word").lemma.to_dict() if path.exists() else {}
    wa = cand[cand.is_wa].set_index("key").cand
    words = set(cand.cand) | set(wa)
    todo = sorted(w for w in words if w not in lem)
    if todo:
        from baselines import _nlp
        for w, doc in zip(todo, _nlp().pipe(todo, batch_size=2000)):
            lem[w] = doc[0].lemma_.lower() if len(doc) else w
        pd.DataFrame(dict(word=list(lem), lemma=list(lem.values()))).to_parquet(path, index=False)
    target = cand.key.map(wa.map(lem))
    out = cand.copy()
    out["wa_lemma"] = (out.cand.map(lem) == target) & ~out.is_wp
    return out


def listener(df: pd.DataFrame, gamma: float, restrict: str = "C+wa", dist: str = "d",
             beta: float = 0.0, use_lm: bool = True, wa_mode: str = "strict") -> pd.DataFrame:
    """항목별 지표표.

    restrict: "C"(원래 후보만, E1), "C+wa"(w_a 덧붙임, E2·E3), "pair"({w_p, w_a}만)
    dist: 발음 항에 쓰는 거리 열 ("d" 자질 거리, "d_unit" 단순 편집 거리)
    beta: 빈도 편향 보정(PMI 형태) — log q에서 β·log p_unigram(T)를 뺀다
    use_lm: False면 LM 항 없이 발음 항만 쓴다
    wa_mode: "strict"(SemEval alter_word 그대로) 또는 "lemma"(표제어가 같은 후보 중 log q 최대를 w_a로;
             df에 add_lemma_flags의 wa_lemma 열이 있어야 한다)
    """
    if restrict == "C":
        df = df[df.in_C]
    elif restrict == "pair":
        keep = df.is_wp | df.is_wa | (df.wa_lemma if wa_mode == "lemma" else False)
        df = df[keep]
    df = df.sort_values("key", kind="stable")
    keys = df.key.to_numpy()
    starts, gid = _segments(keys)
    logq = (df.logprob.to_numpy() if use_lm else np.zeros(len(df))) - gamma * df[dist].to_numpy()
    if beta:
        logq = logq - beta * (df.zipf.to_numpy() - 9.0) * LN10
    is_wp = df.is_wp.to_numpy()
    is_wa = df.is_wa.to_numpy().astype(bool)
    if wa_mode == "lemma":
        cand_wa = df.wa_lemma.to_numpy().astype(bool) | is_wa
        best = np.maximum.reduceat(np.where(cand_wa, logq, -np.inf), starts)
        hit = cand_wa & (logq == best[gid]) & np.isfinite(best[gid])
        first = np.full(len(starts), -1)
        idx = np.nonzero(hit)[0]
        first[gid[idx[::-1]]] = idx[::-1]
        is_wa = np.zeros(len(df), bool)
        is_wa[first[first >= 0]] = True
    if restrict == "pair" and wa_mode == "lemma":
        keep = is_wp | is_wa
        df, logq, is_wp, is_wa = df[keep], logq[keep], is_wp[keep], is_wa[keep]
        keys = df.key.to_numpy()
        starts, gid = _segments(keys)
    logZ = _lse(logq, starts, gid)
    logpi = logq - logZ[gid]
    logD = _lse(np.where(is_wp, -np.inf, logq), starts, gid) - logZ

    n = len(starts)
    lp_wp = np.full(n, np.nan)
    lp_wp[gid[is_wp]] = logpi[is_wp]
    lp_wa = np.full(n, np.nan)
    lp_wa[gid[is_wa]] = logpi[is_wa]

    # w_p를 뺀 후보 중 π 최대 (F*, T̂)
    masked = np.where(is_wp, -np.inf, logpi)
    mx = np.maximum.reduceat(masked, starts)
    is_max = (masked == mx[gid]) & ~is_wp
    first_max = np.full(n, -1)
    idx = np.nonzero(is_max)[0]
    first_max[gid[idx[::-1]]] = idx[::-1]  # 같으면 앞(후보표 순서상 거리 가까운 쪽)
    cands = df.cand.to_numpy()
    t_hat = np.where(first_max >= 0, cands[np.maximum(first_max, 0)], None)

    # w_a 순위: w_p를 뺀 후보 중 π가 w_a보다 큰 후보 수 + 1
    above = np.where(~is_wp & (logpi > lp_wa[gid]), 1, 0)
    rank = np.add.reduceat(above, starts) + 1.0
    rank[np.isnan(lp_wa)] = np.nan

    # A: {w_p, w_a}로 재정규화한 π의 엔트로피 (최대 ln 2)
    pair_lse = np.logaddexp(lp_wp, lp_wa)
    a = np.exp(lp_wp - pair_lse)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = -(a * np.log(a) + (1 - a) * np.log1p(-a))
    A = np.where((a == 0) | (a == 1), 0.0, A)

    # A*, L*: 정답 없이 w_a 자리에 T̂를 넣은 버전 (E2b)
    pair_star = np.logaddexp(lp_wp, mx)
    a_s = np.exp(lp_wp - pair_star)
    with np.errstate(divide="ignore", invalid="ignore"):
        A_star = -(a_s * np.log(a_s) + (1 - a_s) * np.log1p(-a_s))
    A_star = np.where((a_s == 0) | (a_s == 1), 0.0, A_star)

    out = pd.DataFrame(dict(
        key=keys[starts], n_cand=np.diff(np.r_[starts, len(keys)]),
        logpi_wp=lp_wp, logpi_wa=lp_wa, logD=logD, logF=lp_wa - logD,
        logFstar=mx - logD, t_hat=t_hat, A=A, L=lp_wa - lp_wp, rank=rank,
        A_star=A_star, L_star=mx - lp_wp,
    ))
    out["D"] = np.exp(out.logD)
    return out.set_index("key")


def metrics_both(cands, lm, gamma, **kw) -> pd.DataFrame:
    """국소·전역 지표를 한 표로 (열 이름 뒤에 _local/_global)."""
    parts = []
    for ctx in ("local", "global"):
        m = listener(load_scores(cands, lm, ctx), gamma, **kw)
        parts.append(m.add_suffix(f"_{ctx}"))
    return parts[0].join(parts[1], how="outer")


def fit_gamma(df_global: pd.DataFrame, keys, grid=np.arange(0, 5.0001, 0.25), restrict="C",
              dist="d", wa_mode="strict") -> float:
    """학습 항목에서 Σ log π_global(w_a)가 최대인 γ. w_a가 후보에 없는 항목은 뺀다."""
    sub = df_global[df_global.key.isin(set(keys))]
    best, best_ll = 0.0, -np.inf
    for g in grid:
        m = listener(sub, g, restrict=restrict, dist=dist, wa_mode=wa_mode)
        ll = m.logpi_wa.dropna().sum()
        if ll > best_ll:
            best, best_ll = float(g), ll
    return best
