"""E3 특징 (보완 규칙): γ·Kao z·r는 dev에서 맞추고, 지표는 표제어 합 π_lex로 계산한다.

M0  = 기준선 묶음 BASE
M1  = M0 + H1_FEATS (원안: log F_local, log D_local, A_global, log F_global)
M1r = M0 + H3_FEATS (읽기 시점: S_←, log F_←, log F_→, A_→)
pair0 변형: 후보 {w_p} ∪ M(w_a), γ = 0으로 같은 지표 (H1c의 비교 조건)
"""
from v1_common import *  # noqa: F403
from baselines import kao_cosines, kao_features, lm_baselines
from rules import fit_gamma

BASE = ["S_ratio", "unusualness", "ambiguity", "distinctiveness", "phon_dist", "sent_lp"]
H1_FEATS = ["logF_local", "logD_local", "A_global", "logF_global"]
H3_FEATS = ["S_left", "logF_left", "logF_global", "A_global"]
KAO_R_GRID = (0.5, 1.0, 2.0, 4.0)


def base_features(D: Data, mu, sd, r) -> pd.DataFrame:
    items = list(D.items.values())
    X = lm_baselines(items, D.cand, D.lm)
    wa = D.cand[D.cand.is_wa].set_index("key")
    X["phon_dist"] = wa.d_unit_norm
    cos = kao_cosines(items)
    return X.join(kao_features(cos, D.keys, mu, sd, r)).reindex(D.keys)


def a_features(D: Data, gamma: float, variant="full", reading=False) -> pd.DataFrame:
    restrict, g = ("pair", 0.0) if variant == "pair0" else ("C+wa", gamma)
    spec = ("gamma", g)
    mg = D.listener("global", restrict).metrics(spec)
    ml = D.listener("local", restrict).metrics(spec)
    X = pd.DataFrame(dict(logF_local=ml.logF, logD_local=ml.logD, A_global=mg.A, logF_global=mg.logF,
                          L_global=mg.L, logD_global=mg.logD, logpi_wp_global=mg.logpi_wp))
    if reading:
        mf = D.listener("left", restrict).metrics(spec)
        X["S_left"] = mf.S
        X["logF_left"] = mf.logF
        X["logD_left"] = mf.logD
    return X.reindex(D.keys)


def fit_dev_params(model: str, extra_lm=()) -> dict:
    """dev에서 γ(E1 목적, 표제어 합)와 Kao z·r(기준선 회귀의 dev 적합 ρ 최대)를 맞춘다."""
    from baselines import kao_fit_params
    from sklearn.linear_model import Ridge
    D = Data(model, "dev", extra_lm=extra_lm)
    gam = fit_gamma(D.listener("global", "C"))
    cos = kao_cosines(list(D.items.values()))
    mu, sd = kao_fit_params(cos, D.keys)
    y = D.y()
    best = None
    for r in KAO_R_GRID:
        X = base_features(D, mu, sd, r)[BASE].to_numpy(float)
        m, s = np.nanmean(X, 0), np.nanstd(X, 0)
        Z = np.nan_to_num((X - m) / np.where(s == 0, 1, s))
        p = Ridge(alpha=1.0).fit(Z, y.to_numpy()).predict(Z)
        score = rho(p, y)
        if best is None or score > best[0]:
            best = (score, r)
    return dict(gamma=gam, kao_mu=mu, kao_sd=sd, kao_r=best[1], fit_split="dev",
                dev_n=len(D.keys), dev_excluded=D.excluded)


def e3_table(D: Data, P: dict, reading=False) -> dict:
    """특징표: full과 pair0 두 벌."""
    B = base_features(D, P["kao_mu"], P["kao_sd"], P["kao_r"])
    full = B.join(a_features(D, P["gamma"], "full", reading))
    pair0 = B.join(a_features(D, P["gamma"], "pair0", False))
    return dict(full=full, pair0=pair0)


def paired_rho_tests(y: pd.Series, preds: dict, comparisons: dict):
    """같은 부트스트랩 표본으로 여러 ρ와 ρ 차이를 함께 구한다. comparisons = {이름: (a, b)} → ρ(a) − ρ(b)."""
    from rules import boot
    names = list(preds)
    Y = y.to_numpy()
    P = np.column_stack([preds[k].reindex(y.index).to_numpy() for k in names])

    def stat(i):
        rs = [rho(P[i, j], Y[i]) for j in range(len(names))]
        d = dict(zip(names, rs))
        return rs + [d[a] - d[b] for a, b in comparisons.values()]
    p, lo, hi = boot(stat, len(Y))
    out = {k: dict(rho=p[j], ci=[lo[j], hi[j]]) for j, k in enumerate(names)}
    for j, c in enumerate(comparisons, len(names)):
        out[c] = dict(diff=p[j], ci=[lo[j], hi[j]])
    return out
