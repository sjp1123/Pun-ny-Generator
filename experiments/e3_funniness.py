"""E3 말장난끼리 웃김 예측 (Q3, H1b·H1c, 핵심).

데이터: UKP 웃김 점수가 붙은 SemEval 이형 말장난 1,074개(숨은 단어 한 단어).
단일 특징: 웃김 점수와의 스피어만 ρ와 부트스트랩 95% CI(2,000회).
핵심 검정: 10겹 교차검증 선형 회귀(표준화 특징, 약한 릿지).
  모델 0 = 기준선 묶음, 모델 1 = 모델 0 + log F_local, log D_local, A_global, log F_global.
  교차검증 예측값과 정답의 ρ 차이 Δρ에 대해 항목 재표집 부트스트랩 95% CI.
H1c: 후보를 {w_p, w_a}로 제한하고 γ = 0으로 둔 A 지표로 바꾸면 Δρ가 사라진다.
γ, Kao의 z·r, 회귀 계수는 모두 학습 분할에서만 맞춘다.
"""
from common import *  # noqa: F403
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from wordfreq import zipf_frequency

from baselines import kao_cosines, kao_features, kao_fit_params, lm_baselines, phon_baselines
from data import load_he_annotation, semeval_items
from listener import add_lemma_flags, fit_gamma, listener, load_scores

N_FOLDS = 10
RIDGE_ALPHA = 1.0
KAO_R_GRID = (0.5, 1.0, 2.0, 4.0)
BASE = ["S_ratio", "unusualness", "ambiguity", "distinctiveness", "phon_dist", "sent_lp"]
A_FEATS = ["logF_local", "logD_local", "A_global", "logF_global"]


def a_features(g_df, l_df, keys, gamma, restrict="C+wa", **kw):
    """A 지표 묶음. kw는 listener로 넘긴다(wa_mode 등)."""
    sub_g, sub_l = g_df[g_df.key.isin(keys)], l_df[l_df.key.isin(keys)]
    mg = listener(sub_g, gamma, restrict=restrict, **kw)
    ml = listener(sub_l, gamma, restrict=restrict, **kw)
    return pd.DataFrame(dict(logF_local=ml.logF, logD_local=ml.logD, A_global=mg.A,
                             logF_global=mg.logF, A_local=ml.A, logD_global=mg.logD,
                             L_global=mg.L, L_local=ml.L))


def _fit_predict(Xtr, ytr, Xte):
    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
    sd[sd == 0] = 1
    f = lambda X: np.nan_to_num((X - mu) / sd)  # 결측은 학습 평균(=0)으로
    m = Ridge(alpha=RIDGE_ALPHA).fit(f(Xtr), ytr)
    return m.predict(f(Xte)), m.predict(f(Xtr))


class E3:
    def __init__(self, model, K, wa_mode="strict"):
        self.items = {it.id: it for it in semeval_items(require_single_word=True)}
        cand, lm = load_cache(model, "semeval", K)
        self.cand = cand[cand.key.isin(self.items)]
        self.wa_mode = wa_mode
        if wa_mode == "lemma":
            self.cand = add_lemma_flags(self.cand)
        self.lm = lm
        self.g = load_scores(self.cand, lm, "global")
        self.l = load_scores(self.cand, lm, "local")
        self.keys = np.array(sorted(set(self.cand.key), key=lambda k: int(k.split("_")[1])))
        self.y = pd.Series({k: self.items[k].meta["score"] for k in self.keys})
        self.base = lm_baselines(list(self.items.values()), self.cand, lm).join(phon_baselines(self.cand))
        self.cos = kao_cosines(list(self.items.values()))

    def features(self, train, test, variant="full"):
        """학습 분할로 γ·Kao 모수를 맞추고 (train+test) 특징표를 만든다."""
        gam = fit_gamma(self.g, train, restrict="C", wa_mode=self.wa_mode) if variant in ("full", "pair_fitgamma") else 0.0
        restrict = "pair" if variant in ("pair", "pair_fitgamma") else "C+wa"
        if variant == "gamma0":
            restrict = "C+wa"
        allk = np.r_[train, test]
        A = a_features(self.g, self.l, set(allk), gam, restrict=restrict, wa_mode=self.wa_mode)
        mu, sd = kao_fit_params(self.cos, train)
        best = None
        for r in KAO_R_GRID:
            kao = kao_features(self.cos, allk, mu, sd, r)
            X = self.base.join(kao)
            Xtr = X.loc[train, BASE].to_numpy(float)
            _, ptr = _fit_predict(Xtr, self.y[train].to_numpy(), Xtr)
            s = rho(ptr, self.y[train])
            if best is None or s > best[0]:
                best = (s, r, kao)
        X = self.base.join(best[2]).join(A)
        return X, gam, best[1]

    def cv(self, variant="full", extra=A_FEATS):
        kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
        p0, p1 = pd.Series(index=self.keys, dtype=float), pd.Series(index=self.keys, dtype=float)
        info = []
        for tr, te in kf.split(self.keys):
            train, test = self.keys[tr], self.keys[te]
            X, gam, r = self.features(train, test, variant)
            ytr = self.y[train].to_numpy()
            p0[test] = _fit_predict(X.loc[train, BASE].to_numpy(float), ytr, X.loc[test, BASE].to_numpy(float))[0]
            cols = BASE + list(extra)
            p1[test] = _fit_predict(X.loc[train, cols].to_numpy(float), ytr, X.loc[test, cols].to_numpy(float))[0]
            info.append(dict(gamma=gam, kao_r=r))
        y = self.y.to_numpy()
        a, b = p0.to_numpy(), p1.to_numpy()
        point, lo, hi = boot_ci(lambda i: [rho(a[i], y[i]), rho(b[i], y[i]), rho(b[i], y[i]) - rho(a[i], y[i])], len(y))
        return dict(variant=variant, extra=list(extra), folds=info,
                    rho0=dict(point=point[0], ci=[lo[0], hi[0]]),
                    rho1=dict(point=point[1], ci=[lo[1], hi[1]]),
                    delta=dict(point=point[2], ci=[lo[2], hi[2]]))


def single_features(e: E3):
    gam = fit_gamma(e.g, e.keys, restrict="C", wa_mode=e.wa_mode)
    X, _, r = e.features(e.keys, np.array([], dtype=object))
    y = e.y.to_numpy()
    out = {}
    cols = BASE + A_FEATS + ["A_local", "logD_global", "L_global", "L_local"]
    for c in cols:
        x = X.loc[e.keys, c].to_numpy(float)
        p, lo, hi = boot_ci(lambda i: rho(x[i], y[i]), len(y))
        out[c] = dict(rho=float(p), ci=[float(lo), float(hi)], n=int((~np.isnan(x)).sum()))
    return out, X, gam


def partial_rho(x, y, Z):
    """순위 변환 뒤 통제변수 Z를 선형으로 빼고 남은 잔차의 피어슨 상관 (편 스피어만)."""
    from scipy.stats import rankdata
    ok = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(Z).any(1)
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    Zr = np.column_stack([np.ones(ok.sum())] + [rankdata(z) for z in Z[ok].T])
    res = lambda v: v - Zr @ np.linalg.lstsq(Zr, v, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def he33(model, K, gamma):
    """He et al.의 pun 33개에서 He z 점수와의 단일 특징 ρ (검정력이 낮아 비교용)."""
    try:
        cand, lm = load_cache(model, "he", K)
    except SystemExit:
        return None
    items = {it.id: it for it in load_he_annotation()[0]}
    keys = [k for k, it in items.items() if it.meta["type"] == "pun" and k in set(cand.key)]
    cand = cand[cand.key.isin(keys)]
    A = a_features(load_scores(cand, lm, "global"), load_scores(cand, lm, "local"), set(keys), gamma)
    B = lm_baselines(list(items.values()), cand, lm).join(phon_baselines(cand))
    X = B.join(A)
    z = pd.Series({k: items[k].meta["z"] for k in keys})
    return {c: rho(X.loc[keys, c], z[keys]) for c in ["S_ratio", "unusualness", "phon_dist", "sent_lp"] + A_FEATS}


def h2(e: E3):
    """발음 거리 구간별 평균 웃김과 2차항 회귀 (탐색)."""
    d = e.base.phon_dist_unit.reindex(e.keys).to_numpy(float)
    y = e.y.to_numpy()
    bins = {}
    for name, m in (("0", d == 0), ("1", d == 1), ("2", d == 2), ("3+", d >= 3)):
        yy = y[m]
        p, lo, hi = boot_ci(lambda i: yy[i].mean(), len(yy)) if len(yy) else (np.nan,) * 3
        bins[name] = dict(n=int(m.sum()), mean=float(p), ci=[float(lo), float(hi)])
    ok = ~np.isnan(d)

    def quad(i):
        dd, yy = d[ok][i], y[ok][i]
        X = np.column_stack([np.ones_like(dd), dd, dd ** 2])
        return np.linalg.lstsq(X, yy, rcond=None)[0]
    p, lo, hi = boot_ci(quad, ok.sum())
    vertex = -p[1] / (2 * p[2]) if p[2] != 0 else np.nan
    return dict(bins=bins, quad_coef=dict(b1=[p[1], lo[1], hi[1]], b2=[p[2], lo[2], hi[2]]), vertex=vertex,
                inverse_U=bool(hi[2] < 0 and 0 < vertex < np.nanmax(d)), note="탐색 분석")


def main():
    ap = parser("E3 웃김 예측")
    ap.add_argument("--wa-mode", default="strict", choices=["strict", "lemma"])
    args = ap.parse_args()
    od = out_dir(args.model)
    sfx = "" if args.wa_mode == "strict" else f"_{args.wa_mode}"
    e = E3(args.model, args.k, args.wa_mode)
    single, X, gam = single_features(e)
    X.loc[e.keys].assign(funniness=e.y).to_csv(od / f"e3_features{sfx}.csv")
    res = dict(model=args.model, wa_mode=args.wa_mode, n=len(e.keys), gamma_all=gam, single=single)
    res["H1b"] = e.cv("full")
    res["H1b"]["supported"] = res["H1b"]["delta"]["ci"][0] > 0
    res["H1c"] = e.cv("pair")  # 후보 {w_p, w_a}, γ = 0
    res["H1c"]["supported"] = res["H1c"]["delta"]["ci"][0] <= 0 <= res["H1c"]["delta"]["ci"][1] and res["H1b"]["supported"]
    # 보조: 문장 길이·단어 빈도 통제 편상관
    nwords = np.array([len(e.items[k].text.split()) for k in e.keys], float)
    zp = np.array([zipf_frequency(e.items[k].w_p, "en") for k in e.keys], float)
    za = np.array([zipf_frequency(e.items[k].w_a, "en") for k in e.keys], float)
    Z = np.column_stack([nwords, zp, za])
    y = e.y.to_numpy()
    res["partial"] = {c: partial_rho(X.loc[e.keys, c].to_numpy(float), y, Z) for c in BASE + A_FEATS}
    res["he33"] = he33(args.model, args.k, gam)
    res["H2"] = h2(e)
    seen = seen_flags()
    if seen is not None:
        s = seen.reindex(e.keys).fillna(False).to_numpy()
        res["by_E0"] = {nm: {c: rho(X.loc[e.keys[m], c], y[m]) for c in BASE + A_FEATS} | {"n": int(m.sum())}
                        for nm, m in (("seen", s), ("unseen", ~s))}
    save_json(od / f"e3{sfx}.json", res)

    L = [f"# E3 웃김 예측 — {args.model} (w_a: {args.wa_mode})", "", f"n = {res['n']}, γ(전체) = {gam}", "",
         "## 단일 특징 (스피어만 ρ, 95% CI)", "", "| 특징 | ρ | n | 편상관(길이·빈도 통제) |", "|---|---|---|---|"]
    for c, v in single.items():
        L.append(f"| {c} | {fmt_ci(v['rho'], *v['ci'])} | {v['n']} | {res['partial'].get(c, float('nan')):.3f} |")
    L += ["", "## 교차검증 회귀 (10겹, 릿지)", "", "| 검정 | ρ 모델0 | ρ 모델1 | Δρ | 판정 |", "|---|---|---|---|---|"]
    for h in ("H1b", "H1c"):
        r = res[h]
        L.append(f"| {h} ({r['variant']}) | {fmt_ci(r['rho0']['point'], *r['rho0']['ci'])} | "
                 f"{fmt_ci(r['rho1']['point'], *r['rho1']['ci'])} | {fmt_ci(r['delta']['point'], *r['delta']['ci'])} | "
                 f"{'지지' if r['supported'] else '지지 안 됨'} |")
    if args.model != MAIN_MODEL:
        L.append(f"\n주의: 확인 검정은 주 모델 {MAIN_MODEL}로만 판정한다. 이 표는 개발용 수치다.")
    L += ["", "## H2 (탐색): 발음 거리 구간별 평균 웃김", ""]
    for b, v in res["H2"]["bins"].items():
        L.append(f"- 거리 {b}: n={v['n']}, 평균 {fmt_ci(v['mean'], *v['ci'])}")
    q = res["H2"]["quad_coef"]
    L.append(f"- 2차항 계수 {fmt_ci(*q['b2'])}, 꼭짓점 {res['H2']['vertex']:.2f}, 역 U자: {res['H2']['inverse_U']}")
    if res["he33"]:
        L += ["", "## He 33개 (He z 점수와 ρ, 비교용)", "", ", ".join(f"{k}={v:.2f}" for k, v in res["he33"].items())]
    (od / f"e3{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
