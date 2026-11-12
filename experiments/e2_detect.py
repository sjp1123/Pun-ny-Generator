"""E2 말장난 판별과 위치 (Q2).

E2a, 위치·쌍을 아는 경우: He 주석 6유형 194문장. 유형별 지표 평균이 예측대로 나오는지 보고,
     He Table 2와 같은 비교(말장난 vs 일반, 말장난 vs depun)에 말장난 vs retrieved_aw_alter를 더해 스피어만 ρ.
E2b, 위치를 모르는 경우: UKP 이형 말장난 / 말장난 아닌 농담 / 농담 아닌 문장. 내용어 위치 j마다
     T̂를 w_a로 두고 점수를 구한 뒤 문장 점수는 최댓값. 말장난 vs 각 음성 집단의 ROC-AUC.
위치 맞히기: 점수가 가장 높은 위치가 실제 pun 위치인 비율(SemEval 정답).
γ는 E1과 같은 목적함수로 SemEval 전체에서 맞춘 값을 쓴다(웃김 점수는 쓰지 않음).
"""
from common import *  # noqa: F403
from sklearn.metrics import roc_auc_score

from baselines import lm_baselines, phon_baselines
from data import load_he_annotation, semeval_items
from listener import fit_gamma, listener, load_scores

TYPES = ["pun", "depun", "retrieved_aw_alter", "retrieved_aw", "retrieved_pw_alter", "retrieved_pw"]
RETRIEVED = [t for t in TYPES if t.startswith("retrieved")]
METRICS = ["logD_local", "logF_local", "A_global", "logF_global", "L_global", "A_local"]


def semeval_gamma(model, K):
    items = {it.id for it in semeval_items(require_single_word=True)}
    cand, lm = load_cache(model, "semeval", K)
    cand = cand[cand.key.isin(items)]
    return fit_gamma(load_scores(cand, lm, "global"), list(items), restrict="C")


def e2a(model, K, gamma):
    items, review = load_he_annotation()
    by = {it.id: it for it in items}
    cand, lm = load_cache(model, "he", K)
    g, l = load_scores(cand, lm, "global"), load_scores(cand, lm, "local")
    mg, ml = listener(g, gamma), listener(l, gamma)
    X = pd.DataFrame(dict(logD_local=ml.logD, logF_local=ml.logF, A_global=mg.A, logF_global=mg.logF,
                          L_global=mg.L, A_local=ml.A, D_local=ml.D))
    X = X.join(lm_baselines(items, cand, lm)[["S_ratio", "S_local", "S_global"]]).join(phon_baselines(cand))
    X["type"] = [by[k].meta["type"] for k in X.index]
    X["z"] = [by[k].meta["z"] for k in X.index]
    means = {}
    for t in TYPES:
        sub = X[X.type == t]
        means[t] = dict(n=len(sub), z=float(sub.z.mean()))
        for c in METRICS + ["S_ratio"]:
            v = sub[c].to_numpy(float)
            p, lo, hi = boot_ci(lambda i: np.nanmean(v[i]), len(v))
            means[t][c] = [float(p), float(lo), float(hi)]
    comps = {"pun_vs_retrieved": ["pun"] + RETRIEVED, "pun_vs_depun": ["pun", "depun"],
             "pun_vs_retrieved_aw_alter": ["pun", "retrieved_aw_alter"]}
    corr = {}
    for name, ts in comps.items():
        sub = X[X.type.isin(ts)]
        corr[name] = {}
        for c in METRICS + ["S_ratio", "phon_dist"]:
            x, z = sub[c].to_numpy(float), sub.z.to_numpy(float)
            p, lo, hi = boot_ci(lambda i: rho(x[i], z[i]), len(x))
            corr[name][c] = [float(p), float(lo), float(hi)]
    X.to_csv(out_dir(model) / "e2a_items.csv")
    review.to_csv(RESULTS / "he_pun_review.csv", index=False)
    return dict(gamma=gamma, means=means, spearman_vs_z=corr,
                he_review=review.verdict.value_counts().to_dict(),
                he_alter_mismatch=review[~review.alter_match][["sent", "pair", "semeval_pair"]].to_dict("records"))


def e2b(model, K, gamma):
    from e2b_positions import e2b_items
    items = e2b_items()
    try:
        cand, lm = load_cache(model, "e2b", K)
    except SystemExit:
        return None
    by = {it.id: it for it in items}
    g, l = load_scores(cand, lm, "global"), load_scores(cand, lm, "local")
    mg, ml = listener(g, gamma), listener(l, gamma)
    P = pd.DataFrame(dict(logD_local=ml.logD, logFstar_local=ml.logFstar, A_star_global=mg.A_star,
                          logFstar_global=mg.logFstar, L_star_global=mg.L_star, logD_global=mg.logD))
    P["combo"] = P.logD_local + P.logFstar_local + np.log(np.clip(P.A_star_global, 1e-12, None))
    P["ukp_id"] = [by[k].meta["ukp_id"] for k in P.index]
    P["category"] = [by[k].meta["category"] for k in P.index]
    P["start"] = [by[k].start for k in P.index]
    P["gold_start"] = [by[k].meta["gold_start"] for k in P.index]
    cols = ["logD_local", "logFstar_local", "A_star_global", "logFstar_global", "L_star_global", "combo"]
    S = P.groupby("ukp_id").agg({**{c: "max" for c in cols}, "category": "first"})
    res = dict(n_sentences=len(S), n_positions=len(P), sentences_by_cat=S.category.value_counts().to_dict(), auc={})
    for neg in ("nonpun_joke", "nonjoke"):
        sub = S[S.category.isin(["hetero_pun", neg])]
        yb = (sub.category == "hetero_pun").to_numpy(int)
        res["auc"][f"pun_vs_{neg}"] = {}
        for c in cols:
            x = sub[c].to_numpy(float)
            ok = ~np.isnan(x)
            p, lo, hi = boot_ci(lambda i: roc_auc_score(yb[ok][i], x[ok][i]) if len(set(yb[ok][i])) > 1 else np.nan, ok.sum())
            res["auc"][f"pun_vs_{neg}"][c] = [float(p), float(lo), float(hi)]
    loc = {}
    G = P[P.gold_start.notna()]
    for c in cols:
        top = G.loc[G.groupby("ukp_id")[c].idxmax().dropna()]
        loc[c] = float((top.start == top.gold_start).mean())
    n_pos = G.groupby("ukp_id").size()
    loc["chance"] = float((1 / n_pos).mean())
    loc["n"] = int(G.ukp_id.nunique())
    res["location_acc"] = loc
    P.to_csv(out_dir(model) / "e2b_positions.csv")
    return res


def main():
    ap = parser("E2 판별과 위치")
    ap.add_argument("--part", nargs="+", default=["a", "b"])
    ap.add_argument("--e2b-k", type=int, default=None, help="E2b 후보 수 (기본 --k와 같음)")
    args = ap.parse_args()
    gamma = semeval_gamma(args.model, args.k)
    res = dict(model=args.model, gamma=gamma)
    if "a" in args.part:
        res["E2a"] = e2a(args.model, args.k, gamma)
    if "b" in args.part:
        res["E2b"] = e2b(args.model, args.e2b_k or args.k, gamma)
        res["E2b_K"] = args.e2b_k or args.k
    od = out_dir(args.model)
    tag = "".join(args.part)
    save_json(od / f"e2{tag}.json", res)
    L = [f"# E2 판별과 위치 — {args.model}", "", f"γ = {gamma} (SemEval E1 목적함수로 맞춤)"]
    if res.get("E2a"):
        a = res["E2a"]
        L += ["", "## E2a 유형별 평균 (95% CI)", "",
              "| 유형 | n | He z | " + " | ".join(METRICS + ["S_ratio"]) + " |",
              "|---|---|---|" + "---|" * (len(METRICS) + 1)]
        for t, m in a["means"].items():
            L.append(f"| {t} | {m['n']} | {m['z']:+.2f} | " + " | ".join(f"{m[c][0]:.2f}" for c in METRICS + ["S_ratio"]) + " |")
        L += ["", "## E2a 스피어만 ρ (지표 vs He z)", "",
              "| 비교 | " + " | ".join(METRICS + ["S_ratio", "phon_dist"]) + " |", "|---|" + "---|" * (len(METRICS) + 2)]
        for name, d in a["spearman_vs_z"].items():
            L.append(f"| {name} | " + " | ".join(f"{d[c][0]:.2f}" for c in METRICS + ["S_ratio", "phon_dist"]) + " |")
    if res.get("E2b"):
        b = res["E2b"]
        L += ["", f"## E2b ROC-AUC (문장 {b['n_sentences']}, 위치 {b['n_positions']})", ""]
        for name, d in b["auc"].items():
            L.append(f"- {name}: " + ", ".join(f"{c}={fmt_ci(*v, nd=3)}" for c, v in d.items()))
        L.append(f"- 위치 맞히기 (n={b['location_acc']['n']}): " +
                 ", ".join(f"{c}={v:.3f}" for c, v in b["location_acc"].items() if c != "n"))
    (od / f"e2{tag}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
