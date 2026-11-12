"""E1 숨은 단어 찾기 (Q1, H1a).

SemEval 이형 말장난 중 숨은 단어가 한 단어인 1,074개, 5겹 교차검증.
γ는 학습 겹에서 Σ log π_global(w_a)가 최대인 값(0~5, 0.25 간격). w_a가 후보에 없는 항목은 γ 학습에서만 뺀다.
조건: LM만(γ=0), 발음만(거리 같으면 빈도 순), 전체 모델.
"""
from common import *  # noqa: F403
from sklearn.model_selection import KFold

from data import semeval_items
from listener import add_lemma_flags, fit_gamma, listener, load_scores

N_FOLDS = 5


def phon_only_rank(df: pd.DataFrame, dist="d", wa_mode="strict") -> pd.Series:
    """발음만: w_p를 뺀 원래 후보를 (거리, −빈도) 순으로 줄 세운 w_a 순위."""
    out = {}
    for k, g in df[df.in_C & ~df.is_wp].groupby("key", sort=False):
        g = g.sort_values([dist, "zipf"], ascending=[True, False], kind="stable")
        hit = g.is_wa.to_numpy() | (g.wa_lemma.to_numpy() if wa_mode == "lemma" else False)
        pos = np.nonzero(hit)[0]
        out[k] = pos[0] + 1.0 if len(pos) else np.nan
    return pd.Series(out)


def rank_metrics(rank: pd.Series) -> dict:
    r = rank.to_numpy(float)
    return dict(acc1=float(np.nanmean(np.where(np.isnan(r), 0, r <= 1))),
                acc5=float(np.nanmean(np.where(np.isnan(r), 0, r <= 5))),
                mrr=float(np.mean(np.where(np.isnan(r), 0, 1 / np.where(np.isnan(r), 1, r)))))


def run_e1(model, K, dist="d", beta=0.0, label="full", wa_mode="strict"):
    items = {it.id: it for it in semeval_items(require_single_word=True)}
    cand, lm = load_cache(model, "semeval", K)
    cand = cand[cand.key.isin(items)]
    if wa_mode == "lemma":
        cand = add_lemma_flags(cand)
    g_df, l_df = load_scores(cand, lm, "global"), load_scores(cand, lm, "local")
    keys = np.array(sorted(set(cand.key), key=lambda k: int(k.split("_")[1])))
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    ranks = {c: pd.Series(dtype=float) for c in ("lm_global", "lm_local", "full_global", "full_local")}
    t_hat, gammas = {}, []
    for tr, te in kf.split(keys):
        gam = fit_gamma(g_df, keys[tr], restrict="C", dist=dist, wa_mode=wa_mode)
        gammas.append(gam)
        test = set(keys[te])
        for ctx, df in (("global", g_df), ("local", l_df)):
            sub = df[df.key.isin(test)]
            m0 = listener(sub, 0.0, restrict="C", dist=dist, beta=beta, wa_mode=wa_mode)
            m1 = listener(sub, gam, restrict="C", dist=dist, beta=beta, wa_mode=wa_mode)
            ranks[f"lm_{ctx}"] = pd.concat([ranks[f"lm_{ctx}"], m0["rank"]])
            ranks[f"full_{ctx}"] = pd.concat([ranks[f"full_{ctx}"], m1["rank"]])
            if ctx == "global":
                t_hat.update(m1.t_hat.to_dict())
    ranks["phon"] = phon_only_rank(cand, dist, wa_mode)
    R = pd.DataFrame(ranks).reindex(keys)
    wa = cand[cand.is_wa].set_index("key")
    R["wa_in_C"] = wa.in_C.reindex(keys).fillna(False).to_numpy()
    R["exact"] = (wa.d_unit.reindex(keys) == 0).to_numpy()
    R["t_hat"] = pd.Series(t_hat).reindex(keys)
    R["w_p"] = [items[k].w_p for k in keys]
    R["w_a"] = [items[k].w_a for k in keys]
    R["wa_zipf"] = wa.zipf.reindex(keys).to_numpy()
    return R, gammas


def summarize(R: pd.DataFrame, conds=("lm_global", "phon", "full_global", "lm_local", "full_local")):
    res = {}
    for name, sub in (("all", R), ("exact", R[R.exact]), ("near", R[~R.exact])):
        res[name] = dict(n=len(sub))
        for c in conds:
            res[name][c] = rank_metrics(sub[c])
    return res


def paired_diff(R, a, b, metric="acc1"):
    ra, rb = R[a].to_numpy(float), R[b].to_numpy(float)
    hit = lambda r: np.where(np.isnan(r), 0, r <= 1) if metric == "acc1" else None
    ha, hb = hit(ra), hit(rb)
    return boot_ci(lambda i: ha[i].mean() - hb[i].mean(), len(R))


def error_table(R, n=50):
    wrong = R[R.full_global != 1].copy()
    wrong = wrong.sample(min(n, len(wrong)), random_state=SEED)
    from wordfreq import zipf_frequency

    def cat(r):
        if not r.wa_in_C:
            return "후보 누락"
        if r.w_a in r.w_p or r.w_p in r.w_a or (len(r.w_a) >= 3 and r.w_a[:3] not in r.w_p and r.w_a in r.w_p):
            return "부분 단어"
        if isinstance(r.t_hat, str) and zipf_frequency(r.t_hat, "en") > r.wa_zipf + 0.5:
            return "빈도 편향(자동 추정)"
        return "기타(손 분류 필요: 데이터 오류 등)"
    wrong["category_auto"] = wrong.apply(cat, axis=1)
    wrong["category_manual"] = ""
    return wrong


def main():
    ap = parser("E1 숨은 단어 찾기")
    ap.add_argument("--dist", default="d")
    ap.add_argument("--wa-mode", default="strict", choices=["strict", "lemma"])
    args = ap.parse_args()
    R, gammas = run_e1(args.model, args.k, args.dist, wa_mode=args.wa_mode)
    od = out_dir(args.model)
    sfx = "" if args.wa_mode == "strict" else f"_{args.wa_mode}"
    res = dict(model=args.model, K=args.k, dist=args.dist, wa_mode=args.wa_mode, n=len(R), gammas=gammas,
               candidate_recall=float(R.wa_in_C.mean()),
               candidate_recall_exact=float(R[R.exact].wa_in_C.mean()),
               candidate_recall_near=float(R[~R.exact].wa_in_C.mean()),
               metrics=summarize(R))
    # H1a: 전체 − LM만, 전체 − 발음만 (acc@1, 전역 문맥), 부트스트랩 95% CI
    h = {}
    for other in ("lm_global", "phon"):
        p, lo, hi = paired_diff(R, "full_global", other)
        h[f"full_minus_{other}"] = dict(diff=float(p), ci=[float(lo), float(hi)])
    h["supported"] = all(v["ci"][0] > 0 for v in h.values())
    res["H1a"] = h
    seen = seen_flags()
    if seen is not None:
        s = seen.reindex(R.index).fillna(False)
        res["by_E0"] = {"seen": summarize(R[s]), "unseen": summarize(R[~s])}
    R.to_csv(od / f"e1_items{sfx}.csv")
    error_table(R).to_csv(od / f"e1_errors{sfx}.csv")
    err = pd.read_csv(od / f"e1_errors{sfx}.csv").category_auto.value_counts().to_dict()
    res["error_categories_auto"] = err
    save_json(od / f"e1{sfx}.json", res)

    lines = [f"# E1 숨은 단어 찾기 — {args.model} (w_a: {args.wa_mode})", "",
             f"n = {len(R)}, γ(겹별) = {gammas}, 후보 재현율 = {res['candidate_recall']:.3f}", "",
             "| 부분집합 | 조건 | acc@1 | acc@5 | MRR |", "|---|---|---|---|---|"]
    for sub, d in res["metrics"].items():
        for c in ("lm_global", "phon", "full_global", "lm_local", "full_local"):
            m = d[c]
            lines.append(f"| {sub} (n={d['n']}) | {c} | {m['acc1']:.3f} | {m['acc5']:.3f} | {m['mrr']:.3f} |")
    lines += ["", "H1a (acc@1 차이, 95% 부트스트랩 CI):"]
    for k, v in h.items():
        if k != "supported":
            lines.append(f"- {k}: {fmt_ci(v['diff'], *v['ci'])}")
    lines.append(f"- 판정: {'지지' if h['supported'] else '지지 안 됨'}"
                 + ("" if args.model == MAIN_MODEL else f" (주 모델 {MAIN_MODEL} 아님 — 개발용 수치)"))
    lines += ["", f"오류 50개 자동 분류: {err}"]
    (od / f"e1{sfx}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
