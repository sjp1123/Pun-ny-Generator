"""사전 등록(PREREG.md, 태그 prereg-v1)의 확인 검정. 모수는 dev에서 맞추고 test에 고정값으로 적용한다.

확인 검정은 주 모델 pythia-1.4b로만 판정한다. 다른 모델 결과는 "참고용"으로 표시한다.
모든 비교는 같은 부트스트랩 표본(B = 5,000, 시드 20261112)을 쓰는 짝지은 방식이다.

H1a  E1 acc@1(묶음): 전체(단일 γ) − LM만(γ=0), 전체 − 발음만. 두 CI 하한 모두 > 0
H1b  ρ(M1_full) − ρ(M0) CI 하한 > 0                      (E3, test 안 10겹 CV)
H1c  δ = ρ(M1_full) − ρ(M1_pair0) CI 하한 > 0
H3a  ρ(M0 + H3 특징) − ρ(M0) CI 하한 > 0
H3b  ρ(M0 + H3 특징) − ρ(M1_full) CI 하한 > −0.02         (비열등)
H4   C_union에서 acc@1(단일 γ) − acc@1(γ=0) CI 하한 > 0
H4a  (탐색) C_phon에서 acc@1(구간 가중치, 단조형) − acc@1(단일 γ)
"""
import argparse
import json

from v1_features import *  # noqa: F403
from v1_exp4_phon import phon_only_rank
from rules import B_BOOT, RESULTS, RunDir, boot, ci, fit_bins, fit_gamma, split_sha

MAIN_MODEL = "pythia-1.4b"
NONINF_MARGIN = -0.02


def hits(rank) -> np.ndarray:
    r = np.asarray(rank, float)
    return np.where(np.isnan(r), 0.0, (r <= 1).astype(float))


def paired_acc(named: dict, comps: dict):
    names = list(named)
    H = np.column_stack([named[k] for k in names])

    def stat(i):
        m = H[i].mean(0)
        d = dict(zip(names, m))
        return list(m) + [d[a] - d[b] for a, b in comps.values()]
    p, lo, hi = boot(stat, len(H))
    out = {k: dict(acc1=p[j], ci=[lo[j], hi[j]]) for j, k in enumerate(names)}
    for j, c in enumerate(comps, len(names)):
        out[c] = dict(diff=p[j], ci=[lo[j], hi[j]])
    return out


def rho_max():
    f = RESULTS / "no-lm" / "exp2_noise" / "metrics.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))["bt"]["평가자"]["말장난 test"]["rho_max"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MAIN_MODEL)
    a = ap.parse_args()
    status = "확인 검정" if a.model == MAIN_MODEL else "참고용(주 모델 아님)"
    left = ("semeval_K100_left",)

    # ---- dev에서 모수 고정
    P = fit_dev_params(a.model, extra_lm=left)
    Du = Data(a.model, "dev", cand_set="C_union")
    P["gamma_union"] = fit_gamma(Du.listener("global", "C"))
    Dp = Data(a.model, "dev")
    P["bins_phon_mono"] = fit_bins(Dp.listener("global", "C"), monotone=True)[0]

    # ---- test
    T = Data(a.model, "test", extra_lm=left)
    res = dict(status=status, params=P, n_test=len(T.keys), excluded_same_lemma=T.excluded,
               split_sha256=dict(dev=split_sha("dev"), test=split_sha("test")))
    g = T.listener("global", "C")
    res["H1a"] = paired_acc(dict(full=hits(g.metrics(("gamma", P["gamma"]))["rank"]),
                                 lm=hits(g.metrics(("gamma", 0.0))["rank"]),
                                 phon=hits(phon_only_rank(g))),
                            dict(full_minus_lm=("full", "lm"), full_minus_phon=("full", "phon")))
    res["H1a"]["supported"] = bool(res["H1a"]["full_minus_lm"]["ci"][0] > 0 and res["H1a"]["full_minus_phon"]["ci"][0] > 0)

    tab = e3_table(T, P, reading=True)
    y = T.y()
    preds = dict(M0=cv_predict(tab["full"], y, BASE), M1_full=cv_predict(tab["full"], y, BASE + H1_FEATS),
                 M1_pair0=cv_predict(tab["pair0"], y, BASE + H1_FEATS), M1_read=cv_predict(tab["full"], y, BASE + H3_FEATS))
    E3 = paired_rho_tests(y, preds, dict(H1b=("M1_full", "M0"), H1c=("M1_full", "M1_pair0"),
                                         H3a=("M1_read", "M0"), H3b=("M1_read", "M1_full")))
    res["E3"] = E3
    res["H1b_supported"] = bool(E3["H1b"]["ci"][0] > 0)
    res["H1c_supported"] = bool(E3["H1c"]["ci"][0] > 0)
    res["H3a_supported"] = bool(E3["H3a"]["ci"][0] > 0)
    res["H3b_supported"] = bool(E3["H3b"]["ci"][0] > NONINF_MARGIN)
    rm = rho_max()
    res["rho_max_test"] = rm

    U = Data(a.model, "test", cand_set="C_union")
    gu = U.listener("global", "C")
    res["H4"] = paired_acc(dict(gamma=hits(gu.metrics(("gamma", P["gamma_union"]))["rank"]),
                                gamma0=hits(gu.metrics(("gamma", 0.0))["rank"])),
                           dict(diff=("gamma", "gamma0")))
    res["H4"]["supported"] = bool(res["H4"]["diff"]["ci"][0] > 0)
    res["H4a"] = paired_acc(dict(bins=hits(g.metrics(("bins", P["bins_phon_mono"]))["rank"]),
                                 gamma=hits(g.metrics(("gamma", P["gamma"]))["rank"])),
                            dict(diff=("bins", "gamma")))

    # ---- 기술 통계: E0 비교 변수(log π_global(w_p), L_global)
    from scipy.stats import mannwhitneyu
    cf = RESULTS / "e0" / "counts.csv"
    if cf.exists():
        c = pd.read_csv(cf).set_index("key")
        seen = (c.count_pile.reindex(T.keys) > 0).to_numpy()
        e0 = {}
        for col in ("logpi_wp_global", "L_global"):
            v = tab["full"][col].reindex(T.keys).to_numpy(float)
            e0[col] = dict(n_seen=int(seen.sum()), median_seen=float(np.nanmedian(v[seen])),
                           median_unseen=float(np.nanmedian(v[~seen])),
                           p=float(mannwhitneyu(v[seen][~np.isnan(v[seen])], v[~seen][~np.isnan(v[~seen])]).pvalue))
        res["E0_pile"] = e0

    R = RunDir(a.model, "confirm", dict(status=status, prereg_tag="prereg-v1", split="test (숨은 단어 한 단어)",
                                        bootstrap_B=B_BOOT, params=P))
    R.metrics(res)
    yn = lambda b: "지지" if b else "지지 안 됨"
    L = [f"# 사전 등록 확인 검정 — {a.model} ({status})", "",
         f"test n = {len(T.keys)} (표제어 같아 제외 {len(T.excluded)}), γ = {P['gamma']}, γ(C_union) = {P['gamma_union']}, "
         f"Kao r = {P['kao_r']}, 구간 가중치(C_phon, 단조) = {', '.join(f'{x:.2f}' for x in P['bins_phon_mono'])}", "",
         "| 가설 | 수치 (95% CI) | 판정 |", "|---|---|---|",
         f"| H1a | 전체−LM {ci(res['H1a']['full_minus_lm']['diff'], *res['H1a']['full_minus_lm']['ci'])}; "
         f"전체−발음 {ci(res['H1a']['full_minus_phon']['diff'], *res['H1a']['full_minus_phon']['ci'])} | {yn(res['H1a']['supported'])} |",
         f"| H1b | Δρ {ci(E3['H1b']['diff'], *E3['H1b']['ci'])} | {yn(res['H1b_supported'])} |",
         f"| H1c | δ {ci(E3['H1c']['diff'], *E3['H1c']['ci'])} | {yn(res['H1c_supported'])} |",
         f"| H3a | Δρ {ci(E3['H3a']['diff'], *E3['H3a']['ci'])} | {yn(res['H3a_supported'])} |",
         f"| H3b | ρ차 {ci(E3['H3b']['diff'], *E3['H3b']['ci'])} (하한 > {NONINF_MARGIN}) | {yn(res['H3b_supported'])} |",
         f"| H4 | acc@1 차 {ci(res['H4']['diff']['diff'], *res['H4']['diff']['ci'])} | {yn(res['H4']['supported'])} |",
         f"| H4a (탐색) | acc@1 차 {ci(res['H4a']['diff']['diff'], *res['H4a']['diff']['ci'])} | 탐색 |", "",
         "## E3 모델별 ρ", "", "| 모델 | ρ (95% CI) | 상대 성능 ρ/ρ_max |", "|---|---|---|"]
    for k in preds:
        rel = f"{E3[k]['rho'] / rm:.3f}" if rm else "—"
        L.append(f"| {k} | {ci(E3[k]['rho'], *E3[k]['ci'])} | {rel} |")
    L += ["", f"ρ_max (말장난 test, 평가자 반분) = {rm:.3f}" if rm else "ρ_max: 실험 2 결과 없음"]
    if "E0_pile" in res:
        L += ["", "## E0 (기술 통계, Pile 1회 이상 vs 0회)", ""]
        for col, v in res["E0_pile"].items():
            L.append(f"- {col}: 중앙값 {v['median_seen']:.3f} vs {v['median_unseen']:.3f} (본 {v['n_seen']}개), Mann-Whitney p = {v['p']:.3g}")
    R.table("table.md", L)


if __name__ == "__main__":
    main()
