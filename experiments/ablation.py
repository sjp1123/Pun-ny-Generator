"""Ablation: 이론의 각 부품을 뺐을 때 E1·E3가 어떻게 바뀌는지.

| 설정 | 빼는 것 | 이론의 예측 |
| γ = 0 | 발음 항 | E1 유사 동음 정확도가 떨어진다 |
| 후보 = {w_p, w_a} | 전체 후보 정규화(F) | E3 이득이 줄어든다 |
| 둘 다 | 발음 항과 정규화 | He의 S와 거의 같아진다 (H1c) |
| 국소만 / 전역만 | 한쪽 문맥 | 둘 다 있어야 가장 좋다 |
| 모델 크기·계열 | 없음 | 결과 방향이 유지된다 |
위험 대응용 추가: 빈도 편향 보정(PMI 형태, β), 발음 항을 단순 편집 거리로.
"""
from common import *  # noqa: F403

from e1_recovery import run_e1, summarize
from e3_funniness import E3, a_features
from listener import fit_gamma, listener, load_scores


def main():
    ap = parser("Ablation")
    ap.add_argument("--models", nargs="*", default=None, help="크기·계열 비교에 쓸 모델(캐시 있는 것만)")
    args = ap.parse_args()
    od = out_dir(args.model)
    res = dict(model=args.model, E1={}, E3={})

    # ---- E1: 전체 / γ=0 / PMI 보정 / 단순 편집 거리 발음 항 (전역·국소 모두)
    for label, kw in [("full", {}), ("pmi_0.5", dict(beta=0.5)), ("pmi_1.0", dict(beta=1.0)),
                      ("dist_unit", dict(dist="d_unit")), ("wa_lemma", dict(wa_mode="lemma"))]:
        R, gam = run_e1(args.model, args.k, **kw)
        s = summarize(R)
        res["E1"][label] = dict(gammas=gam, **{sub: {c: s[sub][c]["acc1"] for c in
                                                     ("lm_global", "full_global", "lm_local", "full_local", "phon")}
                                                for sub in ("all", "exact", "near")})

    # ---- E3: Δρ 변형
    e = E3(args.model, args.k)
    variants = {
        "full (H1b)": ("full", ["logF_local", "logD_local", "A_global", "logF_global"]),
        "gamma0": ("gamma0", ["logF_local", "logD_local", "A_global", "logF_global"]),
        "pair, fit γ": ("pair_fitgamma", ["logF_local", "logD_local", "A_global", "logF_global"]),
        "pair, γ=0 (H1c)": ("pair", ["logF_local", "logD_local", "A_global", "logF_global"]),
        "local only": ("full", ["logF_local", "logD_local", "A_local"]),
        "global only": ("full", ["logF_global", "logD_global", "A_global"]),
    }
    for name, (v, extra) in variants.items():
        r = e.cv(v, extra)
        res["E3"][name] = dict(delta=r["delta"], rho1=r["rho1"], rho0=r["rho0"])

    # ---- 표제어 일치 w_a로 H1b·H1c 다시
    el = E3(args.model, args.k, "lemma")
    for name, v in (("wa_lemma: full (H1b)", "full"), ("wa_lemma: pair, γ=0 (H1c)", "pair")):
        r = el.cv(v)
        res["E3"][name] = dict(delta=r["delta"], rho1=r["rho1"], rho0=r["rho0"])

    # ---- "둘 다" 설정에서 L_global이 He의 S_global과 같은지
    A0 = a_features(e.g, e.l, set(e.keys), 0.0, restrict="pair")
    res["pair_gamma0_L_vs_S_global"] = rho(A0.L_global.reindex(e.keys), e.base.S_global.reindex(e.keys))

    # ---- 모델 크기·계열
    if args.models:
        res["models"] = {}
        for m in args.models:
            try:
                R, gam = run_e1(m, args.k)
                em = E3(m, args.k)
                r = em.cv("full")
                res["models"][m] = dict(E1_acc1=summarize(R)["all"]["full_global"]["acc1"],
                                        E1_acc1_near=summarize(R)["near"]["full_global"]["acc1"],
                                        E3_delta=r["delta"], E3_rho1=r["rho1"])
            except SystemExit as ex:
                res["models"][m] = str(ex)
    save_json(od / "ablation.json", res)

    L = [f"# Ablation — {args.model}", "", "## E1 acc@1", "",
         "| 설정 | 부분집합 | LM만(전역) | 전체(전역) | LM만(국소) | 전체(국소) | 발음만 |", "|---|---|---|---|---|---|---|"]
    for lab, d in res["E1"].items():
        for sub in ("all", "exact", "near"):
            x = d[sub]
            L.append(f"| {lab} (γ={d['gammas']}) | {sub} | {x['lm_global']:.3f} | {x['full_global']:.3f} | "
                     f"{x['lm_local']:.3f} | {x['full_local']:.3f} | {x['phon']:.3f} |")
    L += ["", "## E3 Δρ (모델1 − 모델0, 10겹 CV, 95% CI)", "", "| 설정 | ρ 모델1 | Δρ |", "|---|---|---|"]
    for name, d in res["E3"].items():
        L.append(f"| {name} | {fmt_ci(d['rho1']['point'], *d['rho1']['ci'])} | {fmt_ci(d['delta']['point'], *d['delta']['ci'])} |")
    L.append(f"\n{{w_p, w_a}}·γ=0에서 L_global과 He S_global의 스피어만 ρ = {res['pair_gamma0_L_vs_S_global']:.4f}")
    if res.get("models"):
        L += ["", "## 모델 크기·계열", "", "| 모델 | E1 acc@1 | E1 acc@1(유사 동음) | E3 ρ 모델1 | E3 Δρ |", "|---|---|---|---|---|"]
        for m, d in res["models"].items():
            if isinstance(d, str):
                L.append(f"| {m} | 캐시 없음 | | | |")
            else:
                L.append(f"| {m} | {d['E1_acc1']:.3f} | {d['E1_acc1_near']:.3f} | "
                         f"{fmt_ci(d['E3_rho1']['point'], *d['E3_rho1']['ci'])} | {fmt_ci(d['E3_delta']['point'], *d['E3_delta']['ci'])} |")
    (od / "ablation.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
