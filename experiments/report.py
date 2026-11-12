"""결과 정리와 가설별 판정: results/<model>/REPORT.md"""
import json

from common import *  # noqa: F403


def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    ap = parser("결과 정리")
    args = ap.parse_args()
    od = out_dir(args.model)
    e1, e3 = load(od / "e1.json"), load(od / "e3.json")
    e2 = load(od / "e2ab.json") or load(od / "e2a.json")
    e2b = load(od / "e2b.json")
    abl, e0 = load(od / "ablation.json"), load(RESULTS / "e0" / "summary.json")
    e0d = load(od / "e0_D_local.json")
    pre = load(RESULTS / "precheck" / "recall.json")
    main_model = args.model == MAIN_MODEL
    tag = "확인 검정" if main_model else f"개발용(주 모델 {MAIN_MODEL} 아님 — 판정은 참고용)"
    L = [f"# 결과 정리 — {args.model}", "", f"판정 지위: **{tag}**", ""]

    L += ["## 가설별 판정", "", "| 가설 | 핵심 수치 (95% CI) | 판정 |", "|---|---|---|"]
    if e1:
        h = e1["H1a"]
        L.append(f"| H1a 전체 > LM만, 발음만 (E1 acc@1) | 전체−LM {fmt_ci(h['full_minus_lm_global']['diff'], *h['full_minus_lm_global']['ci'])}; "
                 f"전체−발음 {fmt_ci(h['full_minus_phon']['diff'], *h['full_minus_phon']['ci'])} | {'지지' if h['supported'] else '지지 안 됨'} |")
    if e3:
        b, c = e3["H1b"], e3["H1c"]
        L.append(f"| H1b A 지표가 기준선 위에 Δρ (E3) | Δρ {fmt_ci(b['delta']['point'], *b['delta']['ci'])} | {'지지' if b['supported'] else '지지 안 됨'} |")
        L.append(f"| H1c {{w_p,w_a}}·γ=0이면 효과 사라짐 | Δρ {fmt_ci(c['delta']['point'], *c['delta']['ci'])} | "
                 f"{'지지' if c['supported'] else ('판정 불가(H1b 미지지)' if not b['supported'] else '지지 안 됨')} |")
        h2 = e3["H2"]
        e1l, e3l = load(od / "e1_lemma.json"), load(od / "e3_lemma.json")
        if e1l:
            h = e1l["H1a"]
            L.append(f"| (표제어 일치 w_a) H1a | 전체−LM {fmt_ci(h['full_minus_lm_global']['diff'], *h['full_minus_lm_global']['ci'])}; "
                     f"전체−발음 {fmt_ci(h['full_minus_phon']['diff'], *h['full_minus_phon']['ci'])} | {'지지' if h['supported'] else '지지 안 됨'} |")
        if e3l:
            bl, cl = e3l["H1b"], e3l["H1c"]
            L.append(f"| (표제어 일치 w_a) H1b | Δρ {fmt_ci(bl['delta']['point'], *bl['delta']['ci'])} | {'지지' if bl['supported'] else '지지 안 됨'} |")
            L.append(f"| (표제어 일치 w_a) H1c | Δρ {fmt_ci(cl['delta']['point'], *cl['delta']['ci'])} | "
                     f"{'지지' if cl['supported'] else ('판정 불가(H1b 미지지)' if not bl['supported'] else '지지 안 됨')} |")
        L.append(f"| H2 (탐색) 역 U자 | 2차항 {fmt_ci(*h2['quad_coef']['b2'])}, 꼭짓점 {h2['vertex']:.2f} | "
                 f"{'역 U자 패턴' if h2['inverse_U'] else '역 U자 아님'} (탐색) |")
    L.append("")

    if pre:
        L += ["## 사전 점검 재측정 (후보 재현율@100)", "",
              f"단순 편집 거리 {pre['unit']['recall@100']:.3f}, panphon 자질 거리 {pre['feature']['recall@100']:.3f}, "
              f"자질 거리(삽입·삭제 ×0.25) {pre['feature@0.25']['recall@100']:.3f} (n={pre['n']})", ""]
    if e0:
        L += ["## E0 오염 점검", "", ", ".join(f"{k}: {v['seen']}/{v['n']} ({v['seen_rate']:.1%}) 1회 이상" for k, v in e0.items())]
        if e0d:
            for k, v in e0d.items():
                if k.startswith("count_"):
                    L.append(f"- {k}: log D_local 중앙값 본 집단 {v['median_logD_seen']:.2f} vs 안 본 집단 {v['median_logD_unseen']:.2f}, "
                             f"Mann-Whitney p = {v['mannwhitney_p']:.3g}")
        L.append("")
    if e1:
        m = e1["metrics"]
        L += ["## E1", "", f"후보 재현율 {e1['candidate_recall']:.3f} (완전 동음 {e1['candidate_recall_exact']:.3f}, 유사 동음 {e1['candidate_recall_near']:.3f}), γ = {e1['gammas']}", "",
              "| 부분집합 | LM만 | 발음만 | 전체 |", "|---|---|---|---|"]
        for sub in ("all", "exact", "near"):
            L.append(f"| {sub} (n={m[sub]['n']}) | {m[sub]['lm_global']['acc1']:.3f} | {m[sub]['phon']['acc1']:.3f} | {m[sub]['full_global']['acc1']:.3f} |")
        L += ["", f"오류 50개 자동 분류: {e1['error_categories_auto']}", ""]
    if e2 and e2.get("E2a"):
        a = e2["E2a"]
        L += ["## E2a (He 6유형)", "", "| 유형 | He z | log D_local | log F_local | A_global | S_ratio |", "|---|---|---|---|---|---|"]
        for t, v in a["means"].items():
            L.append(f"| {t} | {v['z']:+.2f} | {v['logD_local'][0]:.2f} | {v['logF_local'][0]:.2f} | {v['A_global'][0]:.3f} | {v['S_ratio'][0]:.2f} |")
        L.append("")
    if e2b and e2b.get("E2b"):
        b = e2b["E2b"]
        L += ["## E2b", ""] + [f"- {k}: " + ", ".join(f"{c} {v[0]:.3f}" for c, v in d.items()) for k, d in b["auc"].items()]
        L += [f"- 위치 맞히기: " + ", ".join(f"{c} {v:.3f}" for c, v in b["location_acc"].items() if c != "n"), ""]
    if e3:
        L += ["## E3 단일 특징 ρ", "", ", ".join(f"{k} {v['rho']:+.3f}" for k, v in e3["single"].items()), ""]
    if abl:
        L += ["## Ablation (E3 Δρ)", ""] + [f"- {k}: {fmt_ci(v['delta']['point'], *v['delta']['ci'])}" for k, v in abl["E3"].items()]
        L.append(f"- {{w_p,w_a}}·γ=0의 L_global vs He S_global ρ = {abl['pair_gamma0_L_vs_S_global']:.4f}")
        L.append("")
    (od / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
