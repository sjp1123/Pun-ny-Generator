"""실험 4: 발음 항 개선(4-A)과 발음 기여 분리(4-B). dev 점검.

4-A: log q(T) = ℓ_→(T) − w_{b(d)}, b(d) ∈ {0, 1, 2, 3+}, w0 = 0, d = 단순 음소 편집 거리.
     자유형(L-BFGS-B)과 단조형(w1 ≤ w2 ≤ w3+, SLSQP) 모두 dev에서 Σ log π_→(w_a 묶음) 최대화.
     단일 γ도 같은 목표(격자)로 다시 맞춘다. dev 안 5겹 교차검증으로 검증 겹 로그 우도 비교.
4-B: 후보 집합 C_phon / C_LM / C_union × 발음 항 γ=0 / 단일 γ / 구간 가중치.
     칸마다 E1 acc@1(묶음) / 후보 재현율 / E3 ρ(log F_→).
"""
import argparse

from sklearn.model_selection import KFold

from v1_common import *  # noqa: F403
from rules import RunDir, e1_scores, fit_bins, fit_gamma

SPECS = ("γ=0", "단일 γ", "구간 가중치")


def phon_only_rank(lst) -> np.ndarray:
    """발음만: (거리, −빈도) 순으로 줄 세운 뒤 표제어 묶음이 처음 나오는 순서로 순위."""
    df = lst.df[~lst.df.is_wp].sort_values(["key", "d_unit", "zipf"], ascending=[True, True, False], kind="stable")
    out = {}
    for k, g in df.groupby("key", sort=False):
        order = list(dict.fromkeys(g.lemma))
        tgt = g.target_lemma.iloc[0]
        out[k] = order.index(tgt) + 1.0 if tgt in set(g[g.in_M].lemma) else np.nan
    return pd.Series(out).reindex(lst.index).to_numpy(float)


def exact_mask(D: Data, index) -> np.ndarray:
    wa = D.cand[D.cand.is_wa].set_index("key").d_unit
    return (wa.reindex(index) == 0).to_numpy()


def fit_all(lst, keys=None):
    gam = fit_gamma(lst, keys)
    free, ll_free = fit_bins(lst, keys, monotone=False)
    mono, ll_mono = fit_bins(lst, keys, monotone=True)
    return {"단일 γ": ("gamma", gam), "구간(자유)": ("bins", free), "구간(단조)": ("bins", mono)}


def heldout(lst, keys):
    """dev 안 5겹 교차검증: 검증 겹의 Σ log π_lex(w_a)."""
    tot = {k: 0.0 for k in ("γ=0", "단일 γ", "구간(자유)", "구간(단조)")}
    for tr, te in KFold(5, shuffle=True, random_state=SEED % (2 ** 32)).split(keys):
        specs = fit_all(lst, keys[tr])
        specs["γ=0"] = ("gamma", 0.0)
        sel = lst.index.isin(list(keys[te]))
        for k, s in specs.items():
            tot[k] += float(np.nansum(lst.loglik_lex(s)[sel]))
    return tot


def part_a(model):
    D = Data(model, "dev")
    g = D.listener("global", "C")
    specs = fit_all(g)
    specs = {"γ=0": ("gamma", 0.0), **specs}
    ll = {k: float(np.nansum(g.loglik_lex(s))) for k, s in specs.items()}
    ho = heldout(g, D.keys)
    ex = exact_mask(D, g.index)
    acc = {}
    for k, s in specs.items():
        r = g.metrics(s)["rank"]
        acc[k] = dict(all=e1_scores(r)["acc1"], exact=e1_scores(r[ex])["acc1"], near=e1_scores(r[~ex])["acc1"])
    r = phon_only_rank(g)
    acc["발음만"] = dict(all=e1_scores(r)["acc1"], exact=e1_scores(r[ex])["acc1"], near=e1_scores(r[~ex])["acc1"])
    return dict(n=len(g.index), n_exact=int(ex.sum()), specs={k: list(np.atleast_1d(v[1])) for k, v in specs.items()},
                dev_loglik=ll, heldout_loglik_5fold=ho, e1_acc1=acc)


def part_b(model):
    out = {}
    for cs in ("C_phon", "C_LM", "C_union"):
        D = Data(model, "dev", cand_set=cs)
        g = D.listener("global", "C")
        gw = D.listener("global", "C+wa")
        gam = fit_gamma(g)
        bins, _ = fit_bins(g, monotone=True)
        specs = {"γ=0": ("gamma", 0.0), "단일 γ": ("gamma", gam), "구간 가중치": ("bins", bins)}
        has = g.df.groupby("key").in_M.any().reindex(g.index).fillna(False)
        y = D.y()
        out[cs] = dict(n=len(g.index), recall=float(has.mean()), n_cand_mean=float(g.df.groupby("key").size().mean()),
                       params={k: list(np.atleast_1d(v[1])) for k, v in specs.items()}, cells={})
        for k, s in specs.items():
            r = g.metrics(s)["rank"]
            f = gw.metrics(s).logF.reindex(D.keys)
            out[cs]["cells"][k] = dict(acc1=e1_scores(r)["acc1"], rho_logF=rho(f, y.reindex(D.keys)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m")
    ap.add_argument("--part", nargs="+", default=["a", "b"])
    a = ap.parse_args()
    res = {}
    L = [f"# 실험 4 발음 항 — {a.model} (dev 점검)", ""]
    if "a" in a.part:
        A = res["4A"] = part_a(a.model)
        L += ["## 4-A 거리 구간 가중치", "", f"dev n = {A['n']} (완전 동음 {A['n_exact']})", "",
              "| 발음 항 | 모수 | dev 로그 우도 | 5겹 검증 로그 우도 | acc@1 전체 | 완전 동음 | 유사 동음 |", "|---|---|---|---|---|---|---|"]
        for k, v in A["e1_acc1"].items():
            p = A["specs"].get(k, "")
            p = ", ".join(f"{x:.2f}" for x in p) if p != "" else "—"
            L.append(f"| {k} | {p} | {A['dev_loglik'].get(k, float('nan')):.1f} | {A['heldout_loglik_5fold'].get(k, float('nan')):.1f} | "
                     f"{v['all']:.3f} | {v['exact']:.3f} | {v['near']:.3f} |")
        gap = {k: A["e1_acc1"]["발음만"]["exact"] - A["e1_acc1"][k]["exact"] for k in ("단일 γ", "구간(자유)", "구간(단조)")}
        L += ["", "완전 동음에서 발음만 − 전체 차이: " + ", ".join(f"{k} {v:+.3f}" for k, v in gap.items()), ""]
    if "b" in a.part:
        Bt = res["4B"] = part_b(a.model)
        L += ["## 4-B 후보 집합 × 발음 항 (칸: E1 acc@1 / 후보 재현율 / E3 ρ(log F_→))", "",
              "| 후보 집합 | " + " | ".join(SPECS) + " |", "|---|---|---|---|"]
        for cs, v in Bt.items():
            cells = [f"{v['cells'][s]['acc1']:.3f} / {v['recall']:.3f} / {v['cells'][s]['rho_logF']:.3f}" for s in SPECS]
            L.append(f"| {cs} (평균 후보 {v['n_cand_mean']:.0f}) | " + " | ".join(cells) + " |")
        L += [""] + [f"- {cs} 모수: " + "; ".join(f"{k} {', '.join(f'{x:.2f}' for x in p)}" for k, p in v["params"].items())
                     for cs, v in Bt.items()]
    R = RunDir(a.model, "exp4_phon", dict(split="dev", dist="d_unit", parts=a.part, objective="Σ log π_→(w_a 묶음), restrict C"))
    R.metrics(res)
    R.table("table.md", L)


if __name__ == "__main__":
    main()
