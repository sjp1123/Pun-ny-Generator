"""실험 3: 읽는 순간(π_←, 왼쪽 문맥) vs 다 읽은 뒤(π_→ = π_global). dev 점검 3개.

1. 포화 비교: D_local > 0.99 비율 vs D_← > 0.99 비율, S_← 분포
2. 구현 확인: pun word가 마지막 단어인 문장에서 π_←와 π_→의 차이(총변동거리)가 거의 0인지
3. 방향 확인: He 6유형의 S_← 평균 (pun·retrieved_aw_alter 큼, depun·retrieved_aw·retrieved_pw 작음)
U = KL(π_→ ‖ π_←)는 탐색: 뒤에 단어 2개 이상인 부분집합에서만 의미가 있다.
"""
import argparse
import re

from v1_common import *  # noqa: F403
from data import load_he_annotation, semeval_items
from rules import Listener, RunDir, add_lemma_cols, boot, fit_gamma

TYPES = ["pun", "depun", "retrieved_aw_alter", "retrieved_aw", "retrieved_pw_alter", "retrieved_pw"]


def words_after(it) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", it.text[it.end:]))


def position_counts():
    its = semeval_items()
    wa = pd.Series([min(words_after(i), 3) for i in its]).value_counts().sort_index()
    empty_left = sum(not i.text[: i.start].strip() for i in its)
    no_alnum_left = sum(not re.search(r"[A-Za-z0-9]", i.text[: i.start]) for i in its)
    return dict(n=len(its), words_after={("3+" if k == 3 else str(k)): int(v) for k, v in wa.items()},
                empty_left=int(empty_left), no_alnum_left=int(no_alnum_left))


def pi_frame(lst: Listener, spec) -> pd.DataFrame:
    return lst.df[["key", "cand"]].assign(lp=lst.logpi(spec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m")
    a = ap.parse_args()
    res = dict(position_counts=position_counts())
    D = Data(a.model, "dev", extra_lm=("semeval_K100_left",))
    gam = fit_gamma(D.listener("global", "C"))
    spec = ("gamma", gam)
    Lg, Ll, Lf = (D.listener(c, "C+wa") for c in ("global", "local", "left"))
    mg, ml, mf = Lg.metrics(spec), Ll.metrics(spec), Lf.metrics(spec)

    # 1. 포화
    res["saturation"] = dict(D_local_gt_099=float((ml.D > 0.99).mean()), D_left_gt_099=float((mf.D > 0.99).mean()),
                             D_global_gt_099=float((mg.D > 0.99).mean()),
                             S_left_quantiles={q: float(mf.S.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].hist(mf.S, bins=40, color="#4878a8")
        ax[0].set_title("S_left = -log pi_left(w_p) (dev)")
        ax[1].hist(ml.logD, bins=40, color="#a85848")
        ax[1].set_title("log D_local (dev)")
        fig.tight_layout()
        outpng = True
    except Exception:
        outpng = False

    # 2. 구현 확인: 마지막 단어 pun에서 π_← ≈ π_→
    P = pi_frame(Lg, spec).merge(pi_frame(Lf, spec), on=["key", "cand"], suffixes=("_g", "_f"))
    tv = P.assign(x=(np.exp(P.lp_g) - np.exp(P.lp_f)).abs()).groupby("key").x.sum() / 2
    kl = P.assign(x=np.exp(P.lp_g) * (P.lp_g - P.lp_f)).groupby("key").x.sum()
    wa_n = pd.Series({k: words_after(D.items[k]) for k in D.keys})
    grp = wa_n.clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"})
    res["tv_by_words_after"] = {g: dict(n=int((grp == g).sum()), tv_median=float(tv[grp == g].median()),
                                        tv_p90=float(tv[grp == g].quantile(0.9)))
                                for g in ("0", "1", "2", "3+")}
    last = grp[grp == "0"].index
    trail = pd.Series({k: D.items[k].text[D.items[k].end:].strip() for k in last})
    res["last_word_trailing"] = trail.value_counts().head(8).to_dict()
    bare = [k for k in last if trail[k] == ""]
    res["tv_no_trailing_chars"] = {k: float(tv[k]) for k in bare}  # 문장부호도 없으면 π_← = π_→여야 함
    res["U_KL_dev"] = {g: float(kl[grp == g].median()) for g in ("0", "1", "2", "3+")}

    # 3. 방향: He 6유형 S_← (He 왼쪽 문맥 캐시가 있을 때)
    try:
        he_items, _ = load_he_annotation()
        by = {i.id: i for i in he_items}
        hc = add_lemma_cols(pd.read_parquet(CACHE / "candidates_he_K100.parquet"))
        hl = lm_cache(a.model, "he_K100_left")
        mh = Listener(scored(hc, hl, "left"), restrict="C+wa").metrics(spec)
        mh["type"] = [by[k].meta["type"] for k in mh.index]
        res["he_S_left"] = {}
        for t in TYPES:
            v = mh[mh.type == t].S.to_numpy()
            p, lo, hi = boot(lambda i: v[i].mean(), len(v))
            res["he_S_left"][t] = dict(n=len(v), mean=float(p[0]), ci=[float(lo[0]), float(hi[0])])
        big = min(res["he_S_left"][t]["mean"] for t in ("pun", "retrieved_aw_alter"))
        small = max(res["he_S_left"][t]["mean"] for t in ("depun", "retrieved_aw", "retrieved_pw"))
        res["he_direction_ok"] = bool(big > small)
    except SystemExit as e:
        print("He 방향 점검 건너뜀:", e)

    R = RunDir(a.model, "exp3_reading", dict(split="dev", gamma=gam, dist="d_unit", restrict="C+wa"))
    if outpng:
        fig.savefig(R.dir / "S_left_hist.png", dpi=120)
    R.metrics(res)
    pc = res["position_counts"]
    s = res["saturation"]
    L = [f"# 실험 3 읽는 순간 vs 다 읽은 뒤 — {a.model} (dev 점검)", "",
         f"전체 1,098개의 pun 뒤 단어 수: {pc['words_after']}, 왼쪽 문맥에 영숫자가 없는 문장 {pc['no_alnum_left']}개 "
         f"(완전히 빈 문장 {pc['empty_left']}개 + 따옴표·기호만 {pc['no_alnum_left'] - pc['empty_left']}개)", "",
         f"γ (dev) = {gam}, dev n = {len(D.keys)}", "",
         "## 1. 포화", "", f"- D_local > 0.99: {s['D_local_gt_099']:.3f}", f"- D_global > 0.99: {s['D_global_gt_099']:.3f}",
         f"- D_← > 0.99: {s['D_left_gt_099']:.3f}",
         "- S_← 분위수: " + ", ".join(f"{q}: {v:.2f}" for q, v in s["S_left_quantiles"].items()), "",
         "## 2. π_← vs π_→ 총변동거리 (pun 뒤 단어 수별)", "", "| 뒤 단어 수 | n | 중앙값 | 90% 분위 | U = KL(π_→‖π_←) 중앙값 |", "|---|---|---|---|---|"]
    for g, v in res["tv_by_words_after"].items():
        L.append(f"| {g} | {v['n']} | {v['tv_median']:.3f} | {v['tv_p90']:.3f} | {res['U_KL_dev'][g]:.3f} |")
    L += ["", f"마지막 단어 pun 문장의 pun 뒤 문자열 상위: {res['last_word_trailing']}",
          f"pun 뒤에 아무 문자도 없는 문장의 총변동거리: {res['tv_no_trailing_chars']} → 0이면 구현이 맞고, "
          "마지막 단어 문장에 남는 차이는 뒤따르는 문장부호 확률 P(. | 문맥, T)에서만 온다.", "",
          ]
    if "he_S_left" in res:
        L += ["## 3. He 6유형 S_← 평균 (95% CI)", "", "| 유형 | n | S_← |", "|---|---|---|"]
        for t, v in res["he_S_left"].items():
            L.append(f"| {t} | {v['n']} | {v['mean']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}] |")
        L += ["", f"방향 확인 (pun·retrieved_aw_alter의 최솟값 > depun·retrieved_aw·retrieved_pw의 최댓값): {res['he_direction_ok']}"]
    R.table("table.md", L)


if __name__ == "__main__":
    main()
