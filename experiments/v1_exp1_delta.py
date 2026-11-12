"""실험 1-C: H1c 직접 검정 δ = ρ(M1_full) − ρ(M1_pair0). 160M에서는 구현 동작 확인만 (dev).

두 모델의 10겹 교차검증 예측값을 한 번 계산하고, 같은 부트스트랩 표본으로 δ의 분포를 만든다.
판정 규칙(확인 검정): 95% CI 하한 > 0이면 지지.
"""
import argparse

from v1_features import *  # noqa: F403
from rules import RunDir, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m")
    a = ap.parse_args()
    P = fit_dev_params(a.model)
    D = Data(a.model, "dev")
    T = e3_table(D, P)
    y = D.y()
    preds = dict(M0=cv_predict(T["full"], y, BASE),
                 M1_full=cv_predict(T["full"], y, BASE + H1_FEATS),
                 M1_pair0=cv_predict(T["pair0"], y, BASE + H1_FEATS))
    res = paired_rho_tests(y, preds, dict(delta=("M1_full", "M1_pair0"), H1b_drho=("M1_full", "M0")))
    R = RunDir(a.model, "exp1C_delta", dict(split="dev (구현 확인)", params=P, n=len(y)))
    R.metrics(dict(params=P, n=len(y), tests=res))
    L = [f"# 1-C H1c 직접 검정 — {a.model} (dev, 구현 확인용)", "",
         f"n = {len(y)}, γ = {P['gamma']}, Kao r = {P['kao_r']}", "",
         "| 모델 | ρ (95% CI) |", "|---|---|"]
    for k in preds:
        L.append(f"| {k} | {ci(res[k]['rho'], *res[k]['ci'])} |")
    L += ["", f"δ = ρ(M1_full) − ρ(M1_pair0) = {ci(res['delta']['diff'], *res['delta']['ci'])}",
          f"Δρ (H1b) = ρ(M1_full) − ρ(M0) = {ci(res['H1b_drho']['diff'], *res['H1b_drho']['ci'])}",
          "", "dev 219개라 검정력이 낮다. 이 표는 구현이 끝까지 도는지만 확인한다."]
    R.table("table.md", L)


if __name__ == "__main__":
    main()
