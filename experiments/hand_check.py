"""손 점검: 잘 알려진 말장난 10개의 π를 직접 열어보고 상식에 맞는지 확인한다."""
from common import *  # noqa: F403

from data import semeval_items
from listener import fit_gamma, listener, load_scores

WELL_KNOWN = ["dyed", "orifice", "gnus", "excelling", "hooked", "ties", "harried", "soled", "knows", "psalms"]


def main():
    ap = parser("손 점검")
    args = ap.parse_args()
    items = {it.id: it for it in semeval_items(require_single_word=True)}
    cand, lm = load_cache(args.model, "semeval", args.k)
    cand = cand[cand.key.isin(items)]
    g, l = load_scores(cand, lm, "global"), load_scores(cand, lm, "local")
    gam = fit_gamma(g, list(items), restrict="C")
    picks = []
    for w in WELL_KNOWN:
        k = next((k for k, it in items.items() if it.w_p == w), None)
        if k:
            picks.append(k)
    L = [f"# 손 점검 — {args.model}, γ = {gam}", ""]
    for k in picks:
        it = items[k]
        L.append(f"## {it.text}")
        L.append(f"w_p = {it.w_p}, w_a = {it.w_a}")
        for ctx, df in (("global", g), ("local", l)):
            sub = df[df.key == k].copy()
            logq = sub.logprob - gam * sub.d
            sub["pi"] = np.exp(logq - np.logaddexp.reduce(logq.to_numpy()))
            top = sub.sort_values("pi", ascending=False).head(6)
            m = listener(sub, gam).iloc[0]
            L.append(f"- {ctx}: " + ", ".join(f"{r.cand}{'*' if r.is_wa else ''} {r.pi:.3f}" for r in top.itertuples())
                     + f"  | D={m.D:.3f} F={np.exp(m.logF):.3f} A={m.A:.3f} rank(w_a)={m['rank']}")
        L.append("")
    (out_dir(args.model) / "hand_check.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
