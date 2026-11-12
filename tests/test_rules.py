"""보완 규칙(src/rules.py) 단위 테스트."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from rules import Listener, penalty, split_ids, split_sha


def _toy():
    # w_p, 그리고 목표 표제어 arm의 두 활용형(arms, arm)과 다른 후보 둘
    rows = [("wp", "psalm", -5.0, 0, True, False, False),
            ("arms", "arm", -6.0, 1, False, False, True),
            ("arm", "arm", -9.0, 1, False, True, True),
            ("palms", "palm", -6.5, 1, False, False, False),
            ("times", "time", -7.0, 2, False, False, False)]
    df = pd.DataFrame(rows, columns=["cand", "lemma", "logprob", "d_unit", "is_wp", "is_wa", "in_M"])
    return df.assign(key="k", in_C=True)


def test_lemma_sum_is_sum_over_M():
    df = _toy()
    lst = Listener(df, restrict="C+wa")
    m = lst.metrics(("gamma", 0.5), mode="sum").iloc[0]
    logq = df.logprob - 0.5 * df.d_unit
    pi = np.exp(logq - np.logaddexp.reduce(logq))
    assert np.isclose(np.exp(m.logpi_wa), pi[df.in_M].sum())
    assert np.isclose(np.exp(m.logF), pi[df.in_M].sum() / (1 - pi[0]))
    strict = lst.metrics(("gamma", 0.5), mode="strict").iloc[0]
    assert np.isclose(np.exp(strict.logpi_wa), pi[2])


def test_group_rank_sums_inflections():
    # 개별로는 palms(-6.5)가 arm(-9)보다 높지만, 묶음 arm = arms + arm이 1위
    m = Listener(_toy(), restrict="C+wa").metrics(("gamma", 0.0), mode="sum")
    assert m["rank"].iloc[0] == 1
    s = Listener(_toy(), restrict="C+wa").metrics(("gamma", 0.0), mode="strict")
    assert s["rank"].iloc[0] == 4  # arms, palms, times 다음


def test_bins_penalty():
    d = np.array([0, 1, 2, 3, 7.0])
    assert np.allclose(penalty(d, ("bins", (1, 2, 5))), [0, 1, 2, 5, 5])
    assert np.allclose(penalty(d, ("gamma", 2.0)), 2 * d)


def test_splits_fixed():
    assert len(split_ids("dev")) == 219 and len(split_ids("test")) == 879
    assert not split_ids("dev") & split_ids("test")
    assert split_sha("dev") == "ca164d715859ee1258ecc2310a9a8174a03e0a3375ca11bd8beea0a4108625a5"
    assert split_sha("test") == "48819bef45c6359c8427eba4a336760d3cc8ce8b1419f0d003d3de1a5e54a7c9"
