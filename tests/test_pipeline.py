"""작업 목록의 단위 테스트. 실행: python -X utf8 -m pytest tests -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest

from data import Item, load_he_annotation, semeval_items
from g2p import prons
from listener import listener


def test_semeval_alignment_all_1098():
    items = semeval_items()
    assert len(items) == 1098
    assert sum(it.meta["multiword"] for it in items) == 24
    for it in items:
        assert it.surface.lower() == it.w_p, it.id


def test_he_annotation_types_and_review():
    items, review = load_he_annotation()
    assert len(items) == 194
    assert len(review) == 33
    for it in items:
        assert it.surface.lower() == it.w_p


def test_dyed_died_same_phonemes():
    assert prons("dyed") == prons("died")


def test_oov_word_gets_pronunciation():
    p = prons("bached")
    assert p and p[0]


def test_fill_keeps_case_and_space():
    it = Item(id="x", text="Dentists don't like a hard day at the Orifice.", start=38, end=45, w_p="orifice")
    assert it.surface == "Orifice"
    assert it.fill("office") == "Dentists don't like a hard day at the Office."
    assert it.window("office") == "at the Office."


def test_prefix_split_tokenization_matches():
    tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")
    from lmscore import split_at
    for it in semeval_items()[:200]:
        pre, lead = split_at(it)
        suf = lead + it.surface + it.text[it.end:]
        joint = tok(it.text, add_special_tokens=False)["input_ids"]
        parts = tok(pre, add_special_tokens=False)["input_ids"] + tok(suf, add_special_tokens=False)["input_ids"]
        assert joint == parts, it.text


def _toy(logprob_wp, others):
    rows = [dict(key="a", cand="wp", logprob=logprob_wp, d=0.0, zipf=4.0, in_C=True, is_wp=True, is_wa=False)]
    for i, lp in enumerate(others):
        rows.append(dict(key="a", cand=f"c{i}", logprob=lp, d=1.0, zipf=4.0, in_C=True, is_wp=False, is_wa=i == 0))
    return pd.DataFrame(rows)


def test_pi_sums_to_one_and_metrics():
    df = _toy(-10.0, [-11.0, -12.0, -13.0])
    m = listener(df, gamma=0.5).iloc[0]
    logq = df.logprob - 0.5 * df.d
    pi = np.exp(logq - np.logaddexp.reduce(logq))
    assert np.isclose(pi.sum(), 1.0)
    assert np.isclose(m.D, 1 - pi[0])
    assert np.isclose(np.exp(m.logF), pi[1] / (1 - pi[0]))
    assert 0 <= m.A <= np.log(2) + 1e-12
    assert m["rank"] == 1


def test_D_does_not_collapse():
    df = _toy(0.0, [-200.0, -210.0])
    m = listener(df, gamma=0.0).iloc[0]
    assert np.isfinite(m.logD) and m.logD < -150
    assert np.isfinite(m.logF)
