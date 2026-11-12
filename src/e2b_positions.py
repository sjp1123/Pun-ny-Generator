"""E2b용 항목: UKP 문장의 모든 내용어 위치. w_a는 없고(T̂로 대신), 위치마다 Item 하나."""
from functools import lru_cache

from baselines import _nlp
from data import Item, norm, semeval_items, ukp_sentences

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


@lru_cache(maxsize=1)
def e2b_items() -> list[Item]:
    from spacy.lang.en.stop_words import STOP_WORDS
    gold = {norm(it.text): it for it in semeval_items()}
    out = []
    for r in ukp_sentences().itertuples():
        g = gold.get(norm(r.text))
        for t in _nlp()(r.text):
            if t.pos_ in CONTENT_POS and t.is_alpha and t.text.lower() not in STOP_WORDS and len(t.text) > 1:
                s, e = t.idx, t.idx + len(t.text)
                out.append(Item(
                    id=f"ukp{r.ukp_id}@{s}", text=r.text, start=s, end=e, w_p=t.text.lower(),
                    meta=dict(ukp_id=int(r.ukp_id), category=r.category,
                              gold_start=g.start if g else None, semeval_id=g.id if g else None),
                ))
    return out
