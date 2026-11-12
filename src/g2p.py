"""M1 G2P: 단어 → ARPAbet 음소열.

CMU 사전을 먼저 찾고, 없으면 신경망 G2P(g2p_en)로 만든다.
발음이 여러 개면 모두 두고, 강세 숫자는 따로 보관한다(거리 계산에서는 뺀다).
"""
import re
from functools import lru_cache

import cmudict as _cmudict_pkg

_STRESS = re.compile(r"\d")


@lru_cache(maxsize=1)
def cmu() -> dict[str, list[list[str]]]:
    return _cmudict_pkg.dict()


@lru_cache(maxsize=1)
def _neural():
    # g2p_en은 import 시 NLTK 자원을 내려받으려 하고 생성자에서 NLTK cmudict를 읽는다.
    # 신경망 예측(predict)만 쓰므로 CMU 사전은 같은 판의 pip 패키지로 대신 넣고,
    # 내려받기 검사는 빈 자리표시 파일로 통과시킨다.
    import zipfile
    from pathlib import Path
    import nltk
    stub = Path(__file__).resolve().parents[1] / "cache" / "nltk_stub"
    for f in ("taggers/averaged_perceptron_tagger.zip", "corpora/cmudict.zip"):
        (stub / f).parent.mkdir(parents=True, exist_ok=True)
        if not (stub / f).exists():
            zipfile.ZipFile(stub / f, "w").close()
    nltk.data.path.insert(0, str(stub))
    import g2p_en.g2p as m
    m.cmudict = _cmudict_pkg
    return m.G2p()


def strip_stress(phones) -> tuple[str, ...]:
    return tuple(_STRESS.sub("", p) for p in phones)


def stress_of(phones) -> tuple[int, ...]:
    return tuple(int(p[-1]) for p in phones if p[-1].isdigit())


def in_cmu(word: str) -> bool:
    return word.lower() in cmu()


@lru_cache(maxsize=None)
def prons_with_stress(word: str) -> tuple[tuple[str, ...], ...]:
    """강세 포함 발음 목록. 여러 단어(공백·하이픈)는 각 부분 첫 발음을 이어 붙인다."""
    w = word.lower().strip()
    d = cmu()
    if w in d:
        return tuple(tuple(p) for p in d[w])
    parts = [p for p in re.split(r"[\s\-_]+", w) if p]
    if len(parts) > 1:
        return (tuple(ph for p in parts for ph in prons_with_stress(p)[0]),)
    letters = re.sub(r"[^a-z]", "", w)
    if not letters:
        return ((),)
    return (tuple(_neural().predict(letters)),)


@lru_cache(maxsize=None)
def prons(word: str) -> tuple[tuple[str, ...], ...]:
    """강세를 뺀 서로 다른 발음 목록."""
    out = []
    for p in prons_with_stress(word):
        s = strip_stress(p)
        if s not in out:
            out.append(s)
    return tuple(out)
