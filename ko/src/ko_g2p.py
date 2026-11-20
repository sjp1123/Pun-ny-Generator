"""한국어 표준 발음 (K0). g2pkk의 규칙을 쓰되, 형태소 품사는 MeCab 대신 Kiwi로 준다.

g2pkk는 Windows에서 MeCab(eunjeon)을 빌드하려다 실패한다(Visual Studio 필요). g2pkk가 MeCab에서 쓰는 것은
`pos(문장) → [(표면형, 품사)]`뿐이고(utils.annotate), 품사는 첫 글자(N, V, E, J …)와 NNBC만 본다. Kiwi의 품사표는
세종 계열이라 같은 글자를 쓴다. Kiwi는 '했다 → 하/VV 었/EP 다/EF'처럼 축약을 풀어 내므로, 표면에서 겹치는 형태소는
'VV+EP'처럼 하나로 묶어 MeCab-ko와 같은 꼴로 돌려준다.
"""
from functools import lru_cache


class KiwiTagger:
    def __init__(self):
        from kiwipiepy import Kiwi
        self.kiwi = Kiwi()

    def pos(self, s: str):
        spans = []  # [시작, 끝, 품사]
        for t in self.kiwi.tokenize(s):
            st, en, tag = t.start, t.start + t.len, str(t.tag)
            if spans and st < spans[-1][1]:  # 축약: 앞 형태소와 표면이 겹친다
                spans[-1][1] = max(spans[-1][1], en)
                spans[-1][2] += "+" + tag
            else:
                spans.append([st, en, tag])
        return [(s[a:b], tag) for a, b, tag in spans]


@lru_cache(maxsize=1)
def _g2p():
    from g2pkk import G2p

    class KoG2p(G2p):
        def check_mecab(self):
            pass

        def get_mecab(self):
            return KiwiTagger()

    return KoG2p()


@lru_cache(maxsize=None)
def pron(text: str) -> str:
    """표준 발음(한글). 예: 같이 → 가치, 국물 → 궁물."""
    return _g2p()(text)
