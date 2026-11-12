"""M2 발음 거리: 조음 자질 가중 편집 거리 d(φ_T, φ_p).

panphon의 가중 자질 편집 거리(치환 = Σ w·|f1−f2|, 삽입·삭제 = Σ w)를
ARPAbet→IPA 표로 연결한다. 모음↔자음 치환은 Jaech를 따라 큰 비용으로 막는다.
한 단어를 어휘 전체와 비교하는 DP는 어휘 축으로 벡터화했다.
"""
from functools import lru_cache

import numpy as np

ARPA2IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ",
    "ER": "ɜ˞", "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ",
    "UW": "u", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z",
    "ZH": "ʒ",
}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
PHONES = list(ARPA2IPA)
PIDX = {p: i for i, p in enumerate(PHONES)}
PAD = len(PHONES)
VC_COST = 100.0  # 모음↔자음 치환: 사실상 금지


def _seg_dp(a, b, sub, indel):
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1) * indel
    D[0, :] = np.arange(m + 1) * indel
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + indel, D[i, j - 1] + indel,
                          D[i - 1, j - 1] + sub(a[i - 1], b[j - 1]))
    return D[n, m]


@lru_cache(maxsize=None)
def cost_tables(kind: str = "feature"):
    """(SUB, INS) 표. 마지막 행·열은 PAD(비용 0).

    kind: "unit"(단순 편집 거리), "feature"(panphon 기본), "feature@s"(삽입·삭제 비용 × s).
    """
    if kind.startswith("feature@"):
        SUB, INS = cost_tables("feature")
        s = float(kind.split("@")[1])
        return SUB, INS * s
    n = len(PHONES)
    SUB = np.zeros((n + 1, n + 1))
    INS = np.zeros(n + 1)
    if kind == "unit":
        SUB[:n, :n] = 1 - np.eye(n)
        INS[:n] = 1
        return SUB, INS
    import panphon
    ft = panphon.FeatureTable()
    w = np.array(ft.weights)
    k = len(w)  # panphon도 zip으로 앞쪽 자질만 쓴다
    vec = {p: [np.array(v[:k]) for v in ft.word_to_vector_list(ipa, numeric=True)]
           for p, ipa in ARPA2IPA.items()}
    indel = w.sum()
    sub = lambda u, v: float(np.abs(u - v) @ w)
    for p in PHONES:
        assert vec[p], f"panphon이 {p}={ARPA2IPA[p]}를 모름"
        INS[PIDX[p]] = indel * len(vec[p])
    for p in PHONES:
        for q in PHONES:
            if (p in VOWELS) != (q in VOWELS):
                SUB[PIDX[p], PIDX[q]] = VC_COST
            else:
                SUB[PIDX[p], PIDX[q]] = _seg_dp(vec[p], vec[q], sub, indel)
    return SUB, INS


def encode(pron) -> np.ndarray:
    return np.array([PIDX[p] for p in pron], dtype=np.int64)


class PronTable:
    """어휘 발음표: 발음 하나가 한 행(PAD로 채움), entry → 단어 번호."""

    def __init__(self, words, prons_of):
        rows, owner = [], []
        for wi, w in enumerate(words):
            for p in prons_of(w):
                if p:
                    rows.append(encode(p))
                    owner.append(wi)
        self.words = list(words)
        self.lens = np.array([len(r) for r in rows])
        L = self.lens.max()
        self.E = np.full((len(rows), L), PAD, dtype=np.int64)
        for i, r in enumerate(rows):
            self.E[i, : len(r)] = r
        self.owner = np.array(owner)

    def _entry_distances(self, src, kind):
        SUB, INS = cost_tables(kind)
        s = encode(src)
        m = len(s)
        N, L = self.E.shape
        col = np.zeros((m + 1, N))
        col[:, :] = np.concatenate([[0], np.cumsum(INS[s])])[:, None]
        out = np.full(N, np.inf)
        out[self.lens == 0] = col[m, self.lens == 0]
        for j in range(L):
            t = self.E[:, j]
            ins_t = INS[t]
            new = np.empty_like(col)
            new[0] = col[0] + ins_t
            for i in range(1, m + 1):
                new[i] = np.minimum(np.minimum(col[i] + ins_t, new[i - 1] + INS[s[i - 1]]),
                                    col[i - 1] + SUB[s[i - 1], t])
            col = new
            done = self.lens == j + 1
            out[done] = col[m, done]
        return out

    def distances(self, srcs, kind="feature"):
        """발음 목록 srcs와 모든 단어의 거리. 단어별로 가장 가까운 발음 쌍을 쓴다.

        반환: (d 원거리, 정규화 거리) — 정규화는 그 쌍의 긴 쪽 음소 수로 나눈 값.
        """
        d = np.full(len(self.words), np.inf)
        dn = np.full(len(self.words), np.inf)
        for s in srcs:
            if not s:
                continue
            e = self._entry_distances(s, kind)
            order = np.lexsort((e, self.owner))
            first = np.r_[True, self.owner[order][1:] != self.owner[order][:-1]]
            idx = order[first]
            w = self.owner[idx]
            better = e[idx] < d[w]
            d[w[better]] = e[idx][better]
            dn[w[better]] = normalize(e[idx][better], len(s), self.lens[idx][better])
        return d, dn


def pair_distance(prons_a, prons_b, kind="feature") -> tuple[float, int, int]:
    """두 단어의 최소 거리와 그때의 발음 길이 (d, len_a, len_b)."""
    SUB, INS = cost_tables(kind)
    best = (np.inf, 1, 1)
    for a in prons_a:
        for b in prons_b:
            if not a or not b:
                continue
            ea, eb = encode(a), encode(b)
            D = np.zeros((len(ea) + 1, len(eb) + 1))
            D[:, 0] = np.concatenate([[0], np.cumsum(INS[ea])])
            D[0, :] = np.concatenate([[0], np.cumsum(INS[eb])])
            for i in range(1, len(ea) + 1):
                for j in range(1, len(eb) + 1):
                    D[i, j] = min(D[i - 1, j] + INS[ea[i - 1]], D[i, j - 1] + INS[eb[j - 1]],
                                  D[i - 1, j - 1] + SUB[ea[i - 1], eb[j - 1]])
            if D[-1, -1] < best[0]:
                best = (float(D[-1, -1]), len(a), len(b))
    return best


def normalize(d, len_a, len_b):
    """후보 순위용 정규화 거리: 긴 쪽 음소 수로 나눈다."""
    return d / np.maximum(np.maximum(len_a, len_b), 1)
