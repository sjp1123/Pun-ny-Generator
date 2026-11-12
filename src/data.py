"""데이터 로더: SemEval 이형 말장난, UKP 웃김 점수, He et al. 웃김 주석.

SemEval JSON은 소문자 토큰이라 LM 채점에는 UKP 원문을 쓰고,
pun 위치는 "몇 번째로 나오는 pun_word인가"로 원문 문자 구간에 대응시킨다.
"""
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
SEMEVAL_DIR = DATA / "pungen/data/semeval/hetero"
UKP_DIR = DATA / "gppl/data/pl-humor-full"
HE_ANN = DATA / "pungen/data/funniness_annotation/analysis_zscored_pun_scores.txt"

UKP_CATEGORIES = {
    "hetero_pun": "jokes_heterographic_puns.txt",
    "homo_pun": "jokes_homographic_puns.txt",
    "nonpun_joke": "jokes_nonpuns.txt",
    "nonjoke": "nonjokes.txt",
}


def norm(s: str) -> str:
    """정규화 키: 악센트 제거, 소문자, 영숫자만."""
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


@dataclass
class Item:
    """원문 문장 하나와 그 안의 대상 단어 위치(문자 구간)."""
    id: str
    text: str
    start: int
    end: int
    w_p: str                     # 문장에 쓰인 단어 (소문자)
    w_a: str | None = None       # 숨은 단어 (없으면 None)
    meta: dict = field(default_factory=dict)

    @property
    def surface(self) -> str:
        return self.text[self.start:self.end]

    def fill(self, cand: str) -> str:
        return self.text[: self.start] + match_case(self.surface, cand) + self.text[self.end:]

    def ws_tokens(self):
        """공백 단위 토큰의 (start, end) 목록과 대상 단어가 든 토큰 번호."""
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", self.text)]
        idx = next(i for i, (s, e) in enumerate(spans) if s <= self.start < e)
        return spans, idx

    def window(self, cand: str, before: int = 2, after: int = 2) -> str:
        """앞 before 단어 ~ 뒤 after 단어 구간에 후보를 넣은 문자열."""
        spans, idx = self.ws_tokens()
        s = spans[max(0, idx - before)][0]
        e = spans[min(len(spans) - 1, idx + after)][1]
        filled = self.fill(cand)
        shift = len(filled) - len(self.text)
        return filled[s : e + shift]


def match_case(orig: str, cand: str) -> str:
    if len(orig) > 1 and orig.isupper():
        return cand.upper()
    if orig[:1].isupper():
        return cand[:1].upper() + cand[1:]
    return cand


def find_occurrence(text: str, word: str, k: int):
    """text에서 word의 k번째(0부터) 출현 구간. 대소문자 무시, 영숫자 경계."""
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", re.I)
    hits = [m.span() for m in pat.finditer(text)]
    if k < len(hits):
        return hits[k]
    # 악센트 등으로 직접 안 맞으면 정규화한 사본에서 다시 찾는다
    folded = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    if len(folded) == len(text):
        hits = [m.span() for m in pat.finditer(folded)]
        if k < len(hits):
            return hits[k]
    return None


# ---------------------------------------------------------------- SemEval
def load_semeval() -> list[dict]:
    rows = []
    for split in ("dev", "test"):
        for r in json.loads((SEMEVAL_DIR / f"{split}.json").read_text(encoding="utf-8")):
            r["split"] = split
            rows.append(r)
    return rows


# ---------------------------------------------------------------- UKP
def load_ukp() -> pd.DataFrame:
    scores = pd.read_csv(UKP_DIR / "item_scores.txt", sep="\t", header=None,
                         names=["ukp_id", "score", "text"], quoting=3, encoding="utf-8")
    cat_of = {}
    for cat, fn in UKP_CATEGORIES.items():
        for line in (UKP_DIR / fn).read_text(encoding="utf-8").splitlines():
            if line.strip():
                cat_of.setdefault(norm(line), cat)
    scores["key"] = scores.text.map(norm)
    scores["category"] = scores.key.map(cat_of)
    return scores


# ---------------------------------------------------------------- 대응
def semeval_items(require_single_word=False) -> list[Item]:
    """SemEval 이형 말장난 1,098개를 UKP 원문에 대응시킨 Item 목록."""
    ukp = load_ukp()
    by_key = {}
    for r in ukp.itertuples():
        by_key.setdefault(r.key, r)
    items = []
    for r in load_semeval():
        u = by_key.get(norm("".join(r["tokens"])))
        if u is None:
            raise ValueError(f"UKP 원문 없음: {r['id']}")
        pw = r["pun_word"]
        k = r["tokens"][: r["pun_word_id"]].count(pw)
        span = find_occurrence(u.text, pw, k)
        if span is None:
            raise ValueError(f"pun 위치 대응 실패: {r['id']} {pw!r} in {u.text!r}")
        wa = r["alter_word"].strip().lower()
        multi = bool(re.search(r"[\s_]", wa))
        if require_single_word and multi:
            continue
        items.append(Item(
            id=r["id"], text=u.text, start=span[0], end=span[1], w_p=pw.lower(), w_a=wa,
            meta=dict(split=r["split"], ukp_id=int(u.ukp_id), score=float(u.score),
                      category=u.category, multiword=multi, tokens=r["tokens"],
                      pun_word_id=r["pun_word_id"]),
        ))
    return items


# ---------------------------------------------------------------- He 주석
def detokenize(tokens: list[str]) -> str:
    s = " ".join(tokens)
    s = s.replace("' '", '"').replace("` `", '"')
    s = re.sub(r"(\w) n ' t\b", r"\1n't", s)
    s = re.sub(r"(\w) ' (s|m|re|ve|ll|d|t)\b", r"\1'\2", s)
    s = re.sub(r"\s+([.,!?;:%)])", r"\1", s)
    s = re.sub(r"([($])\s+", r"\1", s)
    s = re.sub(r"(\w) - (\w)", r"\1-\2", s)
    s = re.sub(r'"\s+(.*?)\s+"', r'"\1"', s)
    s = re.sub(r"\bi\b", "I", s)
    return s[:1].upper() + s[1:]


def load_he_annotation(fix_pairs=True) -> tuple[list[Item], pd.DataFrame]:
    """He et al. 6유형 194문장. fix_pairs면 SemEval과 대조해 뒤바뀐 pun/depun 쌍을 바로잡는다.

    반환: (Item 목록, 검수표). 검수표는 pun 유형 33개 각각의 SemEval 대조 결과다.
    """
    rows = []
    for line in HE_ANN.read_text(encoding="utf-8").splitlines():
        sent, pair, typ, z = line.split("\t")
        toks = sent.split()
        # 쌍 "x-y"에서 x는 문장 속 단어. 단어 안에 하이픈이 있을 수 있어 가능한 분할을 모두 본다
        parts = pair.split("-")
        for i in range(1, len(parts)):
            x, y = "-".join(parts[:i]), "-".join(parts[i:])
            if x in toks:
                break
        else:
            raise ValueError(f"쌍의 단어가 문장에 없음: {pair} / {sent}")
        rows.append(dict(sent=sent, toks=toks, x=x, y=y, type=typ, z=float(z)))
    df = pd.DataFrame(rows)

    sem = {norm("".join(r["tokens"])): r for r in load_semeval()}
    review = []
    for i, r in df[df.type == "pun"].iterrows():
        hit = sem.get(norm(r.sent))
        partner = df[(df.type == "depun") & (df.x == r.y) & (df.y == r.x)]
        p_hit = sem.get(norm(partner.sent.iloc[0])) if len(partner) else None
        if hit is not None and hit["pun_word"] == r.x:
            verdict = "ok"
        elif p_hit is not None and p_hit["pun_word"] == r.y:
            verdict = "swapped"
        else:
            verdict = "not_in_semeval"
        ref = hit if verdict == "ok" else p_hit
        review.append(dict(row=i, sent=r.sent, pair=f"{r.x}-{r.y}", z=r.z, verdict=verdict,
                           semeval_id=(ref or {}).get("id"),
                           semeval_pair=f"{ref['pun_word']}-{ref['alter_word']}" if ref else None,
                           alter_match=bool(ref) and ref["alter_word"] == (r.y if verdict == "ok" else r.x),
                           partner_row=partner.index[0] if len(partner) else None))
    review = pd.DataFrame(review)
    if fix_pairs:
        for r in review[review.verdict == "swapped"].itertuples():
            a, b = r.row, int(r.partner_row)
            df.loc[a, "type"], df.loc[b, "type"] = "depun", "pun"

    items = []
    for i, r in df.iterrows():
        text = detokenize(r.toks)
        k = r.toks[: len(r.toks) - 1 - r.toks[::-1].index(r.x)].count(r.x)  # 마지막 출현
        span = find_occurrence(text, r.x, k)
        if span is None:
            raise ValueError(f"He 문장 위치 대응 실패: {r.x} / {text}")
        items.append(Item(id=f"he_{i}", text=text, start=span[0], end=span[1],
                          w_p=r.x, w_a=r.y, meta=dict(type=r.type, z=r.z)))
    return items, review


def ukp_sentences(categories=("hetero_pun", "nonpun_joke", "nonjoke")) -> pd.DataFrame:
    ukp = load_ukp()
    return ukp[ukp.category.isin(categories)].reset_index(drop=True)
