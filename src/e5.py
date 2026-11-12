"""E5 생성 파이프라인(docs/E5_DESIGN.md) 공용: 표본 파일, 사전, 형식, 프롬프트, 규칙 검사.

쌍 목록과 표본은 e5/에 고정해 커밋한다(word2vec 없이 GPU에서 돌리기 위해).
생성·채점 결과는 results/e5/<단계>/ 아래에 쌓인다.
"""
import hashlib
import json
import re
from functools import lru_cache

from data import DATA, RESULTS, ROOT

E5_DIR = ROOT / "e5"
SAMPLES = E5_DIR / "samples.json"
N_CAND = 10
MAX_WORDS = 40
TEMPERATURE = 0.9
CONDITIONS = ["B0", "B1", "P1", "P2", "P3", "P4"]

FORMATS = {
    "riddle": dict(
        name="question-and-answer riddle",
        desc='a question that ends with "?", followed by a short punchline answer on the same line',
    ),
    "oneliner": dict(
        name="one-line situation joke",
        desc="a single sentence (or two short sentences) describing a situation; no question-and-answer",
    ),
    "tom_swifty": dict(
        name="Tom Swifty",
        desc='a quoted line followed by "Tom said ..." (or asked, replied, ...), where the pun comes from '
             "how Tom said it or what he did",
    ),
}


def write_config(path, config: dict):
    """실행 기록: 설정, 시드, 커밋 해시(dirty 여부), 명령줄."""
    import sys
    from rules import SEED, dump, git_state
    path.mkdir(parents=True, exist_ok=True)
    dump(path / "config.json", dict(config, seed=SEED, git=git_state(), argv=sys.argv))


def human_for_cells(C: list[dict]) -> dict:
    """칸 → 사람 말장난(상한선) 하나. 같은 형식끼리 짝짓고, 사람 말장난이 모자라면 돌려 쓴다(본 실행: 각 2칸).
    Tom Swifty 칸에는 붙이지 않는다."""
    import numpy as np
    H = load_samples()["human"]
    out = {}
    for f in ("riddle", "oneliner"):
        hs = [h for h in H if h["format"] == f]
        order = np.random.default_rng(stable_seed("human", f)).permutation(len(hs))
        cs = [c for c in C if c["format"] == f]
        for k, c in enumerate(cs):
            out[c["cell"]] = hs[order[k % len(hs)]]
    return out


def stage_dir(stage: str):
    d = RESULTS / "e5" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def stable_seed(*parts) -> int:
    """호출마다 정해지는 시드(문자열 해시, 실행마다 같음)."""
    from rules import SEED
    h = hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).hexdigest()
    return int(h[:8], 16)


def load_samples() -> dict:
    return json.loads(SAMPLES.read_text(encoding="utf-8"))


def cells(stage: str) -> list[dict]:
    """단계의 칸 목록: 쌍 × 형식. pilot = 10쌍 × 2형식, main = 40쌍 × 2형식 + Tom Swifty 10쌍."""
    S = load_samples()
    out = []
    if stage == "pilot":
        for p in S["pilot"]:
            out += [dict(p, cell=f"{p['pair_id']}:{f}", format=f) for f in ("riddle", "oneliner")]
    elif stage == "main":
        for p in S["main"]:
            out += [dict(p, cell=f"{p['pair_id']}:{f}", format=f) for f in ("riddle", "oneliner")]
        out += [dict(p, cell=f"{p['pair_id']}:tom_swifty", format="tom_swifty") for p in S["tom_swifty"]]
    else:
        raise ValueError(stage)
    return out


# ---------------------------------------------------------------- 사전
@lru_cache(maxsize=1)
def wordnet():
    import nltk
    path = DATA / "nltk_data"
    nltk.data.path.insert(0, str(path))
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", download_dir=str(path), quiet=True)
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
    return wn


WN_POS = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective"}


def gloss(word: str):
    """WordNet에서 word가 표제어 이름 그대로(소문자) 들어 있는 첫 뜻(명·동·형). 없으면 None.

    반환: (품사, 뜻풀이). 대문자 표제어(고유명사)와 굴절형(morphy로만 찾히는 형태)은 걸러진다.
    """
    wn = wordnet()
    for s in wn.synsets(word):
        if s.pos() in WN_POS and word in s.lemma_names():
            return WN_POS[s.pos()], s.definition()
    return None


BLOCK_DOMAINS = {"obscenity.n.01", "obscenity.n.02", "vulgarism.n.01", "ethnic_slur.n.01", "disparagement.n.01"}
BLOCK_GLOSS = re.compile(r"(obscene|vulgar|offensive|derogatory|disparaging|slur|taboo)", re.I)
BLOCK_WORDS = {"whore", "slut", "pussy", "twat", "retard", "douche", "douchebag", "jackass", "bitch", "tits",
               "cock", "dildo", "porn", "porno", "rape", "rapist", "nazi", "hooker", "skank", "wank", "boob", "boobs"}


def offensive(word: str) -> bool:
    """비속어·비하어 거르기: WordNet 용법 표시, 뜻풀이의 obscene·derogatory 등, 짧은 수동 목록."""
    if word in BLOCK_WORDS:
        return True
    for s in wordnet().synsets(word):
        if {u.name() for u in s.usage_domains()} & BLOCK_DOMAINS or BLOCK_GLOSS.search(s.definition()):
            return True
    return False


# ---------------------------------------------------------------- 프롬프트
SYSTEM = "You are a comedy writer who specializes in puns."
JSON_LIST = ('Return only JSON in this form, with no other text: {"puns": [{"text": "...", "pun_word": "...", '
             '"format": "%s"}, ...]}. "pun_word" is the exact form of "%s" as written in "text".')


def pair_lines(c: dict) -> list[str]:
    return [f'Use the word "{c["w_p"]}" ({c["pos_p"]}: {c["gloss_p"]}) so that it also makes the reader think of '
            f'the similar-sounding word "{c["w_a"]}" ({c["pos_a"]}: {c["gloss_a"]}).',
            f'- The text must contain "{c["w_p"]}" (an inflected form is fine).',
            f'- Do not write "{c["w_a"]}" or any form of it anywhere in the text. The reader should hear it, '
            "not see it.",
            f"- At most {MAX_WORDS} words."]


def prompt_p(c: dict) -> str:
    """P1~P4의 후보 10개 생성 프롬프트(쌍 + 뜻 + 형식 틀 + 한 줄 상황형이면 연결 단어)."""
    f = FORMATS[c["format"]]
    L = [f"Write {N_CAND} different puns. Each one must be a {f['name']}.", ""] + pair_lines(c)
    L.append(f"- Format: {f['desc']}.")
    if c["format"] == "oneliner" and c.get("link_words"):
        L.append(f'- Topic words that can make the "{c["w_p"]}" meaning fit the situation (use any of them): '
                 + ", ".join(c["link_words"]) + ".")
    L += ["", JSON_LIST % (c["format"], c["w_p"])]
    return "\n".join(L)


def prompt_b1(c: dict) -> str:
    """B1: 쌍 + 뜻만, 1개."""
    L = ["Write one pun.", ""] + pair_lines(c)
    L += ["", 'Return only JSON in this form, with no other text: {"text": "...", "pun_word": "..."}. '
              f'"pun_word" is the exact form of "{c["w_p"]}" as written in "text".']
    return "\n".join(L)


def prompt_b0() -> str:
    """B0: "말장난 하나 써줘"만. 성공 판정에 필요한 pun_word·alt_word만 JSON으로 함께 받는다."""
    return ("Write one pun.\n\n"
            'Return only JSON in this form, with no other text: {"text": "...", "pun_word": "...", "alt_word": "..."}. '
            '"pun_word" is the word in the text that carries the pun, and "alt_word" is the different word it '
            "sounds like.")


def prompt_pick(texts: list[str]) -> str:
    """P3: 같은 생성기가 후보 중 가장 웃긴 것을 고른다."""
    L = ["Here are some puns, numbered.", ""] + [f"{i}. {t}" for i, t in enumerate(texts, 1)]
    L += ["", 'Which one is the funniest? Return only JSON: {"choice": <number>}.']
    return "\n".join(L)


def extract_json(s: str):
    """출력에서 첫 JSON 값(객체·배열)을 꺼낸다. 코드 블록·앞뒤 말은 무시. 실패하면 None."""
    s = re.sub(r"```(?:json)?", "", s)
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "{[":
            try:
                return dec.raw_decode(s[i:])[0]
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------- 규칙 검사
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SAID_RE = re.compile(r"\bTom\b[^\"“”]*\b(said|says|asked|asks|replied|exclaimed|cried|shouted|whispered|"
                     r"remarked|answered|admitted|announced|muttered|added|declared|sighed|noted|explained)\b|"
                     r"\b(said|asked|replied|exclaimed|cried|shouted|whispered|remarked|answered|admitted|"
                     r"announced|muttered|added|declared|sighed|noted|explained)\s+Tom\b", re.I)


def clean_text(s) -> str:
    """생성물 정리: 마크다운 강조(*단어*, **단어**, _단어_)를 걷어내고 공백을 하나로 줄인다."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"(?<![A-Za-z0-9])_+|_+(?![A-Za-z0-9])", "", s)
    return re.sub(r"\s+", " ", s).strip()


def find_word(text: str, word: str):
    """text에서 word(대소문자 무시, 단어 경계)의 첫 출현 구간."""
    m = re.search(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", text, re.I)
    return m.span() if m else None


def format_ok(text: str, fmt: str | None) -> bool:
    if fmt is None:
        return True
    qa = re.search(r"\?\s*\S", text) is not None
    if fmt == "riddle":
        return qa
    if fmt == "oneliner":
        return not qa
    if fmt == "tom_swifty":
        return bool(re.search(r"[\"“”]", text)) and SAID_RE.search(text) is not None
    raise ValueError(fmt)


def check(text, pun_word, w_p: str, w_a: str, fmt: str | None) -> dict:
    """규칙 검사: w_p 포함(표제어), w_a 미포함(표제어, 하이픈 쪼갬 포함), 40단어 이하, 형식 일치.

    반환: 항목별 통과 여부, 위치(start, end), 탈락 이유 목록. Tom Swifty는 w_p에서 만든 -ly 부사도 받는다.
    """
    from rules import lemmas
    out = dict(ok=False, reasons=[], start=None, end=None)
    if not isinstance(text, str) or not text.strip() or not isinstance(pun_word, str) or not pun_word.strip():
        out["reasons"].append("malformed")
        return out
    text, pw = text.strip(), pun_word.strip()
    words = [w.lower() for w in WORD_RE.findall(text)]
    # 하이픈으로 쪼개 쓴 w_a도 잡는다(page-ant → pageant)
    words_nh = [w.lower() for w in WORD_RE.findall(re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", text))]
    lem = lemmas(set(words) | set(words_nh) | {w_p, w_a, pw.lower()})
    span = find_word(text, pw)
    pw_l = pw.lower()
    wp_match = lem[pw_l] == lem[w_p] or pw_l == w_p or (fmt == "tom_swifty" and pw_l.startswith(w_p[:-1])
                                                       and pw_l.endswith("ly"))
    if span is None or not wp_match:
        out["reasons"].append("no_wp")
    if any(lem[w] == lem[w_a] or w == w_a for w in set(words) | set(words_nh)):
        out["reasons"].append("has_wa")
    if len(text.split()) > MAX_WORDS:
        out["reasons"].append("too_long")
    if not format_ok(text, fmt):
        out["reasons"].append("format")
    out["ok"] = not out["reasons"]
    if span:
        out["start"], out["end"] = span
    return out
