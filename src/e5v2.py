"""E5 v2 세 단계 생성 공용 (docs/E5_V2_DESIGN.md): 프롬프트 파일, 표본, S1·S3 검사, S2 바꿔 끼우기, 최종 규칙 검사.

각 LLM 단계는 한 단어만 안다: S1은 w_a만, S3는 w_p만. 두 단어를 다 아는 건 이 코드뿐이다.
  S1  w_a가 끝에 오는 평범한 수수께끼/문장 (LLM)          → ①
  S2  ①의 w_a를 w_p로 바꿈 + 대소문자·관사 + 보호 구간 (코드) → ②
  S3  앞부분에 w_p 뜻을 받치는 단어 1~4개 (LLM)            → ③
프롬프트는 e5/v2/prompts/*.txt (줄바꿈을 LF로 맞춘 내용의 SHA-256을 PREREG_E5에 적는다).
"""
import hashlib
import json
import re
from difflib import SequenceMatcher
from functools import lru_cache

from e5 import E5_DIR, MAX_WORDS, clean_text

V2_DIR = E5_DIR / "v2"
PROMPTS = V2_DIR / "prompts"
SAMPLES_V2 = E5_DIR / "samples_v2.json"  # 파일럿 2 (기록)
SAMPLES_V3 = E5_DIR / "samples_v3.json"  # 파일럿 3·본 실행 (거리 0~1 쌍, PREREG_E5 6.5)
EXAMPLE_WORDS = ("bagel", "beagle", "bread", "bed")  # 프롬프트 예시 쌍 (표본에서 뺀다)
FORMATS_V2 = ("riddle", "oneliner")

N_CAND = 10
MAX_S1 = 60  # 파일럿 2·3은 30 (PREREG_E5 6.7)
S1_DECODE = dict(temperature=0.9, top_p=0.95, max_new_tokens=60)
S3_DECODE = dict(temperature=0.7, top_p=1.0, max_new_tokens=60)  # 사양에 top_p가 없어 v1 기본값(1.0)
S3_TRIES = 2
BASE_DECODE = dict(temperature=0.9, top_p=1.0)  # B1·B0 (v1 그대로)
LIMITS = dict(riddle_q=15, riddle_a=8, oneliner=25, edit_min=1, edit_max=4,
              riddle_a_bare=3)  # 답이 이 단어 수 이하면 끝 문장부호가 없어도 잘림으로 보지 않는다 (6.5)
S1_POSITION = dict(riddle=2, oneliner=3)  # w_a가 들어 있어야 하는 끝 단어 수 (riddle은 답 기준)

PROMPT_NAMES = ("s1_riddle", "s1_oneliner", "s3_riddle", "s3_oneliner", "b1_single", "b0_free",
                "pick", "judge_success", "judge_works", "judge_funny")


# ---------------------------------------------------------------- 프롬프트
def prompt_text(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")


def prompt_hashes() -> dict:
    return {n: hashlib.sha256(prompt_text(n).encode("utf-8")).hexdigest() for n in PROMPT_NAMES}


def render(name: str, **kw) -> str:
    """{이름} 자리만 한 번에 바꾼다(JSON 예시의 중괄호와 값 안의 중괄호는 건드리지 않음)."""
    return re.sub(r"\{(\w+)\}", lambda m: str(kw[m.group(1)]) if m.group(1) in kw else m.group(0), prompt_text(name))


def prompt_s1(c: dict) -> str:
    return render(f"s1_{c['format']}", w_a=c["w_a"], g_a=c["gloss_a"])


def prompt_s3(c: dict, s2: dict) -> str:
    kw = dict(w_p=c["w_p"], g_p=c["gloss_p"], hints=", ".join(c["hints"]))
    if c["format"] == "riddle":
        return render("s3_riddle", question=s2["q"], answer=s2["a"], **kw)
    return render("s3_oneliner", sentence=s2["sentence"], protected_suffix=s2["protected_suffix"], **kw)


def prompt_b1(c: dict) -> str:
    return render("b1_single", w_p=c["w_p"], w_a=c["w_a"], pos_p=c["pos_p"], g_p=c["gloss_p"],
                  pos_a=c["pos_a"], g_a=c["gloss_a"])


def prompt_pick(texts: list[str]) -> str:
    return render("pick", items="\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1)))


# ---------------------------------------------------------------- 표본
def load_samples_v2() -> dict:
    return json.loads(SAMPLES_V2.read_text(encoding="utf-8"))


def load_samples_v3() -> dict:
    return json.loads(SAMPLES_V3.read_text(encoding="utf-8"))


def samples_file(stage: str):
    return SAMPLES_V2 if stage == "pilot2" else SAMPLES_V3


def cells_v2(stage: str) -> list[dict]:
    """pilot2 = 파일럿 2 10쌍(samples_v2, 기록), pilot3·pilot4·pilot5 = 파일럿 3 10쌍(4·5는 같은 쌍을 새 시드로),
    main = 본 실행 40쌍 (samples_v3) × 2형식.
    Tom Swifty 형식은 아직 정하지 않았다."""
    if stage == "pilot2":
        pairs = load_samples_v2()["pilot2"]
    elif stage in ("pilot3", "pilot4", "pilot5"):
        pairs = load_samples_v3()["pilot3"]
    elif stage == "main":
        pairs = load_samples_v3()["main"]
    else:
        raise ValueError(stage)
    return [dict(p, cell=f"{p['pair_id']}:{f}", format=f) for p in pairs for f in FORMATS_V2]


# ---------------------------------------------------------------- 낱말
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
END_RE = re.compile(r"[.!?][\"'”’)\]]*$")
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


def words(s: str) -> list[str]:
    return WORD_RE.findall(s or "")


def exact_re(w: str):
    """주어진 형태 그대로(대소문자 무시). 붙은 글자·아포스트로피·하이픈이 있으면 다른 형태로 본다."""
    return re.compile(r"(?<![A-Za-z'’\-])" + re.escape(w) + r"(?![A-Za-z'’\-])", re.I)


def lemma_of(ws) -> dict:
    from rules import lemmas
    return lemmas({w.lower() for w in ws})


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def norm_dup(s: str) -> str:
    """중복 판정용: 소문자, 구두점 빼기, 공백 하나로."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (s or "").lower())).strip()


# ---------------------------------------------------------------- 모양
def _strip(raw: str) -> str:
    """코드 블록 표시와 마크다운 강조를 걷어내고 줄마다 공백을 정리한다(줄바꿈은 유지: Q/A 두 줄)."""
    s = re.sub(r"```\w*", "", raw or "")
    return "\n".join(clean_text(ln) for ln in s.splitlines()).strip()


def parse_riddle(raw: str):
    """'Q: …' 한 줄과 'A: …' 한 줄 (이 순서, 다른 줄 없음) → (q, a). 아니면 None."""
    lines = [ln.strip() for ln in _strip(raw).splitlines() if ln.strip()]
    if len(lines) != 2:
        return None
    mq, ma = re.match(r"^Q\s*:\s*(.+)$", lines[0], re.I), re.match(r"^A\s*:\s*(.+)$", lines[1], re.I)
    return (mq.group(1).strip(), ma.group(1).strip()) if mq and ma else None


def parse_sentence(raw: str):
    lines = [ln.strip() for ln in _strip(raw).splitlines() if ln.strip()]
    if len(lines) != 1:
        return None
    s = lines[0]
    if len(s) >= 2 and s[0] in "\"“" and s[-1] in "\"”":
        s = s[1:-1].strip()
    return s


def shape_reasons(parts: dict, fmt: str) -> list[str]:
    """형식 모양과 잘림. riddle: Q는 ?로 끝, A는 문장부호로 끝. oneliner: 한 문장, ? 없음, . 또는 !로 끝."""
    out = []
    if fmt == "riddle":
        q, a = parts["q"], parts["a"]
        if not re.search(r"\?[\"'”’)]*$", q):
            out.append("shape")
        if not END_RE.search(a) and len(words(a)) > LIMITS["riddle_a_bare"]:
            out.append("truncated")
    else:
        s = parts["sentence"]
        if "?" in s or re.search(r"[.!?][\"'”’)]*\s+\S", s):
            out.append("shape")
        if not END_RE.search(s):
            out.append("truncated")
    return out


def s3_body(raw: str) -> str:
    """S3 응답에 프롬프트 예시처럼 'After:' 표시가 있으면 마지막 'After:' 뒤만 결과로 읽는다 (6.5).
    7B가 'Before: … / After: …'를 따라 써서 한 줄형 S3 실패의 71%가 모양 실패였다."""
    s = _strip(raw)
    m = list(re.finditer(r"(?im)^After\s*:[ \t]*", s))
    return s[m[-1].end():] if m else raw


def parse_parts(raw: str, fmt: str):
    if fmt == "riddle":
        qa = parse_riddle(raw)
        return None if qa is None else dict(q=qa[0], a=qa[1], text=f"{qa[0]} {qa[1]}")
    s = parse_sentence(raw)
    return None if s is None else dict(sentence=s, text=s)


# ---------------------------------------------------------------- S1 검사
def s1_check(raw: str, w_a: str, fmt: str) -> dict:
    """모양, w_a 정확히 1회(주어진 형태 그대로, 다른 굴절형 없음), 위치(끝 단어), 길이, 잘림."""
    parts = parse_parts(raw, fmt)
    if parts is None:
        return dict(ok=False, reasons=["shape"], parts=None)
    reasons = shape_reasons(parts, fmt)
    text = parts["text"]
    n_exact = len(exact_re(w_a).findall(text))
    lem = lemma_of(words(text) + [w_a])
    other_forms = [t for t in words(text) if t.lower() != w_a.lower() and lem[t.lower()] == lem[w_a.lower()]]
    if n_exact != 1 or other_forms:
        reasons.append("wa_count")
    tail = words(parts["a"] if fmt == "riddle" else parts["sentence"])[-S1_POSITION[fmt]:]
    if w_a.lower() not in [t.lower() for t in tail]:
        reasons.append("wa_position")
    if fmt == "riddle":
        if len(words(parts["q"])) > LIMITS["riddle_q"] or len(words(parts["a"])) > LIMITS["riddle_a"]:
            reasons.append("length")
    elif len(words(parts["sentence"])) > LIMITS["oneliner"]:
        reasons.append("length")
    return dict(ok=not reasons, reasons=reasons, parts=parts)


# ---------------------------------------------------------------- S2 바꿔 끼우기
def starts_with_vowel_sound(word: str) -> bool:
    from g2p import prons
    p = prons(word.lower())
    return bool(p and p[0]) and re.sub(r"\d", "", p[0][0]) in VOWELS


def match_case(src: str, w: str) -> str:
    if len(src) > 1 and src.isupper():
        return w.upper()
    return w[:1].upper() + w[1:] if src[:1].isupper() else w


def swap_in(s: str, w_a: str, w_p: str):
    """s 안의 w_a(한 번)를 w_p로. 바로 앞이 a/an이면 w_p 첫 음소로 고친다. → (새 문자열, 시작, 끝)"""
    m = exact_re(w_a).search(s)
    if m is None:
        raise ValueError(f"{w_a!r} not in {s!r}")
    rep = match_case(m.group(0), w_p)
    before, after = s[:m.start()], s[m.end():]
    art = re.search(r"(?<![A-Za-z'’\-])(an|a)(\s+)$", before, re.I)
    if art:
        new = "an" if starts_with_vowel_sound(w_p) else "a"
        new = match_case(art.group(1), new) if art.group(1)[:1].isupper() else new
        before = before[:art.start(1)] + new + art.group(2)
    return before + rep + after, len(before), len(before) + len(rep)


def protected_from(sentence: str, start: int) -> str:
    """w_p 두 단어 앞부터 끝까지 (공백 단위 토큰)."""
    head = sentence[:start]
    idx = len(head.split()) - (1 if head and not head[-1].isspace() else 0)
    toks = sentence.split()
    return " ".join(toks[max(0, idx - 2):])


def s2_swap(parts: dict, w_a: str, w_p: str, fmt: str) -> dict:
    if fmt == "riddle":
        a2, s, e = swap_in(parts["a"], w_a, w_p)
        q = parts["q"]
        return dict(q=q, a=a2, text=f"{q} {a2}", protected_suffix=a2, surface=a2[s:e],
                    start=len(q) + 1 + s, end=len(q) + 1 + e)
    s2, s, e = swap_in(parts["sentence"], w_a, w_p)
    return dict(sentence=s2, text=s2, protected_suffix=protected_from(s2, s), surface=s2[s:e], start=s, end=e)


# ---------------------------------------------------------------- S3 검사
def wa_leak(text: str, w_a: str) -> bool:
    toks = words(text) + words(text.replace("-", ""))
    lem = lemma_of(toks + [w_a])
    return any(t.lower() == w_a.lower() or lem[t.lower()] == lem[w_a.lower()] for t in toks)


def wp_count(text: str, w_p: str) -> int:
    toks = words(text)
    lem = lemma_of(toks + [w_p])
    return sum(lem[t.lower()] == lem[w_p.lower()] for t in toks)


def edit_words(before: list[str], after: list[str]) -> list[str]:
    """after 쪽에서 바뀌거나 들어간 단어(단어 단위 difflib)."""
    sm = SequenceMatcher(None, before, after, autojunk=False)
    return [w for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag in ("replace", "insert") for w in after[j1:j2]]


def removed_words(before: list[str], after: list[str]) -> list[str]:
    """before 쪽에서 지워지거나 바뀐 단어(단어 단위 difflib). 비어 있으면 before가 after의 부분열이다."""
    sm = SequenceMatcher(None, before, after, autojunk=False)
    return [w for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag in ("replace", "delete") for w in before[i1:i2]]


def s3_check(raw: str, s2: dict, w_p: str, w_a: str, fmt: str, hints=()) -> dict:
    """끝 보존(riddle: 답 전체), 넣기만(②의 앞부분 단어가 모두 순서대로 남음), 넣은 단어 1~4개,
    w_p 표제어 정확히 1회, w_a 누출 없음, 모양.
    넣기만은 파일럿 5부터다(PREREG_E5 6.11): S3가 w_p만 보고 앞부분을 바꾸면 w_a 뜻을 받치던 단어를 지울 수 있다."""
    parts = parse_parts(s3_body(raw), fmt)
    if parts is None:
        return dict(ok=False, reasons=["shape"], parts=None, edited=[], n_edit=None, hint_used=None)
    reasons = shape_reasons(parts, fmt)
    lw = lambda s: [t.lower() for t in words(s)]
    if fmt == "riddle":
        kept = norm_space(parts["a"]) == norm_space(s2["a"])
        head2, head3 = lw(s2["q"]), lw(parts["q"])
    else:
        kept = norm_space(parts["sentence"]).endswith(norm_space(s2["protected_suffix"]))
        n_suf = len(words(s2["protected_suffix"]))
        head2 = lw(s2["sentence"])[:-n_suf] if n_suf else lw(s2["sentence"])
        head3 = lw(parts["sentence"])[:-n_suf] if n_suf else lw(parts["sentence"])
    if not kept:
        reasons.append("suffix_changed")
    edited = edit_words(head2, head3) if kept else []
    if kept and removed_words(head2, head3):
        reasons.append("removed")
    if kept and not (LIMITS["edit_min"] <= len(edited) <= LIMITS["edit_max"]):
        reasons.append("edit_width")
    if wp_count(parts["text"], w_p) != 1:
        reasons.append("wp_count")
    if wa_leak(parts["text"], w_a):
        reasons.append("wa_leak")
    hl = lemma_of(list(hints) + edited)
    hint_used = any(hl[e] in {hl[h.lower()] for h in hints} for e in edited) if edited else False
    return dict(ok=not reasons, reasons=reasons, parts=parts, edited=edited, n_edit=len(edited) if kept else None,
                hint_used=hint_used)


# ---------------------------------------------------------------- 최종 규칙 검사
def final_check(text: str, w_p: str, w_a: str, fmt: str | None, parts: dict | None = None, seen=None) -> list[str]:
    """w_p 정확히 1회·w_a 없음(표제어), 형식 모양(fmt가 있으면), 40단어 이하, 끝 문장부호, 칸 안 중복."""
    reasons = []
    if not text or not str(text).strip():
        return ["malformed"]
    if wp_count(text, w_p) != 1:
        reasons.append("wp_count")
    if wa_leak(text, w_a):
        reasons.append("wa_leak")
    if fmt is not None:
        reasons += [r for r in shape_reasons(parts, fmt) if r not in reasons] if parts else ["shape"]
    elif not END_RE.search(text.strip()):
        reasons.append("truncated")
    if len(words(text)) > MAX_WORDS:
        reasons.append("too_long")
    if seen is not None and norm_dup(text) in seen:
        reasons.append("duplicate")
    return reasons


# ---------------------------------------------------------------- B0 쌍 유효성
@lru_cache(maxsize=None)
def unit_distance(a: str, b: str) -> float:
    from g2p import prons
    from phon import pair_distance
    return pair_distance(prons(a), prons(b), "unit")[0]


def b0_pair_reasons(pun_word: str, alt_word: str) -> list[str]:
    """두 단어 모두 CMUdict에 있음, 음소 편집 거리 ≤ 2, 서로 다른 표제어."""
    from g2p import in_cmu
    pw, aw = (pun_word or "").strip().lower(), (alt_word or "").strip().lower()
    if not (pw.isalpha() and aw.isalpha()):
        return ["pair_malformed"]
    out = []
    if not (in_cmu(pw) and in_cmu(aw)):
        out.append("pair_not_in_cmu")
    elif unit_distance(pw, aw) > 2:
        out.append("pair_too_far")
    lem = lemma_of([pw, aw])
    if lem[pw] == lem[aw]:
        out.append("pair_same_lemma")
    return out


# ---------------------------------------------------------------- 판정기 답
def judge_outcome(answer: str, pun_word: str, w_a: str) -> dict:
    """답의 첫 단어. w_p(말장난 단어)와 표제어가 같으면 되풀이(실패), w_a와 같으면 성공."""
    ws = words(answer or "")
    ans = ws[0].lower() if ws else ""
    if not ans:
        return dict(answer="", echo=False, success=False)
    lem = lemma_of([ans, pun_word, w_a])
    echo = lem[ans] == lem[(pun_word or "").lower()] or ans == (pun_word or "").lower()
    return dict(answer=ans, echo=echo, success=(not echo) and lem[ans] == lem[w_a.lower()])
