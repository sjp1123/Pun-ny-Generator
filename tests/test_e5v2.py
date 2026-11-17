"""E5 v2 세 단계 생성: S1·S3 검사, S2 바꿔 끼우기, 최종 검사, B0 쌍 유효성, 고르기 동점, 판정 (docs/E5_V2_DESIGN.md 9절)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import e5  # noqa: E402
from e5v2 import (EXAMPLE_WORDS, b0_pair_reasons, cells_v2, final_check, load_samples_v2, load_samples_v3,  # noqa: E402
                  norm_dup, s3_body,  # noqa: E402
                  parse_parts, prompt_b1, prompt_pick, render, s1_check, s2_swap, s3_check, swap_in)


# 1. S2 바꿔 끼우기
def test_s2_swap_case_article_punct():
    assert swap_in("Bed.", "bed", "bread")[0] == "Bread."
    assert swap_in("She ate a pear.", "pear", "apple")[0] == "She ate an apple."
    assert swap_in("He found an owl.", "owl", "towel")[0] == "He found a towel."
    assert swap_in("A pear!", "pear", "apple")[0] == "An apple!"
    s, a, b = swap_in("all he wanted was to crawl into bed.", "bed", "bread")
    assert s == "all he wanted was to crawl into bread." and s[a:b] == "bread"
    r = s2_swap(dict(q="What kind of dog did she adopt?", a="A beagle."), "beagle", "bagel", "riddle")
    assert r["a"] == "A bagel." and r["protected_suffix"] == "A bagel." and r["text"][r["start"]:r["end"]] == "bagel"
    o = s2_swap(dict(sentence="After a long day at work, all he wanted was to crawl into bed."), "bed", "bread",
                "oneliner")
    assert o["protected_suffix"] == "crawl into bread." and o["sentence"].endswith("into bread.")


# 2. S1 검사
def test_s1_check():
    ok = s1_check("Q: What kind of dog did she adopt from the shelter?\nA: A beagle.", "beagle", "riddle")
    assert ok["ok"], ok["reasons"]
    assert "wa_count" in s1_check("Q: Beagle or beagle?\nA: A beagle.", "beagle", "riddle")["reasons"]
    assert "wa_count" in s1_check("After work he crawled into the beds.", "bed", "oneliner")["reasons"]
    assert "wa_position" in s1_check("Q: What dog?\nA: A beagle, of course, yes.", "beagle", "riddle")["reasons"]
    assert "wa_position" in s1_check("His bed was where he went after work.", "bed", "oneliner")["reasons"]
    assert s1_check("After a long day at work, all he wanted was to crawl into bed.", "bed", "oneliner")["ok"]
    assert "shape" in s1_check("Where do you sleep? In bed.", "bed", "oneliner")["reasons"]
    assert "truncated" in s1_check("After a long day he crawled into bed", "bed", "oneliner")["reasons"]
    assert s1_check("**Q:** What dog?\n**A:** A beagle.", "beagle", "riddle")["ok"]  # 마크다운 강조


# 3. S3 검사
S2_ONE = s2_swap(dict(sentence="After a long day at work, all he wanted was to crawl into bed."), "bed", "bread",
                 "oneliner")


def test_s3_check():
    good = s3_check("After a long day at work in the bakery, all he wanted was to crawl into bread.", S2_ONE, "bread", "bed",
                    "oneliner", ["bakery", "loaf"])
    assert good["ok"], good["reasons"]
    assert good["edited"] == ["in", "the", "bakery"] and good["hint_used"]
    # 넣기만 (6.11): 기존 단어를 바꾸거나 지우면 실패. 문두 대문자만 바뀐 것은 괜찮다
    assert "removed" in s3_check("After a long day at the bakery, all he wanted was to crawl into bread.", S2_ONE, "bread",
                                 "bed", "oneliner")["reasons"]
    assert "removed" in s3_check("After a day at the bakery, all he wanted was to crawl into bread.", S2_ONE, "bread",
                                 "bed", "oneliner")["reasons"]
    assert s3_check("Fresh from baking, after a long day at work, all he wanted was to crawl into bread.", S2_ONE, "bread",
                    "bed", "oneliner")["ok"]
    assert "suffix_changed" in s3_check("After a long day at the bakery, all he wanted was a slice of bread.", S2_ONE,
                                        "bread", "bed", "oneliner")["reasons"]
    assert "edit_width" in s3_check("After a long day at work, all he wanted was to crawl into bread.", S2_ONE, "bread",
                                    "bed", "oneliner")["reasons"]  # 수정 0개
    assert "edit_width" in s3_check("Once upon a sunny bakery morning, all he wanted was to crawl into bread.", S2_ONE,
                                    "bread", "bed", "oneliner")["reasons"]  # 수정 5개 이상
    assert "wa_leak" in s3_check("After a long day making beds, all he wanted was to crawl into bread.", S2_ONE,
                                 "bread", "bed", "oneliner")["reasons"]
    s2r = s2_swap(dict(q="What kind of dog did she adopt from the shelter?", a="A beagle."), "beagle", "bagel", "riddle")
    assert s3_check("Q: What kind of dog did she adopt from the shelter near the bakery?\nA: A bagel.", s2r, "bagel",
                    "beagle", "riddle")["ok"]
    assert "removed" in s3_check("Q: What kind of dog did the baker adopt from the shelter?\nA: A bagel.", s2r, "bagel",
                                 "beagle", "riddle")["reasons"]
    assert "suffix_changed" in s3_check("Q: What did the baker adopt?\nA: A sesame bagel.", s2r, "bagel", "beagle",
                                        "riddle")["reasons"]


# 4. 최종 검사
def test_final_check():
    p = parse_parts("After a long day at the bakery, he crawled into bread.", "oneliner")
    assert final_check(p["text"], "bread", "bed", "oneliner", p) == []
    seen = {norm_dup("After a long day at the bakery he crawled into bread")}
    assert "duplicate" in final_check(p["text"], "bread", "bed", "oneliner", p, seen)
    assert "truncated" in final_check("A loaf of bread walks into", "bread", "bed", None)
    assert "wp_count" in final_check("Bread and more bread.", "bread", "bed", None)


# 5. B0 쌍 유효성
def test_b0_pair():
    assert "pair_not_in_cmu" in b0_pair_reasons("elevator", "evelator")
    assert "pair_too_far" in b0_pair_reasons("sack", "days")
    assert b0_pair_reasons("high", "eye") == []
    assert "pair_same_lemma" in b0_pair_reasons("beds", "bed")
    assert b0_pair_reasons("Shakes", "Shakes-beer") == ["pair_malformed"]


# 6. 고르기: 같은 시드면 P1·P4 결과가 같고, S_ratio가 모두 -1이면 P4는 undefined
def test_select_ties_and_undefined():
    import pandas as pd
    from e5v2_select import select_v2
    C = cells_v2("pilot2")[:2]
    rows = []
    for c in C:
        for k in range(4):
            rows.append(dict(cand_id=f"{c['cell']}#{k}", source="P", cell=c["cell"], pair_id=c["pair_id"],
                             format=c["format"], w_p=c["w_p"], w_a=c["w_a"], rule_ok=True))
    cand = pd.DataFrame(rows)
    sr = [0.5, 0.5, 0.2, -1.0] + [-1.0] * 4
    X = pd.DataFrame(dict(gate=[False] * 8, score=[1, 2, 3, 3, 0, 0, 0, 0], S_ratio=sr), index=cand.cand_id)
    a = select_v2("pilot2", C, cand, X, pd.DataFrame())
    b = select_v2("pilot2", C, cand, X, pd.DataFrame())
    assert a.equals(b)
    p4 = a[a.condition == "P4"].set_index("cell")
    assert p4.loc[C[0]["cell"], "cand_id"] in (f"{C[0]['cell']}#0", f"{C[0]['cell']}#1")
    assert bool(p4.loc[C[1]["cell"], "undefined"]) and pd.isna(p4.loc[C[1]["cell"], "cand_id"])
    p2 = a[a.condition == "P2"].set_index("cell")
    assert bool(p2.loc[C[0]["cell"], "gate_fail"])


# 7. 판정기: w_p를 그대로 답하면 실패
def test_judge_outcome():
    from e5v2 import judge_outcome
    assert judge_outcome("Bagel.", "bagel", "beagle") == dict(answer="bagel", echo=True, success=False)
    assert judge_outcome("beagle", "bagel", "beagle")["success"]
    assert judge_outcome("Beagles", "bagel", "beagle")["success"]
    assert not judge_outcome("", "bagel", "beagle")["success"]


# 프롬프트: B1·B0·P3·웃김 비교는 v1과 글자까지 같다
def test_prompts_match_v1():
    c = e5.cells("pilot")[0]
    assert prompt_b1(c) == e5.prompt_b1(c)
    assert render("b0_free") == e5.prompt_b0()
    assert prompt_pick(["a b", "c d"]) == e5.prompt_pick(["a b", "c d"])
    from e5_judge import prompt_pair
    assert render("judge_funny", a="X", b="Y") == prompt_pair("X", "Y")


# 파일럿 2 표본: 새 10쌍 × 2형식, 파일럿 1·본 실행·Tom Swifty·예시·SemEval과 겹치지 않음
def test_pilot2_samples():
    V = load_samples_v2()
    S = e5.load_samples()
    C = cells_v2("pilot2")
    assert len(C) == 20 and len(V["pilot2"]) == 10
    used = {w for k in ("pilot", "main", "tom_swifty") for p in S[k] for w in (p["w_p"], p["w_a"])}
    new = [w for p in V["pilot2"] for w in (p["w_p"], p["w_a"])]
    assert len(new) == len(set(new)) and not set(new) & used and not set(new) & set(EXAMPLE_WORDS)
    assert all(p["hints"] and p["w_p"] not in p["hints"] for p in V["pilot2"])
    assert set(V["hints_main"]) == {p["pair_id"] for p in S["main"]}
    json.dumps(C)


# 파일럿 3·본 실행 표본(v3): 거리 0·1 반씩, 흔한 뜻 뜻풀이, 파일럿 1·2·예시 단어와 겹치지 않음, 단어 중복 없음
def test_v3_samples():
    V = load_samples_v3()
    sizes = dict(pilot3=10, main=40, tom_swifty=10)
    allw = []
    for split, n in sizes.items():
        ps = V[split]
        assert len(ps) == n
        assert sum(p["d_unit"] == 0 for p in ps) == n // 2 and sum(p["d_unit"] == 1 for p in ps) == n // 2
        assert all(p["gloss_p"] and p["gloss_a"] and p["hints"] and p["w_p"] not in p["hints"] for p in ps)
        allw += [w for p in ps for w in (p["w_p"], p["w_a"])]
    assert len(allw) == len(set(allw))
    old = {w for p in e5.load_samples()["pilot"] for w in (p["w_p"], p["w_a"])}
    old |= {w for p in load_samples_v2()["pilot2"] for w in (p["w_p"], p["w_a"])} | set(EXAMPLE_WORDS)
    assert not set(allw) & old
    assert len(cells_v2("pilot3")) == 20 and len(cells_v2("main")) == 80
    assert [c["cell"] for c in cells_v2("pilot4")] == [c["cell"] for c in cells_v2("pilot3")]


# 6.5 완화: 짧은 답은 끝 문장부호 없이도 받고, S3의 'After:' 뒤만 읽는다
def test_relaxed_rules():
    assert s1_check("Q: What part of a pizza shows a slice?\nA: A sector", "sector", "riddle")["ok"]
    r = s1_check("Q: Where do you keep the bikes?\nA: Out back on the old rack", "rack", "riddle")
    assert "truncated" in r["reasons"]
    assert s3_body("Before: He sat by the bed.\nAfter: He sat by the warm bread.") == "He sat by the warm bread."
    assert s3_body("He sat by the bread.") == "He sat by the bread."
    s2 = s2_swap(dict(sentence="After a long day at work, all he wanted was to crawl into bed."), "bed", "bread",
                 "oneliner")
    raw = ("Before: After a long day at work, all he wanted was to crawl into bread.\n"
           "After: After a long day at work in the bakery, all he wanted was to crawl into bread.")
    assert s3_check(raw, s2, "bread", "bed", "oneliner")["ok"]


# 판정 2판: 조건별 성립·웃김, 판정 못 한 B0은 성립 아님, ②③ 차이와 관문별 값
def test_judge2_summarize():
    import pandas as pd
    from e5v2_judge2 import summarize, targets
    cand = pd.DataFrame([
        dict(cand_id="c#0", source="P", cell="x:riddle", pair_id="x", text="Q? A bagel.", pun_word="bagel", w_a="beagle",
             text2="Q? A bagel.", pun_word2="bagel"),
        dict(cand_id="c#1", source="P", cell="x:riddle", pair_id="x", text="Q2? A bagel.", pun_word="bagel", w_a="beagle",
             text2="Q2? A bagel.", pun_word2="bagel"),
        dict(cand_id="B0#0", source="B0", cell=None, pair_id=None, text="A pun.", pun_word="stellar", w_a="stellar",
             text2=None, pun_word2=None),
    ]).set_index("cand_id")
    S = pd.DataFrame([dict(condition="P1", cell="x:riddle", pair_id="x", cand_id="c#0"),
                      dict(condition="P2", cell="x:riddle", pair_id="x", cand_id="c#1"),
                      dict(condition="B0", cell="x:riddle", pair_id="x", cand_id="B0#0")])
    T = targets(cand, S)
    assert list(T.judgeable) == [True, True, False]  # 쌍의 두 단어가 같은 B0은 판정하지 않는다
    items = T[["cand_id", "source", "cell", "pair_id", "judgeable"]].assign(works=[False, True, None], works2=[False, False, None],
                                                                            funny=[2, 4, 3])
    res, M, bp = summarize(S, items, pd.Series({"c#0": False, "c#1": True}))
    assert res["by_condition"]["B0"]["works"] == 0.0 and res["by_condition"]["P2"]["works"] == 1.0
    assert res["diffs"]["P2-P1"]["works"]["est"] == 1.0 and res["diffs"]["P2-P1"]["funny"]["est"] == 2.0
    assert res["all_P"]["works3"]["est"] == 0.5 and res["all_P"]["s3_effect"]["est"] == 0.5
    assert res["all_P"]["works_gate_pass"]["est"] == 1.0 and res["all_P"]["works_gate_fail"]["est"] == 0.0
    assert res["pairs_with_any_works"] == 1 and res["pairs_all_works"] == 0


# 파일럿 5: pilot5는 파일럿 3 쌍을 쓰고, 보고서는 judge2로 C4를 바꾸고 S3 확인을 더한다
def test_pilot5_stage_and_report(tmp_path):
    import json
    from e5v2_report import apply_judge2
    assert [c["cell"] for c in cells_v2("pilot5")] == [c["cell"] for c in cells_v2("pilot3")]
    (tmp_path / "metrics.json").write_text(json.dumps(dict(all_P=dict(
        works3=dict(est=0.7, lo=0.5, hi=0.9, n=200), s3_effect=dict(est=-0.01, lo=-0.1, hi=0.1, n=200)))), encoding="utf-8")
    R = dict(criteria=dict(C1=dict(passed=True), C2=dict(passed=True), C3=dict(passed=True), C4=dict(passed=False)))
    apply_judge2(R, tmp_path)
    assert R["criteria"]["C4"]["passed"] and not R["criteria"]["S3"]["passed"] and not R["criteria"]["all"]
