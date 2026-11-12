"""E5 규칙 검사·JSON 읽기·표본 구성 테스트."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from e5 import cells, check, clean_text, extract_json, format_ok, human_for_cells, load_samples  # noqa: E402
from e5_novelty import longest_match  # noqa: E402


def test_extract_json():
    assert extract_json('```json\n{"puns": [{"text": "a"}]}\n```') == {"puns": [{"text": "a"}]}
    assert extract_json('Sure! [1, 2] trailing') == [1, 2]
    assert extract_json("no json here") is None


def test_clean_text():
    assert clean_text("A **bold** *packet*ing _x_ snake_case") == "A bold packeting x snake_case"


def test_formats():
    assert format_ok("Why did the cow cross? To moo.", "riddle")
    assert not format_ok("The cow crossed the road.", "riddle")
    assert format_ok("The cow crossed the road.", "oneliner")
    assert format_ok('"I need a match," Tom said lightly.', "tom_swifty")
    assert not format_ok("Tom liked matches.", "tom_swifty")


def test_check_rules():
    ok = check("A packet of rulers? Sure.", "packet", "packet", "pageant", "riddle")
    assert ok["ok"] and ok["start"] == 2
    assert "has_wa" in check("A packet of page-ant fudge? Yes.", "packet", "packet", "pageant", "riddle")["reasons"]
    assert "has_wa" in check("Packets for pageants? Yes.", "packets", "packet", "pageant", "riddle")["reasons"]
    assert "no_wp" in check("A box of rulers? Sure.", "packet", "packet", "pageant", "riddle")["reasons"]
    assert check("Packets galore? Sure.", "Packets", "packet", "pageant", "riddle")["ok"]  # 굴절형은 표제어로 통과
    assert "too_long" in check(" ".join(["word"] * 41) + " packet?", "packet", "packet", "pageant", None)["reasons"]
    assert check('"Quick," Tom said swiftly.', "swiftly", "swift", "thrift", "tom_swifty")["ok"]
    assert check(None, "x", "x", "y", None)["reasons"] == ["malformed"]


def test_longest_match():
    corpus = " the cat sat on the mat today "
    f = lambda q: 1 if f" {q} " in corpus else 0
    assert longest_match("a dog the cat sat on the mat yes".split(), f) == (6, False)
    assert longest_match([], f) == (0, False)
    assert longest_match("the cat zzz".split(), lambda q: None)[1] is True


def test_samples_and_cells():
    S = load_samples()
    assert (len(S["pilot"]), len(S["main"]), len(S["tom_swifty"]), len(S["human"])) == (10, 40, 10, 40)
    words = [w for k in ("pilot", "main", "tom_swifty") for p in S[k] for w in (p["w_p"], p["w_a"])]
    assert len(words) == len(set(words))  # 같은 단어가 두 쌍에 나오지 않음
    assert all(1 <= p["d_unit"] <= 2 and p["cos"] < 0.3 for k in ("pilot", "main", "tom_swifty") for p in S[k])
    C = cells("main")
    assert len(C) == 90 and sum(c["format"] == "tom_swifty" for c in C) == 10
    assert len(cells("pilot")) == 20
    H = human_for_cells(C)
    assert len(H) == 80 and all(H[c["cell"]]["format"] == c["format"] for c in C if c["cell"] in H)
    assert max(list(H.values()).count(h) for h in S["human"]) == 2
