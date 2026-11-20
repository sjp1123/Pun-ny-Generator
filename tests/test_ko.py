"""한국어 K0: 표준 발음(g2pkk 규칙 + Kiwi 품사). Kiwi·g2pkk가 없는 환경(GPU 머신 등)에서는 건너뛴다."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("kiwipiepy")
pytest.importorskip("g2pkk")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ko" / "src"))

CASES = [("같이", "가치"), ("굳이", "구지"), ("국물", "궁물"), ("낮", "낟"), ("닭", "닥"), ("맑다", "막따"), ("신라", "실라"),
         ("독립", "동닙"), ("꽃잎", "꼰닙"), ("밟다", "밥따"), ("값이", "갑씨"), ("넓다", "널따"), ("앉다", "안따"),
         ("해돋이", "해도지"), ("식물", "싱물"), ("학교", "학꾜"), ("좋다", "조타"), ("놓고", "노코"), ("읽고", "일꼬"), ("가치", "가치")]


def test_standard_pronunciation():
    from ko_g2p import pron
    assert [pron(w) for w, _ in CASES] == [p for _, p in CASES]
    assert pron("낮에는 낫으로 풀을 벤다.") == "나제는 나스로 푸를 벤다."


def test_kiwi_tagger_merges_contractions():
    from ko_g2p import KiwiTagger
    toks = KiwiTagger().pos("친구와 같이 했다.")
    assert "".join(t for t, _ in toks) == "친구와같이했다."
    assert ("했", "VV+EP") in toks
