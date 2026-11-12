"""Kao 기준선용 word2vec 코사인 캐시 (SemEval 1,098개 + He 194문장)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import kao_cosines
from data import load_he_annotation, semeval_items

if __name__ == "__main__":
    items = semeval_items() + load_he_annotation()[0]
    cos = kao_cosines(items)
    ok = sum(cos[i.id]["ok"] for i in items)
    print(f"{len(items)}개 중 w_p·w_a 모두 word2vec에 있음: {ok}")
