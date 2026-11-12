"""실험 스크립트 공용: 캐시 읽기, 부트스트랩, 결과 저장."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from candidates import K_DEFAULT
from data import CACHE, RESULTS
from lmscore import CacheWriter

MAIN_MODEL = "pythia-1.4b"   # 확인 검정용 주 모델 (실험 전에 고정)
B_BOOT = 2000
SEED = 0


def parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--model", default="pythia-160m")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    return ap


def load_cache(model: str, dataset: str, K: int = K_DEFAULT):
    path = CACHE / f"candidates_{dataset}_K{K}.parquet"
    lm = CacheWriter.read(CACHE / "lm" / model / f"{dataset}_K{K}")
    if lm.empty or not path.exists():
        raise SystemExit(f"LM 캐시 없음: {model}/{dataset}. build_cache.py를 먼저 돌려.")
    cand = pd.read_parquet(path)
    done = set(lm.key)
    return cand[cand.key.isin(done)].reset_index(drop=True), lm


def out_dir(model: str) -> Path:
    d = RESULTS / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def boot_ci(stat, n: int, B: int = B_BOOT, seed: int = SEED, alpha=0.05):
    """항목 재표집 부트스트랩. stat(idx) → 실수 또는 실수 배열."""
    rng = np.random.default_rng(seed)
    point = np.asarray(stat(np.arange(n)), dtype=float)
    reps = np.array([stat(rng.integers(0, n, n)) for _ in range(B)], dtype=float)
    lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return point, lo, hi


def rho(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def fmt_ci(p, lo, hi, nd=3):
    return f"{p:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def save_json(path: Path, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv), encoding="utf-8")


def seen_flags(model_dir_unused=None) -> pd.Series | None:
    """E0 결과가 있으면 항목별 '본 적 있을 수 있음' 플래그 (어느 색인이든 1회 이상)."""
    f = RESULTS / "e0" / "counts.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f).set_index("key")
    return (df.filter(like="count_").fillna(0) > 0).any(axis=1)
