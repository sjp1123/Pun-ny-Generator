"""보완 실험(v1) 공용: 분할별 데이터, 문맥별 Listener, E3 특징·교차검증."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data import CACHE, semeval_items
from lmscore import CacheWriter
from rules import SEED, Listener, add_lemma_cols, split_ids

K = 100


def items(split: str | None = None, single=True) -> dict:
    """SemEval 항목. split = "dev"/"test"/None, single이면 숨은 단어 한 단어만."""
    ids = None if split is None else split_ids(split)
    return {it.id: it for it in semeval_items(require_single_word=single) if ids is None or it.id in ids}


def lm_cache(model: str, name: str) -> pd.DataFrame:
    lm = CacheWriter.read(CACHE / "lm" / model / name)
    if lm.empty:
        raise SystemExit(f"LM 캐시 없음: {model}/{name}")
    return lm


def scored(cand: pd.DataFrame, lm: pd.DataFrame, ctx: str) -> pd.DataFrame:
    s = lm[lm.ctx == ctx][["key", "cand", "logprob", "ntok"]].drop_duplicates(["key", "cand"])
    return cand.merge(s, on=["key", "cand"], how="inner", sort=False)


CAND_SETS = {"C_phon": ("semeval_K100",), "C_LM": ("semevalLM_K100",), "C_union": ("semeval_K100", "semevalLM_K100")}
CAND_COLS = ["key", "cand", "rank", "d", "d_norm", "d_unit", "d_unit_norm", "zipf", "in_C", "is_wp", "is_wa"]


def cand_path(table: str, model: str):
    """후보표 경로. 발음 후보(semeval_K100)는 모델과 무관하고, C_LM 후보는 그 모델의 다음 토큰 분포로
    뽑으므로 모델마다 따로 둔다(한 파일을 여러 모델이 같이 쓰면 후보가 섞인다)."""
    return CACHE / (f"candidates_{table}_{model}.parquet" if "LM_" in table else f"candidates_{table}.parquet")


def candidate_set(name: str, model: str) -> pd.DataFrame:
    """후보 집합. C_union은 두 표를 합치고 in_C는 어느 쪽이든 원래 후보였으면 참."""
    parts = [pd.read_parquet(cand_path(n, model))[CAND_COLS] for n in CAND_SETS[name]]
    df = pd.concat(parts, ignore_index=True)
    inC = df.groupby(["key", "cand"]).in_C.any()
    df = df.drop_duplicates(["key", "cand"]).copy()
    df["in_C"] = pd.MultiIndex.from_frame(df[["key", "cand"]]).map(inC).to_numpy(bool)
    return df


class Data:
    """한 분할의 후보표(표제어 열 포함)와 문맥별 점수. 표제어가 w_p와 같은 항목은 뺀다."""

    def __init__(self, model: str, split: str | None, cand_set="C_phon", extra_lm=()):
        self.items = items(split)
        cand = candidate_set(cand_set, model)
        if cand_set != "C_phon":
            extra_lm = tuple(extra_lm) + ("semevalLM_K100",)
        cand = add_lemma_cols(cand[cand.key.isin(self.items)])
        self.excluded = sorted(set(cand[cand.same_lemma].key))
        self.cand = cand[~cand.same_lemma].reset_index(drop=True)
        lms = [lm_cache(model, "semeval_K100")] + [lm_cache(model, n) for n in extra_lm]
        self.lm = pd.concat(lms, ignore_index=True).drop_duplicates(["key", "ctx", "cand"])
        done = set.intersection(*[set(x.key) for x in lms])  # 모든 캐시에 있는 항목만
        self.cand = self.cand[self.cand.key.isin(done)].reset_index(drop=True)
        self.keys = np.array(sorted(set(self.cand.key), key=lambda k: int(k.split("_")[1])))

    def listener(self, ctx: str, restrict="C+wa") -> Listener:
        return Listener(scored(self.cand, self.lm, ctx), restrict=restrict)

    def y(self) -> pd.Series:
        return pd.Series({k: self.items[k].meta["score"] for k in self.keys})


def rho(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    return float(spearmanr(x[ok], y[ok]).statistic) if ok.sum() > 2 else np.nan


def cv_predict(X: pd.DataFrame, y: pd.Series, cols, n_folds=10, alpha=1.0) -> pd.Series:
    """10겹 교차검증 릿지(표준화, 결측은 학습 평균). 겹 나누기는 SEED로 고정."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    keys = np.asarray(y.index)
    pred = pd.Series(index=keys, dtype=float)
    for tr, te in KFold(n_folds, shuffle=True, random_state=SEED % (2 ** 32)).split(keys):
        A, Bm = X.loc[keys[tr], cols].to_numpy(float), X.loc[keys[te], cols].to_numpy(float)
        mu, sd = np.nanmean(A, 0), np.nanstd(A, 0)
        sd[sd == 0] = 1
        f = lambda M: np.nan_to_num((M - mu) / sd)
        pred[keys[te]] = Ridge(alpha=alpha).fit(f(A), y[keys[tr]].to_numpy()).predict(f(Bm))
    return pred
