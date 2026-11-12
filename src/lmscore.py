"""M4 언어 모델 채점: 후보를 빈칸에 넣은 문장의 로그 확률.

전역 점수 ℓ_g는 문장 전체, 국소 점수 ℓ_l은 앞뒤 2단어 구간(둘 다 문장 시작 토큰 포함).
원문의 대소문자와 앞 공백은 그대로 둔다. 한 문장의 후보는 한 배치로 채점하고,
전역 문맥에서는 후보 앞부분(접두부)을 한 번만 계산해 KV 캐시를 후보 수만큼 복제한다.
"""
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

MODELS = {
    "pythia-160m": "EleutherAI/pythia-160m-deduped",
    "pythia-410m": "EleutherAI/pythia-410m-deduped",
    "pythia-1.4b": "EleutherAI/pythia-1.4b-deduped",
    "pythia-2.8b": "EleutherAI/pythia-2.8b-deduped",
    "olmo2-1b": "allenai/OLMo-2-0425-1B",
    "olmo2-7b": "allenai/OLMo-2-1124-7B",
}


class LMScorer:
    def __init__(self, name: str, device: str | None = None, dtype: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name
        repo = MODELS.get(name, name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = "float16" if self.device == "cuda" else "float32"
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=getattr(torch, dtype)).to(self.device).eval()
        bos = self.tok.bos_token_id
        self.bos = bos if bos is not None else self.tok.eos_token_id
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.bos

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @torch.inference_mode()
    def _batch_logprobs(self, seqs: list[list[int]], past=None, prev_logits=None):
        """seqs 각각의 토큰 로그 확률 합. past가 있으면 seqs는 그 뒤에 이어지는 토큰이다.

        prev_logits: past의 마지막 위치 로짓(첫 토큰 확률용), past와 함께 준다.
        """
        B = len(seqs)
        L = max(len(s) for s in seqs)
        x = torch.full((B, L), self.pad, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        x, mask = x.to(self.device), mask.to(self.device)
        if past is None:
            out = self.model(input_ids=x, attention_mask=mask)
            logits = out.logits[:, :-1].float()
            tgt, tmask = x[:, 1:], mask[:, 1:]
        else:
            plen = past.get_seq_length()
            full_mask = torch.cat([torch.ones((B, plen), dtype=torch.long, device=self.device), mask], 1)
            out = self.model(input_ids=x, attention_mask=full_mask, past_key_values=past)
            logits = torch.cat([prev_logits.expand(B, 1, -1), out.logits[:, :-1]], 1).float()
            tgt, tmask = x, mask
        lp = torch.gather(F.log_softmax(logits, -1), 2, tgt.unsqueeze(-1)).squeeze(-1)
        return (lp * tmask).sum(1).cpu().tolist()

    def score_texts(self, texts: list[str], batch_size: int = 128):
        """문장 시작 토큰 뒤에 이어지는 각 text의 (로그 확률, 토큰 수)."""
        seqs = [[self.bos] + self.ids(t) for t in texts]
        out = []
        for i in range(0, len(seqs), batch_size):
            out += self._batch_logprobs(seqs[i : i + batch_size])
        return list(zip(out, [len(s) - 1 for s in seqs]))

    @torch.inference_mode()
    def score_shared_prefix(self, prefix: str, suffixes: list[str], batch_size: int = 128):
        """prefix + suffix_i 각각의 (로그 확률, 토큰 수). 접두부는 한 번만 계산한다.

        따로 토큰화한 결과가 통째 토큰화와 다르면 그 항목만 통째로 채점한다.
        """
        pre = [self.bos] + self.ids(prefix)
        sufs = [self.ids(s) for s in suffixes]
        ok = [pre + s == [self.bos] + self.ids(prefix + suf) for s, suf in zip(sufs, suffixes)]
        res = [None] * len(suffixes)
        out = self.model(input_ids=torch.tensor([pre], device=self.device), use_cache=True)
        pre_lp = 0.0
        if len(pre) > 1:
            lg = F.log_softmax(out.logits[0, :-1].float(), -1)
            pre_lp = float(lg.gather(1, torch.tensor(pre[1:], device=self.device)[:, None]).sum())
        last = out.logits[:, -1:]
        idx = [i for i in range(len(suffixes)) if ok[i] and sufs[i]]
        for b in range(0, len(idx), batch_size):
            chunk = idx[b : b + batch_size]
            past = _repeat_cache(out.past_key_values, len(chunk))
            lps = self._batch_logprobs([sufs[i] for i in chunk], past=past, prev_logits=last)
            for i, lp in zip(chunk, lps):
                res[i] = (pre_lp + lp, len(pre) - 1 + len(sufs[i]))
        rest = [i for i in range(len(suffixes)) if res[i] is None]
        if rest:
            for i, r in zip(rest, self.score_texts([prefix + suffixes[i] for i in rest], batch_size)):
                res[i] = r
        return res


def _repeat_cache(cache, n):
    import copy
    c = copy.deepcopy(cache)
    c.batch_repeat_interleave(n)
    return c


def split_at(item):
    """전역 문맥을 접두부와 (후보를 넣을) 접미부로 나눈다. 대상 단어 앞 공백은 접미부로 보낸다."""
    pre = item.text[: item.start]
    lead = pre[len(pre.rstrip(" ")):]
    return pre.rstrip(" "), lead


def score_item(scorer: LMScorer, item, cands: list[str], he_window=True):
    """한 항목·위치의 후보 채점 행 목록 (ctx: global, local, he_local)."""
    from data import match_case
    prefix, lead = split_at(item)
    sur = item.surface
    tail = item.text[item.end :]
    sufs = [lead + match_case(sur, c) + tail for c in cands]
    rows = []
    for c, (lp, n) in zip(cands, scorer.score_shared_prefix(prefix, sufs)):
        rows.append(("global", c, lp, n))
    for c, (lp, n) in zip(cands, scorer.score_texts([item.window(c, 2, 2) for c in cands])):
        rows.append(("local", c, lp, n))
    if he_window and item.w_a:
        pair = [item.w_p, item.w_a]
        for c, (lp, n) in zip(pair, scorer.score_texts([item.window(c, 2, 1) for c in pair])):
            rows.append(("he_local", c, lp, n))
    return rows


class CacheWriter:
    """항목 단위로 쌓다가 주기적으로 parquet 조각을 쓴다. 다시 실행하면 끝난 항목을 건너뛴다."""

    def __init__(self, path: Path, every: int = 50):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.buf = []
        self.n = 0
        self.done = set()
        for f in self.dir.glob("part-*.parquet"):
            d = pd.read_parquet(f, columns=["key"])
            self.done |= set(d.key)

    def add(self, key: str, rows: list[dict]):
        for r in rows:
            r["key"] = key
        self.buf += rows
        self.n += 1
        self.done.add(key)
        if self.n % self.every == 0:
            self.flush()

    def flush(self):
        if self.buf:
            k = len(list(self.dir.glob("part-*.parquet")))
            pd.DataFrame(self.buf).to_parquet(self.dir / f"part-{k:05d}.parquet", index=False)
            self.buf = []

    @staticmethod
    def read(path: Path) -> pd.DataFrame:
        parts = sorted(Path(path).glob("part-*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True) if parts else pd.DataFrame()


def set_threads():
    n = int(os.environ.get("TORCH_THREADS", os.cpu_count() or 4))
    torch.set_num_threads(n)


class Progress:
    def __init__(self, total, label=""):
        self.total, self.label, self.t0, self.i = total, label, time.time(), 0

    def step(self, k=1):
        self.i += k
        if self.i % 25 == 0 or self.i == self.total:
            el = time.time() - self.t0
            eta = el / self.i * (self.total - self.i)
            print(f"[{self.label}] {self.i}/{self.total}  {el/60:.1f}분 경과, 남은 {eta/60:.1f}분", flush=True)
