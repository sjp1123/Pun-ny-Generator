"""E5 생성 (파이프라인 4~5단계 + P3 고르기). 이어서 돌릴 수 있다(끝난 호출은 건너뜀).

칸마다
  P  : 쌍 + 뜻 + 형식 틀 (+ 한 줄 상황형이면 연결 단어) → 후보 10개 (P1~P4가 같이 쓰는 후보)
  B1 : 쌍 + 뜻만 → 1개
B0 : "Write one pun."만 → Tom Swifty 아닌 칸 수만큼 따로 (쌍이 없으므로 칸에 묶이지 않음)
규칙 검사 뒤 P3: 같은 생성기가 규칙을 통과한 후보 중 가장 웃긴 것을 고른다(온도 0).
JSON을 못 읽으면 시드를 바꿔 최대 3번까지 다시 부른다(시도 횟수를 남김).

사용 (GPU):
  python -X utf8 experiments/e5_generate.py --stage pilot --backend hf --model allenai/OLMo-2-1124-7B-Instruct
CPU 시험:
  python -X utf8 experiments/e5_generate.py --stage pilot --backend ollama --model olmo2:7b --limit 2
출력: results/e5/<stage>/<생성기>/ generations.jsonl, candidates.csv, picks.csv, config.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from e5 import (N_CAND, TEMPERATURE, cells, check, clean_text, extract_json, prompt_b0, prompt_b1, prompt_p, prompt_pick,
                stable_seed, stage_dir)
from e5_gen import Generator, tag_of
from rules import dump, git_state

TRIES = 3
PICK_TEMPERATURE = 0.0


class Log:
    def __init__(self, path: Path):
        self.path = path
        self.rec = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.rec[r["call_id"]] = r

    def add(self, r: dict):
        self.rec[r["call_id"]] = r
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_list(raw):
    obj = extract_json(raw)
    if isinstance(obj, dict):
        obj = obj.get("puns", obj.get("items"))
    if isinstance(obj, list):
        obj = [x for x in obj if isinstance(x, dict)]
        return obj[:N_CAND] if obj else None
    return None


def parse_one(raw, keys=("text", "pun_word")):
    obj = extract_json(raw)
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
    if isinstance(obj, dict) and all(k in obj for k in keys):
        return obj
    return None


def parse_pick(raw, n):
    obj = extract_json(raw)
    try:
        c = int(obj["choice"]) if isinstance(obj, dict) else int(obj)
    except (TypeError, ValueError, KeyError):
        return None
    return c if 1 <= c <= n else None


def call(gen, log, call_id_base, prompt, parse, temperature, max_tokens, seed_parts, meta):
    """같은 호출의 시도 1~3 중 첫 성공을 쓴다. 이미 있으면 다시 부르지 않는다."""
    for a in range(TRIES):
        cid = f"{call_id_base}|a{a}"
        r = log.rec.get(cid)
        if r is None:
            seed = stable_seed(*seed_parts, a)
            t0 = time.time()
            raw = gen.chat(prompt, seed=seed, temperature=temperature, max_new_tokens=max_tokens)
            parsed = parse(raw)
            r = dict(meta, call_id=cid, attempt=a, seed=seed, temperature=temperature, prompt=prompt, raw=raw,
                     parsed=parsed, ok=parsed is not None, sec=round(time.time() - t0, 1))
            log.add(r)
        if r["ok"]:
            return r
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot", "main"])
    ap.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--think", default=None, choices=["true", "false"], help="ollama 추론형 모델의 생각 모드")
    ap.add_argument("--limit", type=int, default=None, help="앞의 칸 몇 개만 (시험용)")
    a = ap.parse_args()
    C = cells(a.stage)[: a.limit]
    out = stage_dir(a.stage) / tag_of(a.model)
    out.mkdir(parents=True, exist_ok=True)
    gen = Generator(a.backend, a.model, a.device, think=None if a.think is None else a.think == "true")
    dump(out / "config.json", dict(stage=a.stage, generator=gen.info, temperature=TEMPERATURE,
                                   pick_temperature=PICK_TEMPERATURE, tries=TRIES, n_cells=len(C), limit=a.limit,
                                   git=git_state(), argv=sys.argv))
    log = Log(out / "generations.jsonl")
    n_b0 = sum(c["format"] != "tom_swifty" for c in C)
    t0 = time.time()
    for i, c in enumerate(C):
        meta = dict(cell=c["cell"], pair_id=c["pair_id"], format=c["format"])
        call(gen, log, f"P|{c['cell']}", prompt_p(c), parse_list, TEMPERATURE, 1500,
             (a.stage, "P", c["cell"]), dict(meta, kind="P"))
        call(gen, log, f"B1|{c['cell']}", prompt_b1(c), parse_one, TEMPERATURE, 300,
             (a.stage, "B1", c["cell"]), dict(meta, kind="B1"))
        print(f"[{i + 1}/{len(C)}] {c['cell']}  {(time.time() - t0) / 60:.1f}분", flush=True)
    for j in range(n_b0):
        call(gen, log, f"B0|{j:03d}", prompt_b0(), lambda r: parse_one(r, ("text", "pun_word", "alt_word")),
             TEMPERATURE, 300, (a.stage, "B0", j), dict(kind="B0", cell=None, pair_id=None, format=None, b0_index=j))
    print(f"B0 {n_b0}개 끝  {(time.time() - t0) / 60:.1f}분", flush=True)

    cand = candidate_table(C, log, n_b0)
    cand.to_csv(out / "candidates.csv", index=False)

    picks = []
    for c in C:
        sub = cand[(cand.cell == c["cell"]) & (cand.source == "P")]
        pool = sub[sub.rule_ok] if sub.rule_ok.any() else sub[sub.text.str.len() > 0]
        if pool.empty:
            picks.append(dict(cell=c["cell"], cand_id=None, pool="none", n_pool=0))
            continue
        texts = pool.text.tolist()
        r = call(gen, log, f"P3|{c['cell']}", prompt_pick(texts), lambda raw, n=len(texts): parse_pick(raw, n),
                 PICK_TEMPERATURE, 100, (a.stage, "P3", c["cell"]),
                 dict(kind="P3", cell=c["cell"], pair_id=c["pair_id"], format=c["format"], pool=pool.cand_id.tolist()))
        k = r["parsed"]
        picks.append(dict(cell=c["cell"], cand_id=pool.cand_id.iloc[k - 1] if k else None,
                          pool="rule_ok" if sub.rule_ok.any() else "all", n_pool=len(pool), attempts=r["attempt"] + 1))
    pd.DataFrame(picks).to_csv(out / "picks.csv", index=False)
    print(f"끝: {out}  ({(time.time() - t0) / 60:.1f}분)")
    print(cand.groupby("source").rule_ok.agg(["size", "mean"]).to_string())


def candidate_table(C, log, n_b0) -> pd.DataFrame:
    """생성물 한 줄 = 후보 하나. 규칙 검사 결과와 pun 위치를 붙인다."""
    rows = []

    def last_ok(base):
        rs = [log.rec[k] for k in (f"{base}|a{t}" for t in range(TRIES)) if k in log.rec]
        ok = [r for r in rs if r["ok"]]
        return (ok[0] if ok else (rs[-1] if rs else None)), len(rs)

    for c in C:
        r, n_try = last_ok(f"P|{c['cell']}")
        items = (r or {}).get("parsed") or []
        for k, x in enumerate(items):
            rows.append(dict(cand_id=f"{c['cell']}#{k}", source="P", cell=c["cell"], pair_id=c["pair_id"],
                             format=c["format"], idx=k, text=x.get("text"), pun_word=x.get("pun_word"),
                             w_p=c["w_p"], w_a=c["w_a"], declared_format=x.get("format"), attempts=n_try))
        r, n_try = last_ok(f"B1|{c['cell']}")
        x = (r or {}).get("parsed") or {}
        rows.append(dict(cand_id=f"{c['cell']}#B1", source="B1", cell=c["cell"], pair_id=c["pair_id"],
                         format=c["format"], idx=0, text=x.get("text"), pun_word=x.get("pun_word"),
                         w_p=c["w_p"], w_a=c["w_a"], attempts=n_try))
    for j in range(n_b0):
        r, n_try = last_ok(f"B0|{j:03d}")
        x = (r or {}).get("parsed") or {}
        pw, aw = str(x.get("pun_word") or "").strip().lower(), str(x.get("alt_word") or "").strip().lower()
        rows.append(dict(cand_id=f"B0#{j:03d}", source="B0", cell=None, pair_id=None, format=None, idx=j,
                         text=x.get("text"), pun_word=x.get("pun_word"), w_p=pw, w_a=aw, attempts=n_try))
    df = pd.DataFrame(rows)
    df["raw_text"] = df.text
    df["text"] = df.text.map(clean_text)
    df["pun_word"] = df.pun_word.map(clean_text)
    res = []
    for r in df.itertuples():
        fmt = r.format if r.source == "P" else None
        single = all(w.isalpha() for w in (r.w_p, r.w_a)) and r.w_p and r.w_a
        if not single:
            res.append(dict(rule_ok=False, reasons="malformed", start=None, end=None))
            continue
        ck = check(r.text, r.pun_word, r.w_p, r.w_a, fmt)
        res.append(dict(rule_ok=ck["ok"], reasons=",".join(ck["reasons"]), start=ck["start"], end=ck["end"]))
    return pd.concat([df, pd.DataFrame(res)], axis=1)


if __name__ == "__main__":
    main()
