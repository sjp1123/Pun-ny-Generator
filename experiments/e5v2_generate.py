"""E5 v2 생성: 세 단계(S1 → S2 → S3) + 최종 규칙 검사로 칸마다 규칙 통과 후보 10개를 채운다 (docs/E5_V2_DESIGN.md 1~5절).

칸마다 시도 k = 0, 1, … (S1 최대 MAX_S1회, 파일럿 4부터 60):
  S1  w_a 문장 (LLM, 온도 0.9, top_p 0.95, 60토큰, 시드 = 칸 시드 + k) → S1 검사
  S2  w_a → w_p (코드: 대소문자, 관사 a/an, 보호 구간)
  S3  w_p 주제어 넣기 (LLM, 온도 0.7, 60토큰, 실패하면 시드를 바꿔 한 번 더) → S3 검사
  S4  (파일럿 6만, S4_STAGES — 본 실행에서는 뺐다, PREREG_E5 6.14) 두 단어를 다 아는 LLM이 앞부분을 두 뜻이 함께 서는 농담으로 다듬음 → S4 검사.
      S1 검사에 품사 검사도 더한다: S1 문장의 w_a 자리 품사를 w_p가 가질 수 있어야 한다(PREREG_E5 6.13)
  최종 규칙 검사 (칸 안 중복 포함) → 통과하면 후보. 10개가 차면 멈춘다. 상한까지 못 채우면 underfilled.
규칙에 걸린 후보에서 고르는 대체 규칙은 없다(통과 0개 칸은 생성 실패).
기준선: B1 = v1 단일 프롬프트(쌍 + 뜻) 1회, B0 = 쌍 없이 "말장난 하나"(칸 수만큼). 둘 다 v1 디코딩·시스템 프롬프트 그대로.
P3: 같은 생성기에게 칸의 후보(고정 시드로 순서를 섞음)를 보여 주고 가장 웃긴 것을 고르게 한다(온도 0).
이어서 돌릴 수 있다: 끝난 시도·호출은 기록에서 읽고 다시 부르지 않는다.

사용 (GPU):
  python -X utf8 experiments/e5v2_generate.py --stage pilot4 --model allenai/OLMo-2-0325-32B-Instruct --load-4bit
출력: results/e5/<stage>/<생성기>[-4bit]/ attempts.jsonl, baselines.jsonl, candidates.csv, cells.csv, picks.csv, config.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from e5 import SYSTEM, clean_text, find_word, stable_seed, stage_dir
from e5_gen import Generator, tag_of
from e5_generate import Log, parse_one, parse_pick
from e5v2 import (BASE_DECODE, MAX_S1, N_CAND, S1_DECODE, S3_DECODE, S3_TRIES, POS_CHECK_STAGES, S4_DECODE, S4_STAGES, S4_TRIES, b0_pair_reasons,
                  cells_v2, final_check, lemma_of, norm_dup, prompt_b1, prompt_hashes, prompt_pick, prompt_s1, samples_file,
                  pos_fits, prompt_s3, prompt_s4, prompt_text, s1_check, s2_swap, s3_check, s4_check, words)
from rules import dump, git_state

JSON_TRIES = 3


def pun_span(text: str, w_p: str):
    """③에서 w_p(표제어가 같은 단어)의 표면형과 위치."""
    toks = words(text)
    lem = lemma_of(toks + [w_p])
    for t in toks:
        if lem[t.lower()] == lem[w_p.lower()]:
            sp = find_word(text, t)
            if sp:
                return t, sp[0], sp[1]
    return None, None, None


def run_attempt(gen, stage, c, k, seen) -> dict:
    w_p, w_a, fmt = c["w_p"], c["w_a"], c["format"]
    seed1 = stable_seed(stage, "S1", c["cell"]) + k
    t0 = time.time()
    raw1 = gen.chat(prompt_s1(c), seed=seed1, system=None, **S1_DECODE)
    ck1 = s1_check(raw1, w_a, fmt)
    use_s4 = stage in S4_STAGES
    if ck1["ok"] and stage in POS_CHECK_STAGES and c.get("pos_set_p") and not pos_fits(ck1["parts"]["text"], w_a, c["pos_set_p"]):
        ck1 = dict(ck1, ok=False, reasons=ck1["reasons"] + ["pos_mismatch"])
    rec = dict(cell=c["cell"], pair_id=c["pair_id"], format=fmt, k=k, seed_s1=seed1, s1_raw=raw1,
               s1_ok=ck1["ok"], s1_reasons=ck1["reasons"], s1_text=(ck1["parts"] or {}).get("text"))
    if not ck1["ok"]:
        return dict(rec, fail_stage="S1", final_ok=False, sec=round(time.time() - t0, 1))
    s2 = s2_swap(ck1["parts"], w_a, w_p, fmt)
    rec.update(s2_text=s2["text"], s2=s2, protected_suffix=s2["protected_suffix"])
    tries, ck3 = [], None
    for t in range(S3_TRIES):
        seed3 = stable_seed(stage, "S3", c["cell"], k, t)
        raw3 = gen.chat(prompt_s3(c, s2), seed=seed3, system=None, **S3_DECODE)
        ck3 = s3_check(raw3, s2, w_p, w_a, fmt, c.get("hints", ()))
        tries.append(dict(t=t, seed=seed3, raw=raw3, ok=ck3["ok"], reasons=ck3["reasons"], edited=ck3["edited"],
                          n_edit=ck3["n_edit"], hint_used=ck3["hint_used"], text=(ck3["parts"] or {}).get("text")))
        if ck3["ok"]:
            break
    rec.update(s3_tries=tries, s3_ok=ck3["ok"], s3_retried=len(tries) > 1)
    if not ck3["ok"]:
        return dict(rec, fail_stage="S3", final_ok=False, sec=round(time.time() - t0, 1))
    parts3 = ck3["parts"]
    rec.update(s3_text=parts3["text"], s3_parts=parts3, edited=ck3["edited"], n_edit=ck3["n_edit"],
               hint_used=ck3["hint_used"], s3_removed=ck3.get("removed"))
    parts = parts3
    if use_s4:
        tries4, ck4 = [], None
        for t in range(S4_TRIES):
            seed4 = stable_seed(stage, "S4", c["cell"], k, t)
            raw4 = gen.chat(prompt_s4(c, s2, parts3), seed=seed4, system=None, **S4_DECODE)
            ck4 = s4_check(raw4, s2, parts3["text"], w_p, w_a, fmt)
            tries4.append(dict(t=t, seed=seed4, raw=raw4, ok=ck4["ok"], reasons=ck4["reasons"], unchanged=ck4["unchanged"],
                               text=(ck4["parts"] or {}).get("text")))
            if ck4["ok"]:
                break
        rec.update(s4_tries=tries4, s4_ok=ck4["ok"], s4_retried=len(tries4) > 1)
        if not ck4["ok"]:
            return dict(rec, fail_stage="S4", final_ok=False, sec=round(time.time() - t0, 1))
        parts = ck4["parts"]
        rec.update(s4_text=parts["text"], s4_parts=parts, s4_unchanged=ck4["unchanged"])
    reasons = final_check(parts["text"], w_p, w_a, fmt, parts, seen)
    rec.update(final_text=parts["text"], final_reasons=reasons, final_ok=not reasons,
               fail_stage="final" if reasons else None, sec=round(time.time() - t0, 1))
    return rec


def call_json(gen, log, cid_base, prompt, parse, seed_parts, meta, system=SYSTEM, temperature=BASE_DECODE["temperature"],
              max_tokens=300):
    """v1과 같은 방식: JSON을 못 읽으면 시드를 바꿔 최대 3번."""
    for a in range(JSON_TRIES):
        cid = f"{cid_base}|a{a}"
        r = log.rec.get(cid)
        if r is None:
            seed = stable_seed(*seed_parts, a)
            raw = gen.chat(prompt, seed=seed, temperature=temperature, max_new_tokens=max_tokens, system=system,
                           top_p=BASE_DECODE["top_p"])
            parsed = parse(raw)
            r = dict(meta, call_id=cid, attempt=a, seed=seed, prompt=prompt, raw=raw, parsed=parsed, ok=parsed is not None)
            log.add(r)
        if r["ok"]:
            return r
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot2", "pilot3", "pilot4", "pilot5", "pilot6", "pilot7", "main"])
    ap.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--load-4bit", action="store_true", help="bitsandbytes 4비트로 올림 (폴더 이름에 -4bit)")
    ap.add_argument("--think", default=None, choices=["true", "false"])
    ap.add_argument("--limit", type=int, default=None, help="앞의 칸 몇 개만 (시험용)")
    ap.add_argument("--max-s1", type=int, default=MAX_S1, help="칸당 S1 최대 횟수 (시험용으로만 줄임, config에 남음)")
    a = ap.parse_args()
    C = cells_v2(a.stage)[: a.limit]
    out = stage_dir(a.stage) / (tag_of(a.model) + ("-4bit" if a.load_4bit else ""))
    out.mkdir(parents=True, exist_ok=True)
    gen = Generator(a.backend, a.model, a.device, think=None if a.think is None else a.think == "true",
                    load_4bit=a.load_4bit)
    dump(out / "config.json", dict(
        stage=a.stage, generator=gen.info, n_cells=len(C), limit=a.limit, n_cand=N_CAND, max_s1=a.max_s1,
        s1_decode=S1_DECODE, s3_decode=S3_DECODE, s3_tries=S3_TRIES, baseline_decode=BASE_DECODE,
        s4=a.stage in S4_STAGES, s4_decode=S4_DECODE, s4_tries=S4_TRIES, s1_pos_check=a.stage in POS_CHECK_STAGES,
        json_tries=JSON_TRIES, pick_temperature=0.0, prompts_sha256=prompt_hashes(),
        samples_file=samples_file(a.stage).name,
        samples_sha256=hashlib.sha256(samples_file(a.stage).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        git=git_state(), argv=sys.argv))
    att = Log(out / "attempts.jsonl", key="key")
    base = Log(out / "baselines.jsonl")
    t0 = time.time()
    cand_rows, cell_rows = [], []
    for i, c in enumerate(C):
        seen, passed, k = set(), [], 0
        while len(passed) < N_CAND and k < a.max_s1:
            key = f"{c['cell']}|k{k}"
            rec = att.rec.get(key)
            if rec is None:
                rec = dict(run_attempt(gen, a.stage, c, k, seen), key=key)
                att.add(rec)
            if rec["final_ok"]:
                seen.add(norm_dup(rec.get("final_text") or rec["s3_text"]))
                passed.append(rec)
            k += 1
        for n, rec in enumerate(passed):
            text3, s2 = rec["s3_text"], rec["s2"]
            text = rec.get("final_text") or text3  # 파일럿 2~5의 기록에는 final_text가 없다(③이 최종)
            surf, s, e = pun_span(text, c["w_p"])
            s3s, s3a, s3b = pun_span(text3, c["w_p"])
            cand_rows.append(dict(cand_id=f"{c['cell']}#{n}", source="P", cell=c["cell"], pair_id=c["pair_id"],
                                  format=c["format"], idx=n, k=rec["k"], text=text, pun_word=surf, start=s, end=e,
                                  text3=text3, pun_word3=s3s, start3=s3a, end3=s3b,
                                  s4_used="s4_text" in rec, s4_unchanged=rec.get("s4_unchanged"),
                                  text2=s2["text"], pun_word2=s2["surface"], start2=s2["start"], end2=s2["end"],
                                  text1=rec["s1_text"], w_p=c["w_p"], w_a=c["w_a"], n_edit=rec["n_edit"],
                                  edited=" ".join(rec["edited"]), hint_used=rec["hint_used"],
                                  s3_retried=rec["s3_retried"], rule_ok=True, reasons=""))
        cell_rows.append(dict(cell=c["cell"], pair_id=c["pair_id"], format=c["format"], n_cand=len(passed),
                              n_s1_calls=k, underfilled=len(passed) < N_CAND, generation_failed=not passed))
        # B1: v1 단일 프롬프트 1회
        call_json(gen, base, f"B1|{c['cell']}", prompt_b1(c), parse_one, (a.stage, "B1", c["cell"]),
                  dict(kind="B1", cell=c["cell"], pair_id=c["pair_id"], format=c["format"]))
        print(f"[{i + 1}/{len(C)}] {c['cell']}  후보 {len(passed)}개 / S1 {k}회  {(time.time() - t0) / 60:.1f}분", flush=True)
    for j in range(len(C)):
        call_json(gen, base, f"B0|{j:03d}", prompt_text("b0_free"),
                  lambda r: parse_one(r, ("text", "pun_word", "alt_word")), (a.stage, "B0", j),
                  dict(kind="B0", cell=None, b0_index=j))
    cand = pd.DataFrame(cand_rows + baseline_rows(C, base))
    cand.to_csv(out / "candidates.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(out / "cells.csv", index=False)

    # P3: 같은 생성기가 고른다 (보여 주는 순서는 칸마다 고정 시드로 섞음)
    picks = []
    for c in C:
        pool = cand[(cand.cell == c["cell"]) & (cand.source == "P")]
        if pool.empty:
            picks.append(dict(cell=c["cell"], cand_id=None, n_pool=0, order=""))
            continue
        order = list(np.random.default_rng(stable_seed(a.stage, "P3order", c["cell"])).permutation(len(pool)))
        ids = [pool.cand_id.iloc[i] for i in order]
        texts = [pool.text.iloc[i] for i in order]
        r = call_json(gen, base, f"P3|{c['cell']}", prompt_pick(texts), lambda raw, n=len(texts): parse_pick(raw, n),
                      (a.stage, "P3", c["cell"]), dict(kind="P3", cell=c["cell"], order=ids), temperature=0.0,
                      max_tokens=100)
        k = r["parsed"]
        picks.append(dict(cell=c["cell"], cand_id=ids[k - 1] if k else None, n_pool=len(pool), order=" ".join(ids),
                          attempts=r["attempt"] + 1))
    pd.DataFrame(picks).to_csv(out / "picks.csv", index=False)
    cs = pd.DataFrame(cell_rows)
    print(f"끝: {out}  ({(time.time() - t0) / 60:.1f}분)  후보 {len(cand_rows)}개, 칸당 S1 호출 중앙값 "
          f"{cs.n_s1_calls.median():.0f}, underfilled {int(cs.underfilled.sum())}/{len(cs)}")
    print(cand.groupby("source").rule_ok.agg(["size", "mean"]).to_string())


def baseline_rows(C, base) -> list[dict]:
    """B1·B0 후보 줄. B1은 최종 규칙 검사(형식 틀이 없는 프롬프트라 모양 검사는 뺌), B0은 + 쌍 유효성."""
    def first_ok(cid):
        rs = [base.rec[k] for k in (f"{cid}|a{t}" for t in range(JSON_TRIES)) if k in base.rec]
        ok = [r for r in rs if r["ok"]]
        return (ok[0] if ok else (rs[-1] if rs else None)), len(rs)

    rows = []
    for c in C:
        r, n = first_ok(f"B1|{c['cell']}")
        x = (r or {}).get("parsed") or {}
        text, pw = clean_text(x.get("text")), clean_text(x.get("pun_word"))
        reasons = final_check(text, c["w_p"], c["w_a"], None) if text else ["malformed"]
        sp = find_word(text, pw) if text and pw else None
        rows.append(dict(cand_id=f"{c['cell']}#B1", source="B1", cell=c["cell"], pair_id=c["pair_id"],
                         format=c["format"], idx=0, text=text, pun_word=pw, start=sp[0] if sp else None,
                         end=sp[1] if sp else None, w_p=c["w_p"], w_a=c["w_a"], rule_ok=not reasons,
                         reasons=",".join(reasons), attempts=n))
    for j in range(len(C)):
        r, n = first_ok(f"B0|{j:03d}")
        x = (r or {}).get("parsed") or {}
        text, pw = clean_text(x.get("text")), clean_text(x.get("pun_word"))
        aw = str(x.get("alt_word") or "").strip().lower()
        pwl = (pw or "").strip().lower()
        reasons = b0_pair_reasons(pwl, aw)
        if not reasons:
            reasons = final_check(text, pwl, aw, None) if text else ["malformed"]
        sp = find_word(text, pw) if text and pw else None
        rows.append(dict(cand_id=f"B0#{j:03d}", source="B0", cell=None, pair_id=None, format=None, idx=j, text=text,
                         pun_word=pw, start=sp[0] if sp else None, end=sp[1] if sp else None, w_p=pwl, w_a=aw,
                         rule_ok=not reasons, reasons=",".join(reasons), attempts=n))
    return rows


if __name__ == "__main__":
    main()
