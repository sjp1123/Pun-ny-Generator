# GPU에서 돌릴 것

- **A. 확인 검정 (prereg-v1)**: 끝 (2026-11-13, `results/pythia-1.4b/confirm/`).
- **B. E5 파일럿 1 (v1 단일 프롬프트)**: 끝 (2026-11-13, `results/e5/pilot/`). 생성기가 말장난을 못 만들어 v2로 고쳤다.
- **C. E5 파일럿 2 (v2 세 단계 생성)**: **지금 돌릴 것.** 사양 `docs/E5_V2_DESIGN.md`, 사전 등록 `PREREG_E5.md` 6절.
  7B와 32B를 둘 다 돌려 통과 기준 C1~C4로 본 실행 생성기를 정한다.

## 0. 준비물

- NVIDIA GPU
  - A만: fp16, 메모리 8GB 이상(Colab T4로 된다)
  - B·C 7B: OLMo-2 7B Instruct bf16 → **24GB 이상**(L4, A10, A100 등). T4(16GB)는 모자란다.
  - C 32B: OLMo-2 32B Instruct bf16 → **80GB**(A100 80GB, H100). 4비트(`--load-4bit`, bitsandbytes)로는 약 20GB라
    24GB GPU에서도 돌지만 40GB 이상이 넉넉하다. 4비트로 돌리면 본 실행도 4비트로 돌려야 한다(사전 등록에 남는다).
- 이 저장소 (`git clone https://github.com/sjp1123/Pun-ny-Generator.git`)
- `gpu_bundle.zip` (깃에 없는 데이터·캐시, 약 9MB). 원래 PC의 저장소 폴더에 있다. 저장소 최상위에서 풀면
  `data/pungen`, `data/gppl/data/pl-humor-full`, `cache/*.parquet`, `cache/kao_cosines.pkl`, `cache/nltk_stub`이 생긴다.
  - 발음 후보표와 표제어 캐시가 들어 있어서 word2vec(1.7GB)과 WordNet은 필요 없다(E5 쌍 목록은 `e5/`에 고정돼 있다).
  - SemEval·UKP 원자료가 들어 있으니 공개 저장소에는 올리지 않는다.

## 1. 설치

표제어 결과가 같아야 하므로 spaCy와 영어 모델 버전을 고정한다. OLMo 2는 transformers 4.47 이상이 필요하다.

```bash
pip install "transformers>=4.47" accelerate pyarrow scikit-learn scipy wordfreq gensim "spacy==3.8.18" cmudict requests pytest choix nltk
pip install --no-deps panphon g2p_en
pip install unicodecsv munkres regex pyyaml inflect distance
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

GPU용 torch 확인: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.
OLMo 2·Qwen 2.5 가중치는 Hugging Face에서 처음 실행할 때 내려받는다(각 약 15GB).

## 2. A — 확인 검정 (저장소 최상위에서)

```bash
unzip gpu_bundle.zip
python -X utf8 -m pytest -q tests
python -X utf8 experiments/build_cache.py --model pythia-1.4b --dataset semeval he --device cuda
python -X utf8 experiments/build_v1_cache.py --model pythia-1.4b --dataset semeval --device cuda
python -X utf8 experiments/v1_confirm.py --model pythia-1.4b
```

- `build_cache.py`: 발음 후보 100개를 전역·국소 문맥으로 채점 → `cache/lm/pythia-1.4b/semeval_K100`, `he_K100`
  (He는 확인 검정에는 안 쓰지만 E5 관문 문턱에 쓴다)
- `build_v1_cache.py`: 읽는 순간(왼쪽 문맥) 점수와 1.4B 자신의 C_LM 후보
- 캐시 단계는 이어서 만들어지므로, 끊겨도 같은 명령을 다시 실행하면 된다. 정밀도는 fp16 기본값(`--dtype` 바꾸지 않음).
- 결과: `results/pythia-1.4b/confirm/` (`table.md`가 가설별 판정표). `config.json`의 `git.dirty`가 `false`인지 확인.

## 3. C — E5 파일럿 4 (32B만, S1 상한 60)

파일럿 2·3은 실패했다(PREREG_E5 6.4, 6.6). 파일럿 3에서 32B는 C2~C4를 넘고 C1에서 한 칸만 모자랐다. 파일럿 4는
S1 상한을 30 → 60회로 올리고(6.7) 같은 10쌍을 새 시드로 32B만 돌린다. 7B는 돌리지 않는다.

```bash
git pull
python -X utf8 -m pytest -q tests
G=olmo-2-0325-32b-instruct-4bit
python -X utf8 experiments/e5v2_generate.py --stage pilot4 --model allenai/OLMo-2-0325-32B-Instruct --load-4bit
python -X utf8 experiments/e5v2_score.py  --stage pilot4 --gen $G --model pythia-1.4b --device cuda
python -X utf8 experiments/e5v2_select.py --stage pilot4 --gen $G --model pythia-1.4b
python -X utf8 experiments/e5v2_judge.py  --stage pilot4 --gen $G --model pythia-1.4b --backend hf --judge Qwen/Qwen2.5-7B-Instruct
```

| 단계 | 하는 일 | 출력 (`results/e5/pilot4/$G/`) |
|---|---|---|
| `e5v2_generate` | 칸 20개마다 S1 → S2 → S3 → 최종 검사로 후보 10개(S1 최대 60회), B1, B0 20개, P3 고르기 | `attempts.jsonl`, `candidates.csv`, `cells.csv`, `baselines.jsonl`, `picks.csv` |
| `e5v2_score` | 후보의 ③과 ②, B1·B0·사람 말장난을 Pythia-1.4B로 채점 | `A_pythia-1.4b/scores.csv`, `candidates.jsonl` |
| `e5v2_select` | P1~P4·B0·B1·H 고르기 | `A_pythia-1.4b/selections.csv`, `selections.jsonl` |
| `e5v2_judge` | (탐색) Qwen 판정 | `A_pythia-1.4b/judge_qwen2.5-7b-instruct/` |

- 끊겨도 같은 명령을 다시 실행하면 끝난 시도부터 이어 간다. `config.json`의 `max_s1`이 60인지, `generator.revision`이 해시인지 확인해 줘.
- 끝나면 `results/e5/pilot4/`를 커밋해서 올려 줘. 보고서는 원래 PC에서
  `python -X utf8 experiments/e5v2_report.py --stage pilot4 --gens olmo-2-0325-32b-instruct-4bit`로 만든다.
- 파일럿 2·3 결과는 `results/e5/pilot2/`, `results/e5/pilot3/`에 그대로 둔다.

## 3b. D — 판정기 검증 (본 실행 전)

파일럿 4는 통과했지만(PREREG_E5 6.8) LLM 판정기가 못 미더워서, 이미 있는 사람 평가 자료로 판정기 후보를 먼저 거른다.
설계와 합격 기준은 [JUDGE_VALIDATION.md](JUDGE_VALIDATION.md). 자료는 `gpu_bundle.zip`에 이미 들어 있다(`data/pungen/MTurk_aggregate_results/`,
`data/gppl/data/pl-humor-full/results.tsv`).

```bash
git pull
python -X utf8 experiments/judge_validate.py --judge Qwen/Qwen2.5-7B-Instruct
python -X utf8 experiments/judge_validate.py --judge Qwen/Qwen2.5-14B-Instruct --load-4bit
python -X utf8 experiments/judge_validate.py --judge microsoft/phi-4 --load-4bit
python -X utf8 experiments/judge_validate.py --judge allenai/OLMo-2-0325-32B-Instruct --load-4bit
```

- 판정기 하나당 호출 약 5,900번(대부분 답이 한 단어). 끊겨도 같은 명령을 다시 실행하면 이어 간다.
- 14B와 phi-4는 새로 내려받는다(각각 약 30GB). 시간이 없으면 첫 줄(7B)과 넷째 줄(이미 받아 둔 32B)만 돌려도 된다.
- 끝날 때마다 화면에 합격 표가 나온다. 결과는 `results/judge_val/<판정기>/`(문장은 들어 있지 않고 항목 id와 답만 있다). 커밋해서 올려 줘.

## 3c. E — 파일럿 4 다시 판정 (검증한 판정기)

판정기 검증 결과([JUDGE_VALIDATION.md](JUDGE_VALIDATION.md) "결과")에 따라 LLM 판정을 phi-4(성립은 쌍을 알려 줌, 웃김은 1~5점)로 바꿨다
(PREREG_E5 6.9). 생성·채점·고르기는 다시 하지 않고 판정만 새로 한다. phi-4는 검증 때 받아 둔 것을 쓴다.

```bash
git pull
python -X utf8 experiments/e5v2_judge2.py --stage pilot4 --gen olmo-2-0325-32b-instruct-4bit
```

- 호출 약 720번(성립 460 + 웃김 260), 답이 한 단어라 몇 분이면 끝난다.
- 결과는 `results/e5/pilot4/olmo-2-0325-32b-instruct-4bit/A_pythia-1.4b/judge2_phi-4-4bit/`. 커밋해서 올려 줘.

## 3d. F — 파일럿 5 (S3 넣기만, 32B)

S3가 앞부분 단어를 바꾸며 w_a 단서를 지우는 문제(PREREG_E5 6.10)를 고쳐, S3가 단어를 넣기만 하게 했다(6.11). 같은 10쌍을 새 시드로 돌린다.

```bash
git pull
python -X utf8 -m pytest -q tests
G=olmo-2-0325-32b-instruct-4bit
python -X utf8 experiments/e5v2_generate.py --stage pilot5 --model allenai/OLMo-2-0325-32B-Instruct --load-4bit
python -X utf8 experiments/e5v2_score.py  --stage pilot5 --gen $G --model pythia-1.4b --device cuda
python -X utf8 experiments/e5v2_select.py --stage pilot5 --gen $G --model pythia-1.4b
python -X utf8 experiments/e5v2_judge2.py --stage pilot5 --gen $G
```

- 옛 판정(`e5v2_judge.py`, Qwen-7B)은 돌리지 않는다.
- 끝나면 `results/e5/pilot5/`를 커밋해서 올려 줘. 판정은 원래 PC에서 `python -X utf8 experiments/e5v2_report.py --stage pilot5 --gens $G --judge2 phi-4-4bit`.

## 3e. G — 파일럿 6 (v4 쌍, S1 품사 검사, S4 다듬기, 32B)

쌍을 다시 골랐고(대표 발음, 사람 말장난만큼 비슷한 소리, 품사) S3 뒤에 두 단어를 다 아는 S4 다듬기를 넣었다(PREREG_E5 6.13).
S1 품사 검사에 spaCy `en_core_web_sm`을 쓴다(1절에서 이미 설치).

```bash
git pull
python -X utf8 -m pytest -q tests
G=olmo-2-0325-32b-instruct-4bit
python -X utf8 experiments/e5v2_generate.py --stage pilot6 --model allenai/OLMo-2-0325-32B-Instruct --load-4bit
python -X utf8 experiments/e5v2_score.py  --stage pilot6 --gen $G --model pythia-1.4b --device cuda
python -X utf8 experiments/e5v2_select.py --stage pilot6 --gen $G --model pythia-1.4b
python -X utf8 experiments/e5v2_judge2.py --stage pilot6 --gen $G
```

- 생성은 S4 호출이 더해져 파일럿 4보다 오래 걸린다.
- 판정기는 ④·③·② 세 문장을 본다(호출 약 1,100번).
- 끝나면 `results/e5/pilot6/`를 커밋해서 올려 줘. S4 A/B 평가지는 원래 PC에서 만들어 연구자가 채운다:
  `python -X utf8 experiments/e5v2_s4_sheet.py make --stage pilot6 --gen $G`

## 4. (끝난 것) B — E5 파일럿 1
 (A의 캐시가 있으면 바로)

```bash
G=olmo-2-1124-7b-instruct
python -X utf8 experiments/e5_gate.py --model pythia-1.4b
python -X utf8 experiments/e5_generate.py --stage pilot --backend hf --model allenai/OLMo-2-1124-7B-Instruct
python -X utf8 experiments/e5_score.py  --stage pilot --gen $G --model pythia-1.4b --device cuda
python -X utf8 experiments/e5_select.py --stage pilot --gen $G --model pythia-1.4b
python -X utf8 experiments/e5_judge.py  --stage pilot --gen $G --model pythia-1.4b --backend hf --judge Qwen/Qwen2.5-7B-Instruct
python -X utf8 experiments/e5_novelty.py --stage pilot --gen $G --model pythia-1.4b
```

| 단계 | 하는 일 | 출력 |
|---|---|---|
| `e5_gate` | γ(SemEval dev)와 A_→ 문턱(He pun vs 오타형, 유든 J). 생성물은 안 봄 | `results/pythia-1.4b/e5_gate/` |
| `e5_generate` | 칸 20개(10쌍 × 2형식)마다 후보 10개 + B1, B0 20개, 규칙 검사, P3 고르기 | `results/e5/pilot/$G/` |
| `e5_score` | 생성물·사람 말장난마다 π_→, π_←, S_ratio, 관문, 점수 | `.../A_pythia-1.4b/scores.csv` |
| `e5_select` | P1~P4·B0·B1·H 고르기와 자동 지표 | `selections.csv`, `table.md` |
| `e5_judge` | (탐색) 다른 계열 LLM의 성공·웃김 판정 | `judge_qwen2.5-7b-instruct/` |
| `e5_novelty` | infini-gram 최장 일치·8단어 오염 표시 (인터넷 필요, 느림) | `novelty_olmo.csv` |

- 생성기와 판정기는 따로 실행되므로 GPU에 한 번에 하나만 올라간다. 모든 단계는 끊겨도 다시 실행하면 이어서 한다.
- 끝나면 `results/e5/pilot/`과 `results/pythia-1.4b/e5_gate/`를 커밋해서 올려 줘. 그걸 보고 프롬프트·관문을 한 번
  고치고 `PREREG_E5.md`를 채워 태그 `prereg-e5`를 단다.
- 본 실행은 태그 뒤에 `--stage main`으로 같은 명령을 돌리고, 사람 평가지는
  `python -X utf8 experiments/e5_human.py export --stage main --gen $G --model pythia-1.4b`로 만든다.
