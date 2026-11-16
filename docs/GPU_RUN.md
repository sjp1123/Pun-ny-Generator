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

## 3. C — E5 파일럿 2 (v2)

```bash
git pull
pip install bitsandbytes   # 32B를 4비트로 돌릴 때만
python -X utf8 -m pytest -q tests
```

생성기마다 아래 네 단계를 돌린다(`--gen`은 생성 단계가 만든 폴더 이름). 7B 먼저:

```bash
G=olmo-2-1124-7b-instruct
python -X utf8 experiments/e5v2_generate.py --stage pilot2 --model allenai/OLMo-2-1124-7B-Instruct
python -X utf8 experiments/e5v2_score.py  --stage pilot2 --gen $G --model pythia-1.4b --device cuda
python -X utf8 experiments/e5v2_select.py --stage pilot2 --gen $G --model pythia-1.4b
python -X utf8 experiments/e5v2_judge.py  --stage pilot2 --gen $G --model pythia-1.4b --backend hf --judge Qwen/Qwen2.5-7B-Instruct
```

그다음 32B. 80GB GPU면 `--load-4bit` 없이, 아니면 붙인다(폴더 이름이 `...-4bit`가 된다):

```bash
G=olmo-2-0325-32b-instruct          # 4비트면 G=olmo-2-0325-32b-instruct-4bit
python -X utf8 experiments/e5v2_generate.py --stage pilot2 --model allenai/OLMo-2-0325-32B-Instruct   # [--load-4bit]
python -X utf8 experiments/e5v2_score.py  --stage pilot2 --gen $G --model pythia-1.4b --device cuda
python -X utf8 experiments/e5v2_select.py --stage pilot2 --gen $G --model pythia-1.4b
python -X utf8 experiments/e5v2_judge.py  --stage pilot2 --gen $G --model pythia-1.4b --backend hf --judge Qwen/Qwen2.5-7B-Instruct
```

| 단계 | 하는 일 | 출력 (`results/e5/pilot2/$G/`) |
|---|---|---|
| `e5v2_generate` | 칸 20개마다 S1 → S2 → S3 → 최종 검사로 후보 10개(S1 최대 30회), B1, B0 20개, P3 고르기 | `attempts.jsonl`, `candidates.csv`, `cells.csv`, `baselines.jsonl`, `picks.csv` |
| `e5v2_score` | 후보의 ③과 ②, B1·B0·사람 말장난을 Pythia-1.4B로 채점(관문 값은 파일럿 1의 e5_gate) | `A_pythia-1.4b/scores.csv`, `candidates.jsonl` |
| `e5v2_select` | P1~P4·B0·B1·H 고르기 | `A_pythia-1.4b/selections.csv`, `selections.jsonl` |
| `e5v2_judge` | (탐색) Qwen 판정: 모든 P 후보의 성공·성립, 칸 안 웃김 비교 | `A_pythia-1.4b/judge_qwen2.5-7b-instruct/` |

- 생성은 칸당 LLM 호출이 최대 90번(S1 30 + S3 60)이지만 한 번에 60토큰이라 짧다. 끊겨도 같은 명령을 다시 실행하면 끝난 시도부터 이어 간다.
- 생성기와 판정기는 따로 올라가므로 GPU에는 한 번에 하나만 있다.
- 끝나면 `results/e5/pilot2/`를 커밋해서 올려 줘(크기는 수 MB). 보고서와 C1~C4 판정(`e5v2_report.py`)은 word2vec이 있는
  원래 PC에서 돌린다. GPU에서 미리 보고 싶으면:
  `python -X utf8 experiments/e5v2_report.py --gens olmo-2-1124-7b-instruct $G --no-w2v`

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
