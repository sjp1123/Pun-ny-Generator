# GPU에서 돌릴 것: Pythia-1.4B 확인 검정 (prereg-v1)

GPU에서 돌릴 것은 사전 등록(`PREREG.md`, 태그 `prereg-v1`)의 확인 검정 하나다. 캐시 두 개를 만든 뒤 판정 스크립트를 실행한다.
E5(생성 파이프라인)는 아직 코드가 없어서 돌릴 것이 없다.

## 0. 준비물

- NVIDIA GPU (fp16, 메모리 8GB 이상이면 충분. Colab T4로 된다)
- 이 저장소 (`git clone https://github.com/sjp1123/Pun-ny-Generator.git`)
- `gpu_bundle.zip` (깃에 없는 데이터·캐시, 약 9MB). 원래 PC의 저장소 폴더에 있다. 저장소 최상위에서 풀면
  `data/pungen`, `data/gppl/data/pl-humor-full`, `cache/*.parquet`, `cache/kao_cosines.pkl`이 생긴다.
  - 발음 후보표와 표제어 캐시가 들어 있어서 g2p와 word2vec(1.7GB)는 필요 없다.
  - SemEval·UKP 원자료가 들어 있으니 공개 저장소에는 올리지 않는다.

## 1. 설치

표제어 결과가 같아야 하므로 spaCy와 영어 모델 버전을 고정한다.

```bash
pip install transformers accelerate pyarrow scikit-learn scipy wordfreq gensim "spacy==3.8.18" cmudict requests pytest choix
pip install --no-deps panphon g2p_en
pip install unicodecsv munkres regex pyyaml inflect distance
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

GPU용 torch가 깔려 있는지 확인: `python -c "import torch; print(torch.cuda.is_available())"` → `True`

## 2. 실행 (저장소 최상위에서)

```bash
unzip gpu_bundle.zip
python -X utf8 -m pytest -q tests
python -X utf8 experiments/build_cache.py --model pythia-1.4b --dataset semeval --device cuda
python -X utf8 experiments/build_v1_cache.py --model pythia-1.4b --dataset semeval --device cuda
python -X utf8 experiments/v1_confirm.py --model pythia-1.4b
```

- `build_cache.py`: 발음 후보(C_phon) 100개를 전역·국소 문맥으로 채점 → `cache/lm/pythia-1.4b/semeval_K100`
- `build_v1_cache.py`: 읽는 순간(왼쪽 문맥) 점수와 1.4B 자신의 C_LM 후보 → `cache/lm/pythia-1.4b/semeval_K100_left`,
  `semevalLM_K100`, `cache/candidates_semevalLM_K100_pythia-1.4b.parquet`
- 두 캐시 단계는 이어서 만들어지므로, 중간에 끊겨도 같은 명령을 다시 실행하면 된다.
- 기본 정밀도는 GPU에서 fp16이다(PREREG 고정 설정과 같음). `--dtype`은 바꾸지 않는다.
- `--dataset he`는 확인 검정에 쓰지 않으니 생략해도 된다.

## 3. 결과 가져오기

`v1_confirm.py`가 `results/pythia-1.4b/confirm/`에 `config.json`(시드, 커밋 해시), `metrics.json`, `table.md`를 쓴다.
이 폴더를 그대로 커밋해 올리거나 원래 PC로 복사한다. `table.md`가 가설별 판정표다.
`config.json`의 `git.dirty`가 `false`인지 확인한다(코드를 고치지 않고 돌렸다는 기록).
