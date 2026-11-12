#!/usr/bin/env bash
# 한 모델로 캐시부터 결과 정리까지. GPU(예: Colab T4)에서 주 모델을 돌릴 때:
#   bash scripts/run_model.sh pythia-1.4b cuda
# 캐시는 이어서 만들어지므로 중간에 끊겨도 다시 실행하면 된다.
set -euo pipefail
MODEL=${1:-pythia-160m}
DEVICE=${2:-}
DEV_ARG=${DEVICE:+--device $DEVICE}
cd "$(dirname "$0")/.."
PY="python -X utf8"
$PY experiments/build_cache.py --model "$MODEL" --dataset semeval he $DEV_ARG
$PY experiments/build_kao.py
[ -f results/e0/counts.csv ] || $PY experiments/e0_contamination.py
$PY experiments/e0_contamination.py --analyze --model "$MODEL"
$PY experiments/e1_recovery.py --model "$MODEL"
$PY experiments/e2_detect.py --model "$MODEL" --part a
$PY experiments/e3_funniness.py --model "$MODEL"
$PY experiments/ablation.py --model "$MODEL"
$PY experiments/hand_check.py --model "$MODEL"
if [ "${E2B:-0}" = "1" ]; then
  $PY experiments/build_cache.py --model "$MODEL" --dataset e2b --k "${E2B_K:-100}" $DEV_ARG
  $PY experiments/e2_detect.py --model "$MODEL" --part b --e2b-k "${E2B_K:-100}"
fi
$PY experiments/report.py --model "$MODEL"
