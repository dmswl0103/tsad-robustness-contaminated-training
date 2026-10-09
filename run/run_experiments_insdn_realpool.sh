#!/bin/bash
# run_experiments_insdn_realpool.sh — insdn_realpool 전용 실행.
# 기존 run_experiments.sh(config.DATASETS 기반)와 분리, results/insdn_realpool/에 저장.
#
# 사용법:
#   bash run/run_experiments_insdn_realpool.sh <gpu_id> [model1 model2 ...]

set -e

GPU_ID="${1:?사용법: bash run_experiments_insdn_realpool.sh <gpu_id> [model ...]}"
shift 1
MODELS_OVERRIDE="$*"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_DIR=$(python3 -c "import config; print(config.INSDN_REALPOOL_TSB_AD_INPUT_DIR)")
RESULTS_DIR=$(python3 -c "import config; print(config.INSDN_REALPOOL_RESULTS_DIR)")

FILE_LIST=$(ls "$BASE_DIR"/_*_File_List.csv 2>/dev/null | head -n1)
if [ -z "$FILE_LIST" ]; then
    echo "[insdn_realpool] File_List.csv를 못 찾음 ($BASE_DIR) — 먼저 실행하세요:"
    echo "       python prepare/prepare_insdn_realpool.py"
    exit 1
fi

if [ -n "$MODELS_OVERRIDE" ]; then
    MODELS="$MODELS_OVERRIDE"
else
    MODELS=$(python3 -c "import config; print(' '.join(config.MODELS))")
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "GPU=$GPU_ID / MODELS=[$MODELS]"

SCORE_DIR="$RESULTS_DIR/score"
SAVE_DIR="$RESULTS_DIR/metrics"
mkdir -p "$SCORE_DIR" "$SAVE_DIR"

for model in $MODELS; do
    echo "=== [insdn_realpool] $model (GPU $GPU_ID) ==="
    python3 TSB-AD/benchmark_exp/Run_Detector_M.py \
        --dataset_dir "$BASE_DIR" \
        --file_lsit "$FILE_LIST" \
        --score_dir "$SCORE_DIR" \
        --save_dir "$SAVE_DIR" \
        --AD_Name "$model" \
        --save True
done

echo "완료."