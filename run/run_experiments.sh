#!/bin/bash
# run_experiments.sh — run_baselines_M.sh + run_smap_then_msl.sh 대체.
# config.py의 DATASETS × MODELS를 순회하며 Run_Detector_M.py를 돌린다.
#
# 사용법:
#   bash run/run_experiments.sh <dataset_key|all> <gpu_id> [model1 model2 ...]
#
# 예시:
#   bash run/run_experiments.sh smd 0                # smd, config의 MODELS 전체, GPU0
#   bash run/run_experiments.sh insdn 1              # insdn, MODELS 전체, GPU1 (smd와 별도 tmux 세션에서 동시 실행 가능)
#   bash run/run_experiments.sh all 0                # smd+insdn 전부, GPU0 (순차)
#   bash run/run_experiments.sh smd 0 PCA RobustPCA  # smd에서 신규 모델 2개만 추가로 (기존 완료분은 Run_Detector_M.py가 자동 skip)

set -e

DATASET_KEY="${1:?사용법: bash run_experiments.sh <dataset_key|all> <gpu_id> [model ...]}"
GPU_ID="${2:?GPU id를 지정하세요 (예: 0)}"
shift 2
MODELS_OVERRIDE="$*"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ "$DATASET_KEY" = "all" ]; then
    DATASETS=$(python3 -c "import config; print(' '.join(config.DATASETS.keys()))")
else
    DATASETS="$DATASET_KEY"
fi

if [ -n "$MODELS_OVERRIDE" ]; then
    MODELS="$MODELS_OVERRIDE"
else
    MODELS=$(python3 -c "import config; print(' '.join(config.MODELS))")
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "GPU=$GPU_ID / DATASETS=[$DATASETS] / MODELS=[$MODELS]"

for ds in $DATASETS; do
    BASE_DIR=$(python3 -c "import config; print(config.DATASETS['$ds']['base_dir'])" 2>/dev/null)
    if [ -z "$BASE_DIR" ]; then
        echo "[$ds] config.DATASETS에 없는 키입니다. 건너뜀."
        continue
    fi

    FILE_LIST=$(ls "$BASE_DIR"/_*_File_List.csv 2>/dev/null | head -n1)
    if [ -z "$FILE_LIST" ]; then
        echo "[$ds] File_List.csv를 못 찾음 ($BASE_DIR) — 먼저 실행하세요:"
        echo "       python prepare/prepare_redlamp_input.py --dataset $ds"
        continue
    fi

    SCORE_DIR="results/$ds/score"
    SAVE_DIR="results/$ds/metrics"
    mkdir -p "$SCORE_DIR" "$SAVE_DIR"

    for model in $MODELS; do
        echo "=== [$ds] $model (GPU $GPU_ID) ==="
        python3 TSB-AD/benchmark_exp/Run_Detector_M.py \
            --dataset_dir "$BASE_DIR" \
            --file_lsit "$FILE_LIST" \
            --score_dir "$SCORE_DIR" \
            --save_dir "$SAVE_DIR" \
            --AD_Name "$model" \
            --save True
    done
done

echo "완료."