#!/bin/bash
# run_experiments_smd_multi.sh — SMD 여러 머신 버전 실험 실행.
# run_experiments.sh(기존 smd/insdn, config.DATASETS 기반)와는 완전히 분리되어
# tsb_ad_input/smd_multi/<machine>/ 밑을 머신별로 순회하며 results/smd_multi/<machine>/
# 에 저장한다 — 기존 results/smd/ 는 절대 안 건드림.
#
# 사용법:
#   bash run/run_experiments_smd_multi.sh <gpu_id> [model1 model2 ...]
#
# 예시:
#   bash run/run_experiments_smd_multi.sh 0                # 전체 머신 x config의 MODELS 전체
#   bash run/run_experiments_smd_multi.sh 0 PCA RobustPCA  # 특정 모델만
#
# 참고: Run_Detector_M.py는 이미 만들어진 .npy가 있으면 건너뛰므로, 머신이
# 나중에 추가돼도 이 스크립트를 다시 돌리면 새 머신분만 새로 계산된다.
# 이 스킵 특성 덕분에, 다른 GPU에서 REVERSE=1로 같은 스크립트를 하나 더 띄우면
# 머신 순서를 거꾸로 돌면서 안전하게 병렬 가속할 수 있다(완료본은 자동 skip,
# 중간 어디선가 두 프로세스가 같은 머신을 살짝 겹쳐 돌 수는 있지만 결과에 영향 없음):
#   REVERSE=1 bash run/run_experiments_smd_multi.sh <다른 gpu_id>

set -e

GPU_ID="${1:?사용법: bash run_experiments_smd_multi.sh <gpu_id> [model ...]}"
shift 1
MODELS_OVERRIDE="$*"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SMD_MULTI_INPUT_DIR=$(python3 -c "import config; print(config.SMD_MULTI_TSB_AD_INPUT_DIR)")
SMD_MULTI_RESULTS_DIR=$(python3 -c "import config; print(config.SMD_MULTI_RESULTS_DIR)")

if [ ! -d "$SMD_MULTI_INPUT_DIR" ]; then
    echo "[smd_multi] $SMD_MULTI_INPUT_DIR 가 없음 — 먼저 실행하세요:"
    echo "       python prepare/prepare_redlamp_input_smd_multi.py"
    exit 1
fi

if [ -n "$MODELS_OVERRIDE" ]; then
    MODELS="$MODELS_OVERRIDE"
else
    MODELS=$(python3 -c "import config; print(' '.join(config.MODELS))")
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "GPU=$GPU_ID / MODELS=[$MODELS]"

# 머신 목록을 배열로 모아서 REVERSE=1이면 순서를 뒤집는다
MACHINE_DIRS=()
for MACHINE_DIR in "$SMD_MULTI_INPUT_DIR"/*/; do
    [ -d "$MACHINE_DIR" ] && MACHINE_DIRS+=("$MACHINE_DIR")
done
if [ "${REVERSE:-0}" = "1" ]; then
    REV=()
    for ((i=${#MACHINE_DIRS[@]}-1; i>=0; i--)); do REV+=("${MACHINE_DIRS[i]}"); done
    MACHINE_DIRS=("${REV[@]}")
    echo "REVERSE=1: 머신 순서를 거꾸로 돕니다"
fi

MACHINE_COUNT=0
for MACHINE_DIR in "${MACHINE_DIRS[@]}"; do
    MACHINE=$(basename "$MACHINE_DIR")

    FILE_LIST=$(ls "$MACHINE_DIR"_*_File_List.csv 2>/dev/null | head -n1)
    if [ -z "$FILE_LIST" ]; then
        echo "[$MACHINE] File_List.csv를 못 찾음 — 건너뜀 (prepare 스크립트가 실패했을 수 있음)"
        continue
    fi

    SCORE_DIR="$SMD_MULTI_RESULTS_DIR/$MACHINE/score"
    SAVE_DIR="$SMD_MULTI_RESULTS_DIR/$MACHINE/metrics"
    mkdir -p "$SCORE_DIR" "$SAVE_DIR"

    for model in $MODELS; do
        echo "=== [smd_multi/$MACHINE] $model (GPU $GPU_ID) ==="
        python3 TSB-AD/benchmark_exp/Run_Detector_M.py \
            --dataset_dir "$MACHINE_DIR" \
            --file_lsit "$FILE_LIST" \
            --score_dir "$SCORE_DIR" \
            --save_dir "$SAVE_DIR" \
            --AD_Name "$model" \
            --save True
    done
    MACHINE_COUNT=$((MACHINE_COUNT + 1))
done

echo "완료. 총 $MACHINE_COUNT개 머신 처리 -> $SMD_MULTI_RESULTS_DIR"