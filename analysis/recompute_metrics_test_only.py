# -*- coding: utf-8 -*-
"""
recompute_metrics_test_only.py
--------------------------------
Run_Detector_M.py가 train 구간을 포함해 계산한(leakage 있는) 원본 metrics를
저장된 .npy 점수를 재사용해 test 구간만으로 재계산한다. 재학습 불필요.

사용법:
    python recompute_metrics_test_only.py --dataset smd
    python recompute_metrics_test_only.py --dataset smd --models AutoEncoder USAD  # 일부만
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def recompute(dataset_key: str, models: list):
    if dataset_key not in config.DATASETS:
        raise ValueError(f"알 수 없는 dataset: {dataset_key} (지원: {list(config.DATASETS)})")

    base_dir = Path(config.DATASETS[dataset_key]["base_dir"])  # tsb_ad_input/<ds>
    score_dir = Path(config.RESULTS_DIR) / dataset_key / "score"
    metrics_dir = Path(config.RESULTS_DIR) / dataset_key / "metrics_test_only"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    file_list_paths = list(base_dir.glob("_*_File_List.csv"))
    if not file_list_paths:
        print(f"⚠ {base_dir}에 File_List가 없어 {dataset_key} 스킵 "
              f"(먼저 python prepare/prepare_redlamp_input.py --dataset {dataset_key} 실행)")
        return
    file_names = pd.read_csv(file_list_paths[0])["file_name"].tolist()

    for model_name in models:
        model_score_dir = score_dir / model_name
        rows = []
        for fname in file_names:
            # fname 예: "rate_5/SMD-machine-1-1-RedLamp_tr_28479_1st_56958.csv"
            rate = int(fname.split("/")[0].split("_")[1])
            npy_path = model_score_dir / f"{fname.split('.')[0]}.npy"
            if not npy_path.exists():
                print(f"⚠ 없음(아직 학습 안 됐거나 진행 중): {npy_path}")
                continue

            df = pd.read_csv(base_dir / fname).dropna()
            label = df["Label"].astype(int).to_numpy()
            # 원본 raw feature (첫 번째 컬럼) — Run_Detector_M.py가 slidingWindow
            # 추정에 쓰던 것과 동일한 컬럼. score가 아니라 이걸로 주기를 추정한다.
            raw_feature = df.iloc[:, 0].to_numpy(dtype=float)
            train_index = int(fname.split(".")[0].split("_")[-3])

            score = np.load(npy_path)
            n = min(len(score), len(label), len(raw_feature))
            score, label, raw_feature = score[:n], label[:n], raw_feature[:n]

            test_score = score[train_index:]
            test_label = label[train_index:]
            test_raw_feature = raw_feature[train_index:]

            if len(np.unique(test_label)) < 2:
                print(f"⚠ {fname}: test 구간 라벨이 한 종류뿐, 스킵")
                continue

            sliding_window = int(find_length_rank(test_raw_feature.reshape(-1, 1), rank=1))
            result = get_metrics(test_score, test_label, slidingWindow=sliding_window)
            result["file"] = fname
            result["model"] = model_name
            result["contamination_rate"] = rate / 100
            rows.append(result)

        if rows:
            out_path = metrics_dir / f"{model_name}_test_only.csv"
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"[{dataset_key}] {model_name}: {len(rows)}개 파일 재계산 완료 -> {out_path}")
        else:
            print(f"[{dataset_key}] {model_name}: 재계산할 결과 없음 (npy가 아직 하나도 없음)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(config.DATASETS.keys()), required=True)
    ap.add_argument("--models", nargs="+", default=config.MODELS,
                     help="기본값: config.MODELS 전체. 일부만 재계산하려면 모델명 나열")
    args = ap.parse_args()
    recompute(args.dataset, args.models)


if __name__ == "__main__":
    main()