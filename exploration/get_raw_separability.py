# -*- coding: utf-8 -*-
"""
get_raw_separability.py
--------------------------------
"Exathlon이 baseline에서 거의 만점(ceiling)인 이유가, 이상 구간이 raw
신호 레벨에서부터 그냥 눈에 띄게 크게 튀기 때문(=인위 주입이라 신호가
크다) 아니냐"는 가설을 실측으로 확인하기 위한 스크립트. 학습된 모델은
전혀 쓰지 않고, 아주 단순한 "train 분포 기준 z-score" 만으로 이상을
얼마나 잘 구분해낼 수 있는지를 데이터셋별로 비교한다.

방법:
  1. 각 데이터셋의 원본(raw) train 구간(오염 주입 전, 순수 정상)에서
     채널별 평균/표준편차를 구한다 — 실제 실험에서 scaler를 이 구간으로만
     fit하는 것과 동일한 기준.
  2. test 구간의 각 시점 x에 대해 z = |(x - train_mean) / train_std| 를
     채널별로 계산하고, 그 중 최댓값(row-wise max)을 그 시점의
     "raw anomaly score"로 삼는다 (한 채널이라도 크게 튀면 이상으로
     의심된다는, 모델 없이 만들 수 있는 가장 단순한 탐지기).
  3. 이 raw score만으로 test 라벨을 얼마나 잘 맞히는지 AUC-ROC로 계산한다.
     이 AUC가 이미 높다면(예: 0.9+) 그 데이터셋은 "어떤 모델을 쓰든
     원래 쉬운 문제"라는 뜻이고, baseline이 왜 천장(ceiling)에 붙어있는지
     설명이 된다. 반대로 AUC가 낮으면(0.5~0.6대) 이상이 raw 신호에서는
     안 보이는 미묘한 패턴이라는 뜻 — MSL이 왜 바닥(floor)인지의
     참고 자료가 될 수 있다.
  4. 참고용으로 이상 구간 vs 정상 구간의 z-score 중앙값도 같이 출력.

주의: 이건 "raw 신호가 얼마나 쉽게 분리되는가"만 보는 것이지 실제 실험에
쓴 14개 모델의 성능이 아니다 — 어디까지나 "baseline이 왜 그 구간에
있는지"에 대한 보조 설명 자료.

사용법:
    python get_raw_separability.py
    python get_raw_separability.py --datasets msl exathlon smd swat psm nab power swat_uni
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# dataset_key -> (raw_dir_attr, file_attr) config.py 변수명 매핑
# (get_dataset_stats.py의 RAW_SPECS와 동일)
RAW_SPECS = {
    "smd": None,  # SMD는 별도 처리 (train.csv/test.csv/test_label.csv 구조)
    "msl": ("MSL_RAW_DIR", "MSL_FILE"),
    "smap": ("SMAP_RAW_DIR", "SMAP_FILE"),
    "swat": ("SWAT_RAW_DIR", "SWAT_FILE"),
    "exathlon": ("EXATHLON_RAW_DIR", "EXATHLON_FILE"),
    "psm": ("PSM_RAW_DIR", "PSM_FILE"),
    "nab": ("NAB_RAW_DIR", "NAB_FILE"),
    "power": ("POWER_RAW_DIR", "POWER_FILE"),
    "swat_uni": ("SWAT_UNI_RAW_DIR", "SWAT_UNI_FILE"),
}

EPS = 1e-8


def _score_and_summarize(dataset_key: str, train_feat: np.ndarray,
                          test_feat: np.ndarray, test_label: np.ndarray) -> dict | None:
    train_mean = train_feat.mean(axis=0)
    train_std = train_feat.std(axis=0)
    train_std = np.where(train_std < EPS, EPS, train_std)  # 상수 채널 0-division 방지

    z = np.abs((test_feat - train_mean) / train_std)
    score = z.max(axis=1)  # 시점별 raw anomaly score (채널 중 최댓값)

    if test_label.sum() == 0 or test_label.sum() == len(test_label):
        print(f"⚠ [{dataset_key}] test 라벨이 전부 0 또는 전부 1이라 AUC 계산 불가, 스킵")
        return None

    auc = roc_auc_score(test_label, score)
    med_anomaly = float(np.median(score[test_label == 1]))
    med_normal = float(np.median(score[test_label == 0]))

    return {
        "dataset": dataset_key,
        "raw_zscore_auc": round(float(auc), 4),
        "median_z_anomaly": round(med_anomaly, 3),
        "median_z_normal": round(med_normal, 3),
        "ratio(anomaly/normal)": round(med_anomaly / med_normal, 2) if med_normal > EPS else float("inf"),
    }


def separability_from_raw_file(dataset_key: str) -> dict | None:
    spec = RAW_SPECS.get(dataset_key)
    if spec is None:
        return None
    raw_dir_attr, file_attr = spec
    raw_dir = getattr(config, raw_dir_attr, None)
    fname = getattr(config, file_attr, None)
    if raw_dir is None or fname is None:
        return None
    path = Path(raw_dir) / fname
    if not path.exists():
        print(f"⚠ [{dataset_key}] {path} 없음, 스킵")
        return None

    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != "Label"]
    train_index = int(path.stem.split("_")[-3])

    train_feat = df.iloc[:train_index][feature_cols].to_numpy(dtype=float)
    test_feat = df.iloc[train_index:][feature_cols].to_numpy(dtype=float)
    test_label = df.iloc[train_index:]["Label"].astype(int).to_numpy()

    return _score_and_summarize(dataset_key, train_feat, test_feat, test_label)


def separability_from_smd() -> dict | None:
    """SMD 원본은 <SMD_RAW_DIR>/train/<machine>.txt, /test/<machine>.txt,
    /test_label/<machine>.txt 구조 (2026-09-09 실측 확인)."""
    smd_dir = Path(config.SMD_RAW_DIR)
    train_path = smd_dir / "train" / f"{config.SMD_MACHINE}.txt"
    test_path = smd_dir / "test" / f"{config.SMD_MACHINE}.txt"
    label_path = smd_dir / "test_label" / f"{config.SMD_MACHINE}.txt"
    if not (train_path.exists() and test_path.exists() and label_path.exists()):
        print(f"⚠ [smd] {train_path} / {test_path} / {label_path} 중 없는 파일 있음, 스킵")
        return None

    train_feat = pd.read_csv(train_path, header=None).to_numpy(dtype=float)
    test_feat = pd.read_csv(test_path, header=None).to_numpy(dtype=float)
    test_label = pd.read_csv(label_path, header=None).iloc[:, 0].astype(int).to_numpy()

    return _score_and_summarize("smd", train_feat, test_feat, test_label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                     default=["smd", "msl", "smap", "swat", "exathlon", "psm",
                              "nab", "power", "swat_uni"])
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        row = separability_from_smd() if ds == "smd" else separability_from_raw_file(ds)
        if row is not None:
            rows.append(row)

    if not rows:
        print("결과 없음.")
        return

    result_df = pd.DataFrame(rows).sort_values("raw_zscore_auc", ascending=False)
    print("\n===== raw z-score 기반 분리도 (모델 없이, train 분포 기준) =====")
    print(result_df.to_string(index=False))
    print("\n해석: raw_zscore_auc가 높을수록(1.0에 가까울수록) 학습된 모델 없이도"
          "\n채널값이 정상 범위를 크게 벗어나는 것만으로 이상을 잘 잡아낸다는 뜻 —"
          "\n그 데이터셋이 baseline에서 왜 천장(ceiling) 근처인지의 근거가 될 수 있음."
          "\nAUC가 0.5~0.6대로 낮으면 raw 신호로는 이상이 잘 안 보인다는 뜻 —"
          "\nbaseline이 바닥(floor)인 데이터셋(예: MSL)의 설명 후보.")

    out_path = Path(config.RESULTS_DIR) / "raw_separability_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()