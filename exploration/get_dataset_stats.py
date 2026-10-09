# -*- coding: utf-8 -*-
"""
get_dataset_stats.py
--------------------------------
미팅 자료(데이터셋 통계 표)를 위한 실측 수집 스크립트. config.py 주석에 이미
적어둔 값(1st_XXX처럼 신뢰 못 하는 라벨 대신 실제로 검증했던 값들)을 다시
"코드로" 뽑아서 표로 만든다 — 손으로 옮겨적다 숫자 틀리는 일 없게.

각 데이터셋(config.py에 정의된 RAW_DIR/FILE)에 대해:
  - 실제 총 행수/컬럼 수(=채널 수, Label 제외)/train 행수/test 행수
  - train 구간 이상 개수(0이어야 정상 — semi-supervised 전제 확인용)
  - test 구간 실제 이상 비율
을 raw csv에서 직접 읽어서 계산하고, 추가로 tsb_ad_input/<ds>/rate_*/ 안의
파일이 이미 만들어져 있으면 거기서 "achieved rate"(실제 주입된 오염 비율)도
target rate(config.RATES)와 나란히 보여준다 — coarse-injection으로 목표 대비
얼마나 오버슈트됐는지 한눈에 보임.

사용법:
    python get_dataset_stats.py
    python get_dataset_stats.py --datasets msl smap swat exathlon psm nab power swat_uni
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# dataset_key -> (raw_dir_attr, file_attr) config.py 변수명 매핑
RAW_SPECS = {
    "smd": None,  # SMD는 이미 공식 train/test/label 파일로 나뉜 구조라 별도 처리
    "msl": ("MSL_RAW_DIR", "MSL_FILE"),
    "smap": ("SMAP_RAW_DIR", "SMAP_FILE"),
    "swat": ("SWAT_RAW_DIR", "SWAT_FILE"),
    "exathlon": ("EXATHLON_RAW_DIR", "EXATHLON_FILE"),
    "psm": ("PSM_RAW_DIR", "PSM_FILE"),
    "nab": ("NAB_RAW_DIR", "NAB_FILE"),
    "power": ("POWER_RAW_DIR", "POWER_FILE"),
    "swat_uni": ("SWAT_UNI_RAW_DIR", "SWAT_UNI_FILE"),
}


def stats_from_raw_file(dataset_key: str) -> dict | None:
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

    total_len = len(df)
    test_len = total_len - train_index
    train_label_sum = int(df.iloc[:train_index]["Label"].astype(int).sum())
    test_label = df.iloc[train_index:]["Label"].astype(int)
    test_anomaly_ratio = 100 * test_label.sum() / len(test_label)

    return {
        "dataset": dataset_key,
        "n_channels": len(feature_cols),
        "train_len": train_index,
        "test_len": test_len,
        "total_len": total_len,
        "train_anomaly_count": train_label_sum,
        "test_anomaly_ratio_pct": round(test_anomaly_ratio, 2),
    }


def stats_from_smd() -> dict | None:
    """SMD 원본은 <SMD_RAW_DIR>/train/<machine>.txt, /test/<machine>.txt,
    /test_label/<machine>.txt 로 나뉘어 있음 (2026-09-09 실측 확인:
    ls /home/ej/sdn_anomaly/data/raw/smd/{train,test,test_label}/ 결과) —
    다른 데이터셋과 raw 파일 구조가 달라 별도 처리."""
    smd_dir = Path(config.SMD_RAW_DIR)
    train_path = smd_dir / "train" / f"{config.SMD_MACHINE}.txt"
    test_path = smd_dir / "test" / f"{config.SMD_MACHINE}.txt"
    label_path = smd_dir / "test_label" / f"{config.SMD_MACHINE}.txt"
    if not (train_path.exists() and test_path.exists() and label_path.exists()):
        print(f"⚠ [smd] {train_path} / {test_path} / {label_path} 중 없는 파일 있음, 스킵")
        return None

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    test_label = pd.read_csv(label_path, header=None).iloc[:, 0].astype(int)

    return {
        "dataset": "smd",
        "n_channels": train_df.shape[1],
        "train_len": len(train_df),
        "test_len": len(test_df),
        "total_len": len(train_df) + len(test_df),
        "train_anomaly_count": 0,  # SMD 공식 train은 정의상 전부 정상
        "test_anomaly_ratio_pct": round(100 * test_label.sum() / len(test_label), 2),
    }


def achieved_rates(dataset_key: str) -> dict[int, float]:
    """tsb_ad_input/<ds>/rate_{r}/*.csv가 이미 만들어져 있으면, 그 파일의 train
    구간 Label 합으로 실제 achieved rate(%)를 계산해서 target(config.RATES)과
    나란히 비교할 수 있게 한다. 아직 안 만들어졌으면 해당 rate는 결과에서 빠짐."""
    base_dir = Path(config.TSB_AD_INPUT_DIR) / dataset_key
    if not base_dir.exists():
        return {}
    result = {}
    for rate in config.RATES:
        rate_dir = base_dir / f"rate_{rate}"
        if not rate_dir.exists():
            continue
        csvs = list(rate_dir.glob("*.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        train_index = int(csvs[0].stem.split("_")[-3])
        train_label = df.iloc[:train_index]["Label"].astype(int)
        achieved = 100 * train_label.sum() / len(train_label) if len(train_label) else 0.0
        result[rate] = round(achieved, 2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                     default=["smd", "msl", "smap", "swat", "exathlon", "psm",
                              "nab", "power", "swat_uni"])
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        row = stats_from_smd() if ds == "smd" else stats_from_raw_file(ds)
        if row is None:
            continue
        ach = achieved_rates(ds)
        row["achieved_rates"] = ach
        rows.append(row)

    if not rows:
        print("결과 없음.")
        return

    df = pd.DataFrame(rows)
    print("\n===== 데이터셋 기본 통계 (raw, 오염 주입 전) =====")
    print(df[["dataset", "n_channels", "train_len", "test_len", "total_len",
              "train_anomaly_count", "test_anomaly_ratio_pct"]]
          .to_string(index=False))

    print("\n===== target rate(%) vs achieved rate(%) (tsb_ad_input 기준, "
          "이미 생성된 것만) =====")
    for row in rows:
        ach = row["achieved_rates"]
        if not ach:
            print(f"  [{row['dataset']}] tsb_ad_input 아직 없음")
            continue
        parts = [f"{r}%→{ach.get(r, 'N/A')}%" for r in config.RATES]
        print(f"  [{row['dataset']}] " + ", ".join(parts))

    out_path = Path(config.RESULTS_DIR) / "dataset_stats_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["achieved_rates"]).to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()