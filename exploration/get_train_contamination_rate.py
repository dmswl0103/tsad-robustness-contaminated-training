# -*- coding: utf-8 -*-
"""
get_train_contamination_rate.py
--------------------------------
TAO/Stock처럼 "공식 train 구간에 이미 실제 이상 라벨이 섞여 있다"는 데이터셋의
정확한 오염 비율을 실측하는 스크립트. 인트로에서 "기존 벤치마크도 이미 X%
오염돼 있다"는 식으로 인용할 정확한 숫자가 필요할 때 씀 — 지금까지는
"섞여있다"는 사실만 확인했고 정확한 %는 series별로 다 실측된 게 아니었음
(config.py 주석: TAO 19~81건, Stock id_1만 26~28건 확인).

폴더 안의 각 시계열 csv(예: tao_all/116_TAO_id_1_..._tr_500_1st_3.csv)마다:
  - train_index를 파일명에서 파싱(get_dataset_stats.py와 동일 규칙:
    파일명(확장자 제외)을 "_"로 나눈 뒤 뒤에서 3번째 토큰)
  - train 구간(0~train_index)의 Label 합 / train_index = 그 series의 train
    오염 비율(%)
을 계산하고, 전체 series 평균/최소/최대까지 같이 보여줌.

사용법:
    python get_train_contamination_rate.py --dir data/raw/tao_all --name TAO
    python get_train_contamination_rate.py --dir data/raw/stock_all --name Stock
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="시계열 csv들이 들어있는 폴더 경로")
    ap.add_argument("--name", default=None, help="출력에 쓸 데이터셋 이름(기본: 폴더명)")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.exists():
        print(f"⚠ {d} 없음")
        return
    csv_paths = sorted(d.glob("*.csv"))
    if not csv_paths:
        print(f"⚠ {d}에 csv 없음")
        return

    label = args.name or d.name
    rows = []
    for path in csv_paths:
        try:
            train_index = int(path.stem.split("_")[-3])
        except (ValueError, IndexError):
            print(f"⚠ {path.name}: 파일명에서 train_index 파싱 실패, 스킵")
            continue
        df = pd.read_csv(path)
        if "Label" not in df.columns:
            print(f"⚠ {path.name}: Label 컬럼 없음, 스킵")
            continue
        train_label = df.iloc[:train_index]["Label"].astype(int)
        n_anomaly = int(train_label.sum())
        pct = 100 * n_anomaly / train_index if train_index else float("nan")
        rows.append({
            "file": path.name,
            "train_len": train_index,
            "train_anomaly_count": n_anomaly,
            "train_anomaly_pct": round(pct, 2),
        })

    if not rows:
        print("계산된 series가 없음.")
        return

    result = pd.DataFrame(rows)
    print(f"\n===== {label}: series별 train 오염 비율 =====")
    print(result.to_string(index=False))

    print(f"\n===== {label}: 전체 {len(result)}개 series 요약 =====")
    print(f"  train_anomaly_count 범위: {result['train_anomaly_count'].min()}"
          f"~{result['train_anomaly_count'].max()}건")
    print(f"  train_anomaly_pct 범위: {result['train_anomaly_pct'].min():.2f}%"
          f"~{result['train_anomaly_pct'].max():.2f}%")
    print(f"  train_anomaly_pct 평균: {result['train_anomaly_pct'].mean():.2f}%")

    out_path = d / f"{label.lower()}_train_contamination.csv"
    result.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()