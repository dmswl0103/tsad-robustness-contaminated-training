# -*- coding: utf-8 -*-
"""
prepare_redlamp_input_smd_multi.py
--------------------------
SMD "여러 머신" 버전 RedLamp nested 주입 스크립트. 기존
prepare_redlamp_input.py --dataset smd (machine-1-1 하나만 처리, 단일
DATASETS["smd"] 네임스페이스)와는 완전히 분리된 파이프라인이다 —
디렉터리도, config.DATASETS도 안 건드리므로 이미 끝난 machine-1-1 결과와
절대 안 겹친다.

data/raw/smd/{train,test,test_label}/ 밑에 있는 machine-*.txt 를 전부
자동으로 찾아서 각 머신마다 독립적으로 RedLamp 주입을 수행한다
(스케일러도 머신별로 각자 fit — 다른 머신 값이 섞이면 안 됨).

산출물 (기존 smd/insdn과 같은 컨벤션, machine 이름만 한 단계 더 있음):
  tsb_ad_input/smd_multi/<machine>/rate_{0,1,5,10,20}/<tag>_tr_{...}_1st_{...}.csv
  tsb_ad_input/smd_multi/<machine>/_<tag>_File_List.csv

머신이 몇 개 있든(지금은 1개, 나중에 늘어나도) 있는 만큼 전부 처리한다.
평균 내는 집계 스크립트(recompute/analyze의 smd_multi 버전)는 별도로 작성함
(다음 단계 — 오늘 밤엔 데이터 준비 + 학습만 걸어놓으면 됨).

사용법:
    python prepare/prepare_redlamp_input_smd_multi.py
    python prepare/prepare_redlamp_input_smd_multi.py --machines machine-1-1 machine-1-2  # 일부만
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from redlamp_lib.injectors import inject_contamination_mv_nested

SEED = 42
WINDOW = 100


def discover_machines(raw_dir: Path) -> list:
    """data/raw/smd/train/*.txt 에서 실제 존재하는 머신 이름을 전부 찾는다
    (train/test/test_label 세 곳 모두에 파일이 있는 것만 유효한 머신으로 인정)."""
    train_dir = raw_dir / "train"
    test_dir = raw_dir / "test"
    label_dir = raw_dir / "test_label"
    candidates = sorted(p.stem for p in train_dir.glob("*.txt"))
    machines = []
    for name in candidates:
        if (test_dir / f"{name}.txt").exists() and (label_dir / f"{name}.txt").exists():
            machines.append(name)
        else:
            print(f"⚠ {name}: train은 있는데 test/test_label 짝이 안 맞음 — 건너뜀")
    return machines


def load_machine(raw_dir: Path, machine: str):
    train = np.loadtxt(raw_dir / "train" / f"{machine}.txt", delimiter=",")
    test = np.loadtxt(raw_dir / "test" / f"{machine}.txt", delimiter=",")
    test_label = np.loadtxt(raw_dir / "test_label" / f"{machine}.txt", delimiter=",").astype(int)
    feature_cols = [f"col_{i}" for i in range(train.shape[1])]
    train_df = pd.DataFrame(train, columns=feature_cols)
    test_df = pd.DataFrame(test, columns=feature_cols)
    return train_df, test_df, test_label, feature_cols, f"SMD-{machine}-RedLamp"


def make_rate0(train_df: pd.DataFrame):
    contaminated = train_df.to_numpy().copy()
    label = np.zeros(len(train_df), dtype=int)
    return contaminated, label


def build_output_frame(contaminated, label, test_arr, test_label, feature_cols, scaler):
    train_scaled = scaler.transform(contaminated)
    test_scaled = scaler.transform(test_arr)
    train_out = pd.DataFrame(train_scaled, columns=feature_cols)
    train_out["Label"] = label
    test_out = pd.DataFrame(test_scaled, columns=feature_cols)
    test_out["Label"] = test_label
    combined = pd.concat([train_out, test_out], ignore_index=True)
    return combined, len(train_out), len(combined)


def run_one_machine(machine: str, raw_dir: Path, out_root: Path, rates: list, window: int, seed: int):
    train_df, test_df, test_label, feature_cols, tag = load_machine(raw_dir, machine)
    train_arr = train_df.to_numpy()
    test_arr = test_df.to_numpy()

    print(f"[{machine}] train={len(train_df)}, test={len(test_df)} "
          f"(feature 수={len(feature_cols)}) — test는 모든 rate 공통 고정")

    # 스케일러: 머신별로 독립적으로, 오염 주입 "이전"의 순수 정상 train으로만 fit
    scaler = MinMaxScaler().fit(train_arr)

    clean_2d = train_arr.T
    positive_rates = [r for r in rates if r > 0]
    results = inject_contamination_mv_nested(
        clean_2d, ratios_pct=positive_rates, window=window, seed=seed,
        min_features=1, max_features=clean_2d.shape[0],
    )

    out_dir = out_root / machine
    out_dir.mkdir(parents=True, exist_ok=True)
    file_list = []
    prev_label = None

    for rate in sorted(rates):
        if rate == 0:
            contaminated, label = make_rate0(train_df)
            achieved = 0.0
        else:
            contaminated_2d, label, log = results[rate]
            contaminated = contaminated_2d.T
            achieved = log["achieved_pct"]

        if prev_label is not None:
            assert np.all(label[prev_label == 1] == 1), \
                f"nested 관계 깨짐: {machine} rate={rate}%"
        prev_label = label

        combined_df, train_len, total_len = build_output_frame(
            contaminated, label, test_arr, test_label, feature_cols, scaler
        )

        rate_dir = out_dir / f"rate_{rate}"
        rate_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{tag}_tr_{train_len}_1st_{total_len}.csv"
        combined_df.to_csv(rate_dir / filename, index=False)
        file_list.append(f"rate_{rate}/{filename}")

        print(f"  [{machine}] rate={rate:>2}%: 목표={rate}% 달성={achieved:.2f}% "
              f"-> {rate_dir.name}/{filename}")

    file_list_path = out_dir / f"_{tag}_File_List.csv"
    pd.DataFrame({"file_name": file_list}).to_csv(file_list_path, index=False)
    print(f"[{machine}] 완료. File_List: {file_list_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", type=str, nargs="+", default=None,
                     help="특정 머신만 처리(예: machine-1-1 machine-1-2). 생략하면 자동 탐색된 전부.")
    ap.add_argument("--out_root", type=str, default=config.SMD_MULTI_TSB_AD_INPUT_DIR)
    ap.add_argument("--rates", type=int, nargs="+", default=config.RATES)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    raw_dir = Path(config.SMD_RAW_DIR)
    machines = args.machines if args.machines else discover_machines(raw_dir)
    if not machines:
        print(f"⚠ {raw_dir}에서 처리할 머신을 못 찾음. train/test/test_label 확인하세요.")
        return

    print(f"처리할 머신 {len(machines)}개: {machines}")
    out_root = Path(args.out_root)
    for machine in machines:
        run_one_machine(machine, raw_dir, out_root, args.rates, args.window, args.seed)

    print(f"\n전체 완료. 총 {len(machines)}개 머신 -> {out_root}")


if __name__ == "__main__":
    main()