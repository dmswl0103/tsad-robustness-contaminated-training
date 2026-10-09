# -*- coding: utf-8 -*-
"""
prepare_redlamp_input.py
--------------------------

공통 원칙 
  - redlamp_lib.injectors.inject_contamination_mv_nested()로 정상 train 위에
    합성 오염 주입 (누적/nested — 낮은 rate의 오염을 유지한 채 새 오염만 추가)
  - test는 절대 안 건드림 — 모든 rate에서 100% 동일
  - 스케일러는 오염 주입 "이전"의 순수 정상 train으로 한 번만 fit, train/test
    양쪽에 동일 적용 (rate 간 스케일 값까지 bit-for-bit 동일하게)

산출물 (Run_Detector_M.py 컨벤션 그대로):
  tsb_ad_input/<dataset>/rate_{0,1,5,10,20}/<tag>_tr_{train_len}_1st_{total_len}.csv
  tsb_ad_input/<dataset>/<DATASET_TAG>_File_List.csv  (컬럼: file_name, rate_*/*.csv 상대경로)

사용법:
    python prepare/prepare_redlamp_input.py --dataset smd
    python prepare/prepare_redlamp_input.py --dataset insdn
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


# ── 데이터셋별 로더: (train_raw, test_raw, test_label, feature_cols)를 반환 ──
# train_raw/test_raw: pandas.DataFrame, feature_cols만으로 구성 (라벨/메타 컬럼 없음)
# test_label: np.ndarray (0/1), test_raw와 같은 길이

def load_smd():
    raw_dir = Path(config.SMD_RAW_DIR)
    machine = config.SMD_MACHINE
    train = np.loadtxt(raw_dir / "train" / f"{machine}.txt", delimiter=",")
    test = np.loadtxt(raw_dir / "test" / f"{machine}.txt", delimiter=",")
    test_label = np.loadtxt(raw_dir / "test_label" / f"{machine}.txt", delimiter=",").astype(int)
    feature_cols = [f"col_{i}" for i in range(train.shape[1])]
    train_df = pd.DataFrame(train, columns=feature_cols)
    test_df = pd.DataFrame(test, columns=feature_cols)
    return train_df, test_df, test_label, feature_cols, f"SMD-{machine}-RedLamp"


def load_msl():
    """TSB-AD-M 공식 벤치마크 파일(config.MSL_FILE) 그대로 사용. Run_Detector_M.py와
    동일한 방식(filename.split('_')[-3])으로 train_index를 구해서 train/test를 나눈다
    — 파일명의 총 행수 표기(1st_XXX)는 실제 행수와 다를 수 있지만 안 씀(§ config.py 주석)."""
    path = Path(config.MSL_RAW_DIR) / config.MSL_FILE
    raw = pd.read_csv(path)
    train_index = int(path.stem.split("_")[-3])

    feature_cols = [c for c in raw.columns if c != "Label"]
    train_df = raw.iloc[:train_index][feature_cols].reset_index(drop=True)
    test_df = raw.iloc[train_index:][feature_cols].reset_index(drop=True)
    test_label = raw.iloc[train_index:]["Label"].astype(int).to_numpy()

    assert raw.iloc[:train_index]["Label"].sum() == 0, \
        "MSL train 구간에 이상 라벨이 섞여있음 — semi-supervised 전제 깨짐, 원본 확인 필요"

    return train_df, test_df, test_label, feature_cols, "MSL-id1-RedLamp"


def load_smap():
    """MSL과 동일한 방식(TSB-AD-M 공식 벤치마크 파일, filename train_index 파싱).
    27개 시계열 중 id_1 채택 — 파일명의 총 행수 표기(1st_5300)는 실제 행수(8209)와
    다르지만 안 씀(config.py 주석 참고). train 구간 Label=0, test 이상 비율 18.21%
    둘 다 실측 확인됨."""
    path = Path(config.SMAP_RAW_DIR) / config.SMAP_FILE
    raw = pd.read_csv(path)
    train_index = int(path.stem.split("_")[-3])

    feature_cols = [c for c in raw.columns if c != "Label"]
    train_df = raw.iloc[:train_index][feature_cols].reset_index(drop=True)
    test_df = raw.iloc[train_index:][feature_cols].reset_index(drop=True)
    test_label = raw.iloc[train_index:]["Label"].astype(int).to_numpy()

    assert raw.iloc[:train_index]["Label"].sum() == 0, \
        "SMAP train 구간에 이상 라벨이 섞여있음 — semi-supervised 전제 깨짐, 원본 확인 필요"

    return train_df, test_df, test_label, feature_cols, "SMAP-id1-RedLamp"


def load_swat():
    """MSL/SMAP과 동일한 방식. SWaT는 시계열이 id_1/id_2 단 2개뿐인데, id_2는
    test가 100행뿐이라 평가 불가능 수준이라 id_1 채택(train=3749, test=11247).
    파일명의 총 행수 표기(1st_9522)는 실제 행수(14996)와 다르지만 안 씀. train 구간
    Label=0, test 이상 비율 17.66% 둘 다 실측 확인됨."""
    path = Path(config.SWAT_RAW_DIR) / config.SWAT_FILE
    raw = pd.read_csv(path)
    train_index = int(path.stem.split("_")[-3])

    feature_cols = [c for c in raw.columns if c != "Label"]
    train_df = raw.iloc[:train_index][feature_cols].reset_index(drop=True)
    test_df = raw.iloc[train_index:][feature_cols].reset_index(drop=True)
    test_label = raw.iloc[train_index:]["Label"].astype(int).to_numpy()

    assert raw.iloc[:train_index]["Label"].sum() == 0, \
        "SWaT train 구간에 이상 라벨이 섞여있음 — semi-supervised 전제 깨짐, 원본 확인 필요"

    return train_df, test_df, test_label, feature_cols, "SWaT-id1-RedLamp"


def load_exathlon():
    """MSL/SMAP/SWaT와 동일한 방식. 27개 시계열 중 id_1 채택(train=10766로 넉넉해서
    coarse-injection 문제 없음). 파일명의 총 행수 표기(1st_12590)는 실제 행수(43066)와
    다르지만 안 씀. train 구간 Label=0, test 이상 비율 16.83% 둘 다 실측 확인됨."""
    path = Path(config.EXATHLON_RAW_DIR) / config.EXATHLON_FILE
    raw = pd.read_csv(path)
    train_index = int(path.stem.split("_")[-3])

    feature_cols = [c for c in raw.columns if c != "Label"]
    train_df = raw.iloc[:train_index][feature_cols].reset_index(drop=True)
    test_df = raw.iloc[train_index:][feature_cols].reset_index(drop=True)
    test_label = raw.iloc[train_index:]["Label"].astype(int).to_numpy()

    assert raw.iloc[:train_index]["Label"].sum() == 0, \
        "Exathlon train 구간에 이상 라벨이 섞여있음 — semi-supervised 전제 깨짐, 원본 확인 필요"

    return train_df, test_df, test_label, feature_cols, "Exathlon-id1-RedLamp"


def load_nab():
    """TSB-AD-U 공식 벤치마크 파일 그대로 사용. 단변량이라 feature_cols는 항상
    ['Data'] 하나뿐 — 나머지는 MSL/SMAP/SWaT/Exathlon과 동일한 방식. id_23
    (Facility, train=4512) 채택. train 구간 Label=0, test 이상 비율 11.07%
    실측 확인됨."""
    path = Path(config.NAB_RAW_DIR) / config.NAB_FILE
    raw = pd.read_csv(path)
    train_index = int(path.stem.split("_")[-3])

    feature_cols = [c for c in raw.columns if c != "Label"]
    train_df = raw.iloc[:train_index][feature_cols].reset_index(drop=True)
    test_df = raw.iloc[train_index:][feature_cols].reset_index(drop=True)
    test_label = raw.iloc[train_index:]["Label"].astype(int).to_numpy()

    assert raw.iloc[:train_index]["Label"].sum() == 0, \
        "NAB train 구간에 이상 라벨이 섞여있음 — semi-supervised 전제 깨짐, 원본 확인 필요"

    return train_df, test_df, test_label, feature_cols, "NAB-id23-RedLamp"


def load_power():
    """NAB와 동일한 방식. 시계열 1개뿐(단일 발전시설). train 구간 Label=0,
    test 이상 비율 10.97% 실측 확인됨."""
    path = Path(config.POWER_RAW_DIR) / config.POWER_FILE
    raw = pd.read_csv(path)
    train_index = int(path.stem.split("_")[-3])

    feature_cols = [c for c in raw.columns if c != "Label"]
    train_df = raw.iloc[:train_index][feature_cols].reset_index(drop=True)
    test_df = raw.iloc[train_index:][feature_cols].reset_index(drop=True)
    test_label = raw.iloc[train_index:]["Label"].astype(int).to_numpy()

    assert raw.iloc[:train_index]["Label"].sum() == 0, \
        "Power train 구간에 이상 라벨이 섞여있음 — semi-supervised 전제 깨짐, 원본 확인 필요"

    return train_df, test_df, test_label, feature_cols, "Power-id1-RedLamp"


def load_swat_uni():
    """NAB/Power와 동일한 방식. 다변량 SWaT(load_swat)와는 완전히 다른 파일
    (config.SWAT_UNI_FILE) — 시계열 1개, train=43700. train 구간 Label=0,
    test 이상 비율 13.46% 실측 확인됨."""
    path = Path(config.SWAT_UNI_RAW_DIR) / config.SWAT_UNI_FILE
    raw = pd.read_csv(path)
    train_index = int(path.stem.split("_")[-3])

    feature_cols = [c for c in raw.columns if c != "Label"]
    train_df = raw.iloc[:train_index][feature_cols].reset_index(drop=True)
    test_df = raw.iloc[train_index:][feature_cols].reset_index(drop=True)
    test_label = raw.iloc[train_index:]["Label"].astype(int).to_numpy()

    assert raw.iloc[:train_index]["Label"].sum() == 0, \
        "SWaT(단변량) train 구간에 이상 라벨이 섞여있음 — semi-supervised 전제 깨짐, 원본 확인 필요"

    return train_df, test_df, test_label, feature_cols, "SWaT-Uni-id1-RedLamp"


def load_psm():
    """MSL/SMAP/SWaT/Exathlon과 동일한 방식. 시계열 1개, 25채널, train=50000으로
    가장 넉넉함. train 구간 Label=0, test 이상 비율 14.55% 실측 확인됨."""
    path = Path(config.PSM_RAW_DIR) / config.PSM_FILE
    raw = pd.read_csv(path)
    train_index = int(path.stem.split("_")[-3])

    feature_cols = [c for c in raw.columns if c != "Label"]
    train_df = raw.iloc[:train_index][feature_cols].reset_index(drop=True)
    test_df = raw.iloc[train_index:][feature_cols].reset_index(drop=True)
    test_label = raw.iloc[train_index:]["Label"].astype(int).to_numpy()

    assert raw.iloc[:train_index]["Label"].sum() == 0, \
        "PSM train 구간에 이상 라벨이 섞여있음 — semi-supervised 전제 깨짐, 원본 확인 필요"

    return train_df, test_df, test_label, feature_cols, "PSM-id1-RedLamp"


LOADERS = {"smd": load_smd, "msl": load_msl,
           "smap": load_smap, "swat": load_swat, "exathlon": load_exathlon,
           "nab": load_nab, "power": load_power, "swat_uni": load_swat_uni,
           "psm": load_psm}


def make_rate0(train_df: pd.DataFrame):
    """rate=0(baseline)은 오염 없음 — 원본 그대로, 라벨 전부 0."""
    contaminated = train_df.to_numpy().copy()
    label = np.zeros(len(train_df), dtype=int)
    return contaminated, label


def build_output_frame(contaminated: np.ndarray, label: np.ndarray, test_arr: np.ndarray,
                        test_label: np.ndarray, feature_cols: list, scaler: MinMaxScaler):
    """train(오염됨)+test(고정)를 이어붙이고, 정상 train 기준으로 fit된 scaler를
    양쪽에 동일하게 적용해서 최종 CSV용 DataFrame 하나를 만든다."""
    train_scaled = scaler.transform(contaminated)
    test_scaled = scaler.transform(test_arr)

    train_out = pd.DataFrame(train_scaled, columns=feature_cols)
    train_out["Label"] = label
    test_out = pd.DataFrame(test_scaled, columns=feature_cols)
    test_out["Label"] = test_label

    combined = pd.concat([train_out, test_out], ignore_index=True)
    return combined, len(train_out), len(combined)


def run(dataset_key: str, out_root: Path, rates: list, window: int, seed: int):
    if dataset_key not in LOADERS:
        raise ValueError(f"알 수 없는 dataset: {dataset_key} (지원: {list(LOADERS)})")

    train_df, test_df, test_label, feature_cols, tag = LOADERS[dataset_key]()
    train_arr = train_df.to_numpy()
    test_arr = test_df.to_numpy()

    print(f"[{dataset_key}] train={len(train_df)}, test={len(test_df)} "
          f"(feature 수={len(feature_cols)}) — test는 모든 rate 공통 고정")

    # 스케일러: 오염 주입 "이전"의 순수 정상 train으로만 fit (rate와 무관하게 고정)
    scaler = MinMaxScaler().fit(train_arr)

    # RedLamp nested 주입 — (features, T) 형태로 넣어야 함
    clean_2d = train_arr.T
    positive_rates = [r for r in rates if r > 0]
    results = inject_contamination_mv_nested(
        clean_2d, ratios_pct=positive_rates, window=window, seed=seed,
        min_features=1, max_features=clean_2d.shape[0],
    )

    out_dir = out_root / dataset_key
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

        # 누적(nested) 관계 검증: 이전 rate에서 오염된 지점이 이번 rate에서도 전부 유지되는지
        if prev_label is not None:
            assert np.all(label[prev_label == 1] == 1), \
                f"nested 관계 깨짐: rate={rate}% 가 이전 rate의 오염을 유지 못 함"
        prev_label = label

        combined_df, train_len, total_len = build_output_frame(
            contaminated, label, test_arr, test_label, feature_cols, scaler
        )

        rate_dir = out_dir / f"rate_{rate}"
        rate_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{tag}_tr_{train_len}_1st_{total_len}.csv"
        combined_df.to_csv(rate_dir / filename, index=False)
        file_list.append(f"rate_{rate}/{filename}")

        print(f"  rate={rate:>2}%: 목표={rate}% 달성={achieved:.2f}% "
              f"-> {rate_dir.name}/{filename}")

    file_list_path = out_dir / f"_{tag}_File_List.csv"
    pd.DataFrame({"file_name": file_list}).to_csv(file_list_path, index=False)
    print(f"[{dataset_key}] 완료. File_List: {file_list_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--out_root", type=str, default=config.TSB_AD_INPUT_DIR)
    ap.add_argument("--rates", type=int, nargs="+", default=config.RATES)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    run(args.dataset, Path(args.out_root), args.rates, args.window, args.seed)


if __name__ == "__main__":
    main()