# -*- coding: utf-8 -*-
"""
새 모델을 추가하고 싶으면:
  1) TSB-AD/TSB_AD/models/ 안에 해당 모델 파일이 있는지 확인
  2) TSB-AD/TSB_AD/HP_list.py의 Optimal_Multi_algo_HP_dict에 프리셋이 있는지 확인
  3) 아래 MODELS 리스트와 MODEL_COLORS에 추가
     (run_experiments.sh / recompute / analyze / plot 4곳 다 자동으로 반영됨)
"""

import os

# ── 경로 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 3단계 데이터 흐름: raw -> interim(1회성 정리, 주입 전)
#                    -> tsb_ad_input(RedLamp 주입 완료, 모델 입력 최종본)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_INTERIM_DIR = os.path.join(DATA_DIR, "interim")
TSB_AD_INPUT_DIR = os.path.join(PROJECT_ROOT, "tsb_ad_input")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# ── InSDN ────────────────────────────────────────────────────────────
INSDN_RAW_DIR = os.path.join(DATA_RAW_DIR, "insdn")
INSDN_RAW_FILES = {
    "normal": "Normal_data.csv",
    "OVS": "OVS.csv",
    "Metasploitable-2": "metasploitable-2.csv",
}

INSDN_INTERIM_DIR = os.path.join(DATA_INTERIM_DIR, "insdn")

# real(합성 아님) 공격 pool 기반 오염 실험용 — 위 RedLamp 파이프라인과 완전히 분리된
# 네임스페이스. attack_test를 train용/test용 pool로 미리 나눠서 서로 안 겹치게 씀.
INSDN_REALPOOL_TSB_AD_INPUT_DIR = os.path.join(TSB_AD_INPUT_DIR, "insdn_realpool")
INSDN_REALPOOL_RESULTS_DIR = os.path.join(RESULTS_DIR, "insdn_realpool")

# ── SMD ──────────────────────────────────────────────────────────────
# 원본 자체가 이미 공식 train/test/test_label로 나뉜 최종 구조라 interim 단계 불필요
SMD_RAW_DIR = os.path.join(DATA_RAW_DIR, "smd")
SMD_MACHINE = "machine-1-1"

# SMD 여러 머신 평균용 — 위 machine-1-1 파이프라인과 폴더가 완전히 분리되어 안 겹침
SMD_MULTI_TSB_AD_INPUT_DIR = os.path.join(TSB_AD_INPUT_DIR, "smd_multi")
SMD_MULTI_RESULTS_DIR = os.path.join(RESULTS_DIR, "smd_multi")

# ── MSL ──────────────────────────────────────────────────────────────
# TSB-AD-M 공식 벤치마크 파일 그대로 사용(train/test 이미 나뉜 채널 하나, id_1).
# train 구간(첫 500행)이 전부 Label=0으로 확인됨 -> semi-supervised 전제 충족.
# 파일명의 "1st_900"은 TSB-AD 쪽 표기가 오래돼서 실제 행수(1827)와 다르지만,
# Run_Detector_M.py는 그 숫자를 안 쓰고 filename.split('_')[-3](=500, train_index)만
# 써서 동작에는 영향 없음(직접 코드 확인함).
MSL_RAW_DIR = os.path.join(DATA_RAW_DIR, "msl")
MSL_FILE = "002_MSL_id_1_Sensor_tr_500_1st_900.csv"

# ── SMAP ─────────────────────────────────────────────────────────────
# TSB-AD-M 공식 벤치마크 파일 그대로 사용(27개 시계열 중 id_1). MSL과 동일하게
# 파일명의 "1st_5300"은 실제 총 행수(8209)와 다르지만 안 씀(위 MSL 주석과 동일 이유).
# train 구간(첫 2052행) 전부 Label=0 확인됨, test 이상 비율 18.21%로 실측 확인.
SMAP_RAW_DIR = os.path.join(DATA_RAW_DIR, "smap")
SMAP_FILE = "144_SMAP_id_1_Sensor_tr_2052_1st_5300.csv"

# ── SWaT ─────────────────────────────────────────────────────────────
# TSB-AD-M에 시계열이 id_1/id_2 단 2개뿐 -> id_2는 test가 100행뿐이라 평가 불가능
# 수준이라 id_1 채택(train=3749, test=11247). 파일명의 "1st_9522"도 실제 총
# 행수(14996)와 다름(MSL/SMAP와 동일 이유). train 구간 Label=0 확인됨,
# test 이상 비율 17.66%로 실측 확인.
SWAT_RAW_DIR = os.path.join(DATA_RAW_DIR, "swat")
SWAT_FILE = "171_SWaT_id_1_Sensor_tr_3749_1st_9522.csv"

# ── Exathlon ─────────────────────────────────────────────────────────
# TSB-AD-M 공식 벤치마크 파일 그대로 사용(27개 시계열 중 id_1, Spark job 성능 지표
# 18채널). train 구간(첫 10766행) 전부 Label=0 확인됨, test 이상 비율 16.83%로
# 실측 확인. 파일명의 "1st_12590"도 실제 총 행수(43066)와 다름(MSL 등과 동일 이유).
# (TAO는 13개 시계열 전부 train 구간에 실제 이상 라벨이 섞여있어(19~81건) semi-
# supervised 전제가 깨져서 이번 프레임워크에서 제외함 — 다른 데이터셋과 공정 비교 불가.)
EXATHLON_RAW_DIR = os.path.join(DATA_RAW_DIR, "exathlon")
EXATHLON_FILE = "174_Exathlon_id_1_Facility_tr_10766_1st_12590.csv"

# ── PSM ──────────────────────────────────────────────────────────────
# TSB-AD-M 공식 벤치마크 파일 그대로 사용(시계열 1개, 25채널). train=50000으로
# 지금까지 중 가장 넉넉해서 coarse-injection 위험 전혀 없음. train 구간 Label=0,
# test 이상 비율 14.55% 실측 확인. 파일명의 "1st_129872"가 실제 총 행수(217624)와
# 다른 것도 다른 데이터셋과 동일 이유(안 씀).
PSM_RAW_DIR = os.path.join(DATA_RAW_DIR, "psm")
PSM_FILE = "115_PSM_id_1_Facility_tr_50000_1st_129872.csv"

# ── 단변량(TSB-AD-U) ────────────────────────────────────────────────────
# 위 SMD/MSL/SMAP/SWaT/Exathlon은 전부 다변량(TSB-AD-M); 여기부터는 교수님 지시로
# 추가한 단변량(TSB-AD-U) 데이터셋 — 별도 zip(TSB-AD-U.zip), 컬럼도 ['Data','Label']
# 하나뿐이라 feature_cols 길이가 항상 1. 나머지 원칙(train=순수 정상, test 고정,
# window=100, RedLamp nested 주입)은 다변량과 완전히 동일.

# NAB: 28개 시계열 중 id_23(Facility 카테고리, train=4512로 제일 큼) 채택.
# NAB 안에는 id_8/12/15/17/20/24처럼 "Synthetic" 카테고리도 섞여있는데, 이건 아예
# 합성 데이터라 우리 취지(실제 정상 데이터+RedLamp 합성 오염)와 안 맞아서 제외함.
# train 구간 Label=0, test 이상 비율 11.07% 실측 확인.
NAB_RAW_DIR = os.path.join(DATA_RAW_DIR, "nab")
NAB_FILE = "023_NAB_id_23_Facility_tr_4512_1st_16551.csv"

# Power: 시계열 1개뿐(단일 발전시설 전력소비, 실제 데이터). train=7685로 넉넉함.
# train 구간 Label=0, test 이상 비율 10.97% 실측 확인.
POWER_RAW_DIR = os.path.join(DATA_RAW_DIR, "power")
POWER_FILE = "302_Power_id_1_Facility_tr_7685_1st_7785.csv"

# SWaT(단변량): 다변량 SWaT(SWAT_FILE)와는 별개 파일 — TSB-AD-U 쪽 SWaT는 시계열
# 1개, train=43700으로 압도적으로 커서 coarse-injection 위험 거의 없음. 다변량
# SWaT 결과와 직접 비교(같은 도메인, 다변량 vs 단변량) 가능. train 구간 Label=0,
# test 이상 비율 13.46% 실측 확인.
SWAT_UNI_RAW_DIR = os.path.join(DATA_RAW_DIR, "swat_uni")
SWAT_UNI_FILE = "550_SWaT_id_1_Sensor_tr_43700_1st_43800.csv"

# Stock: 20개 시계열 전부 train=500이고, 지금까지 확인한 id_1은 train 구간에
# 실제 이상 라벨이 26~28건 섞여있어(TAO와 동일 문제) semi-supervised 전제가 깨짐
# -> 깨끗한(train_anomaly_count=0) id를 찾을 때까지 보류. 확인되면 여기 추가.

# ── 오염 비율 ───────────────────────────────────────────
RATES = [0, 1, 5, 10, 20]

# ── 데이터셋 ──────────────────────────────────────────────────────────
# base_dir: tsb_ad_input/<key>/rate_{rate}/*.csv 형태로 존재해야 함
DATASETS = {
    "smd": {
        "base_dir": os.path.join(TSB_AD_INPUT_DIR, "smd"),
        "label": "SMD (machine-1-1, RedLamp)",
        "rates": RATES,
    },
    "insdn": {
        "base_dir": os.path.join(TSB_AD_INPUT_DIR, "insdn"),
        "label": "InSDN (Merged, RedLamp)",
        "rates": RATES,
    },
    "msl": {
        "base_dir": os.path.join(TSB_AD_INPUT_DIR, "msl"),
        "label": "MSL (id_1, RedLamp)",
        "rates": RATES,
    },
    "smap": {
        "base_dir": os.path.join(TSB_AD_INPUT_DIR, "smap"),
        "label": "SMAP (id_1, RedLamp)",
        "rates": RATES,
    },
    "swat": {
        "base_dir": os.path.join(TSB_AD_INPUT_DIR, "swat"),
        "label": "SWaT (id_1, RedLamp)",
        "rates": RATES,
    },
    "exathlon": {
        "base_dir": os.path.join(TSB_AD_INPUT_DIR, "exathlon"),
        "label": "Exathlon (id_1, RedLamp)",
        "rates": RATES,
    },
    "psm": {
        "base_dir": os.path.join(TSB_AD_INPUT_DIR, "psm"),
        "label": "PSM (id_1, RedLamp)",
        "rates": RATES,
    },
    "nab": {
        "base_dir": os.path.join(TSB_AD_INPUT_DIR, "nab"),
        "label": "NAB (id_23, univariate, RedLamp)",
        "rates": RATES,
    },
    "power": {
        "base_dir": os.path.join(TSB_AD_INPUT_DIR, "power"),
        "label": "Power (id_1, univariate, RedLamp)",
        "rates": RATES,
    },
    "swat_uni": {
        "base_dir": os.path.join(TSB_AD_INPUT_DIR, "swat_uni"),
        "label": "SWaT (id_1, univariate, RedLamp)",
        "rates": RATES,
    },
}

# ── 모델 ──────────────────────────────────────────────────────────────
# CNN/LSTMAD: 원래 SMD/InSDN에 --models CLI override로만 추가했던 "얕은
# 재구성/예측 모델" 비교군인데, 이제 모든 다변량 데이터셋(SMD/MSL/SMAP/SWaT/
# Exathlon/PSM)에서 통일된 모델 세트로 공정 비교하기 위해 정식으로 편입.
# Run_Detector_M.py가 이미 만들어진 .npy는 건너뛰므로, 이미 CNN/LSTMAD를 돌려둔
# SMD/InSDN은 재계산되지 않고, 아직 없는 MSL/SMAP/SWaT/Exathlon/PSM에만 새로
# 계산됨 — run_experiments.sh <dataset> <gpu> CNN LSTMAD 로 부족한 데이터셋만
# 추가 실행하면 됨. KMeansAD는 원인 불명 실패로 제외 유지(§ 이전 논의).
MODELS = [
    "AutoEncoder",
    "USAD",
    "OmniAnomaly",
    "TranAD",
    "TimesNet",
    "FITS",
    "PCA",
    "RobustPCA",
    # "xLSTMAD",
    "Donut",
    "StreamVAE",
    "OFA",
    "PatchTST",
    "CNN",
    "LSTMAD",
]

MODEL_COLORS = {
    "AutoEncoder": "#4C72B0",
    "USAD": "#DD8452",
    "OmniAnomaly": "#C44E52",
    "TranAD": "#55A868",
    "TimesNet": "#8DA0CB",
    "FITS": "#CCB974",
    "PCA": "#8172B2",
    "RobustPCA": "#937860",
    # "xLSTMAD": "#DA8BC3",
    "CNN": "#E377C2",
    "LSTMAD": "#BCBD22",
    "Donut": "#A1A1A1",
    "StreamVAE": "#64B5CD",
    "OFA": "#B07AA1",
    "PatchTST": "#59A14F",
}

# ── 지표 ──────────────────────────────────────────────────────────────
PRIMARY_METRIC = {
    "smd": ["AUC-PR", "VUS-PR"],
    "insdn": ["AUC-ROC", "VUS-ROC"],
    "msl": ["AUC-PR", "VUS-PR"],  # test 이상 비율 0.83%로 낮음 -> SMD와 동일 이유
    "smap": ["AUC-PR", "VUS-PR"],  # 이상 비율 18.21%로 SMD/MSL과 같은 서버·센서 telemetry 계열
    "swat": ["AUC-PR", "VUS-PR"],  # 이상 비율 17.66%, SMAP과 마찬가지로 InSDN(93%)만큼 극단적이지 않음
    "exathlon": ["AUC-PR", "VUS-PR"],  # 이상 비율 16.83%, SMAP/SWaT와 같은 이유
    "psm": ["AUC-PR", "VUS-PR"],  # 이상 비율 14.55%, 같은 이유
    "nab": ["AUC-PR", "VUS-PR"],  # 이상 비율 11.07%, SMD 계열과 같은 이유
    "power": ["AUC-PR", "VUS-PR"],  # 이상 비율 10.97%, SMD 계열과 같은 이유
    "swat_uni": ["AUC-PR", "VUS-PR"],  # 이상 비율 13.46%, SMD 계열과 같은 이유
}

# ── 데이터셋별 추가 모델 제외 (plot_paper_figures.py에서 사용) ──────────────
# PCA/RobustPCA는 전 데이터셋 공통 제외(leakage, plot_paper_figures.py의
# EXCLUDE_MODELS 참고). 그 외에 특정 데이터셋에서만 수치적으로 신뢰 불가능한
# 모델이 확인되면 여기 추가한다.
# StreamVAE: SMAP/SWaT에서 raw score 최댓값이 1e14~1e16 단위로 튀는 게 실측
# 확인됨(정상 범위는 수십 단위) — 특정 채널의 분산이 0에 가까워 likelihood 계산이
# 수치적으로 폭발(overflow)하는 것으로 추정, 그 결과 rate 0~20% 전 구간에서
# AUC 지표가 byte-for-byte 동일한 1.0000이 나옴(학습된 모델의 판별력이 아니라
# 계산 아티팩트). MSL/SMD에서는 정상 범위(0.94~0.97)로 나와 StreamVAE 자체의
# 고질적 버그는 아님 — SMAP/SWaT 데이터의 값 분포에서만 재현되는 것으로 판단.
EXTRA_EXCLUDE_MODELS = {
    "smap": {"StreamVAE"},
    "swat": {"StreamVAE"},
}