# Robust Anomaly Detection

train 데이터의 오염(contamination) 비율에 따라 비지도/반지도 시계열 이상 탐지 모델의
성능이 어떻게 변하는지, 어떤 모델이 오염에 더 강건한지를 분석하는 연구입니다. 서버
텔레메트리 도메인의 두 데이터셋(SMD, PSM)에서 동일한 오염 실험을 수행해 특정 데이터
분포에 국한되지 않는 강건성을 확인합니다.

TSB-AD(https://github.com/TheDatumOrg/TSB-AD) 공식 벤치마크 코드를 기반으로 하며,
자체 구현한 부분은 (1) RedLamp 기반 오염 비율별 학습 데이터 생성(nested injection),
(2) 평가 후처리입니다.

## Publication

이 연구는 아래 논문으로 정리되어 제7회 한국인공지능학술대회(The 7th Korea Artificial
Intelligence Conference, 2026.09)에 제출되었습니다.

> Eunji Lee, Sangyup Lee. "Robustness Analysis of Unsupervised Time Series
> Anomaly Detection under Contaminated Training Data." The 7th Korea
> Artificial Intelligence Conference, Sep. 2026.
>
> 5개 대표 TSAD 모델을 대상으로 pseudo-anomaly 기반 학습 데이터 오염에 따른 성능
> 저하를 체계적으로 평가. SMD, PSM 서버 텔레메트리 벤치마크에 대해 누적(nested)
> 오염 주입 절차를 설계하여 오염 강건성을 고려한 이상 탐지 모델 설계의 필요성을 제시.

---

## 데이터셋

| | SMD | PSM |
|---|---|---|
| 도메인 | 서버 메트릭(machine-1-1) | 서버 메트릭(Facility) |
| Feature 수 | 38 | 25 |
| Train | 28,479 | 50,000 |
| Test 이상 비율 | 9.46% | 14.55% |

둘 다 TSB-AD-M 공식 벤치마크 파일을 그대로 사용하며, train 구간은 전부 정상(Label=0)
임을 실측으로 확인했습니다.

---

## 오염 주입

SMD, PSM 모두 동일한 방식으로 train에 합성 오염을 주입합니다. window=100 구간을
무작위로 선택해 일부 채널에 RedLamp[6]의 11종 pseudo-anomaly 기법(spike, flip,
speedup, noise, cutoff, average, scale, wander, contextual, upsidedown, mixture)
중 하나를 적용하며, 오염 비율(0/1/5/10/20%)은 누적(nested) 구조입니다 — 낮은 rate에서
주입된 오염은 높은 rate에서도 그대로 유지한 채 새 오염만 추가해서, rate 간 성능
변화가 "오염 종류가 바뀌어서"가 아니라 "오염량이 늘어서"임을 보장합니다. Test는 오염
비율과 무관하게 모든 실험에서 동일하게 고정됩니다. 스케일러는 오염 주입 이전의 정상
train에만 fit해서 train/test에 동일 적용합니다.

이 저장소에는 `redlamp_lib/`(주입 함수 구현)를 포함하지 않습니다. RedLamp[6]의 11종
pseudo-anomaly 기법을 직접 구현하고, 그 위에 누적(nested) 주입 로직을 추가해
`redlamp_lib/injectors.py`에 `inject_contamination_mv_nested()` 함수로 두면
`prepare/prepare_redlamp_input.py`가 동작합니다. 입출력 형태는 해당 스크립트의
호출부를 참고하세요.

---

## 모델

TSB-AD 구현체와 공식 하이퍼파라미터(`Optimal_Multi_algo_HP_dict`)를 사용합니다.
논문 실험은 AutoEncoder, USAD, LSTMAD, TimesNet, Donut 5개 모델 기준이며, 이 저장소
자체는 더 많은 모델을 지원합니다(`config.MODELS` 참고).

새 모델 추가 시 `config.py`의 `MODELS`/`MODEL_COLORS`만 수정하면 됩니다.

---

## 평가 지표

AUC-ROC, AUC-PR, VUS-ROC, VUS-PR을 사용하며, SMD/PSM 모두 주 지표는 AUC-PR,
VUS-PR입니다(`config.PRIMARY_METRIC`).

---

## 폴더 구조

```
sdn_anomaly/
├── config.py
├── data/
│   └── raw/{smd, psm, ...}/
├── tsb_ad_input/{smd, psm, ...}/rate_{0,1,5,10,20}/
├── prepare/
│   └── prepare_redlamp_input.py
├── redlamp_lib/              # 직접 구현 필요 (위 "오염 주입" 참고, git에는 미포함)
│   └── injectors.py
├── run/
│   └── run_experiments.sh
├── recompute_metrics_test_only.py
├── analyze_tsbad_results.py
├── plot_paper_figures.py
└── results/{smd, psm, ...}/
```

> `TSB-AD/`는 공식 벤치마크 레포를 그대로 클론해 사용하며 이 저장소에는 포함하지
> 않습니다.

---

## Setup

```bash
git clone <this-repo-url>
cd sdn_anomaly
pip install -r requirements.txt

git clone https://github.com/TheDatumOrg/TSB-AD.git
```

`redlamp_lib/injectors.py`는 위 "오염 주입" 섹션을 참고해 직접 준비하세요.

---

## 실행 방법

```bash
# 1. 오염 주입
python prepare/prepare_redlamp_input.py --dataset smd
python prepare/prepare_redlamp_input.py --dataset psm

# 2. 모델 실행
bash run/run_experiments.sh smd 0
bash run/run_experiments.sh psm 0

# 3. 평가 후처리
python recompute_metrics_test_only.py --dataset smd
python recompute_metrics_test_only.py --dataset psm

# 4. 결과 취합 및 시각화
python analyze_tsbad_results.py --dataset smd
python analyze_tsbad_results.py --dataset psm
python plot_paper_figures.py --dataset smd psm
```

---

## References

[1] Q. Liu and J. Paparrizos, "The Elephant in the Room: Towards A Reliable
Time-Series Anomaly Detection Benchmark," NeurIPS Datasets and Benchmarks
Track, 2024.

[6] K. Obata, Y. Matsubara, and Y. Sakurai, "Robust and Explainable Detector
of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies," Proc.
ACM SIGKDD Conf. on Knowledge Discovery and Data Mining (KDD), 2025.
