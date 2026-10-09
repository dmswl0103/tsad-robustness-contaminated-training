# -*- coding: utf-8 -*-
"""
summarize_contamination_delta.py
--------------------------------
"어느 데이터셋에서 오염 비율(rate) 증가에 따라 성능이 실제로 떨어지는지"를
감이 아니라 숫자로 보기 위한 스크립트. 각 데이터셋 x 모델별로

    delta = (rate=20%에서의 지표 값) - (rate=0%에서의 지표 값)

를 기본으로는 config.PRIMARY_METRIC[dataset]의 두 지표(대개 AUC-PR/VUS-PR,
InSDN만 AUC-ROC/VUS-ROC) 각각에 대해 계산한다. PCA/RobustPCA(leakage로 항상
제외)와 config.EXTRA_EXCLUDE_MODELS(예: SMAP/SWaT의 StreamVAE, 수치 폭발
아티팩트)도 plot_paper_figures.py와 동일하게 제외한다.

[--metrics로 다른 지표 지정 가능] recompute_metrics_test_only.py가 쓰는
TSB_AD.evaluation.metrics.get_metrics()는 애초에 AUC-ROC/AUC-PR/VUS-ROC/
VUS-PR 4개뿐 아니라 Standard-F1/PA-F1/Event-based-F1/R-based-F1/
Affiliation-F까지 총 9개 지표를 한 번에 계산해서 반환한다 — 그래서
results/<ds>/metrics_test_only/*.csv에는 이미 이 9개 컬럼이 전부 저장돼
있고, 재학습은커녕 재계산(recompute)도 새로 할 필요 없이 --metrics로
원하는 컬럼만 골라서 바로 델타를 뽑을 수 있다. 예:
    python summarize_contamination_delta.py --dataset smd \
        --metrics Affiliation-F PA-F1 Event-based-F1 R-based-F1
(컬럼명이 실제 csv와 다르면 조용히 스킵되니, 먼저
`head -1 results/smd/metrics_test_only/AutoEncoder_test_only.csv`로
정확한 컬럼명부터 확인할 것.)

dataset별 요약(모델 평균 delta, delta<0인 모델 수/전체)까지 같이 찍어서
"이 데이터셋은 오염에 따라 전반적으로 떨어지는 패턴이 보이는지" 한눈에
비교할 수 있게 한다. recompute_metrics_test_only.py가 만든
results/<ds>/metrics_test_only/*.csv만 읽는다(leakage 있는 원본 metrics는 안 씀).

사용법:
    python summarize_contamination_delta.py                     # config.DATASETS에서 all_results_test_only.csv 있는 것 전부
    python summarize_contamination_delta.py --dataset smd msl   # 일부만
    python summarize_contamination_delta.py --dataset smd --metrics Affiliation-F PA-F1  # 다른 지표로
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

EXCLUDE_MODELS_GLOBAL = {"PCA", "RobustPCA"}


def load_dataset(dataset_key: str) -> pd.DataFrame | None:
    metrics_dir = Path(config.RESULTS_DIR) / dataset_key / "metrics_test_only"
    csv_paths = sorted(metrics_dir.glob("*_test_only.csv"))
    if not csv_paths:
        return None
    return pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)


def summarize(dataset_key: str, df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    exclude = EXCLUDE_MODELS_GLOBAL | config.EXTRA_EXCLUDE_MODELS.get(dataset_key, set())
    df = df[~df["model"].isin(exclude)]

    if metrics is None:
        metrics = config.PRIMARY_METRIC.get(dataset_key, ["AUC-PR", "VUS-PR"])
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        print(f"⚠ [{dataset_key}] csv에 없는 지표라 스킵: {missing} "
              f"(실제 컬럼: {list(df.columns)})")
    rates = sorted(df["contamination_rate"].unique())
    if 0 not in rates:
        print(f"⚠ [{dataset_key}] rate=0 데이터가 없어 delta 계산 불가, 스킵")
        return pd.DataFrame()
    r_max = max(rates)

    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        agg = df.groupby(["model", "contamination_rate"])[metric].mean().unstack()
        if 0 not in agg.columns or r_max not in agg.columns:
            continue
        for model_name in agg.index:
            v0 = agg.loc[model_name, 0]
            v_end = agg.loc[model_name, r_max]
            if pd.isna(v0) or pd.isna(v_end):
                continue
            rows.append({
                "dataset": dataset_key,
                "model": model_name,
                "metric": metric,
                f"rate0": v0,
                f"rate{int(r_max*100)}": v_end,
                "delta": v_end - v0,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=list(config.DATASETS.keys()))
    ap.add_argument("--metrics", nargs="+", default=None,
                     help="기본값: config.PRIMARY_METRIC[dataset] (데이터셋별 2개). "
                          "지정하면 전 데이터셋에 이 지표들을 공통으로 씀 — 예: "
                          "Affiliation-F PA-F1 Event-based-F1 R-based-F1")
    args = ap.parse_args()

    all_rows = []
    for ds in args.dataset:
        df = load_dataset(ds)
        if df is None:
            print(f"⚠ [{ds}] results/{ds}/metrics_test_only/*.csv 없음, 스킵 "
                  f"(recompute_metrics_test_only.py --dataset {ds} 먼저 실행)")
            continue
        summary = summarize(ds, df, metrics=args.metrics)
        if summary.empty:
            continue
        all_rows.append(summary)

        print(f"\n===== {ds} =====")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        mean_delta = summary.groupby("metric")["delta"].mean()
        n_neg = summary.groupby("metric")["delta"].apply(lambda s: (s < 0).sum())
        n_total = summary.groupby("metric")["delta"].size()
        for metric in mean_delta.index:
            print(f"  -> {metric}: 모델 평균 delta={mean_delta[metric]:+.4f}, "
                  f"delta<0(하락)인 모델 {n_neg[metric]}/{n_total[metric]}개")

    if not all_rows:
        print("\n요약할 데이터가 없음.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    out_path = Path(config.RESULTS_DIR) / "contamination_delta_summary.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n전체 결과 저장: {out_path}")


if __name__ == "__main__":
    main()