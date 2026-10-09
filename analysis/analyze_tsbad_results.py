# -*- coding: utf-8 -*-
"""
analyze_tsbad_results.py
--------------------------
smd/insdn 공용 결과 취합/시각화 스크립트. 반드시
recompute_metrics_test_only.py가 만든 *_test_only.csv만 읽는다
(leakage 있는 Run_Detector_M.py의 원본 결과는 절대 안 씀)

사용법:
    python analyze_tsbad_results.py --dataset smd
    python analyze_tsbad_results.py --dataset insdn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

PRIMARY_METRICS = ["AUC-ROC", "AUC-PR", "VUS-ROC", "VUS-PR"]


def load_all_results(dataset_key: str) -> pd.DataFrame:
    metrics_dir = Path(config.RESULTS_DIR) / dataset_key / "metrics_test_only"
    csv_paths = sorted(metrics_dir.glob("*_test_only.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"{metrics_dir}에 *_test_only.csv가 없습니다. "
            f"python recompute_metrics_test_only.py --dataset {dataset_key} 먼저 실행하세요."
        )
    # recompute_metrics_test_only.py가 이미 model / contamination_rate 컬럼을 채워둠
    return pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)


def plot_results(df: pd.DataFrame, out_dir: Path, dataset_key: str, label: str,
                  max_rate: float | None = None, suffix: str = ""):
    plot_df = df if max_rate is None else df[df["contamination_rate"] <= max_rate]
    primary_metrics = set(config.PRIMARY_METRIC.get(dataset_key, []))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, PRIMARY_METRICS):
        for model_name, grp in plot_df.groupby("model"):
            agg = grp.groupby("contamination_rate")[metric].mean().sort_index()
            ax.plot(agg.index * 100, agg.values, marker="o", linewidth=2,
                     label=model_name, color=config.MODEL_COLORS.get(model_name))
        ax.set_xlabel("Contamination Rate (%)")
        ax.set_ylabel(f"{metric} (mean)")
        title = f"{metric} vs Contamination Rate ({label})"
        if metric in primary_metrics:
            title += "  ★ primary"
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Anomaly Detection Robustness under Data Contamination ({label})",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = out_dir / f"contamination_results{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"저장: {out_path}")
    plt.close()


def summarize(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    summary = (df.groupby(["model", "contamination_rate"])[PRIMARY_METRICS]
               .agg(["mean", "std"]).round(4))
    out_path = out_dir / "summary.csv"
    summary.to_csv(out_path)
    print(summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(config.DATASETS.keys()), required=True)
    args = ap.parse_args()

    ds_cfg = config.DATASETS[args.dataset]
    out_dir = Path(config.RESULTS_DIR) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = load_all_results(args.dataset)
    all_results.to_csv(out_dir / "all_results_test_only.csv", index=False)

    summarize(all_results, out_dir)
    plot_results(all_results, out_dir, args.dataset, ds_cfg["label"])  # 전체 rate
    plot_results(all_results, out_dir, args.dataset, ds_cfg["label"],
                 max_rate=0.10, suffix="_upto10pct")  # 10%까지만


if __name__ == "__main__":
    main()