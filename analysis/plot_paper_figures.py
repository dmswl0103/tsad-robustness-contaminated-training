# -*- coding: utf-8 -*-
"""
plot_paper_figures.py
--------------------------
논문용 figure 생성 스크립트. analyze_tsbad_results.py가 만든
all_results_test_only.csv를 읽어서, 본문용 2-패널(주 지표 2개) +
부록용 2x2 패널(4개 지표 전체) 그림을 만든다.

사용법:
    python plot_paper_figures.py --dataset insdn
    python plot_paper_figures.py --dataset smd
    python plot_paper_figures.py --dataset insdn smd   # 여러 개 한 번에

[config.py 리팩토링 반영 사항]
- 레거시 9개 데이터셋 키(smap/msl/smd_redlamp_mv*/insdn_ovs_fixed 등)를 지우고
  config.DATASETS(smd/insdn) / config.MODELS / config.MODEL_COLORS를 씀.
- is_shortfall(목표 오염 비율 미달 파일 제외) 로직 삭제 — analyze_tsbad_results.py와
  동일한 이유로, 지금 파이프라인에는 해당 개념 자체가 없음(그쪽 스크립트 주석 참고).
- **본문 2-패널에 쓰는 지표를 데이터셋마다 다르게 바꿈**: 기존엔 AUC-PR/VUS-PR을
  모든 데이터셋에 하드코딩해서 그렸는데, config.PRIMARY_METRIC에 정리해뒀듯
  InSDN은 test 이상 비율이 93%로 매우 높아서 AUC-PR/VUS-PR이 base rate에 눌려
  왜곡됨 — InSDN의 주 지표는 AUC-ROC/VUS-ROC임. 이제 이 스크립트는
  config.PRIMARY_METRIC[dataset_key]를 그대로 읽어서 본문 그림에 씀
  (SMD는 여전히 AUC-PR/VUS-PR, InSDN은 AUC-ROC/VUS-ROC로 자동으로 달라짐).
  부록용 2x2는 그대로 4개 지표(AUC-ROC/AUC-PR/VUS-ROC/VUS-PR) 전부 보여줌.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

ALL_METRICS = ["AUC-ROC", "AUC-PR", "VUS-ROC", "VUS-PR"]

# PCA/RobustPCA는 TSB-AD 공식 wrapper가 train+test 전체로 fit해서(leakage) 나머지
# 모델들과 공정 비교 대상이 아님 — 논문용 figure에서는 항상 제외. config.MODELS
# 자체는 안 건드림(학습/recompute 파이프라인엔 그대로 남아있어야 하니까), 여기
# plot 단계에서만 걸러냄.
EXCLUDE_MODELS = {"PCA", "RobustPCA"}


def get_model_order(df: pd.DataFrame, dataset_key: str | None = None,
                     models: list | None = None) -> list:
    """config.MODELS 순서를 기본으로 쓰되, CSV에 실제 존재하는 모델만 남기고,
    config.MODELS에 없는 모델(예: CNN/LSTMAD처럼 --models로 별도 추가 학습한 것)도
    자동으로 뒤에 붙여서 그린다 — config.py를 안 건드려도 새 모델이 그래프에 반영됨.
    dataset_key가 주어지면 config.EXTRA_EXCLUDE_MODELS[dataset_key]도 추가로 제외
    (예: SMAP/SWaT의 StreamVAE — raw score가 수치적으로 폭발해서 나온 아티팩트,
    § config.py EXTRA_EXCLUDE_MODELS 주석 참고. 학습/recompute 파이프라인이 쓰는
    config.MODELS 자체는 안 건드리고 plot 단계에서만 걸러냄).

    models가 주어지면(예: --models로 하락폭 상위 N개만 명시적으로 지정한 경우)
    EXCLUDE_MODELS/EXTRA_EXCLUDE_MODELS 필터는 적용하지 않고 그 목록 순서
    그대로(CSV에 실제 존재하는 것만) 씀 — 사용자가 이미 명시적으로 고른
    부분집합이므로 자동 제외 로직이 끼어들 이유가 없음. 이 경우 반드시
    파일명에 접미사를 붙여서(main()의 --suffix) 전체-모델 원본 그림을
    덮어쓰지 않도록 할 것."""
    if models is not None:
        present = set(df["model"].unique())
        return [m for m in models if m in present]
    exclude = set(EXCLUDE_MODELS)
    if dataset_key is not None:
        exclude |= config.EXTRA_EXCLUDE_MODELS.get(dataset_key, set())
    present = set(df["model"].unique()) - exclude
    ordered = [m for m in config.MODELS if m in present]
    extra = sorted(present - set(config.MODELS))
    return ordered + extra


def setup_korean_font():
    """한글 폰트가 있으면 적용, 없으면 경고만 출력하고 기본 폰트로 진행."""
    candidates = ["NanumGothic", "Malgun Gothic", "AppleGothic", "NanumBarunGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    print("⚠ 한글 폰트를 찾지 못했습니다. 라벨이 깨져 보이면 "
          "'sudo apt install fonts-nanum' 설치 후 다시 실행하세요.")


def _plot_panels(df: pd.DataFrame, metrics: list, label: str, ncols: int, figsize: tuple,
                  zoom: bool = False, dataset_key: str | None = None, models: list | None = None):
    """zoom=False: 모든 패널 y축 0~1.05 고정(지표 간 절대 비교용).
    zoom=True: 패널마다 실제 데이터 범위에 맞춰 y축을 확대(모델 간 상대적 차이가
    잘 보이게) — SMD의 AUC-ROC/VUS-ROC처럼 baseline이 높아 다닥다닥 붙어 보이는
    지표에 특히 유용. 두 버전 다 만들어서 같이 보는 걸 권장(절대 성능 수준은
    zoom 안 한 쪽에서, 오염 비율에 따른 상대적 변화는 zoom한 쪽에서 확인).
    models: 지정하면 그 모델들만(get_model_order 참고)."""
    fig, axes = plt.subplots(1, ncols, figsize=figsize) if ncols <= 2 else \
        plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    model_order = get_model_order(df, dataset_key, models=models)
    for ax, metric in zip(axes, metrics):
        values_all = []
        for model_name in model_order:
            grp = df[df["model"] == model_name]
            if grp.empty:
                continue
            agg = grp.groupby("contamination_rate")[metric].mean().sort_index()
            values_all.append(agg.values)
            ax.plot(
                agg.index * 100, agg.values,
                marker="o", markersize=5, linewidth=2,
                color=config.MODEL_COLORS.get(model_name), label=model_name,
            )
        ax.set_xlabel("Contamination Rate (%)", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        title = metric
        if zoom:
            title += "  (확대)"
        ax.set_title(title, fontsize=12, fontweight="bold")
        if zoom and values_all:
            import numpy as np
            flat = np.concatenate(values_all)
            lo, hi = float(flat.min()), float(flat.max())
            pad = max((hi - lo) * 0.15, 0.02)  # 범위가 거의 0이어도 최소 여백 확보
            ax.set_ylim(max(0, lo - pad), min(1.05, hi + pad))
        else:
            ax.set_ylim(0, 1.05)
        ax.set_xticks(sorted(df["contamination_rate"].unique() * 100))
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return fig, axes


def plot_one_dataset(df: pd.DataFrame, dataset_key: str, label: str, out_dir: Path,
                      metrics: list | None = None, filename: str = "paper_figure_main.png",
                      models: list | None = None):
    """본문용: 기본은 config.PRIMARY_METRIC[dataset_key] 2개(데이터셋마다 다름).
    metrics를 지정하면(예: --metrics로 Affiliation-F/PA-F1 등) 그걸 대신 씀 —
    이 경우 원래 AUC/VUS 그림을 덮어쓰지 않도록 filename도 같이 바꿔서 호출할 것.
    models를 지정하면 그 모델들만 그림(예: 하락폭 상위 N개) — 이 경우도 filename을
    바꿔서 전체-모델 원본을 덮어쓰지 않을 것."""
    if metrics is None:
        metrics = config.PRIMARY_METRIC.get(dataset_key, ["AUC-PR", "VUS-PR"])
    fig, axes = _plot_panels(df, metrics, label, ncols=2, figsize=(10, 4.2),
                              dataset_key=dataset_key, models=models)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.08), fontsize=9)

    fig.suptitle(
        f"Anomaly Detection Performance vs Contamination Rate ({label})\n"
        f"(primary metrics: {', '.join(metrics)})",
        fontsize=13, fontweight="bold", y=1.05,
    )
    fig.tight_layout()

    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"저장: {out_path}")
    plt.close(fig)


def plot_one_dataset_full(df: pd.DataFrame, label: str, out_dir: Path, zoom: bool = False,
                           dataset_key: str | None = None, metrics: list | None = None,
                           filename_base: str = "paper_figure_full", models: list | None = None):
    """부록용: 기본은 AUC-ROC/AUC-PR/VUS-ROC/VUS-PR 4개 지표 2x2 패널.
    metrics를 지정하면(정확히 4개 권장) 그 지표들로 대신 그림 — 예:
    Affiliation-F/PA-F1/Event-based-F1/R-based-F1. filename_base도 같이 바꿔서
    원래 AUC/VUS 그림(paper_figure_full.png)을 덮어쓰지 않게 할 것.
    zoom=True면 {filename_base}_zoomed.png로 별도 저장(원본은 그대로 유지).
    models를 지정하면 그 모델들만 그림(예: 하락폭 상위 N개)."""
    plot_metrics = metrics if metrics is not None else ALL_METRICS
    fig, axes = _plot_panels(df, plot_metrics, label, ncols=4, figsize=(11, 8.5), zoom=zoom,
                              dataset_key=dataset_key, models=models)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=9)

    metric_desc = "All Metrics" if metrics is None else ", ".join(plot_metrics)
    title = f"Anomaly Detection Performance vs Contamination Rate - {metric_desc} ({label})"
    if zoom:
        title += "\n(y축을 지표별 실제 데이터 범위로 확대 — 절대 성능 비교는 zoom 없는 버전 참고)"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.03 if zoom else 1.01)
    fig.tight_layout()

    suffix = "_zoomed" if zoom else ""
    out_path = out_dir / f"{filename_base}{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"저장: {out_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", choices=list(config.DATASETS.keys()), required=True)
    ap.add_argument("--metrics", nargs="+", default=None,
                     help="기본값: 데이터셋별 PRIMARY_METRIC(본문 2개) + AUC/VUS 4개(부록). "
                          "지정하면(예: Affiliation-F PA-F1 Event-based-F1 R-based-F1) 그 지표들로 "
                          "대신 그리고, 파일명에 --suffix를 붙여 기존 AUC/VUS 그림을 덮어쓰지 않음.")
    ap.add_argument("--suffix", default=None,
                     help="--metrics 또는 --models와 함께 쓸 때 파일명 구분용 접미사(예: "
                          "affiliation, top6decline). 안 주면 --metrics만 줬을 땐 "
                          "'_altmetrics', --models만 줬을 땐 '_selected'를 자동 사용.")
    ap.add_argument("--models", nargs="+", default=None,
                     help="지정한 모델만 그림(예: get_top_decline_models.py로 뽑은 하락폭 "
                          "상위 N개). EXCLUDE_MODELS/EXTRA_EXCLUDE_MODELS 자동 제외는 "
                          "적용 안 됨(이미 명시적으로 고른 목록이므로). 반드시 --suffix와 "
                          "같이 써서 전체-모델 원본 그림을 덮어쓰지 않도록 할 것 — "
                          "본문/부록에 쓸 땐 전체-모델 버전과 같이 제시하고 '하락폭 상위 "
                          "N개(전체 M개 중)'라고 반드시 명시할 것.")
    args = ap.parse_args()

    setup_korean_font()

    suffix = ""
    if args.metrics is not None or args.models is not None:
        if args.suffix:
            suffix = f"_{args.suffix}"
        elif args.metrics is not None and args.models is not None:
            suffix = "_altmetrics_selected"
        elif args.metrics is not None:
            suffix = "_altmetrics"
        else:
            suffix = "_selected"

    for key in args.dataset:
        ds_cfg = config.DATASETS[key]
        out_dir = Path(config.RESULTS_DIR) / key
        csv_path = out_dir / "all_results_test_only.csv"
        if not csv_path.exists():
            print(f"⚠ {csv_path} 없음, {key} 스킵 (analyze_tsbad_results.py --dataset {key} 먼저 실행하세요)")
            continue
        df = pd.read_csv(csv_path)
        missing = [m for m in (args.metrics or []) if m not in df.columns]
        if missing:
            print(f"⚠ [{key}] csv에 없는 지표라 스킵: {missing} (실제 컬럼: {list(df.columns)})")
            continue
        if args.models is not None:
            present = set(df["model"].unique())
            missing_models = [m for m in args.models if m not in present]
            if missing_models:
                print(f"⚠ [{key}] csv에 없는 모델이라 무시됨: {missing_models} "
                      f"(실제 모델: {sorted(present)})")
        main_metrics = args.metrics[:2] if args.metrics else None
        full_metrics = args.metrics if args.metrics else None
        plot_one_dataset(df, key, ds_cfg["label"], out_dir, metrics=main_metrics,
                          filename=f"paper_figure_main{suffix}.png", models=args.models)
        plot_one_dataset_full(df, ds_cfg["label"], out_dir, dataset_key=key,
                               metrics=full_metrics, filename_base=f"paper_figure_full{suffix}",
                               models=args.models)
        plot_one_dataset_full(df, ds_cfg["label"], out_dir, zoom=True, dataset_key=key,
                               metrics=full_metrics, filename_base=f"paper_figure_full{suffix}",
                               models=args.models)


if __name__ == "__main__":
    main()