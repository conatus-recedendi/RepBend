"""
analysis/plot_relearning_analysis.py
=====================================
sweep_unlearn_relearn.sh 가 생성한 results.csv를 읽어
Unlearning 파라미터 ↔ Relearning 취약성 관계를 시각화합니다.

생성 Figure 목록
-----------------
Fig 1. [scatter] ASR_after_unlearn vs ASR_delta (가설 핵심: 가장 잘 지운 설정이 더 취약한가?)
Fig 2. [heatmap] (layer_start × window_size) → ASR_delta  (per relearning_mode)
Fig 3. [bar]     relearning_mode별 평균 ASR 비교 (unlearn / relearn)
Fig 4. [scatter] loss 계수(alpha/epsilon) vs ASR_delta  (파라미터 민감도)
Fig 5. [line]    top-K unlearn 설정 vs ASR 변화 trajectory
Fig 6. [heatmap] 종합 취약성 지도 (run_id × relearning_mode → ASR_delta)

사용법:
  python analysis/plot_relearning_analysis.py \
      --results_csv ./out/sweep_relearning/results.csv \
      --output_dir  ./out/sweep_relearning/figures
"""

import argparse
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

# ─── 스타일 설정 ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = {"direct": "#e6194B", "low_budget": "#f58231", "benign": "#3cb44b"}
MODE_LABELS = {
    "direct":     "(A) Direct Relearning\n[Unlearning Isn't Deletion]",
    "low_budget": "(B) Low-Budget Relearning\n[Do Unlearning Methods Remove Info?]",
    "benign":     "(C) Benign Relearning\n[Unlearning or Obfuscating?]",
}


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["asr_after_unlearn", "asr_after_relearn", "asr_delta"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["asr_after_unlearn", "asr_after_relearn", "asr_delta"])
    # 퍼센트 변환 (0~1 범위면 ×100)
    for col in ["asr_after_unlearn", "asr_after_relearn", "asr_delta"]:
        if df[col].abs().max() <= 1.01:
            df[col] = df[col] * 100
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1. Scatter: ASR_after_unlearn vs ASR_delta
#   가설: unlearning이 잘 된(ASR 낮은) 설정 → delta 높음(relearning에 더 취약)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_scatter_unlearn_vs_delta(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, len(df["relearning_mode"].unique()),
                             figsize=(6 * len(df["relearning_mode"].unique()), 5),
                             sharey=False)
    if not hasattr(axes, "__len__"):
        axes = [axes]

    for ax, mode in zip(axes, sorted(df["relearning_mode"].unique())):
        sub = df[df["relearning_mode"] == mode]
        ax.scatter(
            sub["asr_after_unlearn"], sub["asr_delta"],
            color=PALETTE.get(mode, "steelblue"),
            alpha=0.75, edgecolors="white", linewidths=0.5, s=80,
        )
        # 회귀선
        z = np.polyfit(sub["asr_after_unlearn"], sub["asr_delta"], 1)
        p = np.poly1d(z)
        xs = np.linspace(sub["asr_after_unlearn"].min(), sub["asr_after_unlearn"].max(), 100)
        ax.plot(xs, p(xs), "--", color="gray", linewidth=1.2)

        corr = sub[["asr_after_unlearn", "asr_delta"]].corr().iloc[0, 1]
        ax.set_title(textwrap.fill(MODE_LABELS.get(mode, mode), width=30), fontsize=11)
        ax.set_xlabel("ASR after Unlearning (%)")
        ax.set_ylabel("ΔASR (Relearning − Unlearning, %)")
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.text(0.97, 0.05, f"r = {corr:.2f}", transform=ax.transAxes,
                ha="right", fontsize=10, color="gray")

    fig.suptitle(
        "Fig 1. Do Better-Unlearned Models Recover More Easily?\n"
        "(Hypothesis: lower ASR after unlearn → higher ΔASR after relearning)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_scatter_unlearn_vs_delta.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 1] saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2. Heatmap: layer_start × window_size → ASR_delta
# ─────────────────────────────────────────────────────────────────────────────
def fig2_heatmap_layer_delta(df: pd.DataFrame, out_dir: str):
    modes = sorted(df["relearning_mode"].unique())
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 5))
    if not hasattr(axes, "__len__"):
        axes = [axes]

    for ax, mode in zip(axes, modes):
        sub = df[df["relearning_mode"] == mode]
        pivot = sub.pivot_table(
            index="layer_start", columns="window_size",
            values="asr_delta", aggfunc="mean"
        )
        vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1e-3)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        sns.heatmap(
            pivot, ax=ax, cmap="RdYlGn_r", norm=norm,
            annot=True, fmt=".1f", linewidths=0.4, linecolor="white",
            cbar_kws={"label": "ΔASR (%)"},
        )
        ax.set_title(textwrap.fill(MODE_LABELS.get(mode, mode), width=28), fontsize=10)
        ax.set_xlabel("Window Size (# layers)")
        ax.set_ylabel("Layer Start Index")

    fig.suptitle(
        "Fig 2. Relearning Vulnerability by Layer Configuration\n"
        "(Red = high Δ, Green = resistant to relearning)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_heatmap_layer_delta.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 2] saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3. Bar: relearning_mode별 평균 ASR 비교
# ─────────────────────────────────────────────────────────────────────────────
def fig3_bar_mode_comparison(df: pd.DataFrame, out_dir: str):
    agg = (
        df.groupby("relearning_mode")[["asr_after_unlearn", "asr_after_relearn"]]
        .mean()
        .reset_index()
    )
    x = np.arange(len(agg))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, agg["asr_after_unlearn"], width,
                   label="After Unlearning", color="#4878d0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, agg["asr_after_relearn"], width,
                   label="After Relearning", color="#ee854a", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [textwrap.fill(MODE_LABELS.get(m, m), width=20) for m in agg["relearning_mode"]],
        fontsize=9,
    )
    ax.set_ylabel("Mean ASR (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()
    ax.set_title("Fig 3. Mean ASR Before and After Relearning by Mode")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_bar_mode_comparison.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 3] saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4. Scatter: 손실 계수(alpha, epsilon) vs ASR_delta
# ─────────────────────────────────────────────────────────────────────────────
def fig4_loss_coeff_sensitivity(df: pd.DataFrame, out_dir: str):
    modes = sorted(df["relearning_mode"].unique())
    params = [("alpha", "α (Safe Push)"), ("epsilon", "ε (Unsafe Push)")]
    fig, axes = plt.subplots(len(params), len(modes),
                             figsize=(5 * len(modes), 4 * len(params)))
    if axes.ndim == 1:
        axes = axes.reshape(len(params), len(modes))

    for row, (param, param_label) in enumerate(params):
        for col, mode in enumerate(modes):
            ax = axes[row][col]
            sub = df[df["relearning_mode"] == mode]
            if param not in sub.columns or sub[param].nunique() < 2:
                ax.set_visible(False)
                continue

            for val, grp in sub.groupby(param):
                ax.scatter(
                    [val] * len(grp), grp["asr_delta"],
                    alpha=0.6, s=60, color=PALETTE.get(mode, "steelblue"),
                )
            means = sub.groupby(param)["asr_delta"].mean()
            ax.plot(means.index, means.values, "o-", color="black", linewidth=1.5,
                    markersize=5, label="mean")
            ax.set_xlabel(param_label)
            ax.set_ylabel("ΔASR (%)" if col == 0 else "")
            ax.set_title(
                f"{textwrap.fill(MODE_LABELS.get(mode, mode), 22)}\n(vs {param_label})",
                fontsize=9,
            )
            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")

    fig.suptitle(
        "Fig 4. Loss Coefficient Sensitivity: How Does α/ε Affect Relearning Vulnerability?",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_loss_coeff_sensitivity.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 4] saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5. Line: Top-K / Bottom-K unlearning 설정별 ASR 비교
#   "가장 잘 지운 K개" vs "덜 지운 K개"의 relearning ASR 분포 비교
# ─────────────────────────────────────────────────────────────────────────────
def fig5_topk_trajectory(df: pd.DataFrame, out_dir: str, k: int = 5):
    fig, axes = plt.subplots(1, len(df["relearning_mode"].unique()),
                             figsize=(6 * len(df["relearning_mode"].unique()), 5))
    if not hasattr(axes, "__len__"):
        axes = [axes]

    for ax, mode in zip(axes, sorted(df["relearning_mode"].unique())):
        sub = df[df["relearning_mode"] == mode].copy()
        sub = sub.sort_values("asr_after_unlearn")
        top_k = sub.head(k)      # 가장 잘 지운 K개 (ASR 가장 낮음)
        bot_k = sub.tail(k)      # 덜 지운 K개 (ASR 높음)

        stages = ["After Unlearning", "After Relearning"]
        for i, (label, subset, color) in enumerate([
            (f"Best Unlearn (top-{k})", top_k, "#e6194B"),
            (f"Worst Unlearn (bot-{k})", bot_k, "#4878d0"),
        ]):
            vals = [
                subset["asr_after_unlearn"].mean(),
                subset["asr_after_relearn"].mean(),
            ]
            errs = [
                subset["asr_after_unlearn"].std(),
                subset["asr_after_relearn"].std(),
            ]
            ax.errorbar(stages, vals, yerr=errs, marker="o", label=label,
                        color=color, linewidth=2, capsize=5)

        ax.set_ylabel("ASR (%)")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(fontsize=9)
        ax.set_title(textwrap.fill(MODE_LABELS.get(mode, mode), width=28), fontsize=10)

    fig.suptitle(
        f"Fig 5. Best-Unlearned Configs Are More Vulnerable to Relearning\n"
        f"(Top-{k} vs Bottom-{k} by ASR after unlearning)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_topk_trajectory.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 5] saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6. Heatmap: run_id × relearning_mode → ASR_delta (종합 취약성 지도)
# ─────────────────────────────────────────────────────────────────────────────
def fig6_vulnerability_map(df: pd.DataFrame, out_dir: str):
    pivot = df.pivot_table(
        index="run_id", columns="relearning_mode",
        values="asr_delta", aggfunc="mean",
    )
    # run_id를 직접 unlearn ASR 평균으로 정렬
    order = (
        df.groupby("run_id")["asr_after_unlearn"].mean()
        .sort_values()
        .index
    )
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2.5), max(8, len(pivot) * 0.45)))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1e-3)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn_r", norm=norm,
        annot=True, fmt=".1f", linewidths=0.3, linecolor="white",
        cbar_kws={"label": "ΔASR (%) after relearning"},
    )
    ax.set_xlabel("Relearning Mode")
    ax.set_ylabel("Unlearning Config (sorted by ASR↑ = worse unlearning)")
    ax.set_title(
        "Fig 6. Comprehensive Relearning Vulnerability Map\n"
        "(sorted top→bottom: best→worst unlearning performance)",
        fontsize=11,
    )
    ax.set_xticklabels(
        [MODE_LABELS.get(c, c).split("\n")[0] for c in pivot.columns],
        rotation=15, ha="right",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "fig6_vulnerability_map.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 6] saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 요약 통계 출력
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  Relearning Experiment Summary")
    print("=" * 60)
    print(f"  Total runs      : {df['run_id'].nunique()}")
    print(f"  Relearning modes: {sorted(df['relearning_mode'].unique())}")
    print(f"  Total rows      : {len(df)}")
    print()
    for mode, grp in df.groupby("relearning_mode"):
        print(f"  [{mode}]")
        print(f"    ASR after unlearn  : {grp['asr_after_unlearn'].mean():.1f}% ± {grp['asr_after_unlearn'].std():.1f}%")
        print(f"    ASR after relearn  : {grp['asr_after_relearn'].mean():.1f}% ± {grp['asr_after_relearn'].std():.1f}%")
        print(f"    Mean ΔASR          : {grp['asr_delta'].mean():.1f}%")
        print(f"    Max  ΔASR          : {grp['asr_delta'].max():.1f}%  (run: {grp.loc[grp['asr_delta'].idxmax(), 'run_id']})")
        corr = grp[["asr_after_unlearn", "asr_delta"]].corr().iloc[0, 1]
        print(f"    Corr(unlearn_ASR, delta): {corr:.3f}  {'← 가설 지지 (음의 상관)' if corr < -0.2 else ''}")
        print()
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.results_csv)
    if df.empty:
        print("[ERROR] 데이터가 없습니다. CSV를 확인하세요.")
        return

    print_summary(df)

    fig1_scatter_unlearn_vs_delta(df, args.output_dir)
    fig2_heatmap_layer_delta(df, args.output_dir)
    fig3_bar_mode_comparison(df, args.output_dir)
    fig4_loss_coeff_sensitivity(df, args.output_dir)
    fig5_topk_trajectory(df, args.output_dir, k=args.top_k)
    fig6_vulnerability_map(df, args.output_dir)

    print(f"\n모든 Figure 저장 완료 → {args.output_dir}")


if __name__ == "__main__":
    main()
