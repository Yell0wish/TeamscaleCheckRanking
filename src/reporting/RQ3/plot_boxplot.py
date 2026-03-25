import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
import json
from pathlib import Path

def load_config(config_path: str = "config.json"):
    cfg = {}
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return cfg

config = load_config()

_TEAL = "#6aa6a5"


def _make_box(ax, data, labels):
    colors = []
    for lab in labels:
        if "RED" in lab.upper():
            colors.append("#d62728")   # 红色
        elif "YELLOW" in lab.upper():
            colors.append("#ffbf00")   # 黄色
        else:
            colors.append(_TEAL)       # 默认青色

    bp = ax.boxplot(
        data,
        widths=0.35,
        vert=True,
        whis=1.5,
        patch_artist=True,
        showmeans=True,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        meanprops=dict(marker='s', markerfacecolor="white", markeredgecolor="black", markersize=6),
        flierprops=dict(marker='D', markerfacecolor="black", markeredgecolor="black", markersize=4, alpha=0.6)
    )

    # 分别上色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35)
    return bp


def _legend(fig):
    legend_elems = [
        Patch(facecolor='none', edgecolor="black", label="25%–75%"),
        Line2D([0], [0], color="black", lw=1.0, label="Range within 1.5 IQR"),
        Line2D([0], [0], color="black", lw=1.2, label="Median"),
        Line2D([0], [0], marker='s', linestyle='None', markerfacecolor='white', markeredgecolor='black',
               label="Mean"),
        Line2D([0], [0], marker='D', linestyle='None', color='black', label="Outliers"),
    ]
    # 返回 legend 以便 tight 导出时纳入 bbox
    leg = fig.legend(handles=legend_elems,
                     loc="upper center",
                     bbox_to_anchor=(0.5, 0.965), 
                     ncol=5,
                     frameon=True,
                     fontsize=10,                  # 字号略小，避免拥挤
                     borderaxespad=0.6)
    return leg


def plot_severity_box_from_csv(
    csv_path: str,
    value_col: str = "RP value",
    severity_col: str = "Severity",
    save_svg: str = "severity_rank_box.svg",
):

    df = pd.read_csv(csv_path)
    # 清洗列名与 Severity 值
    df.columns = [c.strip() for c in df.columns]
    if value_col not in df.columns or severity_col not in df.columns:
        raise ValueError(f"列不存在：{value_col} 或 {severity_col}。现有列：{list(df.columns)}")

    df[severity_col] = df[severity_col].astype(str).str.strip().str.upper()

    # 生成 Rank（RP value 越小 Rank 越靠前）
    df = df.sort_values(by=value_col, ascending=True, kind="mergesort").reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)

    # 分组取 rank
    red_ranks = df.loc[df[severity_col] == "RED", "Rank"].astype(float).to_numpy()
    yel_ranks = df.loc[df[severity_col] == "YELLOW", "Rank"].astype(float).to_numpy()

    if red_ranks.size == 0 or yel_ranks.size == 0:
        raise ValueError(f"RED (n={red_ranks.size}) 或 YELLOW (n={yel_ranks.size}) 为空，无法绘图。")

    # 画图
    # 宽度跟随组数（这里只有两个箱子，给一点横向空间即可）
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    data = [yel_ranks.tolist(), red_ranks.tolist()]  # 左黄右红，可按需调换
    labels = [f"YELLOW (n={len(yel_ranks)})", f"RED (n={len(red_ranks)})"]

    _make_box(ax, data, labels) #, color=_TEAL)

    max_rank = int(df["Rank"].max())
    headroom = max(1, int(0.03 * max_rank))
    ax.set_ylim(1, max_rank + headroom)
    ax.set_ylabel("Rank (lower is better)", fontsize=11)

    # ax.set_title(full_title, fontsize=12, pad=8)

    _legend(fig)
    plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.88])
    fig.savefig(save_svg, bbox_inches="tight", dpi=300)
    print(f"Saved: {save_svg}")

if __name__ == "__main__":

    csv_file = Path(config["global_checks_ranking"])
    plot_severity_box_from_csv(
        csv_path=csv_file,
        value_col="RP value",
        severity_col="Severity",
        save_svg= Path(config["output_dir"]) / "severity_rank_box.svg",
    )
