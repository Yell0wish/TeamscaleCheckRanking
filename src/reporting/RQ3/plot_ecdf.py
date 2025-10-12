import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def _prepare_rank_df(csv_path, rank_col="Rank", severity_col="Severity", value_col="RP value"):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if severity_col not in df.columns:
        raise ValueError(f"缺少列：{severity_col}")
    df[severity_col] = df[severity_col].astype(str).str.strip().str.upper()

    if rank_col in df.columns:
        # 已有 Rank：按 Rank 升序并规范成 1..N
        df = df.sort_values(by=rank_col, ascending=True, kind="mergesort").reset_index(drop=True)
        df[rank_col] = np.arange(1, len(df) + 1)
    else:
        # 无 Rank：用 RP value 升序生成 Rank；若也没有 value，则用当前顺序
        if value_col in df.columns:
            df = df.sort_values(by=value_col, ascending=True, kind="mergesort").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        df[rank_col] = np.arange(1, len(df) + 1)

    return df[[rank_col, severity_col]]

def _ecdf_by_rank(ranks, max_k=None):
    """返回等长 x(1..max_k) 与 ECDF y: y[k-1] = P(rank ≤ k)"""
    ranks = np.asarray(ranks, dtype=int)
    n = ranks.size
    if n == 0:
        return np.array([]), np.array([])
    if max_k is None:
        max_k = int(ranks.max())
    ks = np.arange(1, max_k + 1)
    # 计数：每个 k 的累积 ≤k 的数量
    counts = np.zeros(max_k + 1, dtype=int)
    np.add.at(counts, ranks, 1)          # 在对应名次处加1
    cumsum = np.cumsum(counts)            # 从 0..max_k
    y = cumsum[1:] / n                    # 丢掉 index=0，得到 1..max_k
    return ks, y

def plot_ecdf_by_severity(
    csv_path: str,
    rank_col: str = "Rank",
    severity_col: str = "Severity",
    value_col: str = "RP value",
    save_svg: str = "ecdf.svg",
    k_refs = (10, 20, 30, 40, 50, 60) 
):
    df = _prepare_rank_df(csv_path, rank_col, severity_col, value_col)
    df = df.sort_values(by=rank_col, ascending=True, kind="mergesort").reset_index(drop=True)

    red = df.loc[df[severity_col] == "RED", rank_col].to_numpy()
    yel = df.loc[df[severity_col] == "YELLOW", rank_col].to_numpy()

    if red.size == 0 or yel.size == 0:
        raise ValueError(f"RED(n={red.size}) 或 YELLOW(n={yel.size}) 为空，无法绘图。")

    max_k = int(df[rank_col].max())
    x_yel, y_yel = _ecdf_by_rank(yel, max_k=max_k)
    x_red, y_red = _ecdf_by_rank(red, max_k=max_k)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    # 使用阶梯线展示 ECDF
    ax.step(x_yel, y_yel, where="post", label=f"YELLOW (n={yel.size})")
    ax.step(x_red, y_red, where="post", label=f"RED (n={red.size})")

    # 参考阈值线（可选）
    for k in k_refs:
        if 1 <= k <= max_k:
            ax.axvline(k, linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlim(1, max_k)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Rank threshold k")
    ax.set_ylabel("Within-group proportion with rank ≤ k")
    # ax.set_title("ECDF of global ranks by severity")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    fig.savefig(save_svg, bbox_inches="tight", dpi=300)
    print(f"Saved: {save_svg} | N_total={len(df)}, RED={red.size}, YELLOW={yel.size}")

    # 统计在整数 k 上 YELLOW > RED 的情形
    eps = 1e-12  # 容差，避免浮点相等误判
    assert np.array_equal(x_yel, x_red), "x 轴不一致，无法逐点比较"

    mask_yel_gt = (y_yel - y_red) > eps
    ks_yel_gt = x_yel[mask_yel_gt]

    print(f"[YELLOW > RED] 的整数 k 个数: {ks_yel_gt.size}")

plot_ecdf_by_severity(Path(config["global_checks_ranking"]), save_svg=Path(config["output_dir"]) / "ecdf.svg")
