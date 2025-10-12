import pandas as pd
import numpy as np
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

def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    diffs = x[:, None] - y[None, :]
    n = x.size * y.size
    delta = (np.sum(diffs > 0) - np.sum(diffs < 0)) / n
    cles = np.sum(diffs < 0) / n   # “越小越好”：P(x<y)
    return float(delta), float(cles)

def mwu_from_rank_csv(
    csv_path: str,
    rank_col: str = "Rank",
    group_col: str = "Severity",
    group_a: str = "RED",
    group_b: str = "YELLOW",
):
    # 读文件并清理列名的前后空格
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # 如果不存在 rank 列，则按当前顺序补一列（1=最好）
    if rank_col not in df.columns:
        df = df.reset_index(drop=False).rename(columns={"index": rank_col})
        df[rank_col] = df[rank_col].astype(int) + 1

    # 兼容 “Severity” 列里不小心带空格/大小写
    df[group_col] = df[group_col].astype(str).str.strip().str.upper()

    # 取两组数据
    a_vals = df.loc[df[group_col] == group_a, rank_col].astype(float).dropna().to_numpy()
    b_vals = df.loc[df[group_col] == group_b, rank_col].astype(float).dropna().to_numpy()

    if len(a_vals) == 0 or len(b_vals) == 0:
        raise ValueError(
            f"{group_a} 或 {group_b} 组没有数据；"
            f"现有分组值示例：{df[group_col].dropna().unique()[:10]}"
        )


    # 效应量
    delta, cles = cliffs_delta(a_vals, b_vals)

    return {
        "CLES_P(a<b)": cles,           # 在“越小越好”语义下，a 更好的概率
    }


result = mwu_from_rank_csv(Path(config["global_checks_ranking"]), rank_col="Rank",
                           group_a="RED", group_b="YELLOW",
                           )
print(result)
