import re
import pandas as pd
import seaborn as sns
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

RESULTS_DIR = Path(config["output_dir"])

def replace_with_rank(csv_path, output_path=None, drop_cols=None, custom_ascending=None):
    df = pd.read_csv(csv_path)
    # 统一列名：Hit@k -> Accuracy@k
    df.rename(columns=lambda c: re.sub(r'(?i)^hit@', 'Accuracy@', str(c)), inplace=True)


    # 方法名列：兼容 "Method" 或 "Name"
    id_col = next((c for c in ["Method", "Name"] if c in df.columns), None)
    if id_col is None:
        raise ValueError("未找到方法名列，请确保存在 'Method' 或 'Name' 列")

    # —— 标准化要丢的列名（同时兼容是否带百分号）——
    drop_candidates = set()
    if drop_cols:
        for c in drop_cols:
            c_no_pct = c.replace("%", "")
            drop_candidates.add(c_no_pct)
            drop_candidates.add(c_no_pct + "%")  # 同名带百分号版本

    # 实际要丢的列（与现有列取交集）
    to_drop = [c for c in df.columns if c != id_col and (c in drop_candidates or c.replace("%", "") in drop_candidates)]
    df_kept = df.drop(columns=to_drop, errors='ignore').copy()

    # 需要排名的列（数值列）
    num_cols = [c for c in df_kept.columns if c != id_col and pd.api.types.is_numeric_dtype(df_kept[c])]

    ranked_df = df_kept.copy()

    for col in num_cols:
        # 优先使用自定义
        if custom_ascending and col in custom_ascending:
            ascending = custom_ascending[col]
        else:
            # 自动规则：
            #   R@k / Hit@k / Accuracy@k -> 值越大越好 => ascending=False
            #   其余默认值越小越好 => ascending=True
            col_no_pct = col.replace("%", "")
            if re.match(r"^(R@|Hit@|Accuracy@)\d+$", col_no_pct):
                ascending = False
            else:
                ascending = True

        # 使用最小名次并发放相同名次
        ranked_df[col] = df_kept[col].rank(method="min", ascending=ascending).astype(int)

    if output_path:
        ranked_df.to_csv(output_path, index=False)

    return ranked_df, id_col



drop_cols = [
    "IFA",
    "R@20", "R@20%",
    "Accuracy@10", "Hit@10",  # 两个任选其一存在就会被丢
    "Effort@20", "Effort@20%",
    "PMI@20", "PMI@20%",
]


custom_ascending = {

}

# 跑排名并保存
ranked_df, id_col = replace_with_rank(
    "./avg_metrics.csv",
    drop_cols=drop_cols,
    custom_ascending=custom_ascending
)

print(ranked_df)


df_numeric = ranked_df.set_index(id_col)

plt.figure(figsize=(10, 6))
ax = sns.heatmap(
    df_numeric,
    annot=True,
    cmap='RdBu_r',
    cbar_kws={'label': 'Rank'},
    xticklabels=True,
    yticklabels=True
)

ax.xaxis.tick_top()
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

ax.tick_params(axis='x', labelsize=10)
ax.figure.subplots_adjust(top=0.88)
plt.title('')
plt.tight_layout()
ax.set_ylabel("Method")
plt.savefig(RESULTS_DIR / r"heatmap_rank.svg", format="svg")
plt.close()
