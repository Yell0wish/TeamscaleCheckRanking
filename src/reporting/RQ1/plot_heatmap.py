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

def replace_with_rank(csv_path, output_path=None, ascending_dict=None):
    """
    将 CSV 文件中的数值列替换为 rank。
    
    参数：
    - csv_path: 输入 CSV 文件路径
    - output_path: 输出文件路径（可选，默认不保存）
    - ascending_dict: dict 指定每列是否升序排序（数值小的 rank 小），
                      如果不传，默认全部使用升序。
                      例如 {"IFA": True, "R@20%": False}
                      表示 IFA 越小越好（升序），R@20% 越大越好（降序）。
    返回：
    - DataFrame: 替换后的 DataFrame
    """
    df = pd.read_csv(csv_path)
    
    # 复制数据
    ranked_df = df.copy()
    
    # 遍历每一列（跳过 Method）
    for col in df.columns:
        if col == "Method":
            continue
        
        # 默认升序
        ascending = True
        if ascending_dict and col in ascending_dict:
            ascending = ascending_dict[col]
        
        # rank，method='min' 保证相等值取相同最小 rank
        ranked_df[col] = df[col].rank(method="min", ascending=ascending).astype(int)
    
    if output_path:
        ranked_df.to_csv(output_path, index=False)
    
    return ranked_df

# 如果某些列希望是降序（值越大排名越前），例如 R@20%、Accuracy@10、PMI@20%
df_ranked = replace_with_rank(
    Path(config["output_dir"]) / "eval_avg_standard.csv",
    Path(config["output_dir"]) / "eval_avg_ranked.csv",
    ascending_dict={"R@20%": False, "Accuracy@10": False}
)

print(df_ranked)

# 设置 Method 列为 index，只对数值部分作图
df_numeric = df_ranked.set_index("Method")

# 绘制热力图
plt.figure(figsize=(10, 6))
ax = sns.heatmap(
    df_numeric,
    annot=True,
    cmap='RdBu_r',
    cbar_kws={'label': 'Rank'},
    xticklabels=True,
    yticklabels=True
)

# 移动 x 轴标签到上方
ax.xaxis.tick_top()
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

# 去除标题
plt.title('') 

# 自适应布局
plt.tight_layout()

# 保存为 SVG
plt.savefig(Path(config["output_dir"]) / r"RQ1_heatmap_rank.svg", format="svg")
plt.close()