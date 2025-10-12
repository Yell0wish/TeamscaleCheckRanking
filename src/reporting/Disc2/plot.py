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

plt.rcParams["font.family"] = "Arial"

# 读取数据
data = pd.read_csv(Path(config["output_dir"]) / "footrule_similarity.csv")


fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(data["project"], data["footrule_similarity"], color="#1F7A8C", width=0.55)

# 设置标签和标题
ax.set_ylabel("Normalized Spearman footrule similarity", fontsize=14)
ax.set_xlabel("Project", fontsize=14)
ax.set_ylim(0, 1.05)  # 相似度范围 [0,1]
plt.xticks(rotation=45, ha="right", fontsize=11)

# 在柱子上标注数值（保留两位小数）
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}",
            ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(Path(config["output_dir"]) / "footrule_similarity.svg", format="svg")
plt.close()
