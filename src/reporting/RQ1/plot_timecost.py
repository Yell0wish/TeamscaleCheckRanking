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
data = pd.DataFrame({
    "Method": ["AdaBoost", "Bagging", "Balanced Bagging", "Balanced Random Forest", "Naive Bayes",
               "Extra Trees", "Gradient Boosting", "LightGBM", "Logistic Regression", "Pearson",
               "Random Forest", "XGBoost"],
    "Time": [1787.51, 532.35, 533.52, 304.84, 196.75, 300.57, 271.53, 119.65, 132.44, 1.27, 317.46, 97.86]
})

# 按耗时排序
data_sorted = data.sort_values("Time", ascending=False)


# ----------------------------
# 对数坐标版（分钟，log）
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(data_sorted["Method"], data_sorted["Time"], color="#1F7A8C")
ax.set_xscale("log")
ax.tick_params(axis="y", labelsize=11)
ax.set_xlabel("Time (minutes, log scale)", fontsize=14)
ax.set_xlim(left=1, right=data_sorted["Time"].max() * 2)
# 数值依然显示在右侧
ax.bar_label(bars, fmt="%.2f", padding=3)
# 把数值显示在条形末端，不让它超框
# ax.bar_label(bars, fmt="%.2f", label_type="edge")

plt.tight_layout()
plt.savefig(Path(config["output_dir"]) / "RQ1_execution_time_log.svg", format="svg")
plt.close()
