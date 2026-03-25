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

def calc_weighted_ranking(csv_path: str, weight_map: dict, out_csv: str | None = None):
    df = pd.read_csv(csv_path)

    w_sum = sum(weight_map.values())

    # RP（加权几何平均）：exp(sum(w * log(x)))
    df["rp_score"] = np.exp(
        sum(w * np.log(df[col]) for col, w in weight_map.items()) / w_sum
    )

    # rp_score 越小越好
    df["final_rank"] = df["rp_score"].rank(method="min", ascending=True).astype(int)

    result = df[["Method", "final_rank", "rp_score"]]  # 保持原CSV顺序

    if out_csv is not None:
        result.to_csv(out_csv, index=False, encoding="utf-8")

    return result

weight_map = {
    'IFA': 1,
    'R@20%': 1,
    'Accuracy@10': 1,
    'E@20%R': 1,
    'PMI@20%': 1,
}

calc_weighted_ranking(Path(config["output_dir"]) / "eval_avg_ranked.csv", weight_map, Path(config["output_dir"]) / "eval_avg_rp_ranked.csv")
