import os
import pandas as pd
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

def aggregate_csv_metrics(folder_path, output_file="eval_avg.csv"):
    """
    遍历指定文件夹，读取所有 CSV 文件，
    对每个 Name 对应的 5 个指标取平均值，并保存到一个新的 CSV 文件中。

    参数:
        folder_path (str): 包含 CSV 文件的文件夹路径
        output_file (str): 输出文件路径，默认 aggregated_results.csv
    """
    all_dfs = []
    
    # 遍历文件夹，读取所有 .csv 文件
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            all_dfs.append(df)
    
    if not all_dfs:
        print("文件夹内没有找到 CSV 文件")
        return
    
    # 合并所有数据
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 对每个 Name 分组，计算均值
    aggregated_df = combined_df.groupby("Name", as_index=False).mean()
    # 过滤掉 “原始顺序” 的行
    if "Name" in aggregated_df.columns:
        aggregated_df = aggregated_df[aggregated_df["Name"] != "原始顺序"]
    
    # 保存到新的 CSV
    output_file = Path(config["output_dir"]) / output_file
    aggregated_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已保存到 {output_file}")

aggregate_csv_metrics(Path(config["eval_results_dir"]), Path(config["output_dir"]) / "eval_avg.csv")