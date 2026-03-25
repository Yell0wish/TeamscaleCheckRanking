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

DATASET_PROJECT_MAP = {
    "ambari": "ambari",
    "amq": "activemq",
    "bookkeeper": "bookkeeper",
    "calcite": "calcite",
    "cassandra": "cassandra",
    "groovy": "groovy",
    "hbase": "hbase",
    "hive": "hive",
    "log4j2": "logging-log4j2",
    "mahout": "mahout",
    "mng": "maven",
    "nifi": "nifi",
    "nutch": "nutch",
    "storm": "storm",
    "tika": "tika",
    "ww": "struts",
    "zookeeper": "zookeeper"
}

def aggregate_csv_metrics(folder_path, matched_keys=None, exclude_keys=None, output_file="avg_results.csv"):
    """
    遍历指定文件夹，读取满足条件的 CSV 文件，
    对每个 Name 对应的数值指标取平均值，并保存到新的 CSV 文件中。

    参数:
        folder_path (str): 包含 CSV 文件的文件夹路径
        matched_keys (list[str] | None): 只保留文件名中包含这些 key 之一的文件
        exclude_keys (list[str] | None): 排除文件名中包含这些 key 之一的文件
        output_file (str): 输出文件路径
    """
    all_dfs = []

    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue

        # 只处理与 DATASET_PROJECT_MAP 的 key 有关的文件
        if not any(k in file for k in DATASET_PROJECT_MAP):
            continue

        # 如果要求必须匹配某些 key
        if matched_keys is not None and not any(k in file for k in matched_keys):
            continue

        # 如果要求排除某些 key
        if exclude_keys is not None and any(k in file for k in exclude_keys):
            continue

        print(f"处理文件: {file}")
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        all_dfs.append(df)

    if not all_dfs:
        print(f"没有找到符合条件的 CSV 文件 -> {output_file}")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 只对数值列求均值，避免非数值列报错
    aggregated_df = combined_df.groupby("Name", as_index=False).mean(numeric_only=True)

    aggregated_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已保存到 {output_file}")


def generate_self_and_others_results(folder_path, output_dir=None):
    """
    对 DATASET_PROJECT_MAP 中的每个项目 key：
    1. 聚合“只包含自己名字”的文件
    2. 聚合“除自己以外其他项目”的文件

    输出示例：
        ambari_self_avg_results.csv
        ambari_others_avg_results.csv
    """
    if output_dir is None:
        output_dir = folder_path

    os.makedirs(output_dir, exist_ok=True)

    for project_key in DATASET_PROJECT_MAP.keys():
        print(f"\n==================== {project_key} ====================")

        self_output = os.path.join(output_dir, f"{project_key}_self_avg_results.csv")
        others_output = os.path.join(output_dir, f"{project_key}_others_avg_results.csv")

        # 1) 只包含自己
        aggregate_csv_metrics(
            folder_path=folder_path,
            matched_keys=[project_key],
            output_file=self_output
        )

        # 2) 不包含自己（但仍然只统计 DATASET_PROJECT_MAP 中其他项目相关文件）
        aggregate_csv_metrics(
            folder_path=folder_path,
            exclude_keys=[project_key],
            output_file=others_output
        )


# 使用示例
generate_self_and_others_results(Path(config["eval_results_dir"]),
                                  output_dir=Path(config["output_dir"]) / "disc2_results1")