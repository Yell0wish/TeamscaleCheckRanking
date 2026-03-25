import pandas as pd
import os
import re
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_config(config_path: str = "config.json"):
    cfg = {}
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return cfg

config = load_config()

DICT_FILE = config["successful_projects"]
BEST_METHOD_RESULT_DIR = config["best_method_results_dir"]
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

check57_list = []
check65_list = []
check96_list = []
check116_list = []
check163_list = []
check219_list = []
check220_list = []


def get_top20_check_ids(csv_path):
    # 读取 CSV
    df = pd.read_csv(csv_path)
    
    # # 按 value 升序排序，取前 20
    # top20 = df.sort_values(by="value").head(20)
    
    # 提取 check_id 列并转为 list
    return (df.head(20))["check_id"].tolist()

def get_rank_from_triplet(check_id, project_name, version_list) -> list: 
    # check_id就是index
    # 返回一个list, 里面是这个check_id在一个项目中所有triplets中的排名
    rank_in_project = []

    for i in range(len(version_list) - 2):
        checks_ranking_file = os.path.join(BEST_METHOD_RESULT_DIR, f"{project_name}_{version_list[i]}_to_{version_list[i+1]}_pearsonr_feature_ranking.csv")
        df = pd.read_csv(checks_ranking_file)
        # 构造目标特征名
        feature_name = f"Feature_{check_id}"
        assert feature_name in df["feature"].values
        # 找到该行的 index（从 0 开始）+1 = 排名
        rank = df.index[df["feature"] == feature_name][0] + 1
        rank_in_project.append(rank)

        if check_id == 57 and rank > 21:
            check57_list.append((project_name, version_list[i], version_list[i+1]))
        if check_id == 65 and rank > 229:
            check65_list.append((project_name, version_list[i], version_list[i+1]))
        if check_id == 96 and rank > 22:
            check96_list.append((project_name, version_list[i], version_list[i+1]))
        if check_id == 116 and rank > 16.5:
            check116_list.append((project_name, version_list[i], version_list[i+1]))
        if check_id == 163 and rank > 22:
            check163_list.append((project_name, version_list[i], version_list[i+1]))
        if check_id == 219 and rank > 22.5:
            check219_list.append((project_name, version_list[i], version_list[i+1]))
        if check_id == 220 and rank > 14:
            check220_list.append((project_name, version_list[i], version_list[i+1]))

    return rank_in_project

def get_rank_from_all(check_id) -> list:
    rank_in_all = []

    # 读取 CSV 文件
    try:
        df = pd.read_csv(DICT_FILE)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

    for idx, project in enumerate(DATASET_PROJECT_MAP.keys()):

        # 获取该项目的所有版本
        version_list = []
        for _, row in df.iterrows():
            file_name = row["File Name"]

            # 提取第一个 "_" 之前的部分（仅版本号）
            match = re.match(rf'^{project}-(.*?)_', file_name)
            if match:
                version = match.group(1)  # 仅获取 "2.1.0" 这样的版本号
                version_list.append(version)

        # 按照版本号数值大小排序
        def version_sort_key(version):
            # 解析为整数元组以支持 X.Y.Z 或 X.Y
            return tuple(map(int, version.split(".")))

        version_list.sort(key=version_sort_key)

        # 在排序后的版本前重新加上 project-
        version_list = [f"{project}-{v}" for v in version_list]
        # logging.info(idx)
        # logging.info(f"✅ {project} 版本列表（已排序）: {version_list}")

        rank_in_all.extend(get_rank_from_triplet(check_id, project, version_list))

    return rank_in_all

def get_rp(rank_in_all):
    return math.exp(sum([math.log(r) for r in rank_in_all]) / len(rank_in_all))

def minmax_norm(data):
    min_val = min(data)
    max_val = max(data)
    print(f"min: {min_val}, max: {max_val}")
    if max_val == min_val:
        return [0 for _ in data]
    return [(x - min_val) / (max_val - min_val) for x in data]

def build_jaccard_df(lists_dict: dict) -> pd.DataFrame:
    """把 bad-triplet 列表字典转为 Jaccard 相似度矩阵(DataFrame)。"""
    ids = list(lists_dict.keys())
    sets = {k: set(v) for k, v in lists_dict.items()}
    n = len(ids)
    mat = np.zeros((n, n), dtype=float)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i == j:
                mat[i, j] = 1.0
            else:
                u = sets[a] | sets[b]
                inter = sets[a] & sets[b]
                mat[i, j] = len(inter) / len(u) if u else 0.0
    return pd.DataFrame(mat, index=ids, columns=ids)

def plot_jaccard_heatmap_seaborn(df_jac: pd.DataFrame,
                                 savepath=r"jaccard_heatmap.svg"):
    """基于 seaborn 画热力图, 数字标注到格子里, x 轴标签在上方，横向显示。"""
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(
        df_jac,
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        cmap="YlGnBu",
        square=True,
        cbar_kws={'label': 'Jaccard similarity'}
    )

    # x 轴标签移到上方，水平显示
    ax.xaxis.tick_top()
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.xticks(rotation=0)   # 设为水平
    plt.yticks(rotation=0)   # 保持 y 轴标签竖直
    plt.tight_layout()
    plt.savefig(savepath, format="svg", dpi=300)
    plt.close()


def build_jaccard_df(lists_dict: dict) -> pd.DataFrame:
    """把 bad-triplet 列表字典转为 Jaccard 相似度矩阵（DataFrame）。"""
    ids = list(lists_dict.keys())
    sets = {k: set(v) for k, v in lists_dict.items()}
    n = len(ids)
    mat = np.zeros((n, n), dtype=float)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i == j:
                mat[i, j] = 1.0
            else:
                u = sets[a] | sets[b]
                inter = sets[a] & sets[b]
                mat[i, j] = len(inter) / len(u) if u else 0.0
    return pd.DataFrame(mat, index=ids, columns=ids)

def main():

    for check_id in range(235):
        if check_id != 65 and check_id != 116 and check_id != 220 and check_id != 96 and check_id != 163 and check_id != 57 and check_id != 219:
            continue
        rank_in_all = get_rank_from_all(check_id)
        print(f"Check ID: {check_id}")
        median_value = np.median(rank_in_all)
        print(f"Median: {median_value}")
    
    intersection = set(check57_list) & set(check65_list) & set(check96_list) \
               & set(check116_list) & set(check163_list) & set(check219_list) & set(check220_list)
    # print(set(check76_list) & set(check96_list))
    # print(check76_list)
    # print(check96_list)
    # print(check108_list)
    # print(check116_list)
    # print(check163_list)
    # print(check220_list)
    print(intersection)


    lists = {
        57: check57_list,
        65: check65_list,
        96: check96_list,
        116: check116_list,
        163: check163_list,
        219: check219_list,
        220: check220_list
    }
    df_jac = build_jaccard_df(lists)
    print(df_jac.round(3))

    plot_jaccard_heatmap_seaborn(
        df_jac,
        savepath=Path(config["output_dir"]) / "jaccard_heatmap.svg"
    )
    
main()

