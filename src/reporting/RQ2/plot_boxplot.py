import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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

RP_CSV = Path(config["output_dir"]) / "rp_gmean.csv"
DICT_FILE = Path(config["successful_projects"])
BEST_METHOD_RESULT_DIR = Path(config["best_method_results_dir"])

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


def get_top20_check_ids(csv_path):
    """从已按 value 排序的 RP 文件中取前20个 check_id"""
    df = pd.read_csv(csv_path)
    return df.head(20)["check_id"].tolist()

def _version_sort_key(v: str):
    # v 形如 "ambari-2.7.0" 或 "2.7.0"
    core = v.split("-", 1)[-1]
    return tuple(int(x) for x in core.split("."))

def build_project_versions():
    """从 DICT_FILE 中解析各项目的版本列表（升序），返回 dict{project: [project-x.y.z, ...]}"""
    versions_map = {p: [] for p in DATASET_PROJECT_MAP.keys()}
    df = pd.read_csv(DICT_FILE)
    for project in DATASET_PROJECT_MAP.keys():
        vs = []
        for _, row in df.iterrows():
            file_name = row["File Name"]
            m = re.match(rf'^{project}-(.*?)_', file_name)
            if m:
                vs.append(f"{project}-{m.group(1)}")
        vs = sorted(set(vs), key=_version_sort_key)
        versions_map[project] = vs
    return versions_map

def get_rank_from_triplet(check_id, project_name, version_list) -> list:
    """返回某 check_id 在该项目所有连续 triplet 中的排名列表"""
    ranks = []
    for i in range(len(version_list) - 2):
        v0, v1 = version_list[i], version_list[i+1]
        csv_path = os.path.join(
            BEST_METHOD_RESULT_DIR,
            f"{project_name}_{v0}_to_{v1}_pearsonr_feature_ranking.csv"
        )
        df = pd.read_csv(csv_path)
        feature_name = f"Feature_{check_id}"
        if feature_name in df["feature"].values:
            rank = df.index[df["feature"] == feature_name][0] + 1
            ranks.append(int(rank))
        else:
            # 若缺失，用 None 占位（也可改为 len(df)+1 表示“垫底”）
            ranks.append(None)
    # 过滤掉 None，避免影响箱线图
    return [r for r in ranks if r is not None]

def get_rank_from_all(check_id, project_versions_map) -> list:
    """汇总该 check_id 在所有项目 triplets 的排名"""
    out = []
    for project, vlist in project_versions_map.items():
        if len(vlist) >= 3:
            out.extend(get_rank_from_triplet(check_id, project, vlist))
    return out

def plot_rank_boxplots_for_top20(save_path_svg):
    top20_ids = get_top20_check_ids(RP_CSV)
    proj_versions = build_project_versions()

    # 收集每个ID的rank列表（过滤空数据的ID，避免报错）
    ranks_by_id = {cid: get_rank_from_all(cid, proj_versions) for cid in top20_ids}
    data, labels = [], []
    for cid in top20_ids:
        r = ranks_by_id.get(cid, [])
        if r:                      # 只保留有数据的
            data.append(r)
            labels.append(str(cid))

    if not data:
        print("没有有效数据可绘图")
        return

    # 统一 y 轴范围
    max_rank = max(max(r) for r in data)

    # 单轴绘图：把整体宽度再压缩一点，避免“太长”
    n = len(data)
    width_inch = min(20, max(10, 0.65 * n))  
    fig, ax = plt.subplots(figsize=(width_inch, 4.2))

    bp = ax.boxplot(
        data,
        widths=0.35,                 # 让箱子更细
        vert=True,
        whis=1.5,
        patch_artist=True,
        showmeans=True,
        boxprops=dict(facecolor="#6aa6a5", edgecolor="black", linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        meanprops=dict(marker='s', markerfacecolor="white", markeredgecolor="black", markersize=4),
        flierprops=dict(marker='D', markerfacecolor="black", markeredgecolor="black", markersize=3, alpha=0.6)
    )

    # 给顶部留余量，防止最上端“顶到”
    headroom = max(1, int(0.03 * max_rank))   # 至少+1名的空间
    ax.set_ylim(1, max_rank + headroom)

    ax.set_ylabel("Rank (lower is better)", fontsize=11)
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels([f"{x}" for x in labels], fontsize=10, rotation=0)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35)

    # 自定义图例
    legend_elems = [
        Patch(facecolor="#6aa6a5", edgecolor="black", label="25%–75%"),
        Line2D([0], [0], color="black", lw=1.0, label="Range within 1.5 IQR"),
        Line2D([0], [0], color="black", lw=1.2, label="Median Line"),
        Line2D([0], [0], marker='s', linestyle='None', markerfacecolor='white', markeredgecolor='black',
               label="Mean"),
        Line2D([0], [0], marker='D', linestyle='None', color='black', label="Outliers")
    ]
    fig.legend(
        handles=legend_elems,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96), 
        ncol=5,
        frameon=True,
        fontsize=11
    )


    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])
    fig.savefig(save_path_svg)  
    print(f"Saved: {save_path_svg}")


if __name__ == "__main__":
    plot_rank_boxplots_for_top20(Path(config["output_dir"]) / "boxplot_top20_checks.svg")
