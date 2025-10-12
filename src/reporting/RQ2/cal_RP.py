import os
import time
import re
import logging
import pandas as pd
import sys
import math
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


DICT_FILE = Path(config["successful_projects"])
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
BEST_METHOD_RESULT_DIR = Path(config["best_method_results_dir"])
RESULTS_DIR = Path(config["output_dir"])


def log_init():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file_path = RESULTS_DIR / ".log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),  # 终端输出
            logging.FileHandler(log_file_path, mode="a", encoding="utf-8")  # 写入日志文件
        ]
    )

    # 确保 stdout 也是 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")

def cal_log_RP(version_list, project_name):
    assert (len(version_list) - 2) > 0
    # 一共235个feature
    project_rp = [0.0 for i in range(235)]
    for i in range(len(version_list) - 2):
        checks_ranking_file = os.path.join(BEST_METHOD_RESULT_DIR, f"{project_name}_{version_list[i]}_to_{version_list[i+1]}_pearsonr_feature_ranking.csv")
        df = pd.read_csv(checks_ranking_file)
        assert df.shape[0] == 235
        assert df["feature"].str.replace("Feature_", "", regex=False).astype(int).between(0, 234).all()

        for idx, row in df.iterrows():
            feature_name = row["feature"]          # 比如 "Feature_116"
            feature_id = int(feature_name.replace("Feature_", ""))  # 得到 116
            feature_id = int(feature_id)
            project_rp[feature_id] += math.log(idx + 1)  # i从0开始，实际是第i+2个版本
        # 文件内已经排好序了
    return project_rp, len(version_list) - 2

def main():
    log_init()

    start_time = time.time()  # 记录总任务的开始时间

    # 读取 CSV 文件
    try:
        df = pd.read_csv(DICT_FILE)
    except Exception as e:
        logging.info(f"读取文件失败: {e}")
        return None

    global_log_rp = [0.0 for _ in range(235)]  # 全部项目的 sum ln(rank)
    global_triplets = 0

    for idx, project in enumerate(DATASET_PROJECT_MAP.keys()):

        project_start_time = time.time()  # 记录当前项目的开始时间
        logging.info(f"开始处理项目: {project}")
        # log_metric("project", idx)

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
        logging.info(idx)
        logging.info(f"{project} 版本列表（已排序）: {version_list}")

        # 计算 RP
        project_log_rp, project_triplets = cal_log_RP(version_list, project)
        global_log_rp = [a + b for a, b in zip(global_log_rp, project_log_rp)]
        global_triplets += project_triplets

    # 需要除以global_triplets
    assert global_triplets > 0
    global_log_rp = [x / global_triplets for x in global_log_rp]
    logging.info(f"全局版本三元组数量: {global_triplets}")

    global_rp_gmean = [math.exp(x) for x in global_log_rp]
    out_df2 = pd.DataFrame({
        "check_id": list(range(235)),
        "value": global_rp_gmean
    }).sort_values("value", ascending=True).reset_index(drop=True)
    out_df2.to_csv(RESULTS_DIR / 'rp_gmean.csv', index=False, encoding='utf-8')

if __name__ == "__main__":
    main()