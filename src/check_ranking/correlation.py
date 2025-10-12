import os
import pandas as pd
import re
import numpy as np
import logging
import sys
import time
import random
import sklearn
from scipy.stats import spearmanr, kendalltau, pearsonr
from joblib import Parallel, delayed
import argparse
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DICT_FILE = Path(config["successful_projects"])
ABS_DATASET_PATH = Path(config["dataset_output_dir"])
GLOBAL_RANDOM_SEED = 0
MODEL_NAME = None

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


RESULTS_DIR = Path(config["correlation_results_dir"]) / f"{MODEL_NAME}_results"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_args():
    parser = argparse.ArgumentParser(description="相关系数方法")
    parser.add_argument("--method", type=str, default="spearman", 
                        help="选择要运行的相关系数, 例如: spearman kendall pearson")
    return parser.parse_args()

def feature_ranking_parallel(X, y, n_jobs=-1, method='spearmanr'):
    def calc_feature(idx):
        try:
            column = X[:, idx]
            if np.all(column == column[0]):
                r, p_value = 0.0, 1.0
            else:
                if method == 'spearmanr':
                    r, p_value = spearmanr(column, y)
                elif method == 'kendalltau':
                    r, p_value = kendalltau(column, y)
                elif method == 'pearsonr':
                    r, p_value = pearsonr(column, y)

        except Exception:
            r, p_value = 0.0, 1.0
        return {
            "feature": f"Feature_{idx}",
            f"{MODEL_NAME}_r": r,
            f"abs_{MODEL_NAME}_r": abs(r),
            "p_value": p_value
        }

    results = Parallel(n_jobs=n_jobs)(delayed(calc_feature)(i) for i in range(X.shape[1]))
    df = pd.DataFrame(results)
    return df.sort_values(by=f"{MODEL_NAME}_r", ascending=False)

def cross_version_validation(versions, project_name):
    """
    使用跨版本验证方法进行训练和测试：使用当前版本训练，下一版本测试
    计算：
    - 原始测试数据的 IFA 和 R@20%
    - 模型预测后排序的 IFA 和 R@20%
    """
    all_corr_dfs = []
    
    for i in range(len(versions) - 1):
        logging.info(f"版本 {versions[i]} 作为训练集，版本 {versions[i + 1]} 作为测试集")
        
        # 生成数据 因为不需要train，所以没有train的数据
        time_start = time.time()
        df_test = pd.read_parquet(os.path.join(ABS_DATASET_PATH, f'{versions[i+1]}.parquet'), engine='pyarrow')
        time_end = time.time()
        print('生成数据集时间：', f'{time_end - time_start}秒')
        
        # 只保留 label 和规则特征（去掉 file, line, code snippet）
        df_test_features = df_test.iloc[:, 3:]

        X_test = df_test_features.iloc[:, 1:].values
        y_test = df_test_features.iloc[:, 0].values

        logging.info(f"测试集样本: {len(X_test)}")

        time_start = time.time()
        # 使用非训练的方法进行特征排序
        use_parallel = True

        corr_df = feature_ranking_parallel(X_test, y_test, method=MODEL_NAME)

        time_end = time.time()
        logging.info('排序时间: ' + f'{time_end - time_start}秒')


        all_corr_dfs.append(corr_df)
        # 保存排序结果
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_path = f"{RESULTS_DIR}/{project_name}_{versions[i]}_to_{versions[i+1]}_{MODEL_NAME}_feature_ranking.csv"
        corr_df.to_csv(output_path, index=False)
        logging.info(f"{MODEL_NAME} 排序结果保存到: {output_path}")

    return all_corr_dfs

def analyze_corr_feature_ranking(corr_dfs):
    """
    聚合排序结果：计算每个特征的 r、abs_r、p_value 的均值。
    支持根据 MODEL_NAME 自动生成列名。
    """
    df_all = pd.concat(corr_dfs, ignore_index=True)

    r_col = f"{MODEL_NAME}_r"
    abs_r_col = f"abs_{MODEL_NAME}_r"

    r_mean_col = f"{r_col}_mean"
    abs_r_mean_col = f"{abs_r_col}_mean"

    feature_stats = df_all.groupby("feature").agg(
        **{
            r_mean_col: (r_col, "mean"),
            abs_r_mean_col: (abs_r_col, "mean"),
            "p_value_mean": ("p_value", "mean"),
            "count": ("feature", "count")
        }
    ).reset_index()

    feature_stats.sort_values(by=abs_r_mean_col, ascending=False, inplace=True)
    return feature_stats


def log_init():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file_path = RESULTS_DIR / f".log"

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

def main():
    log_init()
    # Python 内置 random 库
    random.seed(GLOBAL_RANDOM_SEED)
    # NumPy 随机数
    np.random.seed(GLOBAL_RANDOM_SEED)
    # Scikit-learn
    sklearn.utils.check_random_state(GLOBAL_RANDOM_SEED)

    start_time = time.time()  # 记录总任务的开始时间

    # 读取 CSV 文件
    try:
        df = pd.read_csv(DICT_FILE)
    except Exception as e:
        logging.info(f"读取文件失败: {e}")
        return None
    
    all_corr_dfs_all_projects = []

    for idx, project in enumerate(DATASET_PROJECT_MAP.keys()):
        project_start_time = time.time()  # 记录当前项目的开始时间
        logging.info(f"开始处理项目: {project}")

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
            return tuple(map(int, version.split(".")))  # 解析为整数元组以支持 X.Y.Z 或 X.Y

        version_list.sort(key=version_sort_key)

        # 在排序后的版本前重新加上 project-
        version_list = [f"{project}-{v}" for v in version_list]

        logging.info(f"{project} 版本列表（已排序）: {version_list}")
        
        # 执行跨版本验证
        corr_dfs = cross_version_validation(version_list, project)

        # 项目级统计保存

        project_corr_stats = analyze_corr_feature_ranking(corr_dfs)
        project_corr_stats.to_csv(f"{RESULTS_DIR}/{project}_{MODEL_NAME}_feature_stats.csv", index=False)

        all_corr_dfs_all_projects.extend(corr_dfs)

        # 计算并打印当前项目 的处理时间
        project_end_time = time.time()
        project_elapsed_time = project_end_time - project_start_time
        logging.info(f"{project} 处理完成，耗时: {project_elapsed_time:.2f} 秒 ({project_elapsed_time/60:.2f} 分钟)")

    # 计算所有项目的总统计
    overall_corr_stats = analyze_corr_feature_ranking(all_corr_dfs_all_projects)
    overall_corr_stats.to_csv(f"{RESULTS_DIR}/overall_{MODEL_NAME}_feature_stats.csv", index=False)

    # 记录总任务的结束时间并计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"任务完成，总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")


if __name__ == '__main__':
    args = parse_args()
    MODEL_NAME = args.method  # 从命令行读取模型名称
    RESULTS_DIR = Path(config["correlation_results_dir"]) / f"{MODEL_NAME}_results"
    main()