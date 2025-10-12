import os
import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
import logging
import sys
import time
from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
import xgboost as xgb
import random
import sklearn
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
MODEL_NAME = 'LogisticRegression'

DATASET_PROJECT_MAP = {
    "ambari": "ambari",
    "amq": "activemq",
    "bookkeeper": "bookkeeper",
    "calcite": "calcite",
    "cassandra": "cassandra",
    "groovy": "groovy",
    "hbase": "hbase",
    "hive": "hive",
    "ignite": "ignite",
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


RESULTS_DIR = Path(config["PFI_results_dir"]) / f"{MODEL_NAME}_results"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_args():
    parser = argparse.ArgumentParser(description="运行机器学习模型")
    parser.add_argument("--model", type=str, default="XGBoost", 
                        help="选择要运行的模型, 例如: LogisticRegression, SVM, XGBoost 等")
    return parser.parse_args()

def log_init():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file_path = RESULTS_DIR / f".log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),  # 终端输出
            logging.FileHandler(log_file_path, mode="w", encoding="utf-8")  # 写入日志文件
        ]
    )

    # 确保 stdout 也是 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")


def get_model(scale_pos_weight):
    model_map = {
        "XGBoost": lambda: xgb.XGBClassifier(
                                # eval_metric="logloss",
                                # scale_pos_weight=scale_pos_weight,
                                n_jobs=-1,
                                random_state=GLOBAL_RANDOM_SEED
                            ),
        "LightGBM": lambda: LGBMClassifier(
                                # objective='binary',
                                # scale_pos_weight=scale_pos_weight,
                                n_jobs=-1,
                                random_state=GLOBAL_RANDOM_SEED
                            ),
        "BalancedBagging": lambda: BalancedBaggingClassifier(n_jobs=-1,random_state=GLOBAL_RANDOM_SEED),
        "BalancedRandomForest": lambda: BalancedRandomForestClassifier(n_jobs=-1,random_state=GLOBAL_RANDOM_SEED),
        "RandomForest": lambda: RandomForestClassifier(n_jobs=-1,random_state=GLOBAL_RANDOM_SEED),
        "AdaBoost": lambda: AdaBoostClassifier(random_state=GLOBAL_RANDOM_SEED),
        "ExtraTrees": lambda: ExtraTreesClassifier(n_jobs=-1,random_state=GLOBAL_RANDOM_SEED),
        "Bagging": lambda: BaggingClassifier(n_jobs=-1,random_state=GLOBAL_RANDOM_SEED),
        "GradientBoost": lambda: GradientBoostingClassifier(random_state=GLOBAL_RANDOM_SEED),
        "BernoulliNB": lambda: BernoulliNB(),
        "LogisticRegression": lambda: LogisticRegression(n_jobs=-1,random_state=GLOBAL_RANDOM_SEED),
    }

    return model_map.get(MODEL_NAME, lambda: "Unknown model")()




def column_permutation_importance(model, X_test, y_test, base_ifa, base_recall_at_20, random_seed=GLOBAL_RANDOM_SEED):
    """
    计算每个特征的列置换重要性 (Column Permutation Importance)
    通过 IFA 增加量和 R@20% 下降量来评估特征的重要性
    """
    feature_importance = {}

    # 设置随机种子以确保可复现性
    np.random.seed(random_seed)

    for col_idx in range(X_test.shape[1]):
        time_start = time.time()
        X_permuted = X_test.copy()
        
        # 置换当前列
        np.random.shuffle(X_permuted[:, col_idx])  # 直接打乱，节省内存

        time_end = time.time()
        # print(f'特征 {col_idx} 打乱时间：', f'{time_end - time_start}秒')
        # 重新预测
        y_pred_proba = model.predict_proba(X_permuted)[:, 1]
        time_end = time.time()
        # print(f'特征 {col_idx} 预测时间：', f'{time_end - time_start}秒')
        # 计算排序后的 IFA 和 R@20%
        test_results = pd.DataFrame({
            "label": y_test,
            "probability": y_pred_proba
        }).sort_values(by="probability", ascending=False)

        # 计算 IFA
        ifa_permuted = 0
        for _, row in test_results.iterrows():
            if row["label"] == 1:
                break
            ifa_permuted += 1

        # 计算 R@20%
        top_20_percent_count = int(len(test_results) * 0.2)
        retrieved_positives = sum(test_results.iloc[:top_20_percent_count]["label"] == 1)
        total_positives = sum(test_results["label"] == 1)
        recall_at_20_permuted = retrieved_positives / total_positives if total_positives > 0 else 0

        # 计算 IFA 增加量 和 R@20% 下降量
        ifa_increase = ifa_permuted - base_ifa
        recall_at_20_drop = base_recall_at_20 - recall_at_20_permuted

        feature_importance[f"Feature_{col_idx}"] = {
            "ifa_increase": ifa_increase,
            "recall_at_20_drop": recall_at_20_drop
        }   
        time_end = time.time()
        # print(f'特征 {col_idx} 计算时间：', f'{time_end - time_start}秒')
    return feature_importance

def cross_version_validation(versions, project_name):
    """
    使用跨版本验证方法进行训练和测试：使用当前版本训练，下一版本测试
    计算：
    - 原始测试数据的 IFA 和 R@20%
    - 模型预测后排序的 IFA 和 R@20%
    """
    all_results = []
    all_feature_importance_results = []
    
    for i in range(len(versions) - 1):
        logging.info(f"版本 {versions[i]} 作为训练集，版本 {versions[i + 1]} 作为测试集")
        
        # 生成训练数据
        time_start = time.time()
        df_train = pd.read_parquet(os.path.join(ABS_DATASET_PATH, f'{versions[i]}.parquet'), engine='pyarrow')
        df_test = pd.read_parquet(os.path.join(ABS_DATASET_PATH, f'{versions[i+1]}.parquet'), engine='pyarrow')
        time_end = time.time()
        print('生成数据集时间：', f'{time_end - time_start}秒')
        
        # 只保留 label 和规则特征（去掉 file, line, code snippet）
        df_train_features = df_train.iloc[:, 3:]
        df_test_features = df_test.iloc[:, 3:]

        X_train = df_train_features.iloc[:, 1:].values  # 规则特征
        y_train = df_train_features.iloc[:, 0].values   # label
        X_test = df_test_features.iloc[:, 1:].values
        y_test = df_test_features.iloc[:, 0].values

        logging.info(f"训练集样本: {len(X_train)}, 测试集样本: {len(X_test)}")

        # 计算测试版本的原始顺序 IFA 和 R@20%
        ifa_natural = 0
        for j, label in enumerate(y_test):  # 按照测试数据原始顺序遍历
            if label == 1:
                break  # 遇到第一个 1，停止
            ifa_natural += 1  # 统计在第一个 1 之前的 0

        total_positives_natural = sum(y_test)  # 测试集总的正类数量
        top_20_percent_count_natural = int(len(y_test) * 0.2)  # 取前 20% 样本的数量
        retrieved_positives_natural = sum(y_test[:top_20_percent_count_natural])  # 计算前 20% 内的 1 的数量
        recall_at_20_natural = retrieved_positives_natural / total_positives_natural if total_positives_natural > 0 else 0

        logging.info(f"原始顺序 IFA: {ifa_natural}, R@20%: {recall_at_20_natural:.4f}")

        time_start = time.time()
        # 训练分类器
        model = get_model(scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1)))
        model.fit(X_train, y_train)
        time_end = time.time()
        logging.info('训练时间: ' + f'{time_end - time_start}秒')

        time_start = time.time()
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # 取正类的概率


        # 预测结果排序
        test_results = pd.DataFrame({
            "label": y_test,
            "probability": y_pred_proba
        })
        test_results = test_results.sort_values(by="probability", ascending=False)

        # 计算预测后排序后的 IFA
        ifa_xgb = 0
        for j, row in test_results.iterrows():
            if row["label"] == 1:
                break  # 找到第一个 label=1，停止
            ifa_xgb += 1  # 统计 label=0 的数量

        # 计算预测后排序后的 R@20%
        top_20_percent_count_xgb = int(len(test_results) * 0.2)
        retrieved_positives_xgb = sum(test_results.iloc[:top_20_percent_count_xgb]["label"] == 1)
        total_positives_xgb = sum(test_results["label"] == 1)
        recall_at_20_xgb = retrieved_positives_xgb / total_positives_xgb if total_positives_xgb > 0 else 0

        logging.info(f"{MODEL_NAME} 排序后 IFA: {ifa_xgb}, R@20%: {recall_at_20_xgb:.4f}")

        time_end = time.time()
        logging.info('测试时间: ' +  f'{time_end - time_start}秒')

        # 计算 Column Permutation 特征重要性
        time_start = time.time()
        feature_importance = column_permutation_importance(model, X_test, y_test, ifa_xgb, recall_at_20_xgb)
        time_end = time.time()
        logging.info('特征重要性计算时间: ' + f'{time_end - time_start}秒')

        # 存储 feature_importance 结果
        for feature, values in feature_importance.items():
            all_feature_importance_results.append({
                "feature": feature,
                "ifa_increase": values["ifa_increase"],
                "recall_at_20_drop": values["recall_at_20_drop"],
                "train_version": versions[i],
                "test_version": versions[i + 1]
            })


        feature_importance_df = pd.DataFrame.from_dict(feature_importance, orient="index").reset_index()
        feature_importance_df.rename(columns={"index": "feature"}, inplace=True)


        feature_importance_df.sort_values(by="ifa_increase", ascending=False).to_csv(
            f"{RESULTS_DIR}/{project_name}_{versions[i]}_to_{versions[i+1]}_feature_importance_ifa.csv", index=False)

        feature_importance_df.sort_values(by="recall_at_20_drop", ascending=False).to_csv(
            f"{RESULTS_DIR}/{project_name}_{versions[i]}_to_{versions[i+1]}_feature_importance_recall.csv", index=False)


        all_results.append({
            'train_version': versions[i],
            'test_version': versions[i + 1],
            'ifa_natural': ifa_natural,
            'recall_at_20_natural': recall_at_20_natural,
            'ifa_model': ifa_xgb,
            'recall_at_20_model': recall_at_20_xgb
        })
    
    return all_results, all_feature_importance_results


def analyze_feature_importance(feature_importance_results):
    """
    计算所有项目的 IFA 增加量和 R@20% 下降量的平均值和中位数，并保存到 CSV
    """
    df_feature_importance = pd.DataFrame(feature_importance_results)

    if df_feature_importance.empty:
        return None

    # 计算每个 feature 的平均值和中位数
    feature_stats = df_feature_importance.groupby("feature").agg(
        ifa_increase_avg=("ifa_increase", "mean"),
        ifa_increase_median=("ifa_increase", "median"),
        recall_at_20_drop_avg=("recall_at_20_drop", "mean"),
        recall_at_20_drop_median=("recall_at_20_drop", "median")
    ).reset_index()

    feature_stats["feature_idx"] = feature_stats["feature"].str.extract(r"(\d+)").astype(int)
    feature_stats = feature_stats.sort_values(by="feature_idx").drop(columns=["feature_idx"])

    return feature_stats

def analyze_results(results):
    """
    计算 IFA, R@20%, ACC, Recall, F1-score 的平均值和中位数
    """
    ifa_natural_values = [r['ifa_natural'] for r in results]
    recall_at_20_natural_values = [r['recall_at_20_natural'] for r in results]
    ifa_xgb_values = [r['ifa_model'] for r in results]
    recall_at_20_xgb_values = [r['recall_at_20_model'] for r in results]



    analysis = {
        # IFA 统计
        "ifa_natural_avg": np.mean(ifa_natural_values),
        "ifa_natural_median": np.median(ifa_natural_values),
        "recall_at_20_natural_avg": np.mean(recall_at_20_natural_values),
        "recall_at_20_natural_median": np.median(recall_at_20_natural_values),
        "ifa_model_avg": np.mean(ifa_xgb_values),
        "ifa_model_median": np.median(ifa_xgb_values),
        "recall_at_20_model_avg": np.mean(recall_at_20_xgb_values),
        "recall_at_20_model_median": np.median(recall_at_20_xgb_values),
    }

    return analysis


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

    all_results = []  # 存储所有项目的结果
    all_feature_importance_results = []  # 存储所有项目的特征重要性结果


    for idx, project in enumerate(DATASET_PROJECT_MAP.keys()):
        # if idx < check_point:
        #     continue
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
        results, feature_importance_results = cross_version_validation(version_list, project)

        # 存储该项目的结果
        all_results.extend(results)
        all_feature_importance_results.extend(feature_importance_results)

        # 计算该项目的结果统计
        project_analysis_results = analyze_results(results)

        # 打印单个项目的统计结果
        logging.info(f"{project} 统计结果:")
        for metric, value in project_analysis_results.items():
            logging.info(f"{metric}: {value:.4f}")

        # 计算并保存该项目的特征重要性统计
        project_feature_stats = analyze_feature_importance(feature_importance_results)
        if project_feature_stats is not None:
            project_feature_stats.to_csv(f"{RESULTS_DIR}/{project}_feature_importance_stats.csv", index=False)
        else:
            logging.error(f"{project} 无特征重要性数据")

        # 计算并打印当前项目 的处理时间
        project_end_time = time.time()
        project_elapsed_time = project_end_time - project_start_time
        logging.info(f"{project} 处理完成，耗时: {project_elapsed_time:.2f} 秒 ({project_elapsed_time/60:.2f} 分钟)")

    # 计算所有项目的总统计
    overall_analysis_results = analyze_results(all_results)

    logging.info("所有项目的总体统计结果:")
    for metric, value in overall_analysis_results.items():
        logging.info(f"{metric}: {value:.4f}")

    # 计算所有项目的特征重要性统计并保存
    overall_feature_stats = analyze_feature_importance(all_feature_importance_results)
    if overall_feature_stats is not None:
        overall_feature_stats.to_csv(f"{RESULTS_DIR}/overall_feature_importance_stats.csv", index=False)
    else:
        logging.error("所有项目无特征重要性数据")

    # 记录总任务的结束时间并计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"任务完成，总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")


if __name__ == '__main__':
    args = parse_args()
    MODEL_NAME = args.model  # 从命令行读取模型名称
    RESULTS_DIR = Path(config["PFI_results_dir"]) / f"{MODEL_NAME}_results"
    print(MODEL_NAME)
    print(RESULTS_DIR)
    main()