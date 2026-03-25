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
import csv
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
FEATURE_IMPORTANTCE_RESULTS_PATH = None
RESULTS_DIR = Path(config["eval_results_dir"])

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_args():
    parser = argparse.ArgumentParser(description="运行模型 + 特征重要性分析")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--feature_file", type=str, required=True, help="特征重要性文件路径")
    parser.add_argument("--do_sort", action="store_true", help="是否对筛选的特征进行排序（默认不排序）")
    return parser.parse_args()

def log_init():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    log_file_path = RESULTS_DIR / "eval.log"
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

def get_all_models(scale_pos_weight):
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

    return {name: builder() for name, builder in model_map.items()}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_model(models, X_train, y_train):
    for model_name, model in models.items():
        logging.info(f"开始训练模型: {model_name}")
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        logging.info(f"模型 {model_name} 训练完成，用时: {end - start:.2f} 秒")

def predict_proba_mean(models, X_test):
    proba_list = [model.predict_proba(X_test)[:, 1] for model in models.values()]
    return np.mean(proba_list, axis=0)

def cal_effort_at_k_recall(test_results: pd.DataFrame, k=20):
    ranked_labels = test_results['label'].to_numpy()
    total_bugs = ranked_labels.sum()
    assert total_bugs * k * 0.01 >= 1
    target_bugs = int(total_bugs * k * 0.01)
    assert target_bugs >= 1
    checked_lines = 0
    checked_bugs = 0

    for i in range(len(ranked_labels)):
        checked_lines += 1
        if ranked_labels[i] == 1:
            checked_bugs += 1
        if checked_bugs == target_bugs:
            break
    else:
        print("Warning: 没有找到足够的 bugs 来计算 R@20%, 出问题啦！")
        assert False

    return checked_lines / len(ranked_labels)

def cal_hit_at_k(test_results: pd.DataFrame, k=10):
    ranked_labels = test_results['label'].to_numpy()
    top_k_bugs = ranked_labels[:k].sum()
    return 1 if top_k_bugs > 0 else 0

def cal_PMI_at_k(test_results: pd.DataFrame, k=20):
    checked_files = set()
    total_lines = len(test_results)

    check_line_num = int(total_lines * k * 0.01)
    assert check_line_num >= 1, "检查的行数不能少于 1"

    for i in range(check_line_num):
        checked_files.add(test_results.iloc[i]['file'])

    total_files = set(test_results['file'])

    return len(checked_files) / len(total_files)

def cal_ifa(test_results: pd.DataFrame):
    count = 0
    for i in range(len(test_results)):
        if test_results.iloc[i]['label'] == 1:
            break
        count += 1
    return count

def cal_recall_at_k(test_results: pd.DataFrame, k=20):
    total_bugs = test_results['label'].sum()
    assert total_bugs > 0

    top_k_count = int(len(test_results) * k * 0.01)
    retrieved_positives = test_results['label'][:top_k_count].sum()
    return retrieved_positives / total_bugs

def cross_version_validation_with_selected_features(versions, project_name):
    """
    使用跨版本验证方法进行训练和测试：使用当前版本训练，下一版本测试
    仅使用 selected_feature_indices 指定的特征进行训练，并计算其列置换重要性。
    
    selected_feature_indices 从 0 开始，需要 +4 对齐数据集。
    """
    k_offsets = [0, -5, +5, +10]
    
    ifa_list_natural = []
    r_natural = {
        20 + k: [] for k in k_offsets
    }
    effort_natural = {
        20 + k: [] for k in k_offsets
    }
    pmi_natural = {
        20 + k: [] for k in k_offsets
    }
    hit_natural = {
        10 + k: [] for k in k_offsets
    }

    ifa_list_sorted = []
    r_sorted = {
        20 + k: [] for k in k_offsets
    }
    effort_sorted = {
        20 + k: [] for k in k_offsets
    }
    pmi_sorted = {
        20 + k: [] for k in k_offsets
    }
    hit_sorted = {
        10 + k: [] for k in k_offsets
    }

    assert (len(versions) - 2) > 0
    for i in range(len(versions) - 2):

        logging.info(f"获取版本 {versions[i]} 到 {versions[i+1]} 的特征重要性文件")
        # 首先要获取版本的selected_feature_indices
        if MODEL_NAME == 'kendalltau' or MODEL_NAME == 'pearsonr' or MODEL_NAME == 'spearmanr':
            checks_ranking_file = os.path.join(FEATURE_IMPORTANTCE_RESULTS_PATH, f"{project_name}_{versions[i]}_to_{versions[i+1]}_{MODEL_NAME}_feature_ranking.csv")
        elif 'deepseek' in MODEL_NAME:
            if 'rank' in MODEL_NAME:
                checks_ranking_file = os.path.join(FEATURE_IMPORTANTCE_RESULTS_PATH, 'llm_rank_results_standard_ranking.csv')
            else :
                checks_ranking_file = os.path.join(FEATURE_IMPORTANTCE_RESULTS_PATH, 'llm_score_results_standard_ranking.csv')

        else:
            checks_ranking_file = os.path.join(FEATURE_IMPORTANTCE_RESULTS_PATH, f"{project_name}_{versions[i]}_to_{versions[i+1]}_feature_importance_recall.csv")
        feature_pd = pd.read_csv(checks_ranking_file)


        feature_sorted = feature_pd
        logging.info("未启用排序，按文件特征顺序选前 k 个")
        # logging.info(feature_sorted)
        k = 10
        top_k_features = feature_sorted.head(k)
        # logging.info(top_k_features)
        # return
        # 打印前 k 个 feature 的值
        logging.info(top_k_features)
        logging.info(top_k_features["feature"].tolist())
        selected_feature_indices = [int(f.split("_")[1]) for f in top_k_features["feature"].tolist()]


        # 确保 selected_feature_indices 按升序排列
        selected_feature_indices = sorted(selected_feature_indices)
        logging.info(selected_feature_indices)


        logging.info(f"版本 {versions[i+1]} 作为训练集，版本 {versions[i+2]} 作为测试集")
        
        # 生成训练数据
        time_start = time.time()
        df_train = pd.read_parquet(os.path.join(ABS_DATASET_PATH, f'{versions[i+1]}.parquet'), engine='pyarrow')
        df_test = pd.read_parquet(os.path.join(ABS_DATASET_PATH, f'{versions[i+2]}.parquet'), engine='pyarrow')
        time_end = time.time()
        logging.info(f'生成数据集时间: {time_end - time_start:.2f}秒')

        
        # 将 selected_feature_indices 从 0 基准转换为 4 基准
        selected_cols = [0, 1, 2, 3] + [idx + 4 for idx in selected_feature_indices]

        # 仅保留 selected_feature_indices 指定的特征
        df_train_selected = df_train.iloc[:, selected_cols]
        df_test_selected = df_test.iloc[:, selected_cols]

        # 获取relative path
        test_file_path = df_test.iloc[:, 0].values

        # 提取特征和标签
        X_train = df_train_selected.iloc[:, 4:].values  # 仅选定特征（去掉 file, line, code snippet, label）
        y_train = df_train_selected.iloc[:, 3].values   # label
        X_test = df_test_selected.iloc[:, 4:].values
        y_test = df_test_selected.iloc[:, 3].values

        logging.info(f"训练集样本: {len(X_train)}, 测试集样本: {len(X_test)}")

        test_results_natural = pd.DataFrame({"file": test_file_path, "label": y_test})

        # Effort@k（达到20%召回所需检查的比例）
        for k in effort_natural.keys():
            effort_at_k_natural = cal_effort_at_k_recall(test_results_natural, k=k)
            effort_natural[k].append(effort_at_k_natural)

        # Hit@k（原始顺序前10行是否命中）
        for k in hit_natural.keys():
            hit_at_k_natural = cal_hit_at_k(test_results_natural, k=k)
            hit_natural[k].append(hit_at_k_natural)

        # PMI@k（原始顺序前20%行覆盖的文件比例）
        for k in pmi_natural.keys():
            pmi_at_k_natural = cal_PMI_at_k(test_results_natural, k=k)
            pmi_natural[k].append(pmi_at_k_natural)

        # Recall@k%
        for k in r_natural.keys():
            recall_at_k_natural = cal_recall_at_k(test_results_natural, k=k)
            r_natural[k].append(recall_at_k_natural)

        # IFA
        ifa_natural = cal_ifa(test_results_natural)
        ifa_list_natural.append(ifa_natural)

        # 训练分类器
        models = get_all_models(scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1)))
        train_model(models, X_train, y_train)

        time_start = time.time()
        # 预测
        y_pred_proba = predict_proba_mean(models, X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # 预测结果排序
        test_results = pd.DataFrame({
            "file": test_file_path,
            "label": y_test,
            "probability": y_pred_proba
        })
        test_results = test_results.sort_values(by="probability", ascending=False)

        # Effort@k（排序后）
        for k in effort_sorted.keys():
            effort_at_k_sorted = cal_effort_at_k_recall(test_results, k=k)
            effort_sorted[k].append(effort_at_k_sorted)

        # Hit@k（排序后）
        for k in hit_sorted.keys():
            hit_at_k_sorted = cal_hit_at_k(test_results, k=k)
            hit_sorted[k].append(hit_at_k_sorted)

        # PMI@k（排序后）
        for k in pmi_sorted.keys():
            pmi_at_k_sorted = cal_PMI_at_k(test_results, k=k)
            pmi_sorted[k].append(pmi_at_k_sorted)

        # Recall@k（排序后）
        for k in r_sorted.keys():
            recall_at_k_sorted = cal_recall_at_k(test_results, k=k)
            r_sorted[k].append(recall_at_k_sorted)

        # IFA（排序后）
        ifa_sorted = cal_ifa(test_results)
        ifa_list_sorted.append(ifa_sorted)

        # 没排序
        # 先记录 IFA
        logging.info(f"原始顺序 IFA: {ifa_natural}")

        # 再把各 k 的 Recall / Effort / PMI / Hit 拼接成一行
        parts = []
        parts.append(" | ".join([f"R@{k}: {r_natural[k][-1]:.4f}"   for k in sorted(r_natural.keys())]))
        parts.append(" | ".join([f"Effort@{k}: {effort_natural[k][-1]:.4f}" for k in sorted(effort_natural.keys())]))
        parts.append(" | ".join([f"PMI@{k}: {pmi_natural[k][-1]:.4f}"       for k in sorted(pmi_natural.keys())]))
        parts.append(" | ".join([f"Hit@{k}: {hit_natural[k][-1]:.4f}"       for k in sorted(hit_natural.keys())]))
        logging.info(";\n ".join(parts))

        # 排序后
        logging.info(f"排序后 IFA: {ifa_sorted}")

        parts_sorted = []
        parts_sorted.append(" | ".join([f"R@{k}: {r_sorted[k][-1]:.4f}"   for k in sorted(r_sorted.keys())]))
        parts_sorted.append(" | ".join([f"Effort@{k}: {effort_sorted[k][-1]:.4f}" for k in sorted(effort_sorted.keys())]))
        parts_sorted.append(" | ".join([f"PMI@{k}: {pmi_sorted[k][-1]:.4f}"       for k in sorted(pmi_sorted.keys())]))
        parts_sorted.append(" | ".join([f"Hit@{k}: {hit_sorted[k][-1]:.4f}"       for k in sorted(hit_sorted.keys())]))

        logging.info(";\n ".join(parts_sorted))

        time_end = time.time()
        logging.info(f'测试时间: {time_end - time_start:.2f}秒')
    
    # =====================================================
    # 项目内版本跑完了，统计均值
    logging.info(f"{project_name} 项目级统计：")

    avg_ifa_natural = float(np.mean(ifa_list_natural)) if ifa_list_natural else 0.0
    avg_ifa_sorted  = float(np.mean(ifa_list_sorted))  if ifa_list_sorted  else 0.0

    # 多 k 的均值汇总（自然顺序/排序后）
    metrics_avg_natural = {
        "R":   {k: (float(np.mean(r_natural[k]))   if r_natural[k]   else 0.0) for k in sorted(r_natural.keys())},
        "Eff": {k: (float(np.mean(effort_natural[k])) if effort_natural[k] else 0.0) for k in sorted(effort_natural.keys())},
        "PMI": {k: (float(np.mean(pmi_natural[k])) if pmi_natural[k] else 0.0) for k in sorted(pmi_natural.keys())},
        "Hit": {k: (float(np.mean(hit_natural[k])) if hit_natural[k] else 0.0) for k in sorted(hit_natural.keys())},
    }
    metrics_avg_sorted = {
        "R":   {k: (float(np.mean(r_sorted[k]))   if r_sorted[k]   else 0.0) for k in sorted(r_sorted.keys())},
        "Eff": {k: (float(np.mean(effort_sorted[k])) if effort_sorted[k] else 0.0) for k in sorted(effort_sorted.keys())},
        "PMI": {k: (float(np.mean(pmi_sorted[k])) if pmi_sorted[k] else 0.0) for k in sorted(pmi_sorted.keys())},
        "Hit": {k: (float(np.mean(hit_sorted[k])) if hit_sorted[k] else 0.0) for k in sorted(hit_sorted.keys())},
    }

    # 打印汇总（自然顺序）
    logging.info(f"    原始顺序 IFA 均值: {avg_ifa_natural:.4f}")
    logging.info("    原始顺序 " + " | ".join([f"R@{k}: {metrics_avg_natural['R'][k]:.4f}" for k in sorted(r_natural.keys())]))
    logging.info("    原始顺序 " + " | ".join([f"Effort@{k}: {metrics_avg_natural['Eff'][k]:.4f}" for k in sorted(effort_natural.keys())]))
    logging.info("    原始顺序 " + " | ".join([f"PMI@{k}: {metrics_avg_natural['PMI'][k]:.4f}" for k in sorted(pmi_natural.keys())]))
    logging.info("    原始顺序 " + " | ".join([f"Hit@{k}: {metrics_avg_natural['Hit'][k]:.4f}" for k in sorted(hit_natural.keys())]))

    # 打印汇总（排序后）
    logging.info(f"    排序后 IFA 均值: {avg_ifa_sorted:.4f}")
    logging.info("    排序后 " + " | ".join([f"R@{k}: {metrics_avg_sorted['R'][k]:.4f}" for k in sorted(r_sorted.keys())]))
    logging.info("    排序后 " + " | ".join([f"Effort@{k}: {metrics_avg_sorted['Eff'][k]:.4f}" for k in sorted(effort_sorted.keys())]))
    logging.info("    排序后 " + " | ".join([f"PMI@{k}: {metrics_avg_sorted['PMI'][k]:.4f}" for k in sorted(pmi_sorted.keys())]))
    logging.info("    排序后 " + " | ".join([f"Hit@{k}: {metrics_avg_sorted['Hit'][k]:.4f}" for k in sorted(hit_sorted.keys())]))

    # ================== 写入 CSV（多 k） ==================
    csv_file = RESULTS_DIR / f"{project_name}.csv"
    file_exists = os.path.exists(csv_file)
    file_has_data = False
    if file_exists:
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            file_has_data = any(row for row in reader)

    # 动态表头：Name, IFA, 然后自然/排序后各类指标分开写，或只写自然/排序一种
    # 这里按照“自然顺序一行 + 模型排序后结果一行”的旧风格，只是把列扩为多 k
    header = ["Name", "IFA"]
    # R/Effort/PMI：用 r_natural 的 key 们（就是 20±offset）作为统一的 k 集
    k_list_rep = sorted(r_natural.keys())
    header += [f"R@{k}" for k in k_list_rep]
    header += [f"Effort@{k}" for k in k_list_rep]
    header += [f"PMI@{k}" for k in k_list_rep]
    # Hit：用 hit_natural 的 key（10±offset）
    k_list_hit = sorted(hit_natural.keys())
    header += [f"Hit@{k}" for k in k_list_hit]

    with open(csv_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        # 如果文件为空，先写表头 + “原始顺序”行
        if not file_has_data:
            writer.writerow(header)
            natural_row = ["原始顺序", f"{avg_ifa_natural:.4f}"]
            natural_row += [f"{metrics_avg_natural['R'][k]:.4f}"   for k in k_list_rep]
            natural_row += [f"{metrics_avg_natural['Eff'][k]:.4f}" for k in k_list_rep]
            natural_row += [f"{metrics_avg_natural['PMI'][k]:.4f}" for k in k_list_rep]
            natural_row += [f"{metrics_avg_natural['Hit'][k]:.4f}" for k in k_list_hit]
            writer.writerow(natural_row)

        # 模型名美化
        model_display_name = MODEL_NAME
        if MODEL_NAME.lower() == "pearsonr":
            model_display_name = "Pearson"
        elif MODEL_NAME.lower() == "kendalltau":
            model_display_name = "Kendall"
        elif MODEL_NAME.lower() == "spearmanr":
            model_display_name = "Spearman"

        # 写入“排序后”行
        model_row = [model_display_name, f"{avg_ifa_sorted:.4f}"]
        model_row += [f"{metrics_avg_sorted['R'][k]:.4f}"   for k in k_list_rep]
        model_row += [f"{metrics_avg_sorted['Eff'][k]:.4f}" for k in k_list_rep]
        model_row += [f"{metrics_avg_sorted['PMI'][k]:.4f}" for k in k_list_rep]
        model_row += [f"{metrics_avg_sorted['Hit'][k]:.4f}" for k in k_list_hit]
        writer.writerow(model_row)

    logging.info(f"项目统计结果已追加保存至 {csv_file}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def main():
    log_init()
    # Python 内置 random 库
    random.seed(GLOBAL_RANDOM_SEED)
    # NumPy 随机数
    np.random.seed(GLOBAL_RANDOM_SEED)
    # Scikit-learn
    sklearn.utils.check_random_state(GLOBAL_RANDOM_SEED)

    # os.makedirs(RESULTS_DIR, exist_ok=True)

    start_time = time.time()  # 记录总任务的开始时间

    # 读取 CSV 文件
    try:
        df = pd.read_csv(DICT_FILE)
    except Exception as e:
        logging.info(f"读取文件失败: {e}")
        return None

    logging.info(f"当前模型名称: {MODEL_NAME}")  

    for idx, project in enumerate(DATASET_PROJECT_MAP.keys()):
        # if idx < 18:
        #     continue
        # save_checkpoint(idx)
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
            return tuple(map(int, version.split(".")))  # 解析为整数元组以支持 X.Y.Z 或 X.Y

        version_list.sort(key=version_sort_key)

        # 在排序后的版本前重新加上 project-
        version_list = [f"{project}-{v}" for v in version_list]
        logging.info(idx)
        logging.info(f"{project} 版本列表（已排序）: {version_list}")
        
        # 执行跨版本验证
        # selected_feature_indices = [224, 8, 95, 173, 70, 134, 175, 82, 176, 87]  
        # results, feature_importance_results = 

        cross_version_validation_with_selected_features(version_list, project)

if __name__ == '__main__':
    args = parse_args()
    MODEL_NAME = args.model  # 从命令行读取模型名称
    FEATURE_IMPORTANTCE_RESULTS_PATH = args.feature_file
    # logging.info(MODEL_NAME)
    # logging.info(RESULTS_DIR)
    # log_init()
    main()