import pandas as pd
import logging
import sys
import re
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
# RESULTS_DIR = Path(config["output_dir"])
DATASETDIR = Path(config["dataset_dir"])

def log_init():
    log_file_path = Path(config["output_dir"]) / "compare.log"
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

def cal_effort_at_20_recall(test_results: pd.DataFrame):
    ranked_labels = test_results['label'].to_numpy()
    total_bugs = ranked_labels.sum()
    assert total_bugs >= 5
    target_bugs = int(total_bugs * 0.2)
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

def cal_PMI_at_20(test_results: pd.DataFrame):
    checked_files = set()
    total_lines = len(test_results)

    check_line_num = int(total_lines * 0.2)
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

def cal_recall_at_20(test_results: pd.DataFrame):
    total_bugs = test_results['label'].sum()
    assert total_bugs > 0

    top_20_percent_count = int(len(test_results) * 0.2)
    retrieved_positives = test_results['label'][:top_20_percent_count].sum()
    return retrieved_positives / total_bugs

def rank_dataset_by_checks(dataset_df: pd.DataFrame, ranked_checks: list, top_k: int = 10):
    """
    根据 ranked_checks 的前 top_k 个对 dataset 排序:
    - 先按命中前 top_k 个 checks 的数量降序排序
    - 若命中数量相同，再按命中的 checks 中最好的 rank（最小 rank）升序排序
    - 若仍相同，则保持原始顺序
    """
    df = dataset_df.copy()
    df['_orig_idx'] = range(len(df))
    df['_hit_count'] = 0
    df['_best_rank'] = float('inf')

    top_checks = ranked_checks[:top_k]

    for rank, fid in enumerate(top_checks):
        fid = f'feature_{fid}'
        if fid not in df.columns:
            print(f"Warning: feature '{fid}' not found in dataset columns.")
            continue

        hit_mask = df[fid] > 0

        # 累加命中数量
        df.loc[hit_mask, '_hit_count'] += 1

        # 更新该行命中的最好 rank（取更小的 rank）
        df.loc[hit_mask, '_best_rank'] = df.loc[hit_mask, '_best_rank'].clip(upper=rank)

    df_sorted = df.sort_values(
        by=['_hit_count', '_best_rank', '_orig_idx'],
        ascending=[False, True, True],
        kind='mergesort'
    ).drop(columns=['_orig_idx', '_hit_count', '_best_rank'])

    return df_sorted

def eval_with_ranked_checks(versions, version_paths, ranked_checks, output_path):
    all_results = []

    for i in range(len(versions)):
        logging.info(f"版本 {versions[i]}")
        data_df = pd.read_parquet(version_paths[i], engine='pyarrow')

        # —— 原始顺序（未排序）
        original_y = data_df.iloc[:, 3].values
        original_file_paths = data_df.iloc[:, 0].values
        original_result = pd.DataFrame({'file': original_file_paths, 'label': original_y})

        ifa_original           = cal_ifa(original_result)
        recall_at_20_original  = cal_recall_at_20(original_result)
        effort_at_20_original  = cal_effort_at_20_recall(original_result)
        hit_at_10_original     = cal_hit_at_k(original_result, k=10)
        pmi_at_20_original     = cal_PMI_at_20(original_result)

        # —— 按 ranked_checks 排序
        ranked_df  = rank_dataset_by_checks(data_df, ranked_checks, top_k=10)
        y_sorted   = ranked_df.iloc[:, 3].values
        file_paths = ranked_df.iloc[:, 0].values
        sort_result = pd.DataFrame({'file': file_paths, 'label': y_sorted})

        ifa_sorted          = cal_ifa(sort_result)
        recall_at_20_sorted = cal_recall_at_20(sort_result)
        effort_at_20_sorted = cal_effort_at_20_recall(sort_result)
        hit_at_10_sorted    = cal_hit_at_k(sort_result, k=10)
        pmi_at_20_sorted    = cal_PMI_at_20(sort_result)

        # —— 收集本版本结果
        all_results.append({
            "version": versions[i],

            # 原始
            "ifa_orig": ifa_original,
            "recall@20_orig": recall_at_20_original,
            "effort@20%R_orig": effort_at_20_original,
            "hit@10_orig": hit_at_10_original,
            "pmi@20_orig": pmi_at_20_original,

            # 排序后
            "ifa_ranked": ifa_sorted,
            "recall@20_ranked": recall_at_20_sorted,
            "effort@20%R_ranked": effort_at_20_sorted,
            "hit@10_ranked": hit_at_10_sorted,
            "pmi@20_ranked": pmi_at_20_sorted,
        })

    # —— 汇总为 DataFrame
    df = pd.DataFrame(all_results)

    # —— 计算均值行（只对数值列）
    avg_row = {"version": "AVG"}
    for col in df.columns:
        if col != "version":
            avg_row[col] = df[col].mean()

    df_out = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # —— 保存
    if output_path:
        df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"结果已保存到 {output_path}")

    return df_out

def collect_project_versions(success_csv_df: pd.DataFrame, project: str):
    """从 success_results.csv 中收集该 project 的所有版本（去重+排序）"""
    versions = set()
    pat = re.compile(rf'^{project}-(.*?)_')  # 提取 {project}-{version}_ 前缀里的 version
    for _, row in success_csv_df.iterrows():
        file_name = row["File Name"]
        m = pat.match(file_name)
        if m:
            versions.add(m.group(1))

    # 版本号排序：按数字段逐段比较（支持 X.Y 或 X.Y.Z 形式）
    def version_sort_key(v):
        return tuple(map(int, v.split(".")))
    return sorted(versions, key=version_sort_key)


def main():
    log_init()

    # 读取 success_results.csv（里面有文件名可提取出版本列表）
    try:
        df = pd.read_csv(DICT_FILE)
    except Exception as e:
        logging.info(f"读取文件失败: {e}")
        return None

    ranked_df = pd.read_csv(config["rp_path"])
    ranked_checks = ranked_df['check_id'].tolist()

    # —— 为每个 project 收集所有版本，并分别做评估（project 内先求均值）
    per_project_avg_rows = []

    for idx, project in enumerate(DATASET_PROJECT_MAP.keys()):
        logging.info(f"开始处理项目: {project}")

        versions = collect_project_versions(df, project)
        if not versions:
            logging.info(f"未在 {project} 中找到任何版本，跳过。")
            continue

        # 组装该项目所有版本的带前缀名与 parquet 路径
        versions_with_prefix = [f"{project}-{v}" for v in versions]

        version_paths = [Path(DATASETDIR) / f"{project}-{v}.parquet" for v in versions]

        logging.info(f"{project} 版本列表（已排序）: {versions}")
        versions_with_prefix = [versions_with_prefix[-1]]
        version_paths = [version_paths[-1]]
        logging.info(f"为测试只使用最新版本: {versions_with_prefix}")
        logging.info(f"对应路径: {version_paths}")
        # 对该项目的所有版本做评估；这里不需要导出每版的结果文件，可设为 None
        df_proj = eval_with_ranked_checks(
            versions_with_prefix,
            version_paths,
            ranked_checks,
            output_path=None   # 若想保存“每版本指标 + AVG”表，可改成 rf'{project}_per_version_metrics.csv'
        )

        # 取该项目内的“均值行”（eval_with_ranked_checks 已经帮你加了一个 version='AVG' 行）
        if 'version' not in df_proj.columns:
            logging.info(f"{project} 评估结果缺少 version 列，跳过。")
            continue

        avg_row = df_proj[df_proj['version'] == 'AVG']
        if avg_row.empty:
            # 兜底：如果没有 AVG 行，就自己算（去掉 version 列，对数值列取均值）
            numeric_cols = [c for c in df_proj.columns if c != 'version']
            avg_series = df_proj[numeric_cols].mean(numeric_only=True)
        else:
            avg_series = avg_row.squeeze()  # 单行 to Series
            # 去掉 'version' 这个字段，只保留数值列
            avg_series = avg_series.drop(labels=['version'], errors='ignore')

        # 记录“项目内均值”
        per_project_avg = {'project': project}
        for col, val in avg_series.items():
            per_project_avg[col] = float(val)
        per_project_avg_rows.append(per_project_avg)

    # —— 汇总所有项目的“项目内均值”
    if not per_project_avg_rows:
        logging.info("没有任何项目的均值被计算出来。")
        return

    per_project_avg_df = pd.DataFrame(per_project_avg_rows)

    # —— 最后一步：对所有 project 的这些均值再取一次平均（各项目等权）
    overall_row = {'project': 'MEAN_ACROSS_PROJECTS'}
    metric_cols = [c for c in per_project_avg_df.columns if c != 'project']
    for col in metric_cols:
        overall_row[col] = per_project_avg_df[col].mean()

    # 拼成最终表（每行一个 project 的“项目内均值”，最后一行是“所有项目的均值”）
    final_df = pd.concat([per_project_avg_df, pd.DataFrame([overall_row])], ignore_index=True)

    # —— 保存总表
    out_csv = Path(config["output_dir"]) / f'compare.csv'
    final_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    logging.info(f"已输出每项目均值 + 跨项目均值：{out_csv}")


if __name__ == "__main__":
    main()