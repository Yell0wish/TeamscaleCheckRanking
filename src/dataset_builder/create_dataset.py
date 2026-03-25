import re
import os
import pandas as pd
import logging
import sys
import time
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

# 配置 logging，写入文件并打印到终端
log_file_path = "create_dataset.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # 终端输出
        logging.FileHandler(log_file_path, mode="a", encoding="utf-8")  # 写入日志文件
    ]
)

# 确保 stdout 也是 UTF-8
sys.stdout.reconfigure(encoding="utf-8") # type: ignore


def read_file_level_dataset(release='', 
                            file_path=Path(config['file_level_bug_labels_dir']), 
                            file_level_path_suffix = '_ground-truth-files_dataset.csv'):
    """
    :param release:项目名
    :param file_path
    :return:
    """
    if release == '':
        return [], [], [], []
    path = file_path / f'{release}{file_level_path_suffix}'
    # print(path)
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        # 文件信息索引列表, 每个文件名不一样该语句才没有错误 TODO line.index(line)
        src_file_indices = [lines.index(line) for line in lines if r'.java,true,"' in line or r'.java,false,"' in line]
        # 源文件路径,需要时返回 OK
        src_files = [lines[index].split(',')[0] for index in src_file_indices]
        # 缺陷标记
        string_labels = [lines[index].split(',')[1] for index in src_file_indices]
        numeric_labels = [1 if label == 'true' else 0 for label in string_labels]

        # 行级别的文本语料库
        texts_lines = []
        for i in range(len(src_file_indices)):
            # 从当前文件名所在行开始到下一个文件名所在行结束的所有代码行
            s_index = src_file_indices[i]
            e_index = src_file_indices[i + 1] if i + 1 < len(src_file_indices) else len(lines)

            # xxx 也许需要过滤掉注释行
            code_lines = [line.strip() for line in lines[s_index:e_index]]
            # 去掉首行中的文件名和标签,以及首行中的引号"
            match = re.match(r'^.*?,(true|false),"(.*)', code_lines[0])
            # code_lines[0] = (code_lines[0].split(',')[-1][1:]).strip()
            code_lines[0] = match.group(2).strip() # type: ignore
            # 删除列表中最后的"
            code_lines = code_lines[:-1]
            texts_lines.append(code_lines)

        # 多行合并后的文本语料库
        texts = [' '.join(line) for line in texts_lines]

        # print(texts[0])
        # print(texts_lines[0])
        # print(numeric_labels[0])
        # print(src_files[0])
        
        return texts, texts_lines, numeric_labels, src_files


def create_dataset(release, abs_ts_path, abs_labels_path, abs_findings_path, abs_repo_path=None):
    """
    构建数据集：格式为 [file, line, code snippet, label]
    - 从 abs_repo_path 读取 Java 代码 (如果是需要Test的话, 则是abs_repo_path)
    - 读取 abs_labels_path 获取缺陷代码行并标记为 1
    - 读取 abs_findings_path 获取违反规则的 finding_type 和 severity
    - 读取 abs_ts_path 获取实际违反规则情况
    """
    # 读取缺陷代码行
    labels_df = []

    with open(abs_labels_path, 'r', encoding='utf-8') as file:
        next(file)  # 跳过第一行标题
        for row in file:
            row = row.strip()
            match = re.match(r'^(.*?),(\d+),(.+)$', row)
            if match:
                file_path = match.group(1).strip()
                line_number = int(match.group(2))
                code_snippet = match.group(3).strip()
                labels_df.append([file_path, line_number, code_snippet])

    labels_df = pd.DataFrame(labels_df, columns=['file', 'line', 'code snippet'])
    labels_dict = {(row["file"], row["line"]): row["code snippet"] for _, row in labels_df.iterrows()}

    logging.info(f"读取 {len(labels_dict)} 个缺陷代码行")

    # 读取finding_type
    findings_df = pd.read_csv(abs_findings_path)

    # 处理 severity
    transformed_findings = []

    for _, row in findings_df.iterrows():
        typeid = row["typeid"]
        severity = row["severity"]

        if severity == "Default:Auto":
            # Auto -> 生成两条记录
            transformed_findings.append((typeid, "RED"))
            transformed_findings.append((typeid, "YELLOW"))
        else:
            # 其他情况，去掉 "Default:" 前缀
            new_severity = "RED" if severity == "Default:Red" else "YELLOW"
            transformed_findings.append((typeid, new_severity))

    # 转换为 DataFrame
    transformed_df = pd.DataFrame(transformed_findings, columns=["typeid", "severity"])

    # 生成唯一 ID
    finding_mapping = {}
    next_id = 4  # ID 从 4 开始

    for _, row in transformed_df.iterrows():
        key = (row["typeid"], row["severity"])
        if key not in finding_mapping:
            finding_mapping[key] = next_id
            next_id += 1
        else:
            print(f"Duplicate finding type and severity: {key}")

    # 读取实际违反规则情况
    ts_df = pd.read_csv(abs_ts_path)
    ts_dict = {}
    # location为文件相对路径，line为行号
    # 遍历数据集
    for _, row in ts_df.iterrows():
        try:
            finding_type = row["type"].strip()
            severity = row["severity"].strip()
            location = row["location"].strip()
            start_line = int(float(str(row["start_line"]).strip()))
            end_line = int(float(str(row["end_line"]).strip()))
        except Exception as e:
            location = str(row["location"]).strip()
            finding_type = str(row["type"]).strip()
            severity = str(row["severity"]).strip()
            ts_dict.setdefault((location, -1), set()).add((finding_type, severity))
            continue


        # 遍历 start_line 到 end_line 之间的每一行
        for line in range(start_line, end_line + 1):
            key = (location, line)
            # ts_dict[key] = (finding_type, severity)
            ts_dict.setdefault(key, set()).add((finding_type, severity))


    logging.info(f"读取 {len(ts_dict)} 条违反规则记录")

    temp_count = 0
    # 从 file-level 数据集获取所有行
    _, texts_lines, _, src_files = read_file_level_dataset(
        release=release,
    )
    # 格式 file, line, code snippet, label, rules(232 + 3 = 235)
    dataset = []

    used_labels = set()

    for file_idx, rel_path in enumerate(src_files):
        for line_no, code_line in enumerate(texts_lines[file_idx], start=1):
            data_item = [rel_path, line_no, code_line, 0] + [0] * (max(finding_mapping.values()) - 4 + 1)

            # 标签
            if (rel_path, line_no) in labels_dict:
                expected_code = labels_dict[(rel_path, line_no)]
                if expected_code != code_line:
                    logging.info(f"代码不匹配: {rel_path} (行 {line_no})")
                    logging.info(f"  期望: {expected_code}")
                    logging.info(f"  实际: {code_line}\n")
                else:
                    data_item[3] = 1
                    used_labels.add((rel_path, line_no))  # 记录已使用的标签

            # 规则特征
            if (rel_path, line_no) in ts_dict:
                # data_item[finding_mapping[ts_dict[(rel_path, line_no)]]] += 1
                # temp_count += 1
                for ft, sev in ts_dict[(rel_path, line_no)]:
                    data_item[finding_mapping[(ft, sev)]] = 1
                    temp_count += 1
            if (rel_path, -1) in ts_dict:
                # data_item[finding_mapping[ts_dict[(rel_path, -1)]]] += 1
                for ft, sev in ts_dict[(rel_path, -1)]:
                    data_item[finding_mapping[(ft, sev)]] = 1

            dataset.append(data_item)
    # 检查没用到的 labels
    unused_labels = set(labels_dict.keys()) - used_labels
    if unused_labels:
        logging.info("以下缺陷行在数据集中未匹配到：")
        for key in unused_labels:
            file, line = key
            logging.info(f"{file} (行 {line}): {labels_dict[key]}")
    else:
        logging.info("所有缺陷行都已匹配到数据集")
    logging.info(f"生成数据集，共 {len(dataset)} 条记录，包含 {temp_count} 条findings")

    return dataset


DICT_FILE = Path(config['successful_projects'])

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

CHECKS_PATH = Path(config['checks'])

def main():
    start_time = time.time()  # 记录总任务的开始时间

    output_dir = Path(config['dataset_output_dir'])
    os.makedirs(output_dir, exist_ok=True)

    # 读取 CSV 文件
    try:
        df = pd.read_csv(DICT_FILE)
    except Exception as e:
        logging.info(f"读取文件失败: {e}")
        return None

    for project in DATASET_PROJECT_MAP.keys():
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



        for version in version_list:
            abs_ts_path = Path(config['finding_churns_processed_dir']) / f"{version}.csv"
            abs_labels_path = Path(config['line_level_bug_labels_dir']) / f"{version}_defective_lines_dataset.csv"

            dataset = create_dataset(version, abs_ts_path, abs_labels_path, CHECKS_PATH)
        
            assert(len(dataset[0]) == 239)
            df_ = pd.DataFrame(dataset, columns=["file", "line", "code snippet", "label"] + [f"feature_{i}" for i in range(235)])
            file_path = os.path.join(output_dir, f"{version}.parquet")
            df_.to_parquet(file_path, engine="pyarrow", compression="snappy")
            logging.info(f"{version} 数据集保存完成：{file_path}")
            logging.info(f"DataFrame 维度: {df_.shape}")  # (行数, 列数)
            logging.info(f"前5行数据:\n{df_.head()}")  # 预览前5行数据
        # 计算并打印当前项目 的处理时间
        project_end_time = time.time()
        project_elapsed_time = project_end_time - project_start_time
        logging.info(f"{project} 处理完成，耗时: {project_elapsed_time:.2f} 秒 ({project_elapsed_time/60:.2f} 分钟)")

    # 记录总任务的结束时间并计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"任务完成，总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")




if __name__ == '__main__':
    main()