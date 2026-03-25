import json
import csv
from pathlib import Path
import re
import pandas as pd

def load_config(config_path: str = "config.json"):
    cfg = {}
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return cfg

config = load_config()

NEW_REPOS_PATH = Path(config["repos_dir"])

# 读取 CSV 文件并生成字典
def load_commit_dict(file_path):
    """
    从 CSV 文件中读取 File Name 和 Commit Hash 并返回一个字典
    """
    try:
        # 使用 pandas 读取 CSV 文件
        df = pd.read_csv(file_path)
        
        # 将 'File Name' 列作为键，'Commit Hash' 列作为值，生成字典
        commit_dict = pd.Series(df['Commit Hash'].values, index=df['File Name']).to_dict()
        
        return commit_dict
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查路径！")
        return {}
    except KeyError:
        print("文件中没有找到 'File Name' 或 'Commit Hash' 列，请确认文件格式！")
        return {}
    

def extract_semgrep_rows(data, project_root):
    """
    从 Semgrep JSON dict 中提取 (check_id, path, start_line, end_line) 行数据。
    """
    project_root = str(project_root).replace("\\", "/").rstrip("/") + "/"
    for r in data.get("results", []) or []:
        check_id = r.get("check_id", "").removeprefix("semgrep-rules.")
        path = str(r.get("path", "")).replace("\\", "/")
        path = path.removeprefix(project_root)

        start_line = (r.get("start") or {}).get("line")
        end_line = (r.get("end") or {}).get("line")

        if start_line is None:
            start_line = ""
        if end_line is None:
            end_line = ""

        yield (check_id, path, start_line, end_line)


def semgrep_json_to_csv(json_input,
                        csv_path, project_root):
    """
    读取 Semgrep JSON（文件路径或已加载的 dict），提取字段并写入 CSV。
    返回写入的 finding 行数。
    """
    # 1) 读取 JSON
    if isinstance(json_input, (str, Path)):
        json_path = Path(json_input)
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(json_input, dict):
        data = json_input
    else:
        raise TypeError("json_input 必须是 JSON 文件路径(str/Path) 或 dict")

    # 2) 写 CSV
    csv_path = Path(csv_path)
    rows = list(extract_semgrep_rows(data, project_root))

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["check_id", "path", "start_line", "end_line"])
        writer.writerows(rows)

    return len(rows)


commit_dict = load_commit_dict(Path(config["successful_projects"]))



for index, file_name in enumerate(commit_dict.keys()):
    pattern = r"(\w+)-([\d\.]+)"
    match = re.search(pattern, file_name)
    assert match is not None
    name = match.group(1)  # 形如"ambari"
    version = match.group(2)  # 形如"1.2.0"

    repo_path=NEW_REPOS_PATH / f'{name}-{version}'
    print(f"Processing {index+1}/{len(commit_dict)}: {repo_path}")

    base_output_dir = Path(config["output_dir"]) / "processed_data"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = base_output_dir / f"{name}-{version}-semgrep.csv"

    semgrep_json_to_csv(Path(config["output_dir"]) / f"raw_data/{name}-{version}-semgrep.json", output_path, str(repo_path))

    print(f"Saved processed data to {output_path}")