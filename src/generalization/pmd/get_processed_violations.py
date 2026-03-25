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

def extract_pmd_sarif_items(sarif_path, project_root):
    project_root = str(project_root).replace("\\", "/").rstrip("/") + "/"

    with open(sarif_path, "r", encoding="utf-8") as f:
        sarif = json.load(f)

    items = []
    for run in sarif.get("runs", []):
        for res in run.get("results", []):
            rule_id = res.get("ruleId")
            for loc in res.get("locations", []):
                phys = loc.get("physicalLocation", {})
                uri = phys.get("artifactLocation", {}).get("uri")
                region = phys.get("region", {})
                start_line = region.get("startLine")
                end_line = region.get("endLine", start_line)

                norm_uri = str(uri).replace("\\", "/")
                rel_path = norm_uri.removeprefix(project_root)

                items.append({
                    "ruleID": rule_id,
                    "location": rel_path,
                    "startLine": start_line,
                    "endLine": end_line,
                })

    return items

def save_items_to_csv(items, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ruleID", "location", "startLine", "endLine"])
        writer.writeheader()
        writer.writerows(items)



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
    


commit_dict = load_commit_dict(Path(config["successful_projects"]))



for index, file_name in enumerate(commit_dict.keys()):
    pattern = r"(\w+)-([\d\.]+)"
    match = re.search(pattern, file_name)
    assert match is not None
    name = match.group(1)  # 形如"ambari"
    version = match.group(2)  # 形如"1.2.0"

    # if f'{name}-{version}' != "amq-5.14.0":
    #     continue

    repo_path=NEW_REPOS_PATH / f'{name}-{version}'
    print(f"Processing {index+1}/{len(commit_dict)}: {repo_path}")

    items = extract_pmd_sarif_items(Path(config["output_dir"]) / f"raw_data/{name}-{version}-pmd.sarif", str(repo_path))

    base_output_dir = Path(config["output_dir"]) / "processed_data"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = base_output_dir / f"{name}-{version}-pmd.csv"

    save_items_to_csv(items, output_path)

    print(f"Saved processed data to {output_path}")