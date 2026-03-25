import subprocess
import pandas as pd
import re
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

    base_output_dir = Path(config["output_dir"]) / "raw_data"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = base_output_dir / f"{name}-{version}-pmd.sarif"


    cmd = [
        config["pmd_bin"],
        "pmd",
        "-d", f"{repo_path}",
        "-R", f"{config['pmd_rules_dir']}/pmd-all-java.xml",
        "-f", "sarif",
        "-r", f"{output_path}",
        "--fail-on-violation", "false",
    ]


    res = subprocess.run(cmd, text=True, capture_output=True)

    # 0: 无问题
    # 4: 有 violations
    if res.returncode in (0, 4):
        print("PMD ok (or violations found).")
        if res.returncode == 4:
            print("PMD return 4")
    else:
        print("PMD failed with returncode:", res.returncode)
        print("stderr:\n", res.stderr)

        raise RuntimeError("PMD failed")

