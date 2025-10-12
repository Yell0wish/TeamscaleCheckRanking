import pandas as pd
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

def merge_check_info(rp_file, id_type_file, typename_file, output_file=None):
    """
    合并三个CSV文件，生成包含 ID, Name, Severity, RP* value 的表
    
    参数：
    rp_file: str, 包含 check_id 和 RP* value 的csv路径
    id_type_file: str, 包含 id, typeid, severity 的csv路径
    typename_file: str, 包含 typename, typeid, severity 的csv路径
    output_file: str, 可选，保存合并结果的csv路径
    
    返回：
    DataFrame: 合并后的结果
    """
    # 读取三个文件
    df_rp = pd.read_csv(rp_file)                # check_id, value
    df_id = pd.read_csv(id_type_file)           # id, typeid, severity
    df_name = pd.read_csv(typename_file)        # typename, typeid, severity
    
    # 合并：先把 id_type_file 跟 rp_file 对齐
    df_rp = df_rp.rename(columns={"check_id": "id", "value": "RP value"})
    df_merge = pd.merge(df_rp, df_id, on="id", how="left")
    
    # 再跟 typename_file 用 typeid 对齐
    df_merge = pd.merge(df_merge, df_name[["typename", "typeid"]], on="typeid", how="left")
    
    # 选出需要的列并重命名
    df_merge = df_merge[["id", "typename", "severity", "RP value"]]
    df_merge = df_merge.rename(columns={
        "id": "ID",
        "typename": "Name",
        "severity": "Severity"
    })
    
    if output_file:
        df_merge.to_csv(output_file, index=False)
    
    return df_merge

merge_check_info(
    Path(config["output_dir"]) / "rp_gmean.csv",
    Path(config["checks_index_color"]),
    Path(config["checks"]),
    Path(config["output_dir"]) /'global_checks_ranking.csv'
)