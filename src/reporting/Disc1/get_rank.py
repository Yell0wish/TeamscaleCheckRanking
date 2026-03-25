import re
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

def replace_with_rank(csv_path, output_path=None, drop_cols=None, custom_ascending=None):
    """
    读取 CSV -> 丢掉指定列 -> 对剩余数值列计算名次（rank）。
    - drop_cols: 需要丢掉的列名列表（会智能匹配是否带 %）
    - custom_ascending: {col: bool} 指定个别列是否升序（True=小值名次靠前）
    """
    df = pd.read_csv(csv_path)
    # df = df[df["Name"] != "原始顺序"]
    # df.rename(columns={"Name": "Method"}, inplace=True)
    # 统一列名：Hit@k -> Accuracy@k
    df.rename(columns=lambda c: re.sub(r'(?i)^hit@', 'Accuracy@', str(c)), inplace=True)

    df["Method"] = df["Method"].replace({"BernoulliNB": "Naive Bayes",
                                           'BalancedBagging': 'Balanced Bagging',
                                           'BalancedRandomForest': 'Balanced Random Forest',
                                           'ExtraTrees': 'Extra Trees',
                                           'GradientBoost': 'Gradient Boosting',
                                           'RandomForest': 'Random Forest',
                                           'LogisticRegression': 'Logistic Regression',
                                           'deepseek-chat-rank': 'deepseek-chat-Rank',
                                           'deepseek-chat-score': 'deepseek-chat-Score',
                                           'deepseek-reasoner-rank': 'deepseek-reasoner-Rank',
                                           'deepseek-reasoner-score': 'deepseek-reasoner-Score',
                                           })


    # 方法名列：兼容 "Method" 或 "Name"
    id_col = next((c for c in ["Method", "Name"] if c in df.columns), None)
    if id_col is None:
        raise ValueError("未找到方法名列，请确保存在 'Method' 或 'Name' 列")

    # —— 标准化要丢的列名（同时兼容是否带百分号）——
    drop_candidates = set()
    if drop_cols:
        for c in drop_cols:
            c_no_pct = c.replace("%", "")
            drop_candidates.add(c_no_pct)
            drop_candidates.add(c_no_pct + "%")  # 同名带百分号版本

    # 实际要丢的列（与现有列取交集）
    to_drop = [c for c in df.columns if c != id_col and (c in drop_candidates or c.replace("%", "") in drop_candidates)]
    df_kept = df.drop(columns=to_drop, errors='ignore').copy()

    # 需要排名的列（数值列）
    num_cols = [c for c in df_kept.columns if c != id_col and pd.api.types.is_numeric_dtype(df_kept[c])]

    ranked_df = df_kept.copy()

    for col in num_cols:
        # 优先使用自定义
        if custom_ascending and col in custom_ascending:
            ascending = custom_ascending[col]
        else:
            # 自动规则：
            #   R@k / Hit@k / Accuracy@k -> 值越大越好 => ascending=False
            #   其余默认值越小越好 => ascending=True
            col_no_pct = col.replace("%", "")
            if re.match(r"^(R@|Hit@|Accuracy@)\d+$", col_no_pct):
                ascending = False
            else:
                ascending = True

        # 使用最小名次并发放相同名次
        ranked_df[col] = df_kept[col].rank(method="min", ascending=ascending).astype(int)

    if output_path:
        ranked_df.to_csv(output_path, index=False)

    return ranked_df, id_col


def prettify_metric_name(col):
    """把列名改成论文里更想展示的形式"""
    col = str(col)

    # R@15 -> R@15%
    col = re.sub(r"^R@(\d+)$", r"R@\1%", col)

    # Effort@15 -> E@15%R
    col = re.sub(r"^Effort@(\d+)$", r"E@\1%R", col)

    # PMI@15 -> PMI@15%
    col = re.sub(r"^PMI@(\d+)$", r"PMI@\1%", col)

    return col


drop_cols = []

# 可选：如果想手动覆盖某些列的升降序
custom_ascending = {

}


ranked_df, id_col = replace_with_rank(
    Path(config["output_dir"]) / 'Teamscale_k5_avg_results_standard.csv',
    output_path=Path(config["output_dir"]) / "Teamscale_k5_ranked.csv",
    drop_cols=drop_cols,
    custom_ascending=custom_ascending
)

ranked_df, id_col = replace_with_rank(
    Path(config["output_dir"]) / 'Teamscale_k10_avg_results_standard.csv',
    output_path=Path(config["output_dir"]) / "Teamscale_k10_ranked.csv",
    drop_cols=drop_cols,
    custom_ascending=custom_ascending
)

ranked_df, id_col = replace_with_rank(
    Path(config["output_dir"]) / 'Teamscale_k15_avg_results_standard.csv',
    output_path=Path(config["output_dir"]) / "Teamscale_k15_ranked.csv",
    drop_cols=drop_cols,
    custom_ascending=custom_ascending
)

ranked_df, id_col = replace_with_rank(
    Path(config["output_dir"]) / 'Teamscale_k20_avg_results_standard.csv',
    output_path=Path(config["output_dir"]) / "Teamscale_k20_ranked.csv",
    drop_cols=drop_cols,
    custom_ascending=custom_ascending
)
