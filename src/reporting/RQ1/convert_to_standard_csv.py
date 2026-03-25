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

def convert_metrics_view(csv_path: str, out_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df = df[df["Name"] != "原始顺序"]

    # 取列 + 改列名
    out = df[["Name", "R@20", "IFA", "Hit@10", "Effort@20", "PMI@20"]].rename(columns={
        "Name": "Method",
        "R@20": "R@20%",
        "Hit@10": "Accuracy@10",
        "Effort@20": "E@20%R",
        "PMI@20": "PMI@20%",
    })

    out["Method"] = out["Method"].replace({"BernoulliNB": "Naive Bayes",
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

    if out_path is not None:
        out.to_csv(out_path, index=False, encoding="utf-8")

    return out

convert_metrics_view(Path(config["output_dir"]) / "eval_avg.csv", Path(config["output_dir"]) / "eval_avg_standard.csv")
