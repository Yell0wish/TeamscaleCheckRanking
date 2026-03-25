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

    out = df.rename(columns=lambda c: (
        "Method" if c == "Name"
        else c.replace("Hit@", "Accuracy@")
    )).copy()

    out.columns = [
        col if col in ["Method", "IFA"]
        else f"R@{col.split('@')[1]}%" if col.startswith("R@")
        else f"Accuracy@{col.split('@')[1]}" if col.startswith("Accuracy@")
        else f"E@{col.split('@')[1]}%R" if col.startswith("Effort@")
        else f"PMI@{col.split('@')[1]}%" if col.startswith("PMI@")
        else col
        for col in out.columns
    ]


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

convert_metrics_view(Path(config["output_dir"]) / 'Teamscale_k5_avg_results.csv', Path(config["output_dir"]) / 'Teamscale_k5_avg_results_standard.csv')
convert_metrics_view(Path(config["output_dir"]) / 'Teamscale_k10_avg_results.csv', Path(config["output_dir"]) / 'Teamscale_k10_avg_results_standard.csv')
convert_metrics_view(Path(config["output_dir"]) / 'Teamscale_k15_avg_results.csv', Path(config["output_dir"]) / 'Teamscale_k15_avg_results_standard.csv')
convert_metrics_view(Path(config["output_dir"]) / 'Teamscale_k20_avg_results.csv', Path(config["output_dir"]) / 'Teamscale_k20_avg_results_standard.csv')