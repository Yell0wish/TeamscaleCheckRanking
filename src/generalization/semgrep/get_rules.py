from pathlib import Path
import csv
import json
import yaml


def load_config(config_path: str = "config.json"):
    cfg = {}
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return cfg

config = load_config()

def extract_yaml_rules_to_csv(root_dir: str | Path, output_csv: str | Path) -> int:
    """
    递归遍历 root_dir 下所有 .yaml/.yml 文件，提取每条 rule 的:
      - id
      - severity
      - languages
    并写入 output_csv（CSV 列为：id,severity,languages）。

    返回：写入的行数（rule 数量）。
    """
    root = Path(root_dir)
    output = Path(output_csv)

    rows: list[dict[str, str]] = []

    # 同时支持 .yaml 和 .yml
    yaml_files = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))

    for ypath in yaml_files:
        
        try:
            with ypath.open("r", encoding="utf-8") as f:

                for doc in yaml.safe_load_all(f):
                    if not isinstance(doc, dict):
                        continue

                    rules = doc.get("rules")
                    if not isinstance(rules, list):
                        continue

                    for rule in rules:
                        if not isinstance(rule, dict):
                            continue

                        rule_id = rule.get("id")
                        assert rule_id is not None, f"Rule in {ypath} is missing 'id' field"
                        rule_id = str(rule_id)

                        file_no_suffix = ypath.with_suffix("")

                        file_parts = list(file_no_suffix.parts)
                        parent_parts = list(ypath.parent.parts)

                        if "semgrep-rules" in file_parts:
                            file_parts = file_parts[file_parts.index("semgrep-rules") + 1:]

                        if "semgrep-rules" in parent_parts:
                            parent_parts = parent_parts[parent_parts.index("semgrep-rules") + 1:]

                        rule_id_real = ".".join(file_parts + [rule_id])
                        rule_id = ".".join(parent_parts + [rule_id])
                        
                        severity = rule.get("severity")
                        languages = rule.get("languages")

                        # languages 通常是 list
                        if isinstance(languages, list):
                            languages_str = ",".join(str(x) for x in languages)
                        elif languages is None:
                            languages_str = ""
                        else:
                            languages_str = str(languages)

                        rows.append(
                            {
                                "id": "" if rule_id is None else str(rule_id),
                                "id_ori": str(rule_id_real),
                                "severity": "" if severity is None else str(severity),
                                "languages": languages_str,
                            }
                        )

        except Exception as e:
            # 解析失败就跳过该文件（也可以改成 raise 或记录日志）
            print(f"[WARN] Failed to parse {ypath}: {type(e).__name__}: {e}")

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "severity", "languages", "id_ori"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    n = extract_yaml_rules_to_csv(
        root_dir=config["semgrep_rules_dir"],
        output_csv=config["output_dir"] + "/semgrep_rules.csv",
    )
    print(f"Done. Wrote {n} rules to semgrep_rules.csv")