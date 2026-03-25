import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional
import csv
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

PMD_RULESET_NS = "http://pmd.sourceforge.net/ruleset/2.0.0"


@dataclass
class RuleInfo:
    name: str
    ruleset: str
    priority: Optional[int]
    description: str


def _norm_text(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    text = "".join(elem.itertext())
    return " ".join(text.split())


def _is_true_attr(val: Optional[str]) -> bool:
    if val is None:
        return False
    return val.strip().lower() in {"true", "1", "yes"}


def parse_pmd_ruleset(xml_path: str) -> List[RuleInfo]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {"p": PMD_RULESET_NS}
    ruleset_name = (root.get("name") or "").strip()

    rules: List[RuleInfo] = []
    for rule in root.findall("p:rule", ns):
        # 跳过 deprecated=true 的占位 rule（例如：<rule ... deprecated="true" ... />）
        if _is_true_attr(rule.get("deprecated")):
            continue

        name = (rule.get("name") or "").strip()

        prio_elem = rule.find("p:priority", ns)
        prio_text = _norm_text(prio_elem)
        priority = int(prio_text) if prio_text.isdigit() else None

        desc_elem = rule.find("p:description", ns)
        description = _norm_text(desc_elem)

        rules.append(
            RuleInfo(
                name=name,
                ruleset=ruleset_name,
                priority=priority,
                description=description,
            )
        )

    return rules


def save_rules_to_csv(rules: List[RuleInfo], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "ruleset", "priority", "description"])
        writer.writeheader()
        for r in rules:
            writer.writerow(
                {
                    "name": r.name,
                    "ruleset": r.ruleset,
                    "priority": "" if r.priority is None else r.priority,
                    "description": r.description,
                }
            )


if __name__ == "__main__":
    base_dir = Path(config.get("pmd_rules_dir"))
    xml_files = sorted(base_dir.rglob("*.xml"))

    all_rules: List[RuleInfo] = []
    for p in xml_files:
        all_rules.extend(parse_pmd_ruleset(str(p)))

    save_rules_to_csv(all_rules, Path(config.get("output_dir")) / "pmd_rules.csv")
    print(f"Saved {len(all_rules)} rules to: {config.get('output_dir')}/pmd_rules.csv")