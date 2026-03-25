from langchain_core.output_parsers import PydanticOutputParser
import pydantic
from typing import Optional
from pathlib import Path

class FeatureImportance(pydantic.BaseModel):
    '''Langchain Pydantic output parsing structure.'''

    reasoning: Optional[str] = pydantic.Field(
        description='Logical reasoning behind feature importance score'
    )
    score: float = pydantic.Field(
        description='Feature importance score'
    )

def load_api_key(key_path):
    with open(key_path, 'r') as f:
        return f.read().strip()


OUTPUT_DIR = Path('../../../data/llm_results/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_DIR = "../../../data/prompts/"
log_score_path = ""
log_rank_path = ""
min_score = 0
max_score = 1
parser = PydanticOutputParser(pydantic_object=FeatureImportance)
llm_model = 'deepseek-chat'
temperature = 0
deepseek_api_key = load_api_key("./API-key.txt") # add your API key in this file
n_samples = 5
verbose = False
add_context = True
fix_with_llm = True
teamscale_checks_path = r'../../../data/checks.csv'
pmd_rules_path = r'../../../data/generalization/pmd/pmd_rules.csv'
semgrep_rules_path = r'../../../data/generalization/semgrep/semgrep_rules.csv'