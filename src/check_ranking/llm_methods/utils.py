import os.path as osp
from langchain_classic.output_parsers import OutputFixingParser
from langchain_deepseek import ChatDeepSeek
import numpy as np
import re
from langchain_core import output_parsers
import config
import pandas as pd
from pathlib import Path
import csv
import logging
import sys

def log_init(log_path):
    log_file_path = Path(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),  # 终端输出
            logging.FileHandler(log_file_path, mode="w", encoding="utf-8")  # 写入日志文件
        ]
    )

    # 确保 stdout 也是 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")

# Helper function for parsing concept outputs
def parse_and_aggregate(concept, concept_outputs, parser, verbose, fix_with_llm):
    scores = []
    expls = []
    n_samples = len(concept_outputs)

    # Check parser type
    if isinstance(parser, output_parsers.PydanticOutputParser):
        parser_type = 'pydantic'
    
    else:
        # 报错
        raise ValueError('Unsupported parser type. Please use PydanticOutputParser.')

    for i in range(n_samples):
        output = concept_outputs[i]

        try:
            parsed = parser.parse(output)
            
            if parser_type == 'pydantic':
                score, expl = parsed.score, parsed.reasoning
                logging.info(f"Parsed output for concept '{concept}': score={score}")

            elif parser_type == 'structured':
                score, expl = parsed['score'], parsed['reasoning']
        
        except Exception as e1:
            logging.error(f"Parsing failed with error: {e1}")
            logging.error(f"Original output: {output}")
            # Try to reformat output with higher-capacity LLM
            if fix_with_llm:
                try:
                    fix_parser = OutputFixingParser.from_llm(
                        parser=parser,
                        llm = ChatDeepSeek(
                            base_url='https://api.deepseek.com',
                            model_name=config.llm_model,
                            temperature=config.temperature,
                            api_key=config.deepseek_api_key,
                        )
                    )
                    fixed_output = fix_parser.retry_chain.invoke({
                        "instructions": parser.get_format_instructions(),
                        "completion": output,
                        "error": str(e1),
                    })

                    # 如果返回的是 message，就取 content；如果是 str 就直接用
                    if hasattr(fixed_output, "content"):
                        fixed_output = fixed_output.content
                    # retry_p = [
                    #     HumanMessage(
                    #         content=fix_parser.retry_chain.prompt.format(
                    #             instructions=parser.get_format_instructions(),
                    #             completion=output,
                    #             error=str(e1)
                    #         )
                    #     )
                    # ]
                    # fixed_output = fix_parser.retry_chain.llm(retry_p).content

                    if verbose:
                        logging.info(f'\n[{concept}] Attempting to parse LLM-fixed output:')
                        logging.info(f'{fixed_output}\n')

                    parsed = parser.parse(fixed_output)

                    if parser_type == 'pydantic':
                        score, expl = parsed.score, parsed.reasoning
                    
                    elif parser_type == 'structured':
                        score, expl = parsed['score'], parsed['reasoning']
                
                # Try manual parsing
                except:
                    if verbose:
                        logging.info(f'[{concept}] Attempting manual parsing...')

                    match = re.search('{.*}', output, re.DOTALL)
                    if match:
                        try:
                            parsed = parser.parse(match.group(0))

                            if parser_type == 'pydantic':
                                score, expl = parsed.score, parsed.reasoning

                            elif parser_type == 'structured':
                                score, expl = parsed['score'], parsed['reasoning']
                        except:
                            parsed = None
                            score, expl = np.nan, match.group(0)
                    else:
                        parsed = None
                        score, expl = np.nan, output
            else:
                if verbose:
                    logging.info(f'[{concept}] Attempting manual parsing...')

                match = re.search('{.*}', output, re.DOTALL)
                if match:
                    try:
                        parsed = parser.parse(match.group(0))

                        if parser_type == 'pydantic':
                            score, expl = parsed.score, parsed.reasoning

                        elif parser_type == 'structured':
                            score, expl = parsed['score'], parsed['reasoning']
                    except:
                        parsed = None
                        score, expl = np.nan, match.group(0)
                else:
                    parsed = None
                    score, expl = np.nan, output


        if parsed is None and verbose:
            logging.error(f'[{concept}] Parsing failed with the following output:')
            logging.error(f'{output}\n')

        scores.append(score)
        expls.append(expl)
    
    # Average score
    try:
        scores = list(map(lambda x: x if isinstance(x,(int, float, np.integer, np.floating)) else np.nan, scores))
        final_score = np.nanmean(scores)
    except:
        final_score = np.nan
    
    # Print all generated explanations
    logging.info(f'\nVariable: {concept}')
    logging.info(f'Score: {final_score:.2f}')
    
    # for i, expl in enumerate(expls):
    #     logging.info(f'Explanation {i+1}: {expl}')
    return concept, dict(score=float(final_score), expl=expls)



def load_context(
    sat='Teamscale',
    prompt_dir=config.PROMPT_DIR
):
    '''Prompt context loading function.'''

    context_path = osp.join(prompt_dir, f'{sat}_context.txt')
    with open(context_path, 'r') as fh:
        context = fh.read()

    return context

def load_template(
    sat='Teamscale',
    prompt_dir=config.PROMPT_DIR,
    rank=False,
):
    '''Prompt template loading function.'''

    if rank:
        template_path = osp.join(prompt_dir, f'{sat}_template_rank.txt')
        with open(template_path, 'r') as fh:
            template = fh.read()

        instruction = 'Rank all {n_concepts} features in the following list:\n{concepts}.'
        #instruction += '\nOnly output the feature names (no explanations) in your answer.'
        
        template += '\n' + instruction + '\n'
    else:
        template_path = osp.join(prompt_dir, f'{sat}_template.txt')
        # print(template_path)
        with open(template_path, 'r') as fh:
            template = fh.read()

        instruction = 'Provide a score and reasoning for "{concept}" ' \
            'formatted according to the output schema above:'
        

        template += '\n' + instruction + '\n'

    return template

def load_teamscale_concepts(teamscale_checks_path):
    # 读取finding_type
    findings_df = pd.read_csv(teamscale_checks_path)

    # 处理 severity
    transformed_findings = []

    for _, row in findings_df.iterrows():
        type_name = row["typename"]
        severity = row["severity"]

        if severity == "Default:Auto":
            # Auto -> 生成两条记录
            transformed_findings.append(type_name + "::RED")
            transformed_findings.append(type_name + "::YELLOW")
        else:
            # 其他情况，去掉 "Default:" 前缀
            new_severity = "RED" if severity == "Default:Red" else "YELLOW"
            transformed_findings.append(type_name + "::" + new_severity)
    
    return transformed_findings

def load_pmd_concepts(pmd_rules_path):
    rules_df = pd.read_csv(pmd_rules_path)
    concepts = []
    for _, row in rules_df.iterrows():
        rule_name = row["name"]
        concepts.append(rule_name)
    return concepts

def load_semgrep_concepts(semgrep_rules_path):
    rules_df = pd.read_csv(semgrep_rules_path)
    concepts = []
    for _, row in rules_df.iterrows():
        rule_id = row["id_ori"]
        concepts.append(rule_id)
    return concepts

def save_llm_score_results(
    out_dict,
    base_dir,
    csv_name="llm_scores.csv",
    expl_dir_name="explanations",
    encoding="utf-8",
) -> None:
    """
    out_dict example:
    {
      "type::red": {"score": 0.73, "expl": ["...", "..."]},
      ...
    }

    Saves:
    1) CSV with columns: concept, score
    2) One text file per concept containing all explanations
    """
    base_dir = Path(base_dir)
    csv_path = base_dir / csv_name
    expl_dir = base_dir / expl_dir_name

    # Ensure directories exist
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    expl_dir.mkdir(parents=True, exist_ok=True)  # :contentReference[oaicite:2]{index=2}

    # ---- 1) Save concept + score to CSV ----
    with csv_path.open("w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=["concept", "score"])  # :contentReference[oaicite:3]{index=3}
        writer.writeheader()

        for concept, payload in out_dict.items():
            score = payload.get("score", None)
            # Make sure score is numeric if possible
            try:
                score_val = float(score)
            except Exception:
                score_val = ""  # leave blank if not parseable

            writer.writerow({"concept": concept, "score": score_val})

    # ---- 2) Save explanations to separate files ----
    for idx, (concept, payload) in enumerate(out_dict.items()):
        expls = payload.get("expl", [])
        if expls is None:
            expls = []
        if not isinstance(expls, list):
            expls = [str(expls)]

        safe_name = idx
        expl_path = expl_dir / f"{safe_name}.txt"

        # Write each explanation with an index
        with expl_path.open("w", encoding=encoding) as f:
            f.write(f"Concept: {concept}\n")
            f.write(f"Num explanations: {len(expls)}\n\n")
            for i, expl in enumerate(expls, start=1):
                f.write(f"Explanation {i}:\n{expl}\n")
                f.write("\n" + ("-" * 40) + "\n\n")