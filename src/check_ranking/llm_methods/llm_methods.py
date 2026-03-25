from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_classic.chains.llm import LLMChain
from langchain_core import output_parsers
from functools import partial
import utils
import config
import time
import logging
import ranky as rk
from thefuzz import fuzz
import pandas as pd
from pathlib import Path



def llm_score(sat='Teamscale', max_tokens = 1024, output_dir='score_results'):
    time_start = time.time()

    template = utils.load_template(sat=sat, rank=False)
    context = utils.load_context(sat=sat)
    # print("Loaded template:")
    # print(template)

    input_template = HumanMessagePromptTemplate.from_template(template)
    # print("Constructed input template:")
    # print(input_template)

    prompt = ChatPromptTemplate(
        messages=[input_template],
        input_variables=['concept'],
        partial_variables={
            'min_score': config.min_score,
            'max_score': config.max_score,
            'context': (context if config.add_context else ''),
            'format_instructions': config.parser.get_format_instructions(),
            'examples': ''
        }
    )
    # print("Final prompt:")
    # print(prompt)


    # print("Formatted prompt with concept='Call to printStackTrace()::yellow':")
    # msgs = prompt.format_messages(concept='"equals(Object obj)" should test the argument\'s type (java:S2097)::YELLOW')
    # print(msgs[0].content)   # 这才是最终发给模型的 prompt 文本

    llm = ChatDeepSeek(
        base_url='https://api.deepseek.com',
        model_name=config.llm_model,
        temperature=config.temperature,
        api_key=config.deepseek_api_key,
        max_tokens=max_tokens,
        n=1, # Number of samples to generate for each concept
        max_retries=15 # Exponential backoff
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=config.verbose)

    if sat == 'Teamscale':
        concepts = utils.load_teamscale_concepts(config.teamscale_checks_path)
    elif sat == 'PMD':
        concepts = utils.load_pmd_concepts(config.pmd_rules_path)
    elif sat == 'Semgrep':
        concepts = utils.load_semgrep_concepts(config.semgrep_rules_path)
    else :
        concepts = ['"equals(Object obj)" should test the argument\'s type (java:S2097)::YELLOW', 'Long File::YELLOW']
    inputs = [{'concept': c} for c in concepts for _ in range(config.n_samples)]
    
    res = llm_chain.generate(inputs)

    # print(outputs.generations[0][0].text)

    # print(len(outputs.generations))
    # for i in range(len(outputs.generations)):
    #     print(f"Number of generations for concept '{concepts[i]}': {len(outputs.generations[i])}")

    flat = [res.generations[i][0].message.content for i in range(len(res.generations))]

    outputs = [flat[i:i+config.n_samples] for i in range(0, len(flat), config.n_samples)]



    # outputs = [
    #     [outputs.generations[i][j].message.content for j in range(config.n_samples)] 
    #     for i in range(len(outputs.generations))
    # ]

    # Parse and aggregate outputs
    helper = partial(
        utils.parse_and_aggregate, parser=config.parser, verbose=config.verbose, fix_with_llm=config.fix_with_llm
    )
    out_dict = {c: out for (c, out) in map(helper, concepts, outputs)}

    utils.save_llm_score_results(out_dict, base_dir=output_dir, csv_name='llm_score_results.csv', expl_dir_name='explanations', encoding='utf-8')

    time_end = time.time()
    logging.info(f"\nLLM scoring completed in {time_end - time_start:.2f} seconds.")


def llm_rank(sat='Teamscale', max_tokens = 8192, output_dir='rank_results'):
    time_start = time.time()

    template = utils.load_template(sat=sat, rank=True)

    parser = output_parsers.NumberedListOutputParser()

    input_template = HumanMessagePromptTemplate.from_template(template)

    context = utils.load_context(sat=sat)

    prompt = ChatPromptTemplate(
        messages=[input_template],
        input_variables=['n_concepts', 'concepts'],
        partial_variables={
            'context': (context if config.add_context else ''),
            'format_instructions': parser.get_format_instructions()
        }
    )

    llm = ChatDeepSeek(
        base_url='https://api.deepseek.com',
        model_name=config.llm_model,
        temperature=config.temperature,
        api_key=config.deepseek_api_key,
        max_tokens=max_tokens,
        n=1, # Number of samples to generate for each concept
        max_retries=15 # Exponential backoff
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=config.verbose)

    if sat == 'Teamscale':
        concepts = utils.load_teamscale_concepts(config.teamscale_checks_path)
    elif sat == 'PMD':
        concepts = utils.load_pmd_concepts(config.pmd_rules_path)
    elif sat == 'Semgrep':
        concepts = utils.load_semgrep_concepts(config.semgrep_rules_path)
    else :
        concepts = ['"equals(Object obj)" should test the argument\'s type (java:S2097)::YELLOW', 'Long File::YELLOW']

    # Generate outputs
    inputs = [{
        'n_concepts': len(concepts),
        'concepts': '\n'.join([f"{i}. {c}" for (i,c) in enumerate(concepts, start=1)])
    } for _ in range(config.n_samples)]

    # msgs = prompt.format_messages(
    #     n_concepts=len(concepts),
    #     concepts="\n".join([f"{i}. {c}" for i, c in enumerate(concepts, start=1)])
    # )
    # print(msgs[0].content)

    res = llm_chain.generate(inputs)

    flat = [res.generations[i][0].message.content for i in range(len(res.generations))]
    logging.info(f"results: {flat}")
    # print(flat[0])

    parsed = [parser.parse(output) for output in flat]

    ranks = []
    
    for p in parsed:
        # Ensure outputs match to concepts
        rank = []
        used = set()
        for c_out in p:
            # 在未使用的 concepts 里找最像的（避免重复映射）
            best_c = None
            best_s = -1
            for c in concepts:
                if c in used:
                    continue
                s = fuzz.ratio(c_out, c)
                if s > best_s:
                    best_s = s
                    best_c = c
            if best_c is not None:
                rank.append(best_c)
                used.add(best_c)
        for c in concepts:
            if c not in used:
                rank.append(c)   # 把缺的按原顺序补到末尾

        # for c_out in p:
        #     fuzz_scores = [fuzz.ratio(c_out, c) for c in concepts]
        #     rank.append(concepts[np.argmax(fuzz_scores)])

        ranks.append(rank)

    logging.info(f"Ranks: {ranks}")
    rank_dict = {f'rank_{i+1}': [len(concepts) - r.index(c) for c in concepts] for i,r in enumerate(ranks)}
    rank_df = pd.DataFrame(rank_dict, index=concepts)
    agg_rank = pd.DataFrame(rk.borda(rank_df)).sort_values(by=0).index.to_list()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / 'llm_rank_results.csv'
    agg_df = pd.DataFrame({
        "rank": range(1, len(agg_rank) + 1),
        "feature": agg_rank
    })

    agg_df.to_csv(output_csv, index=False, encoding="utf-8")
    logging.info(f"llm rank results saved: {output_csv}")
    time_end = time.time()
    logging.info(f"\nLLM ranking completed in {time_end - time_start:.2f} seconds.")


def extract_llm_score_prompt(sat='Teamscale', keep_placeholder='{{feature}}'):
    template = utils.load_template(sat=sat, rank=False)
    context = utils.load_context(sat=sat)

    input_template = HumanMessagePromptTemplate.from_template(template)

    prompt = ChatPromptTemplate(
        messages=[input_template],
        input_variables=['concept'],
        partial_variables={
            'min_score': config.min_score,
            'max_score': config.max_score,
            'context': (context if config.add_context else ''),
            'format_instructions': config.parser.get_format_instructions(),
            'examples': ''
        }
    )

    prompt_text = prompt.format_messages(concept=keep_placeholder)[0].content
    return prompt_text

def extract_llm_rank_prompt(
    sat='Teamscale',
    keep_n_concepts='{{n_features}}',
    keep_concepts='{{features}}'
):
    template = utils.load_template(sat=sat, rank=True)
    parser = output_parsers.NumberedListOutputParser()
    input_template = HumanMessagePromptTemplate.from_template(template)
    context = utils.load_context(sat=sat)

    prompt = ChatPromptTemplate(
        messages=[input_template],
        input_variables=['n_concepts', 'concepts'],
        partial_variables={
            'context': (context if config.add_context else ''),
            'format_instructions': parser.get_format_instructions()
        }
    )

    prompt_text = prompt.format_messages(
        n_concepts=keep_n_concepts,
        concepts=keep_concepts
    )[0].content

    return prompt_text