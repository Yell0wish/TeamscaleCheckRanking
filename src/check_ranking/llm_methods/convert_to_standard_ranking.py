import pandas as pd
from pathlib import Path

def save_to_csv(index_list, ranking_list, input_path, output_path):
    df_output = pd.DataFrame({'feature': index_list, 'name': ranking_list})

    if (output_path is None) or (str(output_path).strip() == ""):
        input_path = Path(input_path)
        out_dir = Path.cwd() / input_path.parent.name                 # 当前目录/Teamscale_deepseek_chat_score_results
        out_dir.mkdir(parents=True, exist_ok=True)               # 自动创建
        output_path = out_dir / f"{input_path.stem}_standard_ranking.csv"
        print("cwd =", Path.cwd())
        print(f"Output path not provided. Saving to: {output_path}")
    else:
        print(f"Saving to provided output path: {output_path}")
    df_output.to_csv(output_path, index=False)

def teamscale_convert_to_standard_ranking(input_path, output_path, llm_method, checks, checks_index_color):
    df_input = pd.read_csv(input_path)
    df_checks = pd.read_csv(checks)
    df_checks_index_color = pd.read_csv(checks_index_color)
    if llm_method == 'rank':
        ranking_list = df_input['feature'].tolist()  
    else:
        df_input["score"] = pd.to_numeric(df_input["score"])
        df_input = df_input.sort_values("score", ascending=False)
        ranking_list = df_input['concept'].tolist()
        # max_row = df_input.loc[df_input["score"].idxmax()]
        # print(max_row)
    index_list = []
    for item in ranking_list:
        item = item.split('::')
        assert len(item) == 2, f"Invalid format in ranking: {item}"
        typeid = df_checks.loc[df_checks["typename"] == item[0], "typeid"]
        assert len(typeid) == 1, f"Expected exactly one match for typename {item[0]}, but found {len(typeid)}"
        typeid = typeid.iloc[0]
        id = df_checks_index_color.loc[(df_checks_index_color["typeid"] == typeid) & (df_checks_index_color['severity'] == item[1]), "id"]
        assert len(id) == 1, f"Expected exactly one match for typeid {typeid} and severity {item[1]}, but found {len(id)}"
        id = id.iloc[0]
        index_list.append(f'Feature_{id}')

    save_to_csv(index_list, ranking_list, input_path, output_path)


def sat_convert_to_standard_ranking(input_path, output_path, llm_method, rules, sat):
    df_input = pd.read_csv(input_path)
    df_rules = pd.read_csv(rules)

    assert sat == 'PMD' or sat == 'Semgrep'

    if sat == 'PMD':
        rules_index_dict = pd.Series(df_rules.index.values, index=df_rules["name"]).to_dict()
    else:
        rules_index_dict = pd.Series(df_rules.index.values, index=df_rules["id_ori"]).to_dict()


    if llm_method == 'rank':
        ranking_list = df_input['feature'].tolist()  
    else:
        df_input["score"] = pd.to_numeric(df_input["score"])
        df_input = df_input.sort_values("score", ascending=False)
        ranking_list = df_input['concept'].tolist()

    index_list = []
    for item in ranking_list:
        idx = rules_index_dict[item]
        index_list.append(f'Feature_{idx}')

    save_to_csv(index_list, ranking_list, input_path, output_path)


teamscale_convert_to_standard_ranking(r'../../../data/llm_results/Teamscale_deepseek_chat_score_results/llm_score_results.csv', 
                                      r'../../../data/llm_results/Teamscale_deepseek_chat_score_results/llm_score_results_standard_ranking.csv', 
                                      'score',
                                      r'../../../data/checks.csv',
                                      r'../../../data/checks_index_color.csv')
teamscale_convert_to_standard_ranking(r'../../../data/llm_results/Teamscale_deepseek_reasoner_score_results/llm_score_results.csv', 
                                      r'../../../data/llm_results/Teamscale_deepseek_reasoner_score_results/llm_score_results_standard_ranking.csv', 
                                      'score',
                                      r'../../../data/checks.csv',
                                      r'../../../data/checks_index_color.csv')
teamscale_convert_to_standard_ranking(r'../../../data/llm_results/Teamscale_deepseek_chat_rank_results/llm_rank_results.csv', 
                                      r'../../../data/llm_results/Teamscale_deepseek_chat_rank_results/llm_rank_results_standard_ranking.csv', 
                                      'rank',
                                      r'../../../data/checks.csv',
                                      r'../../../data/checks_index_color.csv')
teamscale_convert_to_standard_ranking(r'../../../data/llm_results/Teamscale_deepseek_reasoner_rank_results/llm_rank_results.csv', 
                                      r'../../../data/llm_results/Teamscale_deepseek_reasoner_rank_results/llm_rank_results_standard_ranking.csv', 
                                      'rank',
                                      r'../../../data/checks.csv',
                                      r'../../../data/checks_index_color.csv')