import config
import utils
import llm_methods
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["deepseek-reasoner", "deepseek-chat"],
        help="Choose which DeepSeek model to run.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["score", "rank"],
        help="Run llm_score / llm_rank",
    )
    parser.add_argument(
        "--sat",
        required=True,
        default="Teamscale",
        choices=["Teamscale", "PMD", "Semgrep"],
        help="Choose which SAT to evaluate on.",
    )
    args = parser.parse_args()

    if args.model == "deepseek-reasoner":
        config.llm_model = 'deepseek-reasoner'
        if args.task == "score":
            config.log_score_path = config.OUTPUT_DIR / f"{args.sat}_deepseek_reasoner_score.log"
            utils.log_init(config.log_score_path)
            llm_methods.llm_score(sat=args.sat, output_dir= config.OUTPUT_DIR / f'{args.sat}_deepseek_reasoner_score_results', max_tokens=64*2**10)
        else:
            config.log_rank_path = config.OUTPUT_DIR / f"{args.sat}_deepseek_reasoner_rank.log"
            utils.log_init(config.log_rank_path)
            llm_methods.llm_rank(sat=args.sat, output_dir= config.OUTPUT_DIR / f'{args.sat}_deepseek_reasoner_rank_results', max_tokens=64*2**10)
    else:
        config.llm_model = 'deepseek-chat'
        if args.task == "score":
            config.log_score_path = config.OUTPUT_DIR / f"{args.sat}_deepseek_chat_score.log"
            utils.log_init(config.log_score_path)
            llm_methods.llm_score(sat=args.sat, output_dir= config.OUTPUT_DIR / f'{args.sat}_deepseek_chat_score_results', max_tokens=1024)
        else:
            config.log_rank_path = config.OUTPUT_DIR / f"{args.sat}_deepseek_chat_rank.log"
            utils.log_init(config.log_rank_path)
            llm_methods.llm_rank(sat=args.sat, output_dir= config.OUTPUT_DIR / f'{args.sat}_deepseek_chat_rank_results', max_tokens=8192)