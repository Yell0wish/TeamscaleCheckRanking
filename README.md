# TeamscaleCheckRanking

## Paper Title  
**Line-level bug-finding power of static analysis rules: A case study of Teamscale**

---

## 1. Directory Structure

- **[`/data`](./data/)**
  
  Contains the released datasets, prompts, and experimental outputs used in the paper.
  
  - [`/data/dataset`](./data/dataset/)
    
    Teamscale dataset files for the 17 subject projects (134 releases in total).
  
  - [`/data/reporting`](./data/reporting/)
    
    Tables and figures reported in the paper.

- **[`/src`](./src/)**
  
  Contains the Python scripts for each stage of the study.
  
  - [`/src/dataset_builder`](./src/dataset_builder/)
    
    Build the dataset from Teamscale findings and bug labels.
  
  - [`/src/check_ranking`](./src/check_ranking/)
    
    Compute check rankings, including PFI, correlation-based, and LLM-based methods.
  
  - [`/src/ranking_eval`](./src/ranking_eval/)
    
    Evaluate the most reliable ranking method.
  
  - [`/src/reporting`](./src/reporting/)
    
    Generate the tables and figures used in the paper.
  
  - [`/src/generalization`](./src/generalization/)
    
    Scripts for the PMD and Semgrep generalization experiments.

## 2. Requirements

The code can be run in a standard Python environment. In most cases, reproducing the released results does not require heavy setup or a GPU, and the scripts can be run on CPU. Some optional stages, such as the LLM-based ranking scripts, additionally require API access.

- Install the common Python dependencies with `pip install -r requirements.txt`.
- Some submodules have additional dependencies. Install the local `requirements.txt` in the corresponding folder when needed, for example:
  - `src/dataset_builder/requirements.txt`
  - `src/check_ranking/llm_methods/requirements.txt`
  - `src/reporting/RQ1/requirements.txt`, `RQ2/requirements.txt`, `RQ3/requirements.txt`
  - `src/generalization/semgrep/requirements.txt`
- Update the `config.json` files before running scripts, especially the local paths for datasets, output folders, repositories, and external tools.

Additional resources are only needed for some stages:

- To rebuild the dataset from scratch, install and configure **Teamscale**, and download the bug annotation data from [`BugDet/Dataset`](https://github.com/Naplues/BugDet/tree/master/Dataset), and prepare the source code of the corresponding ASF java projects locally.
- To rebuild the PMD or Semgrep datasets, install the required versions of PMD or Semgrep, respectively. For Semgrep, also clone the `semgrep-rules` repository from GitHub.
- To run the LLM-based ranking scripts, provide the required API keys in `src/check_ranking/llm_methods/`.

## 3. Running the Code

In most folders, you can run the provided `run.sh` script directly. If no shell script is provided, execute the Python file manually.

Typical workflow:

1. Build the dataset with scripts in [`src/dataset_builder`](./src/dataset_builder/) if you want to reproduce the data collection process.
2. Compute rankings with scripts in [`src/check_ranking`](./src/check_ranking/).
3. Evaluate the ranking methods with scripts in [`src/ranking_eval`](./src/ranking_eval/).
4. Regenerate paper results with scripts in [`src/reporting`](./src/reporting/).
