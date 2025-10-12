# TeamscaleCheckRanking

## Paper Title  
**Line-level bug-finding power of static analysis rules: A case study of Teamscale**

---

## 1. Directory Structure

Below is a description of the key folders and their contents:

- **[`/data`](./data/)**
  
  Contains both the input data and the experimental outputs used in the paper.
  
  - [`/data/dataset`](./data/dataset/)
  
    Stores the datasets used in this study, covering 17 projects with 134 releases in total.
  
  - [`/data/reporting`](./data/reporting/)
  
    Stores the experimental results reported in the paper.

- **[`/src`](./src/)**

  Contains the source code written in Python.

  - [`/src/dataset_builder`](./src/dataset_builder/)

    Scripts for constructing datasets.

  - [`/src/check_ranking`](./src/check_ranking)

    Scripts for implementing check-ranking methods, including PFI and correlation analysis.

  - [`/src/ranking_eval`](./src/ranking_eval)

    Scripts for evaluating the effectiveness of ranking methods.

  - [`/src/reporting`](./src/reporting)

    Scripts for generating experimental results reported in the paper.

## 2. Running the Code

1. Update the paths in `config.json` to match your local environment.
2. If a shell script (`.sh`) is provided in the folder, run `script_name.sh` directly.
3. If no shell script is provided, run `python file_name.py` instead.

### Additional Requirement

To reproduce the dataset construction: 

- Download the **bug annotation data** from [`BugDet/dataset/`](https://github.com/Naplues/BugDet/tree/master/Dataset).

- Install and configure **Teamscale**.