#!/bin/bash

python correlation.py --method kendalltau
python correlation.py --method spearmanr
python correlation.py --method pearsonr