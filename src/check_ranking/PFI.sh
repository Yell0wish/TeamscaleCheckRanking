#!/bin/bash

python PFI.py --model XGBoost
python PFI.py --model LightGBM
python PFI.py --model BalancedBagging
python PFI.py --model BalancedRandomForest
python PFI.py --model RandomForest
python PFI.py --model AdaBoost
python PFI.py --model ExtraTrees
python PFI.py --model Bagging
python PFI.py --model GradientBoost
python PFI.py --model BernoulliNB
python PFI.py --model LogisticRegression
