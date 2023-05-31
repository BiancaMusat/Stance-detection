#!/usr/bin/env bash

echo "Train and test model"
python train_model.py -train 'datasets/cstance_train.csv' -test 'datasets/cstance_test.csv' -val 'datasets/cstance_val.csv'