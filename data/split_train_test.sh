#!/bin/bash
# Usage: bash split_train_test.sh
set -e
python3 split_train_test.py dataset_by_class final_dataset 0.75
