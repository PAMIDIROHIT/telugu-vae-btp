#!/usr/bin/env python3
"""
Split dataset_by_class into train/test with given train fraction.
Usage: python3 split_train_test.py <input_dir> <output_dir> <train_fraction>
"""
import os
import sys
import shutil
import random

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def split_class(class_dir, out_dir_train, out_dir_test, train_frac):
    files = [f for f in os.listdir(class_dir) if f.lower().endswith(".png")]
    files.sort()
    random.shuffle(files)
    n = len(files)
    n_train = int(round(n * train_frac))
    train_files = files[:n_train]
    test_files = files[n_train:]
    for f in train_files:
        shutil.copy2(os.path.join(class_dir, f), os.path.join(out_dir_train, f))
    for f in test_files:
        shutil.copy2(os.path.join(class_dir, f), os.path.join(out_dir_test, f))
    return len(train_files), len(test_files)

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 split_train_test.py <input_dir> <output_dir> <train_fraction>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    train_frac = float(sys.argv[3])
    train_root = os.path.join(output_dir, "train")
    test_root = os.path.join(output_dir, "test")
    ensure_dir(train_root)
    ensure_dir(test_root)

    classes = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    total_train = total_test = 0
    for c in classes:
        class_in = os.path.join(input_dir, c)
        train_out = os.path.join(train_root, c)
        test_out = os.path.join(test_root, c)
        ensure_dir(train_out); ensure_dir(test_out)
        tcnt, scnt = split_class(class_in, train_out, test_out, train_frac)
        print(f"{c}: train={tcnt}, test={scnt}")
        total_train += tcnt; total_test += scnt
    print("Total -> train:", total_train, "test:", total_test)

if __name__ == "__main__":
    main()
