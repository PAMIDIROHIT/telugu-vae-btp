#!/usr/bin/env python3
"""
Split dataset_by_class into train/test sets for training
"""
import os
import shutil
import random

SOURCE_DIR = "dataset_by_class"
TRAIN_DIR = "train"
TEST_DIR = "test"
TRAIN_RATIO = 0.8

random.seed(42)

print("=" * 70)
print("CREATING TRAIN/TEST SPLITS FROM dataset_by_class")
print("=" * 70)
print()

# Create output directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

class_dirs = sorted([d for d in os.listdir(SOURCE_DIR)
                    if os.path.isdir(os.path.join(SOURCE_DIR, d))
                    and d.startswith('class_')])

print(f"Found {len(class_dirs)} classes")
print(f"Train ratio: {TRAIN_RATIO*100:.0f}% / Test ratio: {(1-TRAIN_RATIO)*100:.0f}%")
print()

total_train = 0
total_test = 0

for class_dir in class_dirs:
    class_path = os.path.join(SOURCE_DIR, class_dir)
    images = sorted([f for f in os.listdir(class_path) if f.endswith('.png')])
    
    random.shuffle(images)
    
    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    test_images = images[split_idx:]
    
    train_class_dir = os.path.join(TRAIN_DIR, class_dir)
    test_class_dir = os.path.join(TEST_DIR, class_dir)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)
    
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_class_dir, img)
        shutil.copy2(src, dst)
    
    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_class_dir, img)
        shutil.copy2(src, dst)
    
    total_train += len(train_images)
    total_test += len(test_images)
    
    print(f"  {class_dir}: {len(train_images)} train, {len(test_images)} test")

print()
print("=" * 70)
print(f"✓ Total train images: {total_train}")
print(f"✓ Total test images: {total_test}")
print(f"✓ Train directory: {TRAIN_DIR}/")
print(f"✓ Test directory: {TEST_DIR}/")
print("=" * 70)
