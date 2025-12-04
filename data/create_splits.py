#!/usr/bin/env python3
"""
Split high-quality dataset into train/test sets (80/20 split)
Maintains class balance
"""
import os
import shutil
import random
from pathlib import Path

# Configuration
SOURCE_DIR = "dataset_organized"
TRAIN_DIR = "dataset_final/train"
TEST_DIR = "dataset_final/test"
TRAIN_RATIO = 0.8

def split_dataset():
    print("=" * 70)
    print("SPLITTING DATASET INTO TRAIN/TEST")
    print("=" * 70)
    print()
    
    # Create output directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(SOURCE_DIR)
                        if os.path.isdir(os.path.join(SOURCE_DIR, d))
                        and d.startswith('class_')])
    
    print(f"Found {len(class_dirs)} classes")
    print(f"Train ratio: {TRAIN_RATIO*100:.0f}%")
    print(f"Test ratio: {(1-TRAIN_RATIO)*100:.0f}%")
    print()
    
    total_train = 0
    total_test = 0
    
    for class_dir in class_dirs:
        class_path = os.path.join(SOURCE_DIR, class_dir)
        
        # Get all images in this class
        images = sorted([f for f in os.listdir(class_path) if f.endswith('.png')])
        
        # Shuffle for random split
        random.seed(42)  # For reproducibility
        random.shuffle(images)
        
        # Split
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Create class directories in train/test
        train_class_dir = os.path.join(TRAIN_DIR, class_dir)
        test_class_dir = os.path.join(TEST_DIR, class_dir)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Copy training images
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst)
        
        # Copy test images
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_class_dir, img)
            shutil.copy2(src, dst)
        
        total_train += len(train_images)
        total_test += len(test_images)
        
        print(f"  {class_dir}: {len(train_images)} train, {len(test_images)} test")
    
    print()
    print("=" * 70)
    print("SPLIT COMPLETE!")
    print("=" * 70)
    print(f"Total train images: {total_train}")
    print(f"Total test images:  {total_test}")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Test directory:  {TEST_DIR}")
    print()

if __name__ == "__main__":
    split_dataset()
