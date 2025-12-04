#!/usr/bin/env python3
"""
Comprehensive dataset verification script.
Checks image count, sizes, and generates a detailed report.
"""

import os
import sys
from PIL import Image
from collections import defaultdict

# Configuration
DATASET_DIR = "dataset_by_class"
LABELS_FILE = "telugu_labels.txt"
EXPECTED_SIZE = (32, 32)
EXPECTED_MODE = "L"

def load_character_mapping():
    """Load class ID to character mapping"""
    mapping = {}
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            idx, char = parts
            mapping[idx.zfill(3)] = char
    return mapping

def verify_dataset():
    """Verify the entire dataset"""
    print("=" * 70)
    print("TELUGU CHARACTER DATASET VERIFICATION REPORT")
    print("=" * 70)
    print()
    
    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: Dataset directory '{DATASET_DIR}' not found!")
        return False
    
    # Load character mapping
    char_mapping = load_character_mapping()
    
    # Collect statistics
    class_dirs = sorted([d for d in os.listdir(DATASET_DIR) 
                        if os.path.isdir(os.path.join(DATASET_DIR, d)) 
                        and d.startswith('class_')])
    
    total_images = 0
    issues = []
    class_stats = []
    
    for class_dir in class_dirs:
        class_id = class_dir.replace('class_', '')
        class_path = os.path.join(DATASET_DIR, class_dir)
        character = char_mapping.get(class_id, 'Unknown')
        
        # Count images
        images = [f for f in os.listdir(class_path) if f.endswith('.png')]
        num_images = len(images)
        total_images += num_images
        
        # Check a sample image
        size_ok = True
        mode_ok = True
        if images:
            sample_path = os.path.join(class_path, images[0])
            try:
                img = Image.open(sample_path)
                if img.size != EXPECTED_SIZE:
                    size_ok = False
                    issues.append(f"{class_dir}: Incorrect size {img.size} (expected {EXPECTED_SIZE})")
                if img.mode != EXPECTED_MODE:
                    mode_ok = False
                    issues.append(f"{class_dir}: Incorrect mode {img.mode} (expected {EXPECTED_MODE})")
            except Exception as e:
                issues.append(f"{class_dir}: Error reading sample image: {e}")
        
        class_stats.append({
            'class_id': class_id,
            'character': character,
            'num_images': num_images,
            'size_ok': size_ok,
            'mode_ok': mode_ok
        })
    
    # Print summary
    print("DATASET SUMMARY")
    print("-" * 70)
    print(f"Total Classes:      {len(class_dirs)}")
    print(f"Total Images:       {total_images}")
    print(f"Expected Size:      {EXPECTED_SIZE}")
    print(f"Expected Mode:      {EXPECTED_MODE} (Grayscale)")
    print()
    
    # Print class breakdown
    print("CLASS BREAKDOWN")
    print("-" * 70)
    print(f"{'Class ID':<12} {'Character':<12} {'Images':<10} {'Status'}")
    print("-" * 70)
    
    for stat in class_stats:
        status = "✓ OK" if stat['size_ok'] and stat['mode_ok'] else "✗ ISSUES"
        print(f"class_{stat['class_id']:<6} {stat['character']:<12} {stat['num_images']:<10} {status}")
    
    print()
    
    # Print issues if any
    if issues:
        print("ISSUES FOUND")
        print("-" * 70)
        for issue in issues:
            print(f"  ✗ {issue}")
        print()
    
    # Overall status
    print("=" * 70)
    if not issues and total_images > 0:
        print("✓ VERIFICATION PASSED - Dataset is ready for training!")
        print("=" * 70)
        print()
        print("NEXT STEPS:")
        print("  1. Split dataset into train/test sets: ./split_train_test.sh")
        print("  2. Train VAE models: python3 ../train_all.py")
        print()
        return True
    else:
        print("✗ VERIFICATION FAILED - Please fix the issues above")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = verify_dataset()
    sys.exit(0 if success else 1)
