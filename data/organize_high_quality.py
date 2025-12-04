#!/usr/bin/env python3
"""
Organize high-quality output into class folders using telugu_labels.txt.
Images are already at 32x32, just organize them by class.
"""
import os
import sys
import shutil
from PIL import Image
import glob

# Config
OCROPUS_DIR = "dataset_high_quality"
LABELS_FILE = "telugu_labels.txt"
OUTPUT_DIR = "dataset_organized"
IMG_SIZE = (32, 32)

def load_labels(labels_path):
    mapping = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            idx, ch = parts
            mapping[ch] = idx.zfill(3)
    return mapping

def main():
    print("=" * 70)
    print("ORGANIZING HIGH-QUALITY DATASET")
    print("=" * 70)
    print()
    
    if not os.path.isdir(OCROPUS_DIR):
        print(f"Error: {OCROPUS_DIR} not found!")
        sys.exit(1)
    
    labels = load_labels(LABELS_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all .bin.png files
    png_files = sorted(glob.glob(os.path.join(OCROPUS_DIR, "*.bin.png")))
    
    if not png_files:
        print(f"No .bin.png files found in {OCROPUS_DIR}")
        sys.exit(1)
    
    print(f"Found {len(png_files)} images to organize")
    print()
    
    count_per_class = {}
    processed = 0
    
    for png in png_files:
        base = os.path.splitext(os.path.basename(png))[0]  # e.g., 000001.bin
        gt_file = os.path.join(OCROPUS_DIR, base + ".gt.txt")
        
        if not os.path.isfile(gt_file):
            continue
        
        with open(gt_file, "r", encoding="utf-8") as g:
            text = g.read().strip()
        
        if text not in labels:
            print(f"Warning: character '{text}' not found in labels. Skipping {png}")
            continue
        
        class_id = labels[text]
        class_dir = os.path.join(OUTPUT_DIR, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy file (already correct size)
        dest_name = f"{base}.png"
        dest_path = os.path.join(class_dir, dest_name)
        
        try:
            # Verify size and copy
            img = Image.open(png)
            if img.size != IMG_SIZE:
                print(f"Warning: {png} is {img.size}, expected {IMG_SIZE}")
            shutil.copy2(png, dest_path)
            
            count_per_class.setdefault(class_id, 0)
            count_per_class[class_id] += 1
            processed += 1
            
            if processed % 30 == 0:
                print(f"  Processed: {processed}/{len(png_files)}")
                
        except Exception as e:
            print(f"Failed processing {png}: {e}")
    
    print()
    print("=" * 70)
    print("ORGANIZATION COMPLETE!")
    print("=" * 70)
    print(f"Total images organized: {processed}")
    print()
    print("Class distribution:")
    for cid in sorted(count_per_class.keys()):
        char = [k for k, v in labels.items() if v == cid][0]
        print(f"  class_{cid} ({char}): {count_per_class[cid]} images")
    print()
    print(f"Output directory: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
