#!/usr/bin/env python3
"""
Organize ocropus output into class folders using telugu_labels.txt,
resize all images to 32x32 grayscale, and save them as PNG.
Produces dataset_by_class/class_XX/*.png
"""
import os
import sys
import shutil
from PIL import Image
import glob

# Config
OCROPUS_DIR = "dataset_pothana"
LABELS_FILE = "telugu_labels.txt"
OUTPUT_DIR = "dataset_by_class"
IMG_SIZE = (32, 32)

def load_labels(labels_path):
    mapping = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts = line.split(maxsplit=1)
            if len(parts)<2: continue
            idx, ch = parts
            mapping[ch] = idx.zfill(3)
    return mapping

def main():
    if not os.path.isdir(OCROPUS_DIR):
        print(f"Error: {OCROPUS_DIR} not found. Run generate_ocropus_dataset.sh first.")
        sys.exit(1)
    labels = load_labels(LABELS_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # find all .bin.png files
    png_files = sorted(glob.glob(os.path.join(OCROPUS_DIR, "*.bin.png")))
    if not png_files:
        print("No .bin.png files found in", OCROPUS_DIR)
        sys.exit(1)

    count_per_class = {}
    for png in png_files:
        base = os.path.splitext(os.path.basename(png))[0]  # e.g., 000001.bin
        gt_file = os.path.join(OCROPUS_DIR, base + ".gt.txt")
        if not os.path.isfile(gt_file):
            # sometimes ocropus outputs files differently; try alternate naming
            # fallback: skip
            continue
        with open(gt_file, "r", encoding="utf-8") as g:
            text = g.read().strip()
        # Expecting a single character (or compound) that maps to labels
        if text not in labels:
            print(f"Warning: character '{text}' not found in labels mapping. Skipping {png}")
            continue
        class_id = labels[text]
        class_dir = os.path.join(OUTPUT_DIR, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)
        # Destination filename: use original index to preserve uniqueness
        dest_name = f"{base}.png"
        dest_path = os.path.join(class_dir, dest_name)
        try:
            im = Image.open(png).convert("L")  # grayscale
            im = im.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            im.save(dest_path)
            count_per_class.setdefault(class_id, 0)
            count_per_class[class_id] += 1
        except Exception as e:
            print(f"Failed processing {png}: {e}")

    print("Finished organizing and resizing.")
    for cid in sorted(count_per_class.keys()):
        print(f"class_{cid}: {count_per_class[cid]} images")

if __name__ == "__main__":
    main()
