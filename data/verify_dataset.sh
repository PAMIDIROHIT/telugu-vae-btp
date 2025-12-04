#!/bin/bash
# Prints counts per class for dataset_by_class and for final_dataset/train & test
set -e
echo "Counts in dataset_by_class:"
for d in dataset_by_class/class_*; do
  if [ -d "$d" ]; then
    echo "$(basename $d): $(ls -1 "$d" | wc -l)"
  fi
done

echo ""
echo "Counts in final_dataset/train:"
for d in final_dataset/train/class_*; do
  if [ -d "$d" ]; then
    echo "$(basename $d): $(ls -1 "$d" | wc -l)"
  fi
done

echo ""
echo "Counts in final_dataset/test:"
for d in final_dataset/test/class_*; do
  if [ -d "$d" ]; then
    echo "$(basename $d): $(ls -1 "$d" | wc -l)"
  fi
done
