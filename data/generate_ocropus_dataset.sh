#!/bin/bash
# Usage: bash generate_ocropus_dataset.sh
set -e
mkdir -p dataset_pothana
# Generate images with ocropus-linegen using Pothana2000 font
ocropus-linegen -o dataset_pothana -f ~/.local/share/fonts/Pothana2000.ttf telugu_lines.txt
echo "OCROPUS generation complete. Output in dataset_pothana/"
