#!/usr/bin/env python3
"""
Generate synthetic Telugu character images using Pothana2000 font.
This script creates images for each character in telugu_lines.txt with variations.
"""

import os
import sys
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# Configuration
FONT_PATH = "Pothana2000.ttf"
INPUT_FILE = "telugu_lines.txt"
OUTPUT_DIR = "dataset_pothana"
IMAGE_SIZE = (128, 128)  # Generate at higher resolution, will resize later
FONT_SIZES = [60, 65, 70, 75, 80]  # Different font sizes for variation

def add_noise_and_variations(img, variation_level=0.1):
    """Add small variations to make images more realistic"""
    img_array = np.array(img, dtype=np.float32)
    
    # Add slight Gaussian noise
    noise = np.random.normal(0, variation_level * 255, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_array, mode='L')
    
    # Randomly apply slight blur
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
    
    return img

def generate_character_image(character, output_path, font_size, variation_idx):
    """Generate a single character image with variations"""
    try:
        # Create white background
        img = Image.new('L', IMAGE_SIZE, color=255)
        draw = ImageDraw.Draw(img)
        
        # Load font
        font = ImageFont.truetype(FONT_PATH, font_size)
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), character, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text with slight random offset for variation
        offset_x = random.randint(-3, 3)
        offset_y = random.randint(-3, 3)
        x = (IMAGE_SIZE[0] - text_width) // 2 + offset_x
        y = (IMAGE_SIZE[1] - text_height) // 2 + offset_y
        
        # Draw the character in black
        draw.text((x, y), character, font=font, fill=0)
        
        # Add variations
        img = add_noise_and_variations(img, variation_level=0.05)
        
        # Crop to remove excess white space
        img_array = np.array(img)
        rows = np.any(img_array < 250, axis=1)
        cols = np.any(img_array < 250, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Add padding
            padding = 10
            rmin = max(0, rmin - padding)
            rmax = min(IMAGE_SIZE[1] - 1, rmax + padding)
            cmin = max(0, cmin - padding)
            cmax = min(IMAGE_SIZE[0] - 1, cmax + padding)
            
            img = img.crop((cmin, rmin, cmax + 1, rmax + 1))
        
        # Save the image
        img.save(output_path)
        
        # Also create a ground truth file (for compatibility with OCROPUS format)
        gt_path = output_path.replace('.png', '.gt.txt')
        with open(gt_path, 'w', encoding='utf-8') as f:
            f.write(character)
        
        return True
        
    except Exception as e:
        print(f"Error generating image for '{character}': {e}")
        return False

def main():
    print("=" * 60)
    print("Telugu Character Image Generation")
    print("=" * 60)
    
    # Check if font exists
    if not os.path.exists(FONT_PATH):
        print(f"ERROR: Font file '{FONT_PATH}' not found!")
        sys.exit(1)
    
    print(f"✓ Font file found: {FONT_PATH}")
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file '{INPUT_FILE}' not found!")
        sys.exit(1)
    
    print(f"✓ Input file found: {INPUT_FILE}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}/")
    print()
    
    # Read all lines
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Total lines to process: {len(lines)}")
    print("Starting image generation...")
    print()
    
    # Generate images
    success_count = 0
    character_counts = {}
    
    for idx, character in enumerate(lines, start=1):
        # Track how many images per character
        character_counts[character] = character_counts.get(character, 0) + 1
        char_idx = character_counts[character]
        
        # Choose font size (rotate through different sizes)
        font_size = FONT_SIZES[char_idx % len(FONT_SIZES)]
        
        # Generate output filename (6-digit index)
        output_filename = f"{str(idx).zfill(6)}.bin.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Generate image
        if generate_character_image(character, output_path, font_size, char_idx):
            success_count += 1
            if idx % 30 == 0 or idx == len(lines):
                print(f"  Progress: {idx}/{len(lines)} images generated ({success_count} successful)")
    
    print()
    print("=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Total images generated: {success_count}/{len(lines)}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print()
    print("Character breakdown:")
    for char in sorted(character_counts.keys()):
        print(f"  '{char}': {character_counts[char]} images")
    print()
    print("Next step: Run postprocess_organize_and_resize.py to organize by class")

if __name__ == "__main__":
    main()
