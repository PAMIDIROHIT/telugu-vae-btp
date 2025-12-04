#!/usr/bin/env python3
"""
High-Quality Telugu Character Image Generator
Fixed font size at 12pt for consistency
Enhanced quality with multiple variations
"""

import os
import sys
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# Configuration
FONT_PATH = "Pothana2000.ttf"
INPUT_FILE = "telugu_lines.txt"
OUTPUT_DIR = "dataset_high_quality"
FONT_SIZE = 72  # Large size for rendering, will scale down
TARGET_SIZE = (64, 64)  # Higher resolution for better quality
FINAL_SIZE = (32, 32)  # Final size after processing

# Variation parameters
ROTATION_RANGE = (-5, 5)  # degrees
SCALE_RANGE = (0.95, 1.05)  # scaling factor
NOISE_LEVEL = 0.03  # reduced noise for clarity
BRIGHTNESS_RANGE = (0.95, 1.05)
CONTRAST_RANGE = (0.95, 1.05)

def add_realistic_variations(img, variation_level='medium'):
    """Add realistic variations while maintaining quality"""
    img_array = np.array(img, dtype=np.float32)
    
    # Add subtle Gaussian noise
    if variation_level in ['medium', 'high']:
        noise = np.random.normal(0, NOISE_LEVEL * 255, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_array, mode='L')
    
    # Random brightness adjustment
    if random.random() > 0.3:
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(*BRIGHTNESS_RANGE)
        img = enhancer.enhance(factor)
    
    # Random contrast adjustment
    if random.random() > 0.3:
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(*CONTRAST_RANGE)
        img = enhancer.enhance(factor)
    
    # Slight blur (occasionally)
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.3)))
    
    return img

def generate_character_image(character, output_path, sample_idx, total_samples):
    """Generate a high-quality character image with variations"""
    try:
        # Create high-resolution white background
        img = Image.new('L', TARGET_SIZE, color=255)
        draw = ImageDraw.Draw(img)
        
        # Load font at fixed size 72 (for high quality rendering)
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), character, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Apply random scaling
        scale_factor = random.uniform(*SCALE_RANGE)
        
        # Calculate position to center text with slight random offset
        offset_x = random.randint(-2, 2)
        offset_y = random.randint(-2, 2)
        x = (TARGET_SIZE[0] - text_width) // 2 + offset_x
        y = (TARGET_SIZE[1] - text_height) // 2 + offset_y
        
        # Draw the character in black
        draw.text((x, y), character, font=font, fill=0)
        
        # Apply random rotation
        if random.random() > 0.5:
            angle = random.uniform(*ROTATION_RANGE)
            img = img.rotate(angle, fillcolor=255, resample=Image.BICUBIC)
        
        # Add variations based on sample index
        if sample_idx % 4 == 0:
            variation_level = 'low'
        elif sample_idx % 4 == 1:
            variation_level = 'medium'
        elif sample_idx % 4 == 2:
            variation_level = 'high'
        else:
            variation_level = 'medium'
        
        img = add_realistic_variations(img, variation_level)
        
        # Crop to remove excess whitespace
        img_array = np.array(img)
        rows = np.any(img_array < 250, axis=1)
        cols = np.any(img_array < 250, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Add padding
            padding = 8
            rmin = max(0, rmin - padding)
            rmax = min(TARGET_SIZE[1] - 1, rmax + padding)
            cmin = max(0, cmin - padding)
            cmax = min(TARGET_SIZE[0] - 1, cmax + padding)
            
            img = img.crop((cmin, rmin, cmax + 1, rmax + 1))
        
        # Resize to final size with high-quality resampling
        img = img.resize(FINAL_SIZE, Image.Resampling.LANCZOS)
        
        # Enhance sharpness slightly
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        
        # Save the image
        img.save(output_path, quality=95, optimize=True)
        
        # Create ground truth file
        gt_path = output_path.replace('.png', '.gt.txt')
        with open(gt_path, 'w', encoding='utf-8') as f:
            f.write(character)
        
        return True
        
    except Exception as e:
        print(f"Error generating image for '{character}': {e}")
        return False

def main():
    print("=" * 70)
    print("HIGH-QUALITY TELUGU CHARACTER IMAGE GENERATION")
    print("Fixed Font Size: 72pt (rendered)")
    print("Target Resolution: 64x64 → Final: 32x32")
    print("=" * 70)
    print()
    
    # Check font
    if not os.path.exists(FONT_PATH):
        print(f"ERROR: Font file '{FONT_PATH}' not found!")
        sys.exit(1)
    
    print(f"✓ Font: {FONT_PATH}")
    
    # Check input
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file '{INPUT_FILE}' not found!")
        sys.exit(1)
    
    print(f"✓ Input: {INPUT_FILE}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output: {OUTPUT_DIR}/")
    print()
    
    # Read lines
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(lines)
    print(f"Processing {total_lines} characters...")
    print()
    
    # Track statistics
    success_count = 0
    character_counts = {}
    
    # Generate images
    for idx, character in enumerate(lines, start=1):
        character_counts[character] = character_counts.get(character, 0) + 1
        char_idx = character_counts[character]
        
        # Output filename
        output_filename = f"{str(idx).zfill(6)}.bin.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Generate image
        if generate_character_image(character, output_path, char_idx, len(lines)):
            success_count += 1
            
            # Progress update
            if idx % 30 == 0 or idx == total_lines:
                progress = (idx / total_lines) * 100
                print(f"  [{progress:6.2f}%] {idx}/{total_lines} images generated")
    
    print()
    print("=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print(f"✓ Total images: {success_count}/{total_lines}")
    print(f"✓ Output directory: {OUTPUT_DIR}/")
    print()
    print("Character distribution:")
    for char in sorted(character_counts.keys()):
        print(f"  '{char}': {character_counts[char]} images")
    print()
    print(f"Quality: High-resolution rendering with 64x64 → 32x32 downsampling")
    print(f"Variations: Rotation, scaling, brightness, contrast, noise")
    print()
    print("Next: Run postprocess_organize_and_resize.py")

if __name__ == "__main__":
    main()
