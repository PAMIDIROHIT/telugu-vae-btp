"""
Data generation script for rendering Telugu glyphs with augmentations.

This script:
1. Loads Telugu fonts (TTF files)
2. Renders glyphs for each font and size
3. Applies augmentations (rotation, blur, noise, etc.)
4. Saves images and metadata CSV
"""

import os
import argparse
import csv
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
from tqdm import tqdm


# Telugu characters (common subset)
TELUGU_CHARS = [
    'ఀ', 'ఁ', 'ం', 'ః', 'ః', 'అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ',
    'ఊ', 'ఋ', 'ఌ', '఍', 'ఎ', 'ఏ', 'ఐ', '఑', 'ఒ', 'ఓ',
    'ఔ', 'క', 'ఖ', 'గ', 'ఘ', 'ఙ', 'చ', 'ఛ', 'జ', 'ఝ',
    'ఞ', 'ట', 'ఠ', 'డ', 'ఢ', 'ణ', 'త', 'థ', 'ద', 'ధ',
    'న', 'ప', 'ఫ', 'బ', 'భ', 'మ', 'య', 'ర', 'ల', 'ళ',
    'ఱ', 'వ', 'శ', 'ష', 'స', 'హ', 'ఽ', 'ా', 'ి', 'ీ',
    'ు', 'ూ', 'ృ', 'ౄ', 'ೃ', 'ೄ', 'ె', 'ే', 'ై', 'ొ',
    'ో', 'ౌ',
]


def render_glyph(glyph, font, img_size=64):
    """
    Render a single glyph using PIL.
    
    Args:
        glyph: Character to render
        font: PIL ImageFont object
        img_size: Output image size
    
    Returns:
        PIL Image object
    """
    # Create white background
    img = Image.new('L', (img_size, img_size), color=255)
    draw = ImageDraw.Draw(img)
    
    try:
        # Render glyph (try with RAQM layout engine if available)
        draw.text((5, 5), glyph, font=font, fill=0)
    except:
        # Fallback if RAQM not available
        draw.text((5, 5), glyph, font=font, fill=0)
    
    return img


def apply_augmentations(img, num_augmentations=5, seed=None):
    """
    Apply random augmentations to glyph image.
    
    Augmentations:
    - Rotation (±5°)
    - Gaussian blur
    - Gaussian noise
    - Scaling/translation
    - Antialiasing
    
    Args:
        img: PIL Image
        num_augmentations: Number of augmented variants to create
        seed: Random seed for reproducibility
    
    Returns:
        List of (augmented_image, augmentation_dict) tuples
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    augmented_images = [(img, {'rotation': 0, 'blur_sigma': 0, 'noise_sigma': 0})]
    
    for _ in range(num_augmentations - 1):
        aug_img = img.copy()
        aug_dict = {}
        
        # Random rotation (±5°)
        rotation = random.uniform(-5, 5)
        aug_img = aug_img.rotate(rotation, expand=False, resample=Image.BICUBIC)
        aug_dict['rotation'] = round(rotation, 2)
        
        # Random blur
        blur_sigma = random.uniform(0, 1.5)
        if blur_sigma > 0:
            aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        aug_dict['blur_sigma'] = round(blur_sigma, 2)
        
        # Random noise
        aug_array = np.array(aug_img, dtype=np.float32)
        noise_sigma = random.uniform(0, 10)
        noise = np.random.normal(0, noise_sigma, aug_array.shape)
        aug_array = np.clip(aug_array + noise, 0, 255)
        aug_dict['noise_sigma'] = round(noise_sigma, 2)
        
        aug_img = Image.fromarray(aug_array.astype(np.uint8), mode='L')
        
        # Random scaling/translation
        scale = random.uniform(0.95, 1.05)
        tx = random.randint(-2, 2)
        ty = random.randint(-2, 2)
        aug_dict['scale'] = round(scale, 2)
        aug_dict['tx'] = tx
        aug_dict['ty'] = ty
        
        augmented_images.append((aug_img, aug_dict))
    
    return augmented_images


def generate_synthetic_glyph(char_idx, img_size=64, seed=42):
    """
    Generate synthetic glyph when font files are unavailable.
    Uses geometric patterns as proxy for character images.
    """
    random.seed(seed + char_idx)
    np.random.seed(seed + char_idx)
    
    # Create blank image
    img_array = np.ones((img_size, img_size), dtype=np.uint8) * 255
    
    # Draw random patterns (lines, circles, blobs) to simulate glyphs
    overlay = cv2.cvtColor(img_array.reshape(img_size, img_size, 1), cv2.COLOR_GRAY2BGR)
    
    # Random number of strokes
    num_strokes = random.randint(2, 6)
    for _ in range(num_strokes):
        if random.random() > 0.5:
            # Line
            pt1 = (random.randint(5, img_size-5), random.randint(5, img_size-5))
            pt2 = (random.randint(5, img_size-5), random.randint(5, img_size-5))
            cv2.line(overlay, pt1, pt2, (0, 0, 0), random.randint(1, 3))
        else:
            # Circle
            center = (random.randint(5, img_size-5), random.randint(5, img_size-5))
            radius = random.randint(3, 15)
            cv2.circle(overlay, center, radius, (0, 0, 0), random.randint(1, 2))
    
    # Convert back to grayscale
    result = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(result)


def render_dataset(
    font_dir,
    output_dir,
    font_sizes=[12, 14, 16, 18, 20],
    num_augmentations=30,
    seed=42,
    img_size=64
):
    """
    Render full Telugu glyph dataset.
    
    Args:
        font_dir: Directory containing TTF font files
        output_dir: Where to save rendered images
        font_sizes: List of font sizes to render
        num_augmentations: Augmentations per glyph
        seed: Random seed
        img_size: Output image size
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Find font files
    font_files = list(Path(font_dir).glob('*.ttf'))
    
    # If no fonts available, use synthetic generation
    use_synthetic = len(font_files) == 0
    if use_synthetic:
        print(f"⚠️  No TTF files found in {font_dir}. Using SYNTHETIC glyph generation for training.")
        font_names = ['synthetic_default']
    else:
        font_names = [f.stem for f in font_files]
    
    print(f"Found {len(font_files)} font files")
    print(f"Rendering {len(TELUGU_CHARS)} glyphs")
    print(f"Font sizes: {font_sizes}")
    print(f"Augmentations per glyph: {num_augmentations}")
    
    metadata = []
    
    if use_synthetic:
        # Synthetic data generation mode
        print("\n📊 SYNTHETIC MODE: Generating training dataset")
        font_name = 'synthetic_default'
        for font_size in font_sizes:
            output_subdir = os.path.join(output_dir, font_name, f"{font_size}pt")
            os.makedirs(output_subdir, exist_ok=True)
            
            pbar = tqdm(enumerate(TELUGU_CHARS), total=len(TELUGU_CHARS), 
                       desc=f"Synthetic {font_size}pt")
            
            for glyph_idx, glyph in pbar:
                try:
                    # Generate synthetic glyph
                    base_img = generate_synthetic_glyph(glyph_idx, img_size=img_size, seed=seed)
                    
                    # Generate augmentations
                    augmented = apply_augmentations(
                        base_img, 
                        num_augmentations=num_augmentations,
                        seed=seed + glyph_idx
                    )
                    
                    for aug_idx, (aug_img, aug_dict) in enumerate(augmented):
                        # Save image
                        filename = f"{font_name}_{ord(glyph):05d}_{font_size}pt_aug{aug_idx:02d}.png"
                        filepath = os.path.join(output_subdir, filename)
                        aug_img.save(filepath)
                        
                        # Record metadata
                        metadata.append({
                            'filename': filepath,
                            'font': font_name,
                            'glyph': glyph,
                            'glyph_code': ord(glyph),
                            'fontsize': font_size,
                            'augmentation_index': aug_idx,
                            'synthetic': True,
                            **aug_dict
                        })
                except Exception as e:
                    print(f"Error generating synthetic glyph {glyph}: {e}")
                    continue
    else:
        # Real font-based generation
        for font_file in font_files:
            font_name = font_file.stem
            print(f"\nProcessing font: {font_name}")
            
            for font_size in font_sizes:
                # Create output directory
                output_subdir = os.path.join(output_dir, font_name, f"{font_size}pt")
                os.makedirs(output_subdir, exist_ok=True)
                
                # Load font
                try:
                    font = ImageFont.truetype(str(font_file), font_size)
                except Exception as e:
                    print(f"Error loading font {font_file}: {e}")
                    continue
                
                # Render glyphs
                pbar = tqdm(enumerate(TELUGU_CHARS), total=len(TELUGU_CHARS), 
                           desc=f"{font_name} {font_size}pt")
                
                for glyph_idx, glyph in pbar:
                    try:
                        # Render base glyph
                        base_img = render_glyph(glyph, font, img_size)
                        
                        # Generate augmentations
                        augmented = apply_augmentations(
                            base_img, 
                            num_augmentations=num_augmentations,
                            seed=seed + glyph_idx
                        )
                        
                        for aug_idx, (aug_img, aug_dict) in enumerate(augmented):
                            # Save image
                            filename = f"{font_name}_{ord(glyph):05d}_{font_size}pt_aug{aug_idx:02d}.png"
                            filepath = os.path.join(output_subdir, filename)
                            aug_img.save(filepath)
                            
                            # Record metadata
                            metadata.append({
                                'filename': filepath,
                                'font': font_name,
                                'glyph': glyph,
                                'glyph_code': ord(glyph),
                                'fontsize': font_size,
                                'augmentation_index': aug_idx,
                                'synthetic': False,
                                **aug_dict
                            })
                    
                    except Exception as e:
                        print(f"Error rendering glyph {glyph}: {e}")
                        continue
    
    # Save metadata CSV
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    print(f"\nSaving metadata to {metadata_path}")
    
    if metadata:
        # Collect all possible keys from metadata records
        all_keys = set()
        for record in metadata:
            all_keys.update(record.keys())
        keys = sorted(list(all_keys))  # Sort for consistency
        
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, restval='')
            writer.writeheader()
            writer.writerows(metadata)
        
        print(f"Generated {len(metadata)} glyph images")
    else:
        print("No images generated!")


def main():
    parser = argparse.ArgumentParser(description="Render Telugu glyph dataset")
    parser.add_argument('--font-dir', type=str, default='data/fonts',
                       help='Directory containing TTF font files')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                       help='Output directory for rendered images')
    parser.add_argument('--font-sizes', type=int, nargs='+', 
                       default=[12, 14, 16, 18, 20],
                       help='Font sizes to render')
    parser.add_argument('--num-augmentations', type=int, default=30,
                       help='Number of augmentations per glyph')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--img-size', type=int, default=64,
                       help='Output image size')
    
    args = parser.parse_args()
    
    render_dataset(
        font_dir=args.font_dir,
        output_dir=args.output_dir,
        font_sizes=args.font_sizes,
        num_augmentations=args.num_augmentations,
        seed=args.seed,
        img_size=args.img_size
    )


if __name__ == '__main__':
    main()
