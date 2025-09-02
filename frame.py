#!/usr/bin/env python3
"""
Frame.py - Add a 3.5% black border to images with optional watermark text.

Usage:
    python frame.py "<image_name>"
    python frame.py "<image_name>" --watermark "Your text here"
    python frame.py "<image_name>" --watermark "Text" --font "custom_font.ttf"

Note: Script will attempt to use VecTerminus12Medium.otf by default.
"""

import argparse
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os


def calculate_border_size(width, height, border_percentage=4):
    """
    Calculate the border size based on the image dimensions and border percentage.
    
    Args:
        width (int): Image width
        height (int): Image height
        border_percentage (float): Border percentage (default 4%)
    
    Returns:
        int: Border size in pixels
    """
    # Use the smaller dimension to calculate border size for consistency
    min_dimension = min(width, height)
    border_size = int(min_dimension * (border_percentage / 100))
    return max(border_size, 10)  # Minimum border of 10 pixels


def parse_aspect_ratio(aspect_str):
    """
    Parse aspect ratio string in format "x:y" or "x/y".
    
    Args:
        aspect_str (str): Aspect ratio string like "16:9" or "4:3"
    
    Returns:
        tuple: (width_ratio, height_ratio) as floats
    """
    try:
        if ':' in aspect_str:
            parts = aspect_str.split(':')
        elif '/' in aspect_str:
            parts = aspect_str.split('/')
        else:
            raise ValueError("Invalid aspect ratio format")
        
        if len(parts) != 2:
            raise ValueError("Aspect ratio must have exactly two parts")
        
        width_ratio = float(parts[0])
        height_ratio = float(parts[1])
        
        if width_ratio <= 0 or height_ratio <= 0:
            raise ValueError("Aspect ratio values must be positive")
        
        return width_ratio, height_ratio
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid aspect ratio '{aspect_str}'. Use format 'x:y' (e.g., '16:9')")


def calculate_aspect_ratio_borders(width, height, target_aspect_width, target_aspect_height, base_border):
    """
    Calculate additional borders needed to achieve target aspect ratio.
    
    Args:
        width (int): Current width (including base border)
        height (int): Current height (including base border)
        target_aspect_width (float): Target aspect ratio width component
        target_aspect_height (float): Target aspect ratio height component
        base_border (int): The base border already applied
    
    Returns:
        tuple: (additional_horizontal_border, additional_vertical_border)
    """
    current_ratio = width / height
    target_ratio = target_aspect_width / target_aspect_height
    
    if abs(current_ratio - target_ratio) < 0.001:  # Already very close
        return 0, 0
    
    if current_ratio < target_ratio:
        # Need to expand width
        target_width = height * target_ratio
        additional_width = target_width - width
        return int(additional_width / 2), 0
    else:
        # Need to expand height
        target_height = width / target_ratio
        additional_height = target_height - height
        return 0, int(additional_height / 2)


def get_font_for_border(border_size, font_path=None):
    """
    Get an appropriate font size that fits nicely in the black border.
    
    Args:
        border_size (int): Size of the border in pixels
        font_path (str, optional): Path to custom font file
    
    Returns:
        ImageFont: Font object
    """
    # Font size should be about 60-70% of border height for good fit
    font_size = int(border_size * 0.65)
    
    # List of fonts to try in order of preference
    font_paths_to_try = []
    
    # If custom font path is provided, try it first
    if font_path:
        font_paths_to_try.append(font_path)
    
    # Default to VecTerminus12Medium.otf in the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mvc_roman_path = os.path.join(script_dir, "VecTerminus12Medium.otf")
    font_paths_to_try.append(mvc_roman_path)
    
    # Fallback system fonts
    if sys.platform.startswith('win'):
        font_paths_to_try.append("C:/Windows/Fonts/arial.ttf")
    elif sys.platform.startswith('darwin'):  # macOS
        font_paths_to_try.append("/System/Library/Fonts/Helvetica.ttc")
    else:  # Linux and others
        font_paths_to_try.append("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    
    # Try each font path
    for font_path_attempt in font_paths_to_try:
        try:
            if os.path.exists(font_path_attempt):
                # Check if it's a WOFF2 file and provide helpful message
                if font_path_attempt.lower().endswith('.woff2'):
                    print(f"⚠️  Warning: WOFF2 fonts are not supported by PIL/Pillow. Skipping '{font_path_attempt}'")
                    print("   Please convert to TTF or OTF format, or use a different font.")
                    continue
                return ImageFont.truetype(font_path_attempt, font_size)
        except (IOError, OSError) as e:
            # Continue to next font if this one fails
            print(f"⚠️  Warning: Could not load font '{font_path_attempt}': {e}")
            continue
    
    # If all else fails, use default font
    return ImageFont.load_default()


def add_border_and_watermark(image_path, watermark_text=None, output_path=None, font_path=None, border_percentage=4, aspect_ratio=None, rotate_angle=None):
    """
    Add a black border to an image and optionally add watermark text with aspect ratio adjustment.
    
    Args:
        image_path (str): Path to the input image
        watermark_text (str, optional): Text to add as watermark
        output_path (str, optional): Path for output image (defaults to adding "_framed" suffix)
        font_path (str, optional): Path to custom font file
        border_percentage (float): Border percentage (default 4%)
        aspect_ratio (tuple, optional): Target aspect ratio as (width, height)
        rotate_angle (int, optional): Rotation angle in degrees (90, -90, 180)
    
    Returns:
        str: Path to the output image
    """
    try:
        # Open the original image
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            
            # Calculate border size based on the specified percentage
            border_size = calculate_border_size(original_width, original_height, border_percentage)
            
            # First, create image with base border only
            base_width = original_width + (2 * border_size)
            base_height = original_height + (2 * border_size)
            
            # Create new image with black background (base border only)
            framed_img = Image.new('RGB', (base_width, base_height), color='black')
            
            # Paste the original image centered with base border
            framed_img.paste(img, (border_size, border_size))
            
            # Add watermark text if provided (before aspect ratio adjustment)
            if watermark_text:
                draw = ImageDraw.Draw(framed_img)
                font = get_font_for_border(border_size, font_path)
                
                # Position text in top-left corner of the base border
                text_x = border_size + 25  # Small padding from the very edge
                text_y = (border_size - font.getbbox(watermark_text)[3]) // 2  # Center vertically in border
                
                # Ensure text doesn't go outside the border area
                if text_y < 5:
                    text_y = 5
                
                # Draw text in orange for visibility on black background
                draw.text((text_x, text_y), watermark_text, fill='orange', font=font)
            
            # Apply rotation if specified (after watermark is applied)
            if rotate_angle:
                if rotate_angle in [90, -90, 180, 270, -270]:
                    if rotate_angle == 90 or rotate_angle == -270:
                        framed_img = framed_img.transpose(Image.Transpose.ROTATE_90)
                    elif rotate_angle == -90 or rotate_angle == 270:
                        framed_img = framed_img.transpose(Image.Transpose.ROTATE_270)
                    elif rotate_angle == 180 or rotate_angle == -180:
                        framed_img = framed_img.transpose(Image.Transpose.ROTATE_180)
                    
                    # Update dimensions after rotation for 90/-90 degree rotations
                    if rotate_angle in [90, -90, 270, -270]:
                        base_width, base_height = base_height, base_width
                else:
                    print(f"⚠️  Warning: Unsupported rotation angle {rotate_angle}. Supported: 90, -90, 180")
            
            # Now apply aspect ratio adjustment if specified
            if aspect_ratio:
                target_width, target_height = aspect_ratio
                additional_h_border, additional_v_border = calculate_aspect_ratio_borders(
                    base_width, base_height, target_width, target_height, border_size
                )
                
                if additional_h_border > 0 or additional_v_border > 0:
                    # Calculate final dimensions
                    final_width = base_width + (2 * additional_h_border)
                    final_height = base_height + (2 * additional_v_border)
                    
                    # Create new image with final dimensions
                    final_img = Image.new('RGB', (final_width, final_height), color='black')
                    
                    # Paste the framed image (with watermark) centered in the final image
                    paste_x = additional_h_border
                    paste_y = additional_v_border
                    final_img.paste(framed_img, (paste_x, paste_y))
                    
                    # Replace framed_img with the final image
                    framed_img = final_img
                    
                    # Update dimensions for reporting
                    new_width = final_width
                    new_height = final_height
                else:
                    # No additional borders needed
                    additional_h_border = 0
                    additional_v_border = 0
                    new_width = base_width
                    new_height = base_height
            else:
                # No aspect ratio adjustment
                additional_h_border = 0
                additional_v_border = 0
                new_width = base_width
                new_height = base_height
            
            # Determine output path
            if output_path is None:
                input_path = Path(image_path)
                output_path = input_path.parent / f"{input_path.stem}_framed{input_path.suffix}"
            
            # Save the framed image
            framed_img.save(output_path, quality=95)
            
            print(f"✓ Successfully created framed image: {output_path}")
            if rotate_angle:
                print(f"  Rotation applied: {rotate_angle}°")
            print(f"  Original size: {original_width}x{original_height}")
            print(f"  New size: {new_width}x{new_height}")
            print(f"  Base border: {border_size}px ({border_percentage:.1f}%)")
            if aspect_ratio:
                print(f"  Target aspect ratio: {aspect_ratio[0]}:{aspect_ratio[1]}")
                print(f"  Additional borders: {additional_h_border}px (H), {additional_v_border}px (V)")
                actual_ratio = new_width / new_height
                target_ratio = aspect_ratio[0] / aspect_ratio[1]
                print(f"  Final aspect ratio: {actual_ratio:.3f} (target: {target_ratio:.3f})")
            
            return str(output_path)
            
    except Exception as e:
        print(f"✗ Error processing image: {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments and process the image."""
    parser = argparse.ArgumentParser(
        description="Add a customizable black border to an image with optional watermark text and aspect ratio adjustment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python frame.py "photo.jpg"
  python frame.py "photo.jpg" --watermark "© 2024 My Photography"
  python frame.py "image.png" --watermark "Sample Text" --font "custom_font.ttf"
  python frame.py "photo.jpg" --percentage 5.0 --aspect "16:9"
  python frame.py "image.png" --watermark "Text" --percentage 2.5 --aspect "1:1"
  python frame.py "photo.jpg" --rotate 90 --aspect "16:9"
  python frame.py "image.png" --rotate -90 --watermark "Rotated" --percentage 3.0
        """
    )
    
    parser.add_argument('image', help='Path to the input image file')
    parser.add_argument('--watermark', type=str, help='Text to add as watermark in top-left corner')
    parser.add_argument('--output', '-o', type=str, help='Output file path (optional)')
    parser.add_argument('--font', type=str, help='Path to custom font file in TTF/OTF format (WOFF2 not supported)')
    parser.add_argument('--percentage', type=float, default=4.0, help='Border percentage (default: 4.0)')
    parser.add_argument('--aspect', type=str, help='Target aspect ratio in format "x:y" (e.g., "16:9", "4:3")')
    parser.add_argument('--rotate', type=int, help='Rotation angle in degrees (90, -90, 180)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.image):
        print(f"✗ Error: Image file '{args.image}' not found.")
        sys.exit(1)
    
    # Check if it's a valid image file
    try:
        with Image.open(args.image) as img:
            pass  # Just check if we can open it
    except Exception as e:
        print(f"✗ Error: Invalid image file '{args.image}': {e}")
        sys.exit(1)
    
    # Parse aspect ratio if provided
    aspect_ratio = None
    if args.aspect:
        try:
            aspect_ratio = parse_aspect_ratio(args.aspect)
        except ValueError as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
    
    # Validate percentage
    if args.percentage <= 0:
        print(f"✗ Error: Border percentage must be positive, got {args.percentage}")
        sys.exit(1)
    
    # Validate rotation angle if provided
    if args.rotate is not None:
        if args.rotate not in [90, -90, 180, 270, -270, -180]:
            print(f"✗ Error: Unsupported rotation angle {args.rotate}. Supported: 90, -90, 180")
            sys.exit(1)
    
    # Process the image
    output_path = add_border_and_watermark(args.image, args.watermark, args.output, args.font, args.percentage, aspect_ratio, args.rotate)
    
    if args.watermark:
        print(f"  Watermark: '{args.watermark}'")


if __name__ == "__main__":
    main()
