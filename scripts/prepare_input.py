import rawpy
import numpy as np
import imageio.v3 as iio
import sys
import os
import argparse

# --- Configuration ---
# LINEAR_PROPHOTO_PROFILE_PATH is defined relative to this script
LINEAR_PROPHOTO_PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ROMM RGB.icc")

# --- Core Conversion Logic ---

def process_raw_image(raw_path: str, highlight_mode: int = 2) -> np.ndarray:
    """
    Reads a raw image and processes it into a linear 32-bit float RGB numpy array.
    
    Args:
        raw_path: Path to the input raw file.
        highlight_mode: Highlight recovery mode (0=clip, 1=unblend, 2=blend, 3+=reconstruct).
        
    Returns:
        np.ndarray: 32-bit float RGB image array (H, W, 3) in ProPhoto RGB space (linear).
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found at {raw_path}")
    
    print(f"Processing raw file: {raw_path}")

    # 1. Linear Raw Extraction (using rawpy/LibRaw)
    try:
        with rawpy.imread(raw_path) as raw:
            # Postprocess with settings for scene-referred, linear light output:
            linear_16bit_rgb = raw.postprocess(
                gamma=(1,1), 
                no_auto_bright=True, 
                output_bps=16,
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.ProPhoto,
                highlight_mode=highlight_mode
            )
        
        # 2. Cast to 32-bit Floating Point (Unbounded Data)
        linear_float32_rgb = linear_16bit_rgb.astype(np.float32) / 65535.0
        
        return linear_float32_rgb
        
    except rawpy.LibRawError as e:
        raise RuntimeError(f"Error during raw processing with rawpy: {e}")

def convert_to_archival_tiff(raw_path: str, output_path: str, profile_path: str, highlight_mode: int = 2):
    """
    Converts a raw image to a 32-bit float TIFF with linear ProPhoto RGB,
    Absolute Colorimetric intent, and Deflate level 6 compression.
    """
    try:
        final_output_array = process_raw_image(raw_path, highlight_mode)
    except Exception as e:
        print(e)
        sys.exit(1)

    # 4. Write to TIFF with Compression and Bit Depth Specifications
    print(f"Exporting to TIFF 32-bit float, lossless Deflate (ZIP) {output_path}")

    # Load ICC Profile Data for embedding
    try:
        with open(profile_path, 'rb') as f:
            icc_profile_data = f.read()
    except IOError as e:
        print(f"Warning: Could not read ICC profile at {profile_path}: {e}")
        icc_profile_data = None

    iio.imwrite(
        output_path, 
        final_output_array,
        # Enforce 32-bit floating point data type
        dtype=np.float32,
        # Compression Type: 'deflate' is the lossless ZIP compression 
        compression='deflate',
        # Embed the ICC profile
        iccprofile=icc_profile_data,
        photometric='rgb',
    )

    print(f"Success! Archival TIFF created at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw images to archival TIFF.")
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input raw image file.')
    parser.add_argument('--highlight-mode', type=int, default=2, help='Highlight recovery mode (0=clip, 1=unblend, 2=blend, 3+=reconstruct). Default is 2.')
    args = parser.parse_args()

    RAW_FILE_PATH = args.input
    base_name = os.path.splitext(os.path.basename(RAW_FILE_PATH))[0]
    output_dir = os.path.dirname(RAW_FILE_PATH) if os.path.dirname(RAW_FILE_PATH) else "."
    OUTPUT_FILE_PATH = os.path.join(output_dir, f"{base_name}.tif")

    convert_to_archival_tiff(RAW_FILE_PATH, OUTPUT_FILE_PATH, LINEAR_PROPHOTO_PROFILE_PATH, args.highlight_mode)