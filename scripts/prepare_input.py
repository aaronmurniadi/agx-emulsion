import rawpy
import numpy as np
import imageio.v3 as iio
import sys
import os
import argparse

# --- Configuration ---
parser = argparse.ArgumentParser(description="Convert raw images to archival TIFF.")
parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input raw image file.')
args = parser.parse_args()

RAW_FILE_PATH = args.input
base_name = os.path.splitext(os.path.basename(RAW_FILE_PATH))[0]
output_dir = os.path.dirname(RAW_FILE_PATH) if os.path.dirname(RAW_FILE_PATH) else "."
OUTPUT_FILE_PATH = os.path.join(output_dir, f"{base_name}.tif")

LINEAR_PROPHOTO_PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ROMM RGB.icc")

# --- Core Conversion Logic ---

def convert_to_archival_tiff(raw_path: str, output_path: str, profile_path: str):
    """
    Converts a raw image to a 32-bit float TIFF with linear ProPhoto RGB,
    Absolute Colorimetric intent, and Deflate level 6 compression.
    """
    if not os.path.exists(raw_path):
        print(f"Error: Raw file not found at {raw_path}")
        sys.exit(1)
    
    print(f"Processing raw file: {raw_path}")

    # 1. Linear Raw Extraction (using rawpy/LibRaw)
    try:
        with rawpy.imread(raw_path) as raw:
            # Postprocess with settings for scene-referred, linear light output:
            # gamma=(1,1): Sets the Tone Response Curve (TRC) to Gamma 1.0 (linear light).
            # no_auto_bright=True: Prevents LibRaw from applying automatic scaling.
            # output_bps=16: Ensures maximum precision extraction from the sensor data (16-bit integer).
            # This output is still likely in a camera-specific color space, but it is linear.
            linear_16bit_rgb = raw.postprocess(
                gamma=(1,1), 
                no_auto_bright=True, 
                output_bps=16,
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.ProPhoto
                # output_color set to ProPhoto RGB (ROMM RGB)
            )
        
        # 2. Cast to 32-bit Floating Point (Unbounded Data)
        # Convert the high-precision 16-bit integer data to a 32-bit float (single precision).
        # Normalizing by 65535.0 (2^16 - 1) makes 1.0 represent the 16-bit maximum, 
        # allowing for 'super-white' values (> 1.0) necessary for 32-bit scene-referred data.
        linear_float32_rgb = linear_16bit_rgb.astype(np.float32) / 65535.0
        
    except rawpy.LibRawError as e:
        print(f"Error during raw processing with rawpy: {e}")
        sys.exit(1)


    # 3. Color Transformation
    # Handled by rawpy.ColorSpace.ProPhoto during extraction.
    # The output is already in Linear ProPhoto RGB space.
    
    final_output_array = linear_float32_rgb 
    
    # 4. Write to TIFF with Compression and Bit Depth Specifications
    print(f"Exporting to TIFF 32-bit float, lossless Deflate (ZIP) {output_path}")

    # Use imageio (via tifffile plugin) to write the data.
    # The 'tiff' format writer supports specific keywords for compression and level.
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
    # Note: Replace 'input_image.nef' and the profile path with actual file paths
    # For a quick test, ensure a raw file (e.g., a.CR2,.NEF, or.DNG) exists 
    # and adjust the file paths above.
    convert_to_archival_tiff(RAW_FILE_PATH, OUTPUT_FILE_PATH, LINEAR_PROPHOTO_PROFILE_PATH)