#!/bin/bash

DARKTABLE_CLI=/Applications/Darktable.app/Contents/MacOS/darktable-cli

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input) INPUT_RAW="$2"; shift ;;
        -x|--xmp) INPUT_XMP="$2"; shift ;;
        -o|--output) OUTPUT="$2"; shift ;;
        -s|--scale) SCALE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check required arguments
if [[ -z "$INPUT_RAW" || -z "$INPUT_XMP" ]]; then
    echo "Usage: $0 -i <input.raw> -x <input.xmp> [-o <output.tif>]"
    exit 1
fi

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="${INPUT_RAW%.raw}"
fi

$DARKTABLE_CLI \
  "$INPUT_RAW" \
  "$INPUT_XMP" \
  "$OUTPUT" \
  --out-ext "tiff" \
  --hq true \
  --core \
  --conf plugins/lighttable/export/pixel_interpolator="lanczos3" \
  --conf plugins/lighttable/export/dimensions_type=3 \
  --conf plugins/lighttable/export/height=$SCALE \
  --conf plugins/lighttable/export/width=$SCALE \
  --conf plugins/lighttable/export/iccprofile="Linear ProPhoto RGB" \
  --conf plugins/lighttable/export/iccintent=3 \
  --conf plugins/imageio/format/tiff/bpp=32 \
  --conf plugins/imageio/format/tiff/pixel_type=1 \
  --conf plugins/imageio/format/tiff/compress=0 \
  --conf plugins/imageio/format/tiff/grayscale=0 \
  --conf plugins/imageio/format/tiff/store_masks=0
