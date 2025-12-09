import rawpy
import sys
import exifread

def inspect_raw(path):
    # Extract EXIF data using exifread
    print("=== EXIF Metadata ===")
    with open(path, 'rb') as f:
        tags = exifread.process_file(f)
        
        # Print relevant tags
        relevant_keys = [
            'Image Make',
            'Image Model', 
            'EXIF LensModel',
            'EXIF LensMake',
            'EXIF FocalLength',
            'EXIF FNumber',
            'EXIF FocalLengthIn35mmFilm',
            'EXIF SubjectDistance',
        ]
        
        for key in relevant_keys:
            if key in tags:
                print(f"  {key}: {tags[key]}")
            else:
                print(f"  {key}: Not found")
    
    print("\n=== Rawpy Attributes ===")
    with rawpy.imread(path) as raw:
        print(f"  Sizes: {raw.sizes}")
        print(f"  White balance: {raw.camera_whitebalance}")

if __name__ == "__main__":
    inspect_raw(sys.argv[1])


