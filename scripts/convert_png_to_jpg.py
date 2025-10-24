
import os
from PIL import Image
from pathlib import Path

def convert_png_to_jpg(input_dir, output_dir, quality=100):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    png_files = list(input_path.glob('**/*.png'))
    
    converted_count = 0
    for png_file in png_files:
        try:
            img = Image.open(png_file)
            # Ensure image is in RGB mode before saving as JPG
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Construct output path, changing extension to .jpg
            relative_path = png_file.relative_to(input_path)
            jpg_file_path = output_path / relative_path.with_suffix('.jpg')
            
            # Ensure parent directory exists for the output JPG file
            jpg_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            img.save(jpg_file_path, 'JPEG', quality=quality)
            converted_count += 1
        except Exception as e:
            print(f"Error converting {png_file}: {e}")
    return converted_count, len(png_files)

if __name__ == "__main__":
    input_directory = "C:/Users/seok436/Documents/VSCode/Projects/Camera-LiDAR-Projection/ncdb-cls-sample/synced_data/image_a6"
    # For in-place replacement, output_directory is the same as input_directory
    output_directory = "C:/Users/seok436/Documents/VSCode/Projects/Camera-LiDAR-Projection/ncdb-cls-sample/synced_data/image_a6_jpg"
    
    print(f"Converting PNGs in {input_directory} to JPGs in {output_directory}...")
    converted, total = convert_png_to_jpg(input_directory, output_directory)
    print(f"Conversion complete. Converted {converted} out of {total} PNG files.")

    # Verification step: Check if the number of JPGs matches the original number of PNGs
    original_png_count = len(list(Path("C:/Users/seok436/Documents/VSCode/Projects/Camera-LiDAR-Projection/ncdb-cls-sample/synced_data/image_a6").glob('**/*.png')))
    current_jpg_count = len(list(Path(output_directory).glob('**/*.jpg')))
    
    print(f"Original PNG count (from backup): {original_png_count}")
    print(f"Current JPG count: {current_jpg_count}")

    if converted == original_png_count and current_jpg_count == original_png_count:
        print("Verification successful: All PNGs were converted to JPGs and counts match.")
        # Remove original PNG files after successful conversion and verification
        print("Removing original PNG files...")
        for png_file in Path(input_directory).glob('**/*.png'):
            try:
                os.remove(png_file)
            except Exception as e:
                print(f"Error removing {png_file}: {e}")
        print("Original PNG files removed.")
    else:
        print("Verification failed: Mismatch in converted file count or original PNG count.")
