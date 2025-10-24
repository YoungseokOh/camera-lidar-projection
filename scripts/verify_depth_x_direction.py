import os
import json
import numpy as np
import cv2 # Import cv2
import sys
from pathlib import Path
import math
import argparse

# Import necessary components from existing scripts
# Assuming these are in the same project directory or accessible via PYTHONPATH
from create_depth_maps import (
    CameraModelBase, VADASFisheyeCameraModel, SensorInfo, CalibrationDB,
    load_pcd_xyz, LidarCameraProjector # Removed save_depth_map, load_image
)
from visualize_depth_comparison import load_depth_map_png
from calibration_data import DEFAULT_CALIB, DEFAULT_LIDAR_TO_WORLD_v2

# Define constants - use relative path from script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent  # Go up one level from scripts/
DEFAULT_DATA_PATH = PROJECT_ROOT / "ncdb-cls-sample" / "synced_data"
CAM_NAME = "a6"

# Custom load_image using cv2
def load_image_cv2(path: Path) -> np.ndarray:
    """Loads an image using OpenCV."""
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not load image from {path}")
    return img

def verify_depth_x_direction(parent_folder: Path):
    print(f"Starting verification for data in: {parent_folder}")

    mapping_file = parent_folder / "mapping_data.json"
    if not mapping_file.exists():
        print(f"Error: mapping_data.json not found at {mapping_file}", file=sys.stderr)
        return

    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    if not isinstance(mapping_data, dict) or "image_a6" not in mapping_data or "pcd" not in mapping_data:
        print(f"Error: mapping_data.json is not in the expected dictionary format.", file=sys.stderr)
        return

    image_rel_paths = mapping_data["image_a6"]
    pcd_rel_paths = mapping_data["pcd"]

    if len(image_rel_paths) != len(pcd_rel_paths):
        print(f"Warning: Mismatch in number of image and PCD entries. Using minimum count.", file=sys.stderr)
    
    num_samples = min(len(image_rel_paths), len(pcd_rel_paths))
    print(f"Processing {num_samples} file pairs for verification.")

    # Initialize CalibrationDB and LidarCameraProjector
    calib_db = CalibrationDB(DEFAULT_CALIB, lidar_to_world=DEFAULT_LIDAR_TO_WORLD_v2)
    projector = LidarCameraProjector(calib_db)

    all_abs_errors = []
    all_rel_errors = []
    points_processed_total = 0
    points_below_threshold_total = 0
    threshold = 0.01 # 1 cm threshold for "close enough"

    for i in range(num_samples):
        image_path_rel = image_rel_paths[i]
        pcd_path_rel = pcd_rel_paths[i]

        image_path = parent_folder / image_path_rel
        pcd_path = parent_folder / pcd_path_rel
        depth_map_png_path = parent_folder / "depth_maps" / f"{Path(image_path_rel).stem}.png"

        # Handle potential .jpg or .bin fallbacks as in create_depth_maps.py
        if not image_path.exists():
            image_path = image_path.with_suffix('.jpg')
            if not image_path.exists():
                print(f"Skipping: Image file not found: {image_path.with_suffix('.png')} or {image_path}", file=sys.stderr)
                continue
        
        if not pcd_path.exists():
            pcd_path = pcd_path.with_suffix('.bin')
            if not pcd_path.exists():
                print(f"Skipping: PCD file not found: {pcd_path.with_suffix('.pcd')} or {pcd_path}", file=sys.stderr)
                continue

        if not depth_map_png_path.exists():
            print(f"Skipping: Depth map PNG not found for {image_path.name} at {depth_map_png_path}", file=sys.stderr)
            continue

        try:
            # Use custom load_image_cv2
            cv2_image = load_image_cv2(image_path)
            cloud_xyz = load_pcd_xyz(pcd_path)
            depth_map_meters = load_depth_map_png(depth_map_png_path)

            if depth_map_meters is None:
                print(f"Skipping {image_path.name}: Failed to load depth map.", file=sys.stderr)
                continue

            # Get image size from cv2_image
            image_height, image_width = cv2_image.shape[:2]
            
            # Re-project points to camera coordinates to get Xc for comparison
            sensor_info = calib_db.get(CAM_NAME)
            camera_model = sensor_info.model
            cam_extrinsic = sensor_info.extrinsic

            if isinstance(camera_model, VADASFisheyeCameraModel) and camera_model.image_size is None:
                camera_model.image_size = (image_width, image_height)

            cloud_xyz_hom = np.hstack((cloud_xyz, np.ones((cloud_xyz.shape[0], 1))))

            lidar_to_camera_transform = cam_extrinsic @ calib_db.lidar_to_world
            points_cam_hom = (lidar_to_camera_transform @ cloud_xyz_hom.T).T
            points_cam = points_cam_hom[:, :3]

            file_abs_errors = []
            file_rel_errors = []
            file_points_processed = 0
            file_points_below_threshold = 0

            for j in range(points_cam.shape[0]):
                Xc, Yc, Zc = points_cam[j]
                
                if Xc <= 0: # Skip points behind the camera or at origin
                    continue

                u, v, valid_projection = camera_model.project_point(Xc, Yc, Zc)

                if valid_projection and 0 <= u < image_width and 0 <= v < image_height:
                    depth_map_value = depth_map_meters[v, u]

                    # Only compare if the depth map actually has a value from a projected point
                    # (i.e., not 0, which is the initialized value)
                    if depth_map_value > 0: 
                        abs_diff = abs(Xc - depth_map_value)
                        file_abs_errors.append(abs_diff)
                        if depth_map_value != 0: # Avoid division by zero for relative error
                            file_rel_errors.append(abs_diff / depth_map_value)
                        
                        if abs_diff <= threshold:
                            file_points_below_threshold += 1
                        file_points_processed += 1
            
            if file_points_processed > 0:
                mean_abs_error = np.mean(file_abs_errors)
                max_abs_error = np.max(file_abs_errors)
                mean_rel_error = np.mean(file_rel_errors) if file_rel_errors else 0
                percent_below_threshold = (file_points_below_threshold / file_points_processed) * 100

                print(f"--- {image_path.name} ---")
                print(f"  Valid Projected Points: {file_points_processed}")
                print(f"  Mean Absolute Error (m): {mean_abs_error:.4f}")
                print(f"  Max Absolute Error (m): {max_abs_error:.4f}")
                print(f"  Mean Relative Error: {mean_rel_error:.4f}")
                print(f"  Points within {threshold*100}cm: {percent_below_threshold:.2f}%")

                all_abs_errors.extend(file_abs_errors)
                all_rel_errors.extend(file_rel_errors)
                points_processed_total += file_points_processed
                points_below_threshold_total += file_points_below_threshold

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

    print("\n--- Overall Verification Results ---")
    if points_processed_total > 0:
        overall_mean_abs_error = np.mean(all_abs_errors)
        overall_max_abs_error = np.max(all_abs_errors)
        overall_mean_rel_error = np.mean(all_rel_errors) if all_rel_errors else 0
        overall_percent_below_threshold = (points_below_threshold_total / points_processed_total) * 100

        print(f"Total Valid Projected Points: {points_processed_total}")
        print(f"Overall Mean Absolute Error (m): {overall_mean_abs_error:.4f}")
        print(f"Overall Max Absolute Error (m): {overall_max_abs_error:.4f}")
        print(f"Overall Mean Relative Error: {overall_mean_rel_error:.4f}")
        print(f"Overall Points within {threshold*100}cm: {overall_percent_below_threshold:.2f}%")
    else:
        print("No valid points were processed for comparison.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify if LiDAR X-direction is stored as depth in generated depth maps.")
    parser.add_argument("--parent", type=str, default=str(DEFAULT_DATA_PATH),
                        help="Parent folder containing the 'synced_data' directory (e.g., 'ncdb-cls-sample/synced_data').")
    
    args = parser.parse_args()
    
    verify_depth_x_direction(Path(args.parent))