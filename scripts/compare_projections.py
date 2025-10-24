import sys
import os
import json
import math
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from PyQt5.QtWidgets import QApplication, QFileDialog

import numpy as np
from PIL import Image, ImageDraw
import cv2

from calibration_data import DEFAULT_CALIB, DEFAULT_LIDAR_TO_WORLD_v1, DEFAULT_LIDAR_TO_WORLD_v2

# Try importing open3d, provide a fallback if not available
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not found. Falling back to basic ASCII PCD parser.", file=sys.stderr)


class CameraModelBase:
    """Base class for camera projection models."""
    def project_point(self, Xc: float, Yc: float, Zc: float) -> Tuple[int, int, bool]:
        raise NotImplementedError

class VADASFisheyeCameraModel(CameraModelBase):
    """VADAS Polynomial Fisheye Camera Model, assuming +X is forward."""
    def __init__(self, intrinsic: List[float], image_size: Optional[Tuple[int, int]] = None):
        if len(intrinsic) < 11:
            raise ValueError("VADAS intrinsic must have at least 11 parameters.")
        self.k = intrinsic[0:7]
        self.s = intrinsic[7]
        self.div = intrinsic[8]
        self.ux = intrinsic[9]
        self.uy = intrinsic[10]
        self.image_size = image_size

    def _poly_eval(self, coeffs: List[float], x: float) -> float:
        res = 0.0
        # Using Horner's method for polynomial evaluation, matching C++ implementation
        for c in reversed(coeffs):
            res = res * x + c
        return res

    def project_point(self, Xc: float, Yc: float, Zc: float) -> Tuple[int, int, bool]:
        # This model expects camera looking along +X axis.
        # The C++ code uses: normPt = cv::Point2f(-extrinsic_result(1), -extrinsic_result(2));
        # This corresponds to nx = -Yc, ny = -Zc
        nx = -Yc
        ny = -Zc
        
        dist = math.hypot(nx, ny)
        
        # C++: dist = dist < DBL_EPSILON ? DBL_EPSILON : dist;
        # This prevents division by zero. sys.float_info.epsilon is the Python equivalent of DBL_EPSILON.
        if dist < sys.float_info.epsilon:
            dist = sys.float_info.epsilon
        
        cosPhi = nx / dist
        sinPhi = ny / dist
        
        # C++: theta = atan2(dist, extrinsic_result(0));
        # This corresponds to theta = atan2(dist, Xc)
        theta = math.atan2(dist, Xc)

        if Xc < 0: # Point is behind the camera
            return 0, 0, False

        xd = theta * self.s

        if abs(self.div) < 1e-9:
            return 0, 0, False
        
        # C++ polynomial evaluation loop is equivalent to this
        rd = self._poly_eval(self.k, xd) / self.div

        if math.isinf(rd) or math.isnan(rd):
            return 0, 0, False

        img_w_half = (self.image_size[0] / 2) if self.image_size else 0
        img_h_half = (self.image_size[1] / 2) if self.image_size else 0

        u = rd * cosPhi + self.ux + img_w_half
        v = rd * sinPhi + self.uy + img_h_half
        
        return int(round(u)), int(round(v)), True

class SensorInfo:
    """Holds camera sensor information."""
    def __init__(self, name: str, model: CameraModelBase, intrinsic: List[float], extrinsic: np.ndarray, image_size: Optional[Tuple[int, int]] = None):
        self.name = name
        self.model = model
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.image_size = image_size

class CalibrationDB:
    """Manages camera calibration data."""
    def __init__(self, calib_dict: Dict[str, Any], lidar_to_world: Optional[np.ndarray] = None, extrinsic_key: str = "extrinsic_v1"):
        self.sensors: Dict[str, SensorInfo] = {}
        self.lidar_to_world = lidar_to_world if lidar_to_world is not None else np.eye(4)

        for cam_name, calib_data in calib_dict.items():
            model_type = calib_data["model"]
            intrinsic = calib_data["intrinsic"]
            extrinsic_raw = calib_data.get(extrinsic_key, calib_data.get("extrinsic_v1")) # Use specified key or fallback to v1
            if extrinsic_raw is None:
                raise ValueError(f"Extrinsic data for key '{extrinsic_key}' or 'extrinsic_v1' not found for camera '{cam_name}'.")
            image_size = tuple(calib_data["image_size"]) if calib_data["image_size"] else None

            extrinsic_matrix = self._rodrigues_to_matrix(extrinsic_raw) if len(extrinsic_raw) == 6 else np.array(extrinsic_raw).reshape(4, 4)

            if model_type == "vadas":
                camera_model = VADASFisheyeCameraModel(intrinsic, image_size=image_size)
            else:
                raise ValueError(f"Unsupported camera model: {model_type}. This script is configured for 'vadas' only.")
            
            self.sensors[cam_name] = SensorInfo(cam_name, camera_model, intrinsic, extrinsic_matrix, image_size)

    def _rodrigues_to_matrix(self, rvec_tvec: List[float]) -> np.ndarray:
        # C++ 참조 코드(rodrigues_to_matrix.cpp)를 기반으로 하며,
        # (tx, ty, tz, rx, ry, rz) 순서의 입력을 예상합니다.
        tvec = np.array(rvec_tvec[0:3]).reshape(3, 1)
        rvec = np.array(rvec_tvec[3:6])
        theta = np.linalg.norm(rvec)
        if theta < 1e-6:
            R = np.eye(3)
        else:
            # 로드리게스 회전 공식. C++ 코드의 각 원소 계산과 수학적으로 동일합니다.
            r = rvec / theta
            K = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
        
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = R
        transform_matrix[0:3, 3:4] = tvec
        return transform_matrix

    def get(self, name: str) -> SensorInfo:
        if name not in self.sensors:
            raise ValueError(f"Sensor '{name}' not found in calibration database.")
        return self.sensors[name]

def load_pcd_xyz(path: Path) -> np.ndarray:
    if OPEN3D_AVAILABLE:
        try:
            pcd = o3d.io.read_point_cloud(str(path))
            return np.asarray(pcd.points, dtype=np.float64) if pcd.has_points() else np.empty((0, 3))
        except Exception as e:
            print(f"Warning: open3d failed to read {path}. Falling back. Error: {e}", file=sys.stderr)

    points = []
    with open(path, 'r', encoding='utf-8') as f:
        data_started = False
        for line in f:
            if data_started:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except (ValueError, IndexError):
                    continue
            elif line.startswith("DATA ascii"):
                data_started = True
    return np.array(points, dtype=np.float64)

def load_image(path: Path) -> Image.Image:
    return Image.open(path)

class LidarCameraProjector:
    """Projects LiDAR point clouds onto camera images, based on C++ reference."""
    def __init__(self, calib_db: CalibrationDB, max_range_m: float = 10.0, point_radius: int = 2):
        self.calib_db = calib_db
        self.max_range_m = max_range_m
        self.point_radius = point_radius

    def _get_color_from_distance(self, distance: float) -> Tuple[int, int, int]:
        """
        Calculates a color based on distance using a JET-like colormap.
        The colormap transitions from deep blue for close objects to dark red for distant objects,
        providing a smooth and perceptually uniform gradient.
        """
        normalized_dist = max(0.0, min(1.0, distance / self.max_range_m))

        # This is a common, simplified implementation of the JET colormap.
        # It maps the [0, 1] range to a blue-cyan-yellow-red-dark red spectrum.
        # The logic is based on piecewise linear functions for R, G, and B channels.
        v = normalized_dist
        
        # The colormap is calculated by defining linear ramps for R, G, and B
        # that are active over different parts of the value range.
        four_v = 4.0 * v
        r = min(four_v - 1.5, -four_v + 4.5)
        g = min(four_v - 0.5, -four_v + 3.5)
        b = min(four_v + 0.5, -four_v + 2.5)

        # Clamp values to [0, 1] range and scale to 0-255
        r_byte = int(max(0.0, min(1.0, r)) * 255)
        g_byte = int(max(0.0, min(1.0, g)) * 255)
        b_byte = int(max(0.0, min(1.0, b)) * 255)

        return (r_byte, g_byte, b_byte)

    def project_cloud_to_image(self, sensor_name: str, cloud_xyz: np.ndarray, pil_image: Image.Image) -> Tuple[Image.Image, int, int]:
        sensor_info = self.calib_db.get(sensor_name)
        camera_model = sensor_info.model
        cam_extrinsic = sensor_info.extrinsic
        image_width, image_height = pil_image.size

        if isinstance(camera_model, VADASFisheyeCameraModel) and camera_model.image_size is None:
            camera_model.image_size = (image_width, image_height)

        output_image = pil_image.copy()
        draw = ImageDraw.Draw(output_image)

        cloud_xyz_hom = np.hstack((cloud_xyz, np.ones((cloud_xyz.shape[0], 1))))

        # C++ logic: L2CExtrinsic = extrinsic * L2WMatrix;
        # This means cam_extrinsic is World->Cam, not Cam->World. No inversion needed.
        lidar_to_camera_transform = cam_extrinsic @ self.calib_db.lidar_to_world
        
        points_cam_hom = (lidar_to_camera_transform @ cloud_xyz_hom.T).T
        points_cam = points_cam_hom[:, :3]

        in_front_of_camera_count = 0
        on_image_count = 0
        
        for i in range(points_cam.shape[0]):
            Xc, Yc, Zc = points_cam[i]
            
            # C++ filter: if (extrinsic_result(0) <= 0 || extrinsic_result(0) >=4.3 || extrinsic_result(2) >= 3) continue;
            # This translates to: if Xc <= 0 or Xc >= 4.3 or Zc >= 3: continue
            # Removed Xc >= 4.3 or Zc >= 3 filter to show more points
            if Xc <= 0:
                continue
            
            in_front_of_camera_count += 1

            u, v, valid_projection = camera_model.project_point(Xc, Yc, Zc)

            if valid_projection and 0 <= u < image_width and 0 <= v < image_height:
                on_image_count += 1
                # C++ uses forward distance (Xc) for color mapping
                color = self._get_color_from_distance(Xc)
                # Draw a circle with specified radius
                r = self.point_radius
                draw.ellipse((u - r, v - r, u + r, v + r), fill=color)
        
        return output_image, in_front_of_camera_count, on_image_count

def main():
    app = None
    if not QApplication.instance():
        app = QApplication(sys.argv)

    parser = argparse.ArgumentParser(description="Compare LiDAR-Camera Projections using different calibration versions.")
    parser.add_argument("--parent", type=str, default=None,
                        help="Parent folder containing image_a6, pcd, synced_data.")
    parser.add_argument("--cam", type=str, default="a6",
                        help="Camera to project to (must be 'a6' with this configuration).")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to compare.")
    
    args = parser.parse_args()

    parent_folder_path_str = args.parent
    if parent_folder_path_str is None:
        default_path = os.getcwd() # Set default path to current working directory
        parent_folder_path_str = QFileDialog.getExistingDirectory(None, "Select Parent Folder for LiDAR-Camera Projection", default_path)
        if not parent_folder_path_str:
            print("No folder selected. Exiting.", file=sys.stderr)
            sys.exit(1)
    
    parent_folder = Path(parent_folder_path_str)
    synced_data_dir = parent_folder / "synced_data"
    mapping_file = synced_data_dir / "mapping_data.json"
    pcd_dir = parent_folder / "pcd"

    if not mapping_file.exists():
        print(f"Error: mapping_data.json not found at {mapping_file}", file=sys.stderr)
        sys.exit(1)

    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    if not mapping_data:
        print("mapping_data.json is empty.", file=sys.stderr)
        sys.exit(1)

    # Select samples
    import random
    samples = random.sample(mapping_data, min(args.num_samples, len(mapping_data)))

    results_images = []

    for i, record in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}: ID {record.get('id', 'N/A')}")

        image_path = Path(record[f"{args.cam}_original_path"])
        pcd_filename_stem = Path(record["a5_original_path"]).stem
        pcd_path = pcd_dir / (pcd_filename_stem + ".pcd")

        if not image_path.exists() or not pcd_path.exists():
            print(f"Warning: Skipping sample due to missing files. Image: {image_path}, PCD: {pcd_path}", file=sys.stderr)
            continue

        initial_pil_image = load_image(image_path)
        cloud_xyz = load_pcd_xyz(pcd_path)

        # --- Projection with v1 calibration ---
        calib_db_v1 = CalibrationDB(DEFAULT_CALIB, lidar_to_world=DEFAULT_LIDAR_TO_WORLD_v1, extrinsic_key="extrinsic_v1")
        projector_v1 = LidarCameraProjector(calib_db_v1)
        sensor_info_v1 = projector_v1.calib_db.get(args.cam)
        if sensor_info_v1.image_size is None:
            sensor_info_v1.image_size = initial_pil_image.size
            if isinstance(sensor_info_v1.model, VADASFisheyeCameraModel):
                sensor_info_v1.model.image_size = initial_pil_image.size

        projected_pil_v1, _, _ = projector_v1.project_cloud_to_image(args.cam, cloud_xyz, initial_pil_image)
        projected_image_cv_v1 = cv2.cvtColor(np.array(projected_pil_v1.convert("RGB")), cv2.COLOR_RGB2BGR)

        # --- Projection with v2 calibration ---
        calib_db_v2 = CalibrationDB(DEFAULT_CALIB, lidar_to_world=DEFAULT_LIDAR_TO_WORLD_v2, extrinsic_key="extrinsic_v2")
        projector_v2 = LidarCameraProjector(calib_db_v2)
        sensor_info_v2 = projector_v2.calib_db.get(args.cam)
        if sensor_info_v2.image_size is None:
            sensor_info_v2.image_size = initial_pil_image.size
            if isinstance(sensor_info_v2.model, VADASFisheyeCameraModel):
                sensor_info_v2.model.image_size = initial_pil_image.size

        projected_pil_v2, _, _ = projector_v2.project_cloud_to_image(args.cam, cloud_xyz, initial_pil_image)
        projected_image_cv_v2 = cv2.cvtColor(np.array(projected_pil_v2.convert("RGB")), cv2.COLOR_RGB2BGR)

        # Combine images for comparison
        combined_image_pil = Image.fromarray(cv2.cvtColor(np.hstack((projected_image_cv_v1, projected_image_cv_v2)), cv2.COLOR_BGR2RGB))
        
        # Resize combined image for smaller display/storage
        original_width, original_height = combined_image_pil.size
        new_width = 1280  # Target width
        new_height = int(original_height * (new_width / original_width))
        resized_combined_image_pil = combined_image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

        results_images.append(resized_combined_image_pil)
    
    if not results_images:
        print("No images were processed for comparison.", file=sys.stderr)
        sys.exit(1)

    # Save all combined images to a folder
    output_dir = parent_folder / "comparison_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving comparison results to: {output_dir}")
    for i, img_pil in enumerate(results_images):
        output_path = output_dir / f"comparison_sample_{i+1}.jpg"
        img_pil.save(output_path, "JPEG", quality=90)
        print(f"  Saved {output_path}")
    
    print("Comparison results saved. Exiting.")

if __name__ == "__main__":
    main()
