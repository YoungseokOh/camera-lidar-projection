import sys
import os
import json
import math
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

import numpy as np
from PIL import Image
import cv2

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
        for c in reversed(coeffs):
            res = res * x + c
        return res

    def project_point(self, Xc: float, Yc: float, Zc: float) -> Tuple[int, int, bool]:
        nx = -Yc
        ny = -Zc
        dist = math.hypot(nx, ny)
        if dist < sys.float_info.epsilon:
            dist = sys.float_info.epsilon
        cosPhi = nx / dist
        sinPhi = ny / dist
        theta = math.atan2(dist, Xc)

        if Xc < 0:
            return 0, 0, False

        xd = theta * self.s
        if abs(self.div) < 1e-9:
            return 0, 0, False
        
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
    def __init__(self, calib_dict: Dict[str, Any], lidar_to_world: Optional[np.ndarray] = None):
        self.sensors: Dict[str, SensorInfo] = {}
        self.lidar_to_world = lidar_to_world if lidar_to_world is not None else np.eye(4)

        for cam_name, calib_data in calib_dict.items():
            model_type = calib_data["model"]
            intrinsic = calib_data["intrinsic"]
            extrinsic_raw = calib_data["extrinsic"]
            image_size = tuple(calib_data["image_size"]) if "image_size" in calib_data and calib_data["image_size"] else None

            extrinsic_matrix = self._rodrigues_to_matrix(extrinsic_raw) if len(extrinsic_raw) == 6 else np.array(extrinsic_raw).reshape(4, 4)

            if model_type == "vadas":
                camera_model = VADASFisheyeCameraModel(intrinsic, image_size=image_size)
            else:
                raise ValueError(f"Unsupported camera model: {model_type}. This script is configured for 'vadas' only.")
            
            self.sensors[cam_name] = SensorInfo(cam_name, camera_model, intrinsic, extrinsic_matrix, image_size)

    def _rodrigues_to_matrix(self, rvec_tvec: List[float]) -> np.ndarray:
        tvec = np.array(rvec_tvec[0:3]).reshape(3, 1)
        rvec = np.array(rvec_tvec[3:6])
        theta = np.linalg.norm(rvec)
        if theta < 1e-6:
            R = np.eye(3)
        else:
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
    """Projects LiDAR point clouds to create depth maps."""
    def __init__(self, calib_db: CalibrationDB):
        self.calib_db = calib_db

    def project_cloud_to_depth_map(self, sensor_name: str, cloud_xyz: np.ndarray, image_size: Tuple[int, int]) -> Optional[np.ndarray]:
        sensor_info = self.calib_db.get(sensor_name)
        camera_model = sensor_info.model
        cam_extrinsic = sensor_info.extrinsic
        image_width, image_height = image_size

        if isinstance(camera_model, VADASFisheyeCameraModel) and camera_model.image_size is None:
            camera_model.image_size = (image_width, image_height)

        depth_map = np.zeros((image_height, image_width), dtype=np.float32)
        
        cloud_xyz_hom = np.hstack((cloud_xyz, np.ones((cloud_xyz.shape[0], 1))))
        lidar_to_camera_transform = cam_extrinsic @ self.calib_db.lidar_to_world
        points_cam_hom = (lidar_to_camera_transform @ cloud_xyz_hom.T).T
        points_cam = points_cam_hom[:, :3]

        for i in range(points_cam.shape[0]):
            Xc, Yc, Zc = points_cam[i]
            
            if Xc <= 0:
                continue

            u, v, valid_projection = camera_model.project_point(Xc, Yc, Zc)

            if valid_projection and 0 <= u < image_width and 0 <= v < image_height:
                # Occlusion check
                if depth_map[v, u] == 0 or depth_map[v, u] > Xc:
                    depth_map[v, u] = Xc
        
        return depth_map

def save_depth_map(path: Path, depth_map: np.ndarray):
    """Saves a depth map as a 16-bit PNG image, following KITTI conventions."""
    depth_map_uint16 = (depth_map * 256.0).astype(np.uint16)
    image = Image.fromarray(depth_map_uint16, mode='I;16')
    image.save(path)

def process_folder(projector: LidarCameraProjector, parent_folder: Path, cam_name: str, output_dir: Path):
    synced_data_dir = parent_folder / "synced_data"
    mapping_file = synced_data_dir / "mapping_data.json"
    
    if not mapping_file.exists():
        print(f"Error: mapping_data.json not found at {mapping_file}", file=sys.stderr)
        return

    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(mapping_data, dict) and "image_a6" in mapping_data and "pcd" in mapping_data:
        image_rel_paths = mapping_data["image_a6"]
        pcd_rel_paths = mapping_data["pcd"]
        
        if len(image_rel_paths) != len(pcd_rel_paths):
            print(f"Warning: Mismatch in number of image and PCD entries in mapping_data.json. Using minimum count.", file=sys.stderr)
        
        num_samples = min(len(image_rel_paths), len(pcd_rel_paths))
        print(f"Processing {num_samples} files. Saving depth maps to {output_dir}")

        for i in tqdm(range(num_samples)):
            image_path = synced_data_dir / image_rel_paths[i]
            pcd_path = synced_data_dir / pcd_rel_paths[i]
            
            if not image_path.exists():
                # Try with .jpg extension as a fallback
                image_path = image_path.with_suffix('.jpg')
                if not image_path.exists():
                    print(f"DEBUG: Image file not found: {image_path.with_suffix('.png')} or {image_path}", file=sys.stderr)
                    continue
            
            if not pcd_path.exists():
                 # Try with .bin extension as a fallback
                pcd_path = pcd_path.with_suffix('.bin')
                if not pcd_path.exists():
                    print(f"DEBUG: PCD file not found: {pcd_path.with_suffix('.pcd')} or {pcd_path}", file=sys.stderr)
                    continue

            try:
                pil_image = load_image(image_path)
                cloud_xyz = load_pcd_xyz(pcd_path)

                depth_map = projector.project_cloud_to_depth_map(cam_name, cloud_xyz, pil_image.size)

                if depth_map is not None:
                    output_filename = output_dir / f"{image_path.stem}.png"
                    save_depth_map(output_filename, depth_map)

            except Exception as e:
                print(f"Error processing {image_path.name}: {e}", file=sys.stderr)
                traceback.print_exc()
    elif isinstance(mapping_data, list): # Keep original list handling for backward compatibility if needed
        print(f"Warning: mapping_data.json is a list of objects. Processing with 'new_filename' assumption.", file=sys.stderr)
        print(f"Processing {len(mapping_data)} files. Saving depth maps to {output_dir}")

        for item in tqdm(mapping_data):
            new_filename = item.get("new_filename")
            if not new_filename:
                continue

            image_path = synced_data_dir / "image_a6" / f"{new_filename}.png"
            pcd_path = synced_data_dir / "pcd" / f"{new_filename}.pcd"
            
            if not image_path.exists():
                # Try with .jpg extension as a fallback
                image_path = image_path.with_suffix('.jpg')
                if not image_path.exists():
                    print(f"DEBUG: Image file not found: {image_path.with_suffix('.png')} or {image_path}", file=sys.stderr)
                    continue
            
            if not pcd_path.exists():
                 # Try with .bin extension as a fallback
                pcd_path = pcd_path.with_suffix('.bin')
                if not pcd_path.exists():
                    print(f"DEBUG: PCD file not found: {pcd_path.with_suffix('.pcd')} or {pcd_path}", file=sys.stderr)
                    continue

            try:
                pil_image = load_image(image_path)
                cloud_xyz = load_pcd_xyz(pcd_path)

                depth_map = projector.project_cloud_to_depth_map(cam_name, cloud_xyz, pil_image.size)

                if depth_map is not None:
                    output_filename = output_dir / f"{image_path.stem}.png"
                    save_depth_map(output_filename, depth_map)

            except Exception as e:
                print(f"Error processing {image_path.name}: {e}", file=sys.stderr)
                traceback.print_exc()
    else:
        print(f"Error: mapping_data.json is not in an expected format (dictionary with 'image_a6' and 'pcd' keys, or list of objects with 'new_filename').", file=sys.stderr)
        return

# As per user instruction, all data is for a6 with vadas model.
DEFAULT_CALIB = {
  "a6": {
    "model": "vadas",
    "intrinsic": [-0.0004, 1.0136, -0.0623, 0.2852, -0.332, 0.1896, -0.0391,
                  1.0447, 0.0021, 44.9516, 2.48822, 0, 0.9965, -0.0067,
                  -0.0956, 0.1006, -0.054, 0.0106],
    # "extrinsic": [0.0900425, -0.00450864, -0.356367, 0.00100918, -0.236104, -0.0219886],
    "extrinsic": [ 0.293769, -0.0542026, -0.631615, -0.00394431, -0.33116, -0.00963617],
    "image_size": None
  }
}

DEFAULT_LIDAR_TO_WORLD = np.array([
    [-0.998752, -0.00237052, -0.0498847,  0.0375091],
    [ 0.00167658, -0.999901,   0.0139481,  0.0349093],
    [-0.0499128,  0.0138471,   0.998658,   0.771878],
    [ 0.,         0.,          0.,         1.       ]
])

def main():
    parser = argparse.ArgumentParser(description="Create KITTI-style depth maps from LiDAR point clouds.")
    parser.add_argument("--parent", type=str, required=True,
                        help="Parent folder containing the 'synced_data' directory.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the generated depth maps.")
    parser.add_argument("--cam", type=str, default="a6",
                        help="Camera to project to (must be 'a6' with this configuration).")
    parser.add_argument("--calib_json", type=str, default=None,
                        help="Path to a JSON file with calibration data. Uses default if not provided.")
    parser.add_argument("--lidar_to_world", type=str, default=None,
                        help="Path to a text file with a 4x4 LiDAR to World matrix. Uses default if not provided.")
    
    args = parser.parse_args()

    if args.cam != 'a6':
        print(f"Warning: This script is configured for '--cam a6' only. You provided '{args.cam}'.", file=sys.stderr)

    parent_folder = Path(args.parent)
    output_dir = Path(args.output_dir)
    
    calib_data = DEFAULT_CALIB
    if args.calib_json:
        with open(args.calib_json, 'r', encoding='utf-8') as f:
            calib_data = json.load(f)

    lidar_to_world_matrix = DEFAULT_LIDAR_TO_WORLD
    if args.lidar_to_world:
        lidar_to_world_matrix = np.loadtxt(args.lidar_to_world).reshape(4, 4)

    try:
        calib_db = CalibrationDB(calib_data, lidar_to_world=lidar_to_world_matrix)
        projector = LidarCameraProjector(calib_db)

        process_folder(
            projector=projector,
            parent_folder=parent_folder,
            cam_name=args.cam,
            output_dir=output_dir
        )
        print("Depth map generation complete.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
