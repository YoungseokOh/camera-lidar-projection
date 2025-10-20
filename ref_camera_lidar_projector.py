import sys
import os
import json
import math
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from PyQt5.QtWidgets import QApplication, QFileDialog
from .ref_calibration_data import DEFAULT_CALIB, DEFAULT_LIDAR_TO_WORLD_v2
import numpy as np
from PIL import Image, ImageDraw
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

        # if Xc < 0: # Point is behind the camera
        #     return 0, 0, False

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
    def __init__(self, calib_dict: Dict[str, Any], lidar_to_world: Optional[np.ndarray] = None):
        self.sensors: Dict[str, SensorInfo] = {}
        self.lidar_to_world = lidar_to_world if lidar_to_world is not None else np.eye(4)

        for cam_name, calib_data in calib_dict.items():
            model_type = calib_data["model"]
            intrinsic = calib_data["intrinsic"]
            extrinsic_raw = calib_data["extrinsic"]
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
    def __init__(self, calib_db: CalibrationDB, max_range_m: float = 100.0):
        self.calib_db = calib_db
        self.max_range_m = max_range_m

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

        # Define the exclusion condition based on Y and X coordinates
        # Exclude points where (Y <= 0.5 and Y >= -0.7) AND (X >= 0.0)
        exclude_y_condition = (cloud_xyz_hom[:, 1] <= 0.5) & (cloud_xyz_hom[:, 1] >= -0.7)
        exclude_x_condition = (cloud_xyz_hom[:, 0] >= 0.0)
        
        # Combine conditions to get points to EXCLUDE
        points_to_exclude = exclude_y_condition & exclude_x_condition
        
        # Keep only the points that are NOT in the exclusion set
        cloud_xyz_hom = cloud_xyz_hom[~points_to_exclude]

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
            # if Xc <= 0 or Xc >= 4.3 or Zc >= 3:
            #     continue

            if Xc <= 0:
                continue
            
            in_front_of_camera_count += 1

            u, v, valid_projection = camera_model.project_point(Xc, Yc, Zc)

            if valid_projection and 0 <= u < image_width and 0 <= v < image_height:
                on_image_count += 1
                # C++ uses forward distance (Xc) for color mapping
                color = self._get_color_from_distance(Xc)
                # C++ uses circle, radius 1. draw.point is equivalent for single pixels.
                draw.point((u, v), fill=color)
        
        return output_image, in_front_of_camera_count, on_image_count

    def project_colored_cloud_to_image(self, sensor_name: str, cloud_xyz: np.ndarray, cloud_colors: np.ndarray, pil_image: Image.Image) -> Image.Image:
        """Projects a point cloud to an image using the points' own colors."""
        sensor_info = self.calib_db.get(sensor_name)
        camera_model = sensor_info.model
        cam_extrinsic = sensor_info.extrinsic
        image_width, image_height = pil_image.size

        if isinstance(camera_model, VADASFisheyeCameraModel) and camera_model.image_size is None:
            camera_model.image_size = (image_width, image_height)

        output_image = pil_image.copy()
        draw = ImageDraw.Draw(output_image)

        cloud_xyz_hom = np.hstack((cloud_xyz, np.ones((cloud_xyz.shape[0], 1))))
        lidar_to_camera_transform = cam_extrinsic @ self.calib_db.lidar_to_world
        points_cam_hom = (lidar_to_camera_transform @ cloud_xyz_hom.T).T
        points_cam = points_cam_hom[:, :3]

        for i in range(points_cam.shape[0]):
            Xc, Yc, Zc = points_cam[i]
            
            if Xc <= 0: # Basic filter for points behind the camera
                continue

            u, v, valid_projection = camera_model.project_point(Xc, Yc, Zc)

            if valid_projection and 0 <= u < image_width and 0 <= v < image_height:
                # Convert color from [0,1] float to [0,255] int tuple
                color_float = cloud_colors[i]
                color_int = tuple((np.array(color_float) * 255).astype(int))
                draw.point((u, v), fill=color_int)
        
        return output_image

    def get_valid_projections_with_camera_coords(self, sensor_name: str, cloud_xyz: np.ndarray, image_size: Tuple[int, int]) -> List[Tuple[int, int, float, float, float, int]]:
        """
        Projects LiDAR point cloud onto camera image and returns valid projected (u,v) coordinates,
        corresponding 3D camera coordinates (Xc, Yc, Zc), and original point indices.
        """
        sensor_info = self.calib_db.get(sensor_name)
        camera_model = sensor_info.model
        cam_extrinsic = sensor_info.extrinsic
        image_width, image_height = image_size

        if isinstance(camera_model, VADASFisheyeCameraModel) and camera_model.image_size is None:
            camera_model.image_size = (image_width, image_height)

        cloud_xyz_hom = np.hstack((cloud_xyz, np.ones((cloud_xyz.shape[0], 1))))
        
        # Define the exclusion condition based on Y and X coordinates
        # Exclude points where (Y <= 0.5 and Y >= -0.7) AND (X >= 0.0)
        exclude_y_condition = (cloud_xyz_hom[:, 1] <= 0.5) & (cloud_xyz_hom[:, 1] >= -0.7)
        exclude_x_condition = (cloud_xyz_hom[:, 0] >= 0.0)
        
        # Combine conditions to get points to EXCLUDE
        points_to_exclude = exclude_y_condition & exclude_x_condition
        
        # Keep only the points that are NOT in the exclusion set
        cloud_xyz_hom = cloud_xyz_hom[~points_to_exclude]

        lidar_to_camera_transform = cam_extrinsic @ self.calib_db.lidar_to_world
        points_cam_hom = (lidar_to_camera_transform @ cloud_xyz_hom.T).T
        points_cam = points_cam_hom[:, :3]

        projected_data = []

        for i in range(points_cam.shape[0]):
            Xc, Yc, Zc = points_cam[i]
            
            # C++ filter: if (extrinsic_result(0) <= 0 || extrinsic_result(0) >=4.3 || extrinsic_result(2) >= 3) continue;
            # if Xc <= 0 or Xc >= 4.3 or Zc >= 3:
            #     continue
            
            u, v, valid_projection = camera_model.project_point(Xc, Yc, Zc)

            if valid_projection and 0 <= u < image_width and 0 <= v < image_height:
                projected_data.append((u, v, Xc, Yc, Zc, i)) # Store u, v, camera coords, and original index
        
        return projected_data

def show_interactive_projection(projector: LidarCameraProjector, parent_folder: Path, cam_name: str):
    synced_data_dir = parent_folder / "synced_data"
    mapping_file = synced_data_dir / "mapping_data.json"
    pcd_dir = parent_folder / "pcd"

    if not mapping_file.exists():
        print(f"Error: mapping_data.json not found at {mapping_file}", file=sys.stderr)
        return

    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    if not mapping_data:
        print("mapping_data.json is empty.", file=sys.stderr)
        return

    # Use the first record in the mapping file
    record = mapping_data[0]
    print(f"Using first record for interactive session: ID {record.get('id', 'N/A')}")

    # Determine image and pcd paths from the record
    image_path = Path(record[f"{cam_name}_original_path"])
    # The pcd filename is always based on the a5 image name, as per user requirement.
    pcd_filename_stem = Path(record["a5_original_path"]).stem
    pcd_path = pcd_dir / (pcd_filename_stem + ".pcd")

    if not image_path.exists() or not pcd_path.exists():
        print(f"Error: File not found. Image: {image_path}, PCD: {pcd_path}", file=sys.stderr)
        return

    initial_pil_image = load_image(image_path)
    cloud_xyz = load_pcd_xyz(pcd_path)

    sensor_info = projector.calib_db.get(cam_name)
    if sensor_info.image_size is None:
        sensor_info.image_size = initial_pil_image.size
        if isinstance(sensor_info.model, VADASFisheyeCameraModel):
            sensor_info.model.image_size = initial_pil_image.size

    initial_image_cv = cv2.cvtColor(np.array(initial_pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    window_name = f"LiDAR Projection on {cam_name} - Press 'p' to project, 'r' to reset, 'q' to quit"
    cv2.imshow(window_name, initial_image_cv)

    projected = False

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            if projected:
                print("Already projected. Press 'r' to reset.")
                continue
            print("Projecting LiDAR points...")
            projected_pil, in_pts, on_img_pts = projector.project_cloud_to_image(cam_name, cloud_xyz, initial_pil_image)
            print(f"  -> Points passing filter: {in_pts}, Points on image: {on_img_pts}")
            projected_image_cv = cv2.cvtColor(np.array(projected_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, projected_image_cv)
            projected = True
        elif key == ord('r'):
            print("Resetting image.")
            cv2.imshow(window_name, initial_image_cv)
            projected = False

    cv2.destroyAllWindows()
    print("Interactive viewer closed.")

# As per user instruction, all data is for a6 with vadas model.
# The old a5 vadas data is now a6. The old a6 pinhole data is removed.

def main():
    app = None
    if not QApplication.instance():
        app = QApplication(sys.argv)

    parser = argparse.ArgumentParser(description="Interactively project LiDAR point cloud onto a camera image.")
    parser.add_argument("--parent", type=str, default=None,
                        help="Parent folder containing image_a6, pcd, synced_data.")
    parser.add_argument("--cam", type=str, default="a6",
                        help="Camera to project to (must be 'a6' with this configuration).")
    parser.add_argument("--max_range", type=float, default=100.0,
                        help="Maximum LiDAR range in meters for coloring.")
    parser.add_argument("--calib_json", type=str, default=None,
                        help="Path to a JSON file with calibration data. Uses default if not provided.")
    parser.add_argument("--lidar_to_world", type=str, default=None,
                        help="Path to a text file with a 4x4 LiDAR to World matrix. Uses default if not provided.")
    
    args = parser.parse_args()

    if args.cam != 'a6':
        print(f"Warning: This script is configured for '--cam a6' only. You provided '{args.cam}'.", file=sys.stderr)
        # Proceeding, but it will likely fail if 'a6' is not in the calib data.

    parent_folder_path_str = args.parent
    if parent_folder_path_str is None:
        default_path = r"Y:\adasip\Temp\20250711_LC_test\20250711_A6_A5_LC_test\ncdb_a6_dataset\2025_07_11"
        parent_folder_path_str = QFileDialog.getExistingDirectory(None, "Select Parent Folder for LiDAR-Camera Projection", default_path)
        if not parent_folder_path_str:
            print("No folder selected. Exiting.", file=sys.stderr)
            sys.exit(1)
    
    parent_folder = Path(parent_folder_path_str)
    
    calib_data = DEFAULT_CALIB
    if args.calib_json:
        with open(args.calib_json, 'r', encoding='utf-8') as f:
            calib_data = json.load(f)

    lidar_to_world_matrix = DEFAULT_LIDAR_TO_WORLD_v2
    if args.lidar_to_world:
        lidar_to_world_matrix = np.loadtxt(args.lidar_to_world).reshape(4, 4)

    try:
        calib_db = CalibrationDB(calib_data, lidar_to_world=lidar_to_world_matrix)
        projector = LidarCameraProjector(calib_db, max_range_m=args.max_range)

        show_interactive_projection(
            projector=projector,
            parent_folder=parent_folder,
            cam_name=args.cam
        )

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    if app:
        app.quit()

if __name__ == "__main__":
    main()
