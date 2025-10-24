import os
import re
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import sys
import json
import math
import traceback
from typing import Dict, List, Tuple, Optional, Any

from PIL import Image, ImageDraw

# Try importing open3d, provide a fallback if not available
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not found. Falling back to basic ASCII PCD parser.", file=sys.stderr)

# =================================================================================
# Part 1: LiDAR Projection Logic
# (from camera_lidar_projector.py)
# =================================================================================

DEFAULT_CALIB = {
  "a6": {
    "model": "vadas",
    "intrinsic": [-0.0004, 1.0136, -0.0623, 0.2852, -0.332, 0.1896, -0.0391,
                  1.0447, 0.0021, 44.9516, 2.48822, 0, 0.9965, -0.0067,
                  -0.0956, 0.1006, -0.054, 0.0106],
    "extrinsic": [0.0900425, -0.00450864, -0.356367, 0.00100918, -0.236104, -0.0219886],
    "image_size": None
  }
}

DEFAULT_LIDAR_TO_WORLD = np.array([
    [-0.998752, -0.00237052, -0.0498847,  0.0375091],
    [ 0.00167658, -0.999901,   0.0139481,  0.0349093],
    [-0.0499128,  0.0138471,   0.998658,   0.771878],
    [ 0.,         0.,          0.,         1.       ]
])

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
            image_size = tuple(calib_data["image_size"]) if calib_data["image_size"] else None
            extrinsic_matrix = self._rodrigues_to_matrix(extrinsic_raw) if len(extrinsic_raw) == 6 else np.array(extrinsic_raw).reshape(4, 4)
            if model_type == "vadas":
                camera_model = VADASFisheyeCameraModel(intrinsic, image_size=image_size)
            else:
                raise ValueError(f"Unsupported camera model: {model_type}.")
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

class LidarProjector:
    """Projects LiDAR point clouds onto camera images."""
    def __init__(self, calib_db: CalibrationDB, max_range_m: float = 100.0):
        self.calib_db = calib_db
        self.max_range_m = max_range_m

    @staticmethod
    def _load_pcd_xyz(path: Path) -> np.ndarray:
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

    def _get_color_from_distance(self, distance: float) -> Tuple[int, int, int]:
        normalized_dist = max(0.0, min(1.0, distance / self.max_range_m))
        if normalized_dist < 0.5:
            r, g, b = 0, int(255 * (normalized_dist * 2)), int(255 * (1 - normalized_dist * 2))
        else:
            r, g, b = int(255 * ((normalized_dist - 0.5) * 2)), int(255 * (1 - (normalized_dist - 0.5) * 2)), 0
        return (r, g, b)

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
        lidar_to_camera_transform = cam_extrinsic @ self.calib_db.lidar_to_world
        points_cam_hom = (lidar_to_camera_transform @ cloud_xyz_hom.T).T
        points_cam = points_cam_hom[:, :3]

        in_front_of_camera_count = 0
        on_image_count = 0

        for i in range(points_cam.shape[0]):
            Xc, Yc, Zc = points_cam[i]
            if Xc <= 0 or Xc >= 4.3 or Zc >= 3:
                continue
            in_front_of_camera_count += 1
            u, v, valid_projection = camera_model.project_point(Xc, Yc, Zc)
            if valid_projection and 0 <= u < image_width and 0 <= v < image_height:
                on_image_count += 1
                color = self._get_color_from_distance(Xc)
                draw.point((u, v), fill=color)
        
        return output_image, in_front_of_camera_count, on_image_count

# =================================================================================
# Part 2: Camera Sync and Interactive Viewer
# =================================================================================

class CameraSyncAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        
        # LiDAR Projector related
        self.projector = None
        self.projection_enabled = False
        self._initialize_projector()

        # Matplotlib state
        self.fig = None
        self.axes = None
        self.frame_pairs = None
        self.display_size = (600, 450)

    def _initialize_projector(self):
        """Initializes the LidarProjector with default calibration data."""
        try:
            calib_db = CalibrationDB(DEFAULT_CALIB, lidar_to_world=DEFAULT_LIDAR_TO_WORLD)
            self.projector = LidarProjector(calib_db)
            print("✅ LiDAR Projector initialized successfully.")
        except Exception as e:
            print(f"❌ Failed to initialize LiDAR Projector: {e}", file=sys.stderr)
            traceback.print_exc()
            self.projector = None

    def read_frame_offset(self):
        """Reads frame offset information from frame_offset.txt."""
        offset_file = self.dataset_path / "frame_offset.txt"
        if not offset_file.exists():
            print(f"❌ frame_offset.txt not found: {offset_file}", file=sys.stderr)
            return None
        offsets = {}
        with open(offset_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    key, value = parts
                    offsets[key] = int(value)
        print("📄 frame_offset.txt read successfully:", offsets)
        return offsets

    def get_frame_pairs(self, offsets, num_samples=10, random_mode=False):
        """Returns frame pairs mapping A6 to A5."""
        a6_start, a6_end = offsets['a6_start'], offsets['a6_end']
        a5_start, a5_end = offsets['a5_start'], offsets['a5_end']
        a6_total = a6_end - a6_start + 1
        a5_total = a5_end - a5_start + 1
        ratio = a5_total / a6_total

        if random_mode:
            a6_frames = sorted(random.sample(range(a6_start, a6_end + 1), num_samples))
        else:
            a6_frames = np.linspace(a6_start, a6_end, num_samples, dtype=int)

        pairs = []
        for idx, a6_frame in enumerate(a6_frames):
            rel_idx = a6_frame - a6_start
            a5_frame = a5_start + int(rel_idx * ratio)
            if a5_frame <= a5_end:
                pairs.append({'idx': idx, 'a6': a6_frame, 'a5': a5_frame})
        print(f"\n🎯 Mapped frame pairs ({len(pairs)}):")
        for p in pairs:
            print(f"  Sample {p['idx']:2d}: A6[{p['a6']}] <-> A5[{p['a5']}]")
        return pairs

    def get_frame_pairs_reverse(self, offsets, num_samples=10, random_mode=False):
        """Returns frame pairs mapping A5 to A6."""
        a6_start, a6_end = offsets['a6_start'], offsets['a6_end']
        a5_start, a5_end = offsets['a5_start'], offsets['a5_end']
        a6_total = a6_end - a6_start + 1
        a5_total = a5_end - a5_start + 1
        ratio = a5_total / a6_total

        if random_mode:
            a5_frames = sorted(random.sample(range(a5_start, a5_end + 1), num_samples))
        else:
            a5_frames = np.linspace(a5_start, a5_end, num_samples, dtype=int)

        pairs = []
        for idx, a5_frame in enumerate(a5_frames):
            rel_idx = a5_frame - a5_start
            a6_frame = a6_start + int(rel_idx / ratio)
            if a6_frame <= a6_end:
                pairs.append({'idx': idx, 'a5': a5_frame, 'a6': a6_frame})
        print(f"\n🎯 Reverse mapped frame pairs ({len(pairs)}):")
        for p in pairs:
            print(f"  Sample {p['idx']:2d}: A5[{p['a5']}] <-> A6[{p['a6']}]")
        return pairs

    def find_image(self, folder, frame_num):
        """Finds an image file for a given frame number."""
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            path = folder / f"{frame_num:010d}{ext}"
            if path.exists():
                return path
        return None

    def _find_pcd_path(self, a5_image_path: Path) -> Optional[Path]:
        """Finds the corresponding .pcd file path based on an A5 image path."""
        if not a5_image_path:
            return None
        pcd_filename_stem = a5_image_path.stem
        # As per original logic, pcd folder is a sibling of the image folders' parent.
        pcd_path = self.dataset_path / "pcd" / (pcd_filename_stem + ".pcd")
        return pcd_path if pcd_path.exists() else None

    def _get_display_image(self, image_path: Path, pcd_path: Optional[Path], cam_name: str) -> Optional[np.ndarray]:
        """Loads, projects (if enabled), and resizes an image for display."""
        if not image_path or not image_path.exists():
            return None

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}", file=sys.stderr)
            return None

        # Apply projection if enabled, projector is available, and it's the correct camera
        if self.projection_enabled and self.projector and cam_name == 'a6' and pcd_path:
            print(f"  Projecting on {image_path.name} using {pcd_path.name}...")
            cloud_xyz = self.projector._load_pcd_xyz(pcd_path)
            if cloud_xyz.size > 0:
                pil_image, in_pts, on_img_pts = self.projector.project_cloud_to_image('a6', cloud_xyz, pil_image)
                print(f"    -> Points passing filter: {in_pts}, Points on image: {on_img_pts}")
        
        # Convert to numpy array for resizing and display
        img_np = np.array(pil_image)
        return cv2.resize(img_np, self.display_size)

    def _on_key_press(self, event):
        """Handles key press events in the matplotlib window."""
        if event.key == 'p':
            self.projection_enabled = not self.projection_enabled
            status = "ON" if self.projection_enabled else "OFF"
            print(f"\n--- LiDAR Projection Toggled: {status} ---")
            self._update_plot()
        elif event.key == 'q':
            plt.close(self.fig)

    def _update_plot(self):
        """Redraws the entire plot based on the current state."""
        a5_folder = self.dataset_path / "image_a5"
        a6_folder = self.dataset_path / "image_a6"

        for i, pair in enumerate(self.frame_pairs):
            ax_a5, ax_a6 = self.axes[i, 0], self.axes[i, 1]
            ax_a5.cla()
            ax_a6.cla()

            a5_path = self.find_image(a5_folder, pair['a5'])
            a6_path = self.find_image(a6_folder, pair['a6'])
            pcd_path = self._find_pcd_path(a5_path)

            a5_img = self._get_display_image(a5_path, pcd_path, 'a5')
            a6_img = self._get_display_image(a6_path, pcd_path, 'a6')

            # Display A5
            if a5_img is not None:
                ax_a5.imshow(a5_img)
                ax_a5.set_title(f'A5 Frame {pair["a5"]}', fontsize=12)
            else:
                ax_a5.text(0.5, 0.5, 'Image Not Found', ha='center', va='center', transform=ax_a5.transAxes)
                ax_a5.set_title(f'A5 Frame {pair["a5"]} (Missing)', fontsize=12, color='red')
            ax_a5.axis('off')

            # Display A6
            if a6_img is not None:
                ax_a6.imshow(a6_img)
                title = f'A6 Frame {pair["a6"]}'
                if self.projection_enabled:
                    title += " (Projected)"
                ax_a6.set_title(title, fontsize=12)
            else:
                ax_a6.text(0.5, 0.5, 'Image Not Found', ha='center', va='center', transform=ax_a6.transAxes)
                ax_a6.set_title(f'A6 Frame {pair["a6"]} (Missing)', fontsize=12, color='red')
            ax_a6.axis('off')

        self.fig.canvas.draw_idle()

    def show_comparison(self, frame_pairs):
        """Shows an interactive comparison window."""
        self.frame_pairs = frame_pairs
        rows = len(self.frame_pairs)
        self.fig, self.axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
        self.fig.suptitle(f'Camera Sync Comparison - {self.dataset_path.name}', fontsize=16)
        if rows == 1:
            self.axes = self.axes.reshape(1, -1)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        self.fig.text(0.5, 0.02, "Press 'p' to toggle LiDAR projection on A6 images. Press 'q' to quit.",
                      ha='center', fontsize=12, style='italic', bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="k", lw=1))

        self._update_plot()
        plt.show()

    def analyze(self, num_samples=10, random_mode=False, reverse=False):
        """Runs the complete analysis and shows the interactive viewer."""
        print(f"\n{'='*60}\n🔍 Analyzing dataset: {self.dataset_path.name}\n{'='*60}")
        offsets = self.read_frame_offset()
        if offsets is None:
            return
        
        if reverse:
            pairs = self.get_frame_pairs_reverse(offsets, num_samples, random_mode)
        else:
            pairs = self.get_frame_pairs(offsets, num_samples, random_mode)
        
        if not pairs:
            print("No frame pairs generated. Exiting.", file=sys.stderr)
            return
            
        self.show_comparison(pairs)

def main():
    print("📷 Camera Sync & LiDAR Projection Tool")
    print("=" * 50)
    
    default_path = r"Y:\adasip\Temp\20250711_LC_test\20250711_A6_A5_LC_test\ncdb_a6_dataset"
    dataset_path_str = input(f"Enter dataset path (press Enter for default):\n{default_path}\n> ").strip().strip('"')
    if not dataset_path_str:
        dataset_path_str = default_path
    
    dataset_path = Path(dataset_path_str)
    if not dataset_path.exists():
        print(f"❌ Path does not exist: {dataset_path}", file=sys.stderr)
        return

    try:
        num_samples = int(input("Enter number of samples (default 10): ") or "10")
        num_samples = max(1, min(num_samples, 20))
    except ValueError:
        num_samples = 10
        print(f"Invalid input. Using default: {num_samples} samples.")

    random_mode = input("Use random sampling? (y/n, default n): ").lower() == 'y'
    reverse = input("Use reverse mapping (A5->A6)? (y/n, default n): ").lower() == 'y'

    analyzer = CameraSyncAnalyzer(dataset_path)
    analyzer.analyze(num_samples, random_mode, reverse)

if __name__ == "__main__":
    main()
