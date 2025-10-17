import sys
import os
import json
import shutil
import re
import math
import traceback
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image, ImageDraw

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QMessageBox, QFileDialog,
                             QTableWidget, QTableWidgetItem, QHeaderView, QShortcut, QSizePolicy, QMenu, QAbstractItemView)
from PyQt5.QtGui import QPixmap, QKeySequence, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QObject

# Try importing open3d, provide a fallback if not available
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not found. Falling back to basic ASCII PCD parser.", file=sys.stderr)

# =================================================================================
# Part 1: LiDAR Projection Logic
# =================================================================================

DEFAULT_CALIB = {
  "a6": {
    "model": "vadas",
    "intrinsic": [-0.0004, 1.0136, -0.0623, 0.2852, -0.332, 0.1896, -0.0391,
                  1.0447, 0.0021, 44.9516, 2.48822, 0, 0.9965, -0.0067,
                  -0.0956, 0.1006, -0.054, 0.0106],
    # "extrinsic": [0.0900425, -0.00450864, -0.356367, 0.00100918, -0.236104, -0.0219886],
    "extrinsic": [ 0.293769, -0.0542026, -0.631615, -0.00394431, -0.33116, -0.00963617 ],
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
        # if Xc < 0:
        #     return 0, 0, False
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
            raise ValueError(f"Sensor '{name}' not found.")
        return self.sensors[name]

class LidarProjector:
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
        # The logic is based on piecewise linear functions for R, G, B channels.
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

        lidar_to_camera_transform = cam_extrinsic @ self.calib_db.lidar_to_world
        points_cam_hom = (lidar_to_camera_transform @ cloud_xyz_hom.T).T
        points_cam = points_cam_hom[:, :3]
        in_front_of_camera_count = 0
        on_image_count = 0
        for i in range(points_cam.shape[0]):
            Xc, Yc, Zc = points_cam[i]

            # Filter points behind the camera, but remove other distance/height restrictions
            # if Xc <= 0:
            #     continue

            in_front_of_camera_count += 1
            u, v, valid_projection = camera_model.project_point(Xc, Yc, Zc)
            if valid_projection and 0 <= u < image_width and 0 <= v < image_height:
                on_image_count += 1
                color = self._get_color_from_distance(Xc)
                r = self.point_radius
                draw.ellipse((u - r, v - r, u + r, v + r), fill=color)
        return output_image, in_front_of_camera_count, on_image_count

# =================================================================================
# Part 2: GUI Application
# =================================================================================

class MappingWindow(QWidget):
    delete_requested = pyqtSignal(list)
    item_double_clicked = pyqtSignal(int) # Add this line

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mapping Data")
        self.resize(800, 700) # Set initial size, position will be handled by main window
        layout = QVBoxLayout(self)
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(4)
        self.mapping_table.setHorizontalHeaderLabels(["ID", "Filename", "A6 Original", "PCD Original"])
        header = self.mapping_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.mapping_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mapping_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mapping_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.mapping_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mapping_table.customContextMenuRequested.connect(self.open_menu)
        self.mapping_table.doubleClicked.connect(self._on_item_double_clicked) # Add this line
        self.mapping_table.setStyleSheet('QTableWidget::item:selected { background-color: #FFFFE0; color: black; }')
        layout.addWidget(self.mapping_table)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.handle_deletion()
        else:
            super().keyPressEvent(event)

    def _on_item_double_clicked(self, index):
        """Emits a signal with the row index when an item is double-clicked."""
        self.item_double_clicked.emit(index.row())

    def open_menu(self, position):
        selected_rows = self.get_selected_rows()
        if not selected_rows: return
        menu = QMenu()
        delete_action = menu.addAction(f"Delete {len(selected_rows)} selected item(s)")
        action = menu.exec_(self.mapping_table.mapToGlobal(position))
        if action == delete_action:
            self.handle_deletion()

    def handle_deletion(self):
        selected_rows = self.get_selected_rows()
        if selected_rows:
            self.delete_requested.emit(selected_rows)

    def get_selected_rows(self) -> List[int]:
        return sorted(list(set(index.row() for index in self.mapping_table.selectedIndexes())))

    def update_table(self, data):
        self.mapping_table.setRowCount(len(data))
        for i, item in enumerate(data):
            self.mapping_table.setItem(i, 0, QTableWidgetItem(str(item['id'])))
            self.mapping_table.setItem(i, 1, QTableWidgetItem(item['new_filename']))
            self.mapping_table.setItem(i, 2, QTableWidgetItem(os.path.basename(item['a6_original_path'])))
            self.mapping_table.setItem(i, 3, QTableWidgetItem(os.path.basename(item.get('pcd_original_path', 'N/A'))))
        self.mapping_table.scrollToBottom()

class PrefetchWorker(QObject):
    """Worker to load files in a background thread."""
    image_loaded = pyqtSignal(str, object)
    pcd_loaded = pyqtSignal(str, object)

    def load_image(self, path):
        try:
            if not os.path.exists(path): return
            image = Image.open(path).convert("RGB")
            self.image_loaded.emit(path, image)
        except Exception as e:
            print(f"PrefetchWorker: Failed to load image {path}: {e}", file=sys.stderr)

    def load_pcd(self, path):
        try:
            if not os.path.exists(path): return
            if OPEN3D_AVAILABLE:
                pcd = o3d.io.read_point_cloud(path)
                points = np.asarray(pcd.points, dtype=np.float64) if pcd.has_points() else np.empty((0, 3))
            else:
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
                points = np.array(points, dtype=np.float64)
            self.pcd_loaded.emit(path, points)
        except Exception as e:
            print(f"PrefetchWorker: Failed to load PCD {path}: {e}", file=sys.stderr)

class ImageSyncTool(QMainWindow):
    request_load_image = pyqtSignal(str)
    request_load_pcd = pyqtSignal(str)
    def __init__(self, parent_folder):
        super().__init__()
        self.parent_folder = os.path.abspath(parent_folder)
        self.valid = False
        self._setup_paths()
        if not self._load_frame_offsets(): return
        if not self._load_and_map_images(): return
        self._initialize_state()
        self._setup_prefetch_thread()
        self._load_mapping_data()
        self._setup_gui()
        self.mapping_window.update_table(self.mapping_data) # Update table after GUI is set up
        self._create_shortcuts()
        self.valid = True

    def _center_on_screen(self):
        """Centers the main window and mapping window together on the primary screen."""
        try:
            screen_geometry = QApplication.primaryScreen().geometry()
            main_geo = self.frameGeometry()
            # Ensure mapping_window is visible and has a valid geometry before using it
            if not self.mapping_window.isVisible():
                # Temporarily show and hide to get geometry if not visible
                self.mapping_window.show()
                self.mapping_window.hide()
            mapping_geo = self.mapping_window.frameGeometry()

            mapping_geo = self.mapping_window.frameGeometry()

            # Calculate total width including the gap (if any, currently 0)
            total_width = main_geo.width() + mapping_geo.width()

            # Calculate the new x position for the main window
            new_x = screen_geometry.center().x() - (total_width // 2)

            # Calculate the new y position for the main window (centered vertically)
            new_y = screen_geometry.center().y() - (main_geo.height() // 2)

            # Move the main window
            self.move(new_x, new_y)

            # Move the mapping window relative to the main window
            self.mapping_window.move(self.x() + main_geo.width(), self.y())

        except AttributeError:
            # Fallback for unusual environments
            pass

    def showEvent(self, event):
        super().showEvent(event)
        # Call update_display for the first time here to ensure widgets are sized correctly.
        if not hasattr(self, '_initial_display_done'):
            self._update_display()
            self._initial_display_done = True

        if self.valid and not self.mapping_window.isVisible():
            main_geo = self.geometry()
            # Set mapping window height to match main window height
            self.mapping_window.setGeometry(main_geo.right(), main_geo.top(), 600, main_geo.height())
            self.mapping_window.show()
            # Center after both windows are shown and sized
            self._center_on_screen()

    def closeEvent(self, event):
        self.mapping_window.close()
        if self.prefetch_thread:
            self.prefetch_thread.quit()
            self.prefetch_thread.wait()
        super().closeEvent(event)

    def _setup_paths(self):
        self.a5_dir = os.path.join(self.parent_folder, 'image_a5')
        self.a6_dir = os.path.join(self.parent_folder, 'image_a6')
        self.pcd_dir = os.path.join(self.parent_folder, 'pcd')
        self.synced_data_dir = os.path.join(self.parent_folder, 'synced_data')
        self.synced_a6_dir = os.path.join(self.synced_data_dir, 'image_a6')
        self.synced_pcd_dir = os.path.join(self.synced_data_dir, 'pcd')
        self.projected_results_dir = os.path.join(self.synced_data_dir, 'projected_results')
        self.mapping_file = os.path.join(self.synced_data_dir, 'mapping_data.json')
        self.offset_file = os.path.join(self.parent_folder, 'frame_offset.txt')

    def _load_frame_offsets(self):
        if not os.path.exists(self.offset_file):
            QMessageBox.critical(None, "Error", f"'frame_offset.txt' not found:\n{self.offset_file}")
            return False
        try:
            with open(self.offset_file, 'r') as f: content = f.read()
            self.a5_start_frame = int(re.search(r"a5_start\s+(\d+)", content).group(1))
            self.a6_start_frame = int(re.search(r"a6_start\s+(\d+)", content).group(1))
            self.a5_end_frame = int(re.search(r"a5_end\s+(\d+)", content).group(1))
            self.a6_end_frame = int(re.search(r"a6_end\s+(\d+)", content).group(1))
            return True
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error parsing 'frame_offset.txt': {e}")
            return False

    def _parse_frame_number(self, filename):
        match = re.search(r'\d+', filename)
        return int(match.group(0)) if match else -1

    def _load_and_map_images(self):
        if not all(os.path.isdir(d) for d in [self.a5_dir, self.a6_dir, self.pcd_dir]):
            QMessageBox.critical(None, "Error", "Required folders ('image_a5', 'image_a6', 'pcd') not found.")
            return False
        pcd_basenames = {os.path.splitext(f)[0] for f in os.listdir(self.pcd_dir)}
        all_a5_images = sorted(os.listdir(self.a5_dir), key=self._parse_frame_number)
        self.a5_images = [f for f in all_a5_images if os.path.splitext(f)[0] in pcd_basenames]
        self.a6_images = sorted([f for f in os.listdir(self.a6_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))], key=self._parse_frame_number)
        if not self.a5_images or not self.a6_images:
            QMessageBox.critical(None, "Error", "Image folders are empty or no A5 images match PCD files.")
            return False
        self.a5_frame_to_index = {self._parse_frame_number(f): i for i, f in enumerate(self.a5_images)}
        self.a6_frame_to_index = {self._parse_frame_number(f): i for i, f in enumerate(self.a6_images)}

        # Find the start index from the frame number. If not found, default to the first image (index 0).
        self.a5_start_index = self.a5_frame_to_index.get(self.a5_start_frame)
        if self.a5_start_index is None:
            print(f"Warning: a5_start_frame '{self.a5_start_frame}' not found. Defaulting to the first image.", file=sys.stderr)
            self.a5_start_index = 0

        self.a6_start_index = self.a6_frame_to_index.get(self.a6_start_frame)
        if self.a6_start_index is None:
            print(f"Warning: a6_start_frame '{self.a6_start_frame}' not found. Defaulting to the first image.", file=sys.stderr)
            self.a6_start_index = 0

        # The end index is now the end of the list, ignoring the file's end_frame.
        self.a5_end_index = len(self.a5_images) - 1 if self.a5_images else 0
        self.a6_end_index = len(self.a6_images) - 1 if self.a6_images else 0

        return True

    def _initialize_state(self):
        self.a5_index, self.a6_index = self.a5_start_index, self.a6_start_index
        self.mapping_data, self.sync_counter = [], 0
        self.last_status_message, self.view_mode, self.view_index = "", False, 0
        self.synced_original_paths = set()
        self.projector, self.projection_enabled = None, False
        self.prefetch_thread, self.prefetch_worker = None, None
        
        # --- Caching Initialization ---
        self.projection_cache = OrderedDict()
        self.image_cache = OrderedDict()
        self.pcd_cache = OrderedDict()
        self.MAX_CACHE_SIZE = 50  # For rendered pixmaps
        self.MAX_DATA_CACHE_SIZE = 100  # For raw data like images and pcd points

        self._initialize_projector()

    def _initialize_projector(self):
        try:
            calib_db = CalibrationDB(DEFAULT_CALIB, lidar_to_world=DEFAULT_LIDAR_TO_WORLD)
            self.projector = LidarProjector(calib_db)
        except Exception as e:
            print(f"❌ Failed to initialize LiDAR Projector: {e}", file=sys.stderr)

    def _load_mapping_data(self):
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r', encoding='utf-8') as f: self.mapping_data = json.load(f)
                if self.mapping_data:
                    self.sync_counter = max(item.get('id', -1) for item in self.mapping_data) + 1
                    for item in self.mapping_data:
                        # Use normcase for case-insensitive path comparison
                        self.synced_original_paths.add(os.path.normcase(item['a5_original_path']))
                        self.synced_original_paths.add(os.path.normcase(item['a6_original_path']))
                        if item.get('pcd_original_path'):
                            self.synced_original_paths.add(os.path.normcase(item['pcd_original_path']))
            except Exception as e:
                QMessageBox.warning(None, "JSON Error", f"Error parsing mapping_data.json: {e}")
                self.mapping_data, self.sync_counter = [], 0
        else:
            pass # No mapping file, initialize empty mapping data

    def _setup_gui(self):
        self.setWindowTitle("Image Sync & LiDAR Projection Tool")
        self.setGeometry(100, 100, 1250, 750)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        image_layout = QHBoxLayout(central_widget)
        
        # Calculate target size based on A6 original ratio and desired scaling
        target_image_width = int(1920 / 2.25)
        target_image_height = int(1536 / 2.25)
        fixed_image_size = (target_image_width, target_image_height)

        for side in ['a5', 'a6']:
            layout = QVBoxLayout()
            filename_label = QLabel(f"{side.upper()} Filename")
            filename_label.setAlignment(Qt.AlignCenter)
            filename_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed) # Fixed height
            setattr(self, f"{side}_filename_label", filename_label)
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed) # Fixed size
            image_label.setFixedSize(*fixed_image_size) # Set fixed size here
            setattr(self, f"{side}_image_label", image_label)
            layout.addWidget(filename_label)
            layout.addWidget(image_label, 1) # Image label takes remaining space
            image_layout.addLayout(layout, 1)
        self.mapping_window = MappingWindow() # Pass no parent to make it a top-level window
        self.mapping_window.delete_requested.connect(self._request_delete_pairs)
        self.mapping_window.item_double_clicked.connect(self._on_mapping_item_double_clicked) # Add this line
        self.statusBar().showMessage("Ready. Press 'P' to toggle LiDAR projection.")

    def _create_shortcuts(self):
        self.shortcuts = {
            'toggle_view': QShortcut(QKeySequence("V"), self, self._toggle_view_mode),
            'go_home': QShortcut(QKeySequence("H"), self, self._go_to_home_state),
            'sync_a5_prev': QShortcut(QKeySequence("Q"), self, lambda: self._navigate_index('a5_index', 'a5_start_index', 'a5_end_index', -1, "A5 Previous")),
            'sync_a5_next': QShortcut(QKeySequence("E"), self, lambda: self._navigate_index('a5_index', 'a5_start_index', 'a5_end_index', 1, "A5 Next")),
            'sync_a6_prev': QShortcut(QKeySequence("A"), self, lambda: self._navigate_index('a6_index', 'a6_start_index', 'a6_end_index', -1, "A6 Previous")),
            'sync_a6_next': QShortcut(QKeySequence("D"), self, lambda: self._navigate_index('a6_index', 'a6_start_index', 'a6_end_index', 1, "A6 Next")),
            'sync_save': QShortcut(QKeySequence(Qt.Key_Return), self, self._save_pair),
            'sync_save_enter': QShortcut(QKeySequence(Qt.Key_Enter), self, self._save_pair),
            'sync_delete': QShortcut(QKeySequence(Qt.Key_Backspace), self, self._delete_last_pair),
            'view_prev': QShortcut(QKeySequence("A"), self, lambda: self._navigate_circular_index('view_index', 'mapping_data', -1, "Previous Pair")),
            'view_next': QShortcut(QKeySequence("D"), self, lambda: self._navigate_circular_index('view_index', 'mapping_data', 1, "Next Pair")),
            'toggle_projection': QShortcut(QKeySequence("P"), self, self._toggle_projection),
            'save_projection': QShortcut(QKeySequence("Ctrl+S"), self, self._save_full_res_projection),
        }
        self._update_shortcut_states()

    def _on_mapping_item_double_clicked(self, row_index: int):
        """Handles double-click on a mapping table item to display the corresponding scene."""
        if not self.mapping_data:
            return
        
        self.view_mode = True
        self.view_index = row_index
        self.projection_enabled = True # Enable projection by default when viewing a scene
        self._update_shortcut_states()
        self._update_display(f"Moved to saved pair ID {self.mapping_data[row_index].get('id', 'N/A')}")

    def _update_shortcut_states(self):
        is_sync_mode = not self.view_mode
        for key, sc in self.shortcuts.items():
            if key.startswith('sync_'):
                sc.setEnabled(is_sync_mode)
            elif key.startswith('view_'):
                sc.setEnabled(self.view_mode)
            elif key == 'save_projection':
                # Enable save projection only when projection is on
                sc.setEnabled(self.projection_enabled)

    def _update_display(self, status_message=None):
        if self.view_mode:
            if not self.mapping_data:
                self.statusBar().showMessage("No saved data to display. Press 'V' to switch to Sync Mode.")
                self.mapping_window.mapping_table.clearSelection()
                for label in [self.a5_image_label, self.a6_image_label]:
                    label.setPixmap(QPixmap())
                    label.setText("")
                for label in [self.a5_filename_label, self.a6_filename_label]:
                    label.setText("")
                return

            if not (0 <= self.view_index < len(self.mapping_data)):
                self.view_index = len(self.mapping_data) - 1 if self.mapping_data else 0
            
            entry = self.mapping_data[self.view_index]
            base_filename = entry['new_filename']
            self.mapping_window.mapping_table.selectRow(self.view_index)

            # Load and display A5 image from original path
            a5_original_path = entry['a5_original_path']
            a5_original_filename = os.path.basename(a5_original_path)
            self._update_panel(os.path.dirname(a5_original_path), [a5_original_filename], 0, self.a5_image_label, self.a5_filename_label)

            # Load and display A6 image (synced version)
            a6_synced_filename = f"{base_filename}.png"
            self._update_panel(self.synced_a6_dir, [a6_synced_filename], 0, self.a6_image_label, self.a6_filename_label, entry['a5_original_path'])
            
            status_text = f"View Mode: {self.view_index + 1}/{len(self.mapping_data)} ({base_filename})"
        else:
            a5_path = os.path.join(self.a5_dir, self.a5_images[self.a5_index])
            self._update_panel(self.a5_dir, self.a5_images, self.a5_index, self.a5_image_label, self.a5_filename_label)
            self._update_panel(self.a6_dir, self.a6_images, self.a6_index, self.a6_image_label, self.a6_filename_label, a5_path)
            a5_frame, a6_frame = self._parse_frame_number(self.a5_images[self.a5_index]), self._parse_frame_number(self.a6_images[self.a6_index])
            status_text = f"Saved Pairs: {len(self.mapping_data)} | A5: {self.a5_index + 1}/{len(self.a5_images)} ({a5_frame}) | A6: {self.a6_index + 1}/{len(self.a6_images)} ({a6_frame})"
        
        if status_message: self.last_status_message = status_message
        if self.last_status_message: status_text += f" | {self.last_status_message}"
        self.statusBar().showMessage(status_text)
        self._request_prefetch()

    def _load_pcd_with_cache(self, pcd_path: str) -> Optional[np.ndarray]:
        """Loads a PCD file, using a cache to avoid re-reading from disk."""
        if pcd_path in self.pcd_cache:
            self.pcd_cache.move_to_end(pcd_path)  # Mark as recently used
            return self.pcd_cache[pcd_path]

        if not os.path.exists(pcd_path):
            return None

        try:
            if OPEN3D_AVAILABLE:
                pcd = o3d.io.read_point_cloud(pcd_path)
                points = np.asarray(pcd.points, dtype=np.float64) if pcd.has_points() else np.empty((0, 3))
            else: # Manual parsing
                points = []
                with open(pcd_path, 'r', encoding='utf-8') as f:
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
                points = np.array(points, dtype=np.float64)
            
            # Cache the loaded points
            self.pcd_cache[pcd_path] = points
            if len(self.pcd_cache) > self.MAX_DATA_CACHE_SIZE:
                self.pcd_cache.popitem(last=False) # Remove least recently used
            
            return points

        except Exception as e:
            print(f"Error loading PCD {pcd_path}: {e}", file=sys.stderr)
            return None

    def _update_panel(self, img_dir, img_list, index, image_label, filename_label, pcd_ref_path=None):
        filename = img_list[index]
        image_path = os.path.join(img_dir, filename)
        
        is_a6_panel = (image_label is self.a6_image_label)
        is_projected = is_a6_panel and self.projection_enabled
        pixmap = None

        # --- Caching Logic ---
        if is_projected and self.projector:
            pcd_path = self._find_pcd_path(pcd_ref_path)
            if pcd_path:
                cache_key = (image_path, pcd_path)
                if cache_key in self.projection_cache:
                    pixmap = self.projection_cache[cache_key]
                    self.projection_cache.move_to_end(cache_key) # Mark as recently used
        
        if pixmap is None: # If not in cache or not a projection case
            try:
                # --- Image Caching Logic ---
                if image_path in self.image_cache:
                    pil_image = self.image_cache[image_path]
                    self.image_cache.move_to_end(image_path) # Mark as recently used
                else:
                    pil_image = Image.open(image_path).convert("RGB")
                    self.image_cache[image_path] = pil_image
                    if len(self.image_cache) > self.MAX_DATA_CACHE_SIZE:
                        self.image_cache.popitem(last=False)

                # Start with the original image; this will be rendered unless projection is successful
                pil_image_to_render = pil_image

                if is_projected and self.projector:
                    pcd_path = self._find_pcd_path(pcd_ref_path) # Re-finding, but it's cheap
                    if pcd_path:
                        try:
                            cloud_xyz = self._load_pcd_with_cache(pcd_path) # Use cached loader
                            if cloud_xyz is not None and cloud_xyz.size > 0:
                                # Project onto a copy so the original in the cache remains clean
                                projected_pil, _, _ = self.projector.project_cloud_to_image('a6', cloud_xyz, pil_image.copy())
                                pil_image_to_render = projected_pil
                            elif cloud_xyz is not None: # cloud_xyz is empty
                                print(f"Warning: PCD file is empty: {pcd_path}", file=sys.stderr)
                        except Exception as e:
                            print(f"Error during projection: {e}", file=sys.stderr)
                            traceback.print_exc()
                    elif pcd_ref_path:
                        print(f"Warning: Could not find corresponding PCD for A5 image: {os.path.basename(pcd_ref_path)}", file=sys.stderr)

                np_image = np.array(pil_image_to_render)
                height, width, channel = np_image.shape
                bytes_per_line = 3 * width
                qimage = QImage(np_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)

                # --- Store in projection_cache ---
                if is_projected and self.projector and pcd_path:
                    cache_key = (image_path, pcd_path)
                    self.projection_cache[cache_key] = pixmap
                    if len(self.projection_cache) > self.MAX_CACHE_SIZE:
                        self.projection_cache.popitem(last=False) # Remove least recently used

            except Exception as e:
                image_label.setText(f"Image load failed:\n{filename}\nError: {e}")
                image_label.setPixmap(QPixmap())
                return

        # Determine border style based on state
        norm_image_path = os.path.normcase(os.path.abspath(image_path))
        is_synced = norm_image_path in self.synced_original_paths
        
        if self.view_mode:
            is_synced = True

        # Base style with padding
        base_style = "padding: 3px;"
        border_color = "transparent"
        inner_bg_color = "transparent"

        if is_synced and is_projected:
            border_color = "green"
            inner_bg_color = "cyan"
        elif is_synced:
            border_color = "green"
        elif is_projected:
            border_color = "cyan"
        
        # The inner color is achieved by setting the background of the padded area.
        # The outer color is the border itself.
        border_style = f"border: 3px solid {border_color}; background-color: {inner_bg_color}; {base_style}"
        
        image_label.setStyleSheet(border_style)
        
        filename_label.setText(f"{filename}{' (Projected)' if is_projected else ''}")
        
        # A5 should fill the panel, A6 should maintain aspect ratio for projection accuracy
        scale_mode = Qt.IgnoreAspectRatio if image_label is self.a5_image_label else Qt.KeepAspectRatio
        scaled_pixmap = pixmap.scaled(image_label.size(), scale_mode, Qt.SmoothTransformation)
        image_label.setPixmap(scaled_pixmap)

    def _find_pcd_path(self, a5_image_path_str: str) -> Optional[str]:
        if not a5_image_path_str: return None
        pcd_filename_stem = os.path.splitext(os.path.basename(a5_image_path_str))[0]
        pcd_path = os.path.join(self.pcd_dir, pcd_filename_stem + ".pcd")
        return pcd_path if os.path.exists(pcd_path) else None

    def _navigate_index(self, idx_attr, start_attr, end_attr, step, msg):
        current = getattr(self, idx_attr)
        start = getattr(self, start_attr)
        end = getattr(self, end_attr)

        new_index = current  # Default to current index

        if step > 0:  # Moving forward
            if current >= end:
                new_index = start  # At the end, loop to the start
            else:
                new_index = current + 1
        elif step < 0:  # Moving backward
            if current <= start:
                new_index = end  # At the start, loop to the end
            else:
                new_index = current - 1
            
        setattr(self, idx_attr, new_index)
        self._update_display(msg)

    def _navigate_circular_index(self, idx_attr, list_attr, step, msg):
        data_list = getattr(self, list_attr)
        if not data_list: self._update_display("No data to display"); return
        new_index = (getattr(self, idx_attr) + step + len(data_list)) % len(data_list)
        setattr(self, idx_attr, new_index)
        self._update_display(msg)

    def _go_to_home_state(self):
        if self.view_mode:
            if self.mapping_data: self.view_index = len(self.mapping_data) - 1
            self._update_display("Moved to last saved pair")
        else:
            if self.mapping_data:
                last = self.mapping_data[-1]
                a5_fn, a6_fn = os.path.basename(last['a5_original_path']), os.path.basename(last['a6_original_path'])
                try:
                    self.a5_index, self.a6_index = self.a5_images.index(a5_fn), self.a6_images.index(a6_fn)
                    self._update_display("Moved to last saved position")
                    return
                except ValueError: pass
            self.a5_index, self.a6_index = self.a5_start_index, self.a6_start_index
            self._update_display("Reset to initial state")

    def _toggle_view_mode(self):
        if not self.mapping_data and not self.view_mode:
            QMessageBox.information(self, "Information", "You must save at least one pair to enter View Mode.")
            return
        self.view_mode = not self.view_mode
        self.view_index = len(self.mapping_data) - 1 if self.mapping_data else 0
        self._update_shortcut_states()
        self._update_display("Switched to View Mode" if self.view_mode else "Switched to Sync Mode")

    def _toggle_projection(self):
        if not self.projector:
            QMessageBox.warning(self, "Error", "LiDAR projector module not initialized.")
            return
        self.projection_enabled = not self.projection_enabled
        self._update_shortcut_states() # Update shortcut availability
        status = "ON" if self.projection_enabled else "OFF"
        self._update_display(f"LiDAR Projection {status}")

    def _save_full_res_projection(self):
        if not self.projection_enabled:
            self._update_display("Projection is not enabled.")
            return

        try:
            # Determine which image paths to use based on the current mode
            if self.view_mode:
                if not self.mapping_data: return
                entry = self.mapping_data[self.view_index]
                a6_original_path = entry['a6_original_path']
                pcd_ref_path = entry['a5_original_path']
                base_filename = f"view_{entry['new_filename']}"
            else: # Sync mode
                a6_original_path = os.path.join(self.a6_dir, self.a6_images[self.a6_index])
                pcd_ref_path = os.path.join(self.a5_dir, self.a5_images[self.a5_index])
                base_filename = os.path.splitext(self.a6_images[self.a6_index])[0]

            pcd_path = self._find_pcd_path(pcd_ref_path)
            if not pcd_path:
                QMessageBox.warning(self, "PCD Error", "Could not find corresponding PCD file to create projection.")
                return

            # Perform projection on full-resolution image
            pil_a6_image = Image.open(a6_original_path).convert("RGB")
            pcd_cloud = self._load_pcd_with_cache(pcd_path)
            
            if pcd_cloud is None or pcd_cloud.size == 0:
                QMessageBox.warning(self, "PCD Error", "PCD file is empty or could not be loaded.")
                return

            projected_image, _, _ = self.projector.project_cloud_to_image('a6', pcd_cloud, pil_a6_image)

            # Save the result
            save_dir = os.path.join(self.synced_data_dir, 'projected_full_res')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{base_filename}_projected.jpg")
            projected_image.save(save_path, "JPEG", quality=95)
            
            self._update_display(f"Saved full-res projection to: {os.path.basename(save_dir)}/{os.path.basename(save_path)}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save full-resolution projection: {e}")
            traceback.print_exc()

    def _save_pair(self):
        src_a5_path = os.path.join(self.a5_dir, self.a5_images[self.a5_index])
        src_a6_path = os.path.join(self.a6_dir, self.a6_images[self.a6_index])
        src_pcd_path = self._find_pcd_path(src_a5_path)

        if not src_pcd_path:
            QMessageBox.warning(self, "PCD Error", "Could not find corresponding PCD file for the A5 image.")
            return

        abs_a5_path = os.path.abspath(src_a5_path)
        abs_a6_path = os.path.abspath(src_a6_path)
        abs_pcd_path = os.path.abspath(src_pcd_path)

        # Check for duplicates using normalized paths
        if any(os.path.normcase(p) in self.synced_original_paths for p in [abs_a5_path, abs_a6_path, abs_pcd_path]):
            QMessageBox.warning(self, "Duplicate Error", "One or more selected files are already part of another pair.")
            return

        os.makedirs(self.synced_a6_dir, exist_ok=True)
        os.makedirs(self.synced_pcd_dir, exist_ok=True)
        os.makedirs(self.projected_results_dir, exist_ok=True)

        base_filename = f"{self.sync_counter:010d}"
        new_synced_img_filename = f"{base_filename}.png"
        new_projected_img_filename = f"{base_filename}.jpg"
        new_pcd_filename = f"{base_filename}.pcd"

        dest_a6_path = os.path.join(self.synced_a6_dir, new_synced_img_filename)
        dest_pcd_path = os.path.join(self.synced_pcd_dir, new_pcd_filename)
        dest_proj_path = os.path.join(self.projected_results_dir, new_projected_img_filename)

        try:
            # Open and save the A6 image to ensure it's in PNG format for consistency in view mode
            Image.open(src_a6_path).convert("RGB").save(dest_a6_path, "PNG")
            shutil.copy2(src_pcd_path, dest_pcd_path)

            if self.projector:
                pil_a6_image = Image.open(src_a6_path).convert("RGB")
                pcd_cloud = self._load_pcd_with_cache(src_pcd_path)
                if pcd_cloud is not None and pcd_cloud.size > 0:
                    projected_image, _, _ = self.projector.project_cloud_to_image('a6', pcd_cloud, pil_a6_image)
                    # Save projected image as JPEG with high quality
                    projected_image.save(dest_proj_path, "JPEG", quality=95)

        except Exception as e:
            QMessageBox.critical(self, "File Save Error", f"An error occurred while saving files: {e}")
            traceback.print_exc()
            return
        
        mapping_entry = {
            "id": self.sync_counter, 
            "new_filename": base_filename, 
            "a5_original_path": abs_a5_path,
            "a6_original_path": abs_a6_path,
            "pcd_original_path": abs_pcd_path
        }
        
        self.mapping_data.append(mapping_entry)
        
        # Correctly update the set with all normalized paths for immediate UI feedback
        self.synced_original_paths.add(os.path.normcase(abs_a5_path))
        self.synced_original_paths.add(os.path.normcase(abs_a6_path))
        self.synced_original_paths.add(os.path.normcase(abs_pcd_path))
        
        self.mapping_data.sort(key=lambda x: x['id'])
        self._save_mapping_data()
        self.mapping_window.update_table(self.mapping_data)

        self.sync_counter += 1
        self._update_display(f"Success: Saved pair ID {self.sync_counter-1}!")

    def _delete_last_pair(self):
        if not self.mapping_data:
            QMessageBox.information(self, "Information", "No saved pairs to delete.")
            return
        last_entry_index = len(self.mapping_data) - 1
        self._request_delete_pairs([last_entry_index])

    def _request_delete_pairs(self, indices: List[int]):
        if not indices: return
        
        count = len(indices)
        reply = QMessageBox.question(self, "Confirm Deletion", f"Are you sure you want to delete {count} selected pair(s)?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            entries_to_delete = [self.mapping_data[i] for i in indices]

            # 1. Perform file system and state modifications
            for entry in entries_to_delete:
                self._perform_delete_actions(entry)

            # 2. Modify the main data list
            for index in sorted(indices, reverse=True):
                del self.mapping_data[index]

            # 3. Update UI and save state
            self.sync_counter = max((item.get('id', -1) for item in self.mapping_data), default=-1) + 1
            self._save_mapping_data()
            self.mapping_window.update_table(self.mapping_data)
            
            status_message = f"Success: Deleted {count} pair(s)!"
            if self.view_mode and not self.mapping_data:
                self.view_mode = False
                self._update_shortcut_states()
                status_message += " Switched to Sync Mode."
            self._update_display(status_message)

    def _perform_delete_actions(self, entry_to_delete):
        """Handles file deletion and updates the synced_original_paths set."""
        base_filename = entry_to_delete['new_filename']
        synced_img_filename = f"{base_filename}.png"
        projected_img_filename = f"{base_filename}.jpg"
        pcd_filename = f"{base_filename}.pcd"

        paths_to_delete = [
            os.path.join(self.synced_a6_dir, synced_img_filename),
            os.path.join(self.synced_pcd_dir, pcd_filename),
            os.path.join(self.projected_results_dir, projected_img_filename)
        ]
        for path in paths_to_delete:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    QMessageBox.warning(self, "File Deletion Error", f"Failed to delete file:\n{path}\nError: {e}")

        # Remove the normalized paths from the tracking set
        self.synced_original_paths.discard(os.path.normcase(entry_to_delete['a5_original_path']))
        self.synced_original_paths.discard(os.path.normcase(entry_to_delete['a6_original_path']))
        if entry_to_delete.get('pcd_original_path'):
            self.synced_original_paths.discard(os.path.normcase(entry_to_delete['pcd_original_path']))

    def _setup_prefetch_thread(self):
        self.prefetch_thread = QThread()
        self.prefetch_worker = PrefetchWorker()
        self.prefetch_worker.moveToThread(self.prefetch_thread)

        # Connect worker signals to main thread slots
        self.prefetch_worker.image_loaded.connect(self._update_cache_from_worker)
        self.prefetch_worker.pcd_loaded.connect(self._update_cache_from_worker)

        # Connect main thread request signals to worker slots
        self.request_load_image.connect(self.prefetch_worker.load_image)
        self.request_load_pcd.connect(self.prefetch_worker.load_pcd)

        self.prefetch_thread.start()

    def _request_prefetch(self):
        if self.view_mode: # Prefetching is only for sync mode for now
            return

        indices_to_check = [-2, -1, 1, 2] # Prefetch 2 frames in each direction
        for offset in indices_to_check:
            # Prefetch A6 image
            a6_idx = self.a6_index + offset
            if 0 <= a6_idx < len(self.a6_images):
                a6_path = os.path.join(self.a6_dir, self.a6_images[a6_idx])
                if a6_path not in self.image_cache:
                    self.request_load_image.emit(a6_path)

            # Prefetch A5 image and corresponding PCD
            a5_idx = self.a5_index + offset
            if 0 <= a5_idx < len(self.a5_images):
                a5_path = os.path.join(self.a5_dir, self.a5_images[a5_idx])
                if a5_path not in self.image_cache:
                    self.request_load_image.emit(a5_path)
                
                pcd_path = self._find_pcd_path(a5_path)
                if pcd_path and pcd_path not in self.pcd_cache:
                    self.request_load_pcd.emit(pcd_path)

    def _update_cache_from_worker(self, path, data):
        if isinstance(data, Image.Image):
            if path not in self.image_cache:
                self.image_cache[path] = data
                if len(self.image_cache) > self.MAX_DATA_CACHE_SIZE:
                    self.image_cache.popitem(last=False)
        elif isinstance(data, np.ndarray):
            if path not in self.pcd_cache:
                self.pcd_cache[path] = data
                if len(self.pcd_cache) > self.MAX_DATA_CACHE_SIZE:
                    self.pcd_cache.popitem(last=False)

    def _save_mapping_data(self):
        try:
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.mapping_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            QMessageBox.critical(self, "JSON Save Error", f"Error saving mapping_data.json: {e}")

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    if len(sys.argv) > 1: parent_folder = sys.argv[1]
    else:
        default_path = r"Y:\adasip\Temp\20250711_LC_test\20250711_A6_A5_LC_test\ncdb_a6_dataset\2025_07_11"
        start_path = default_path if os.path.isdir(default_path) else "."
        parent_folder = QFileDialog.getExistingDirectory(None, "Select Parent Folder", start_path)
        if not parent_folder: sys.exit("No folder selected.")
    main_win = ImageSyncTool(parent_folder)
    if main_win.valid:
        main_win.show()
        sys.exit(app.exec_())
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
