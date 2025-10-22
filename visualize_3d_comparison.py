"""
3D LiDAR-Camera Projection Comparison Viewer
Displays RGB/GT/Pred images on the left and interactive 3D view on the right.

Usage:
    python visualize_3d_comparison.py --results_dir "output/ResNet-SAN_0.05to100_results"
"""
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFileDialog, QSpinBox, QGroupBox, QSlider,
    QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

try:
    import matplotlib
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    import matplotlib.cm as mpl_cm
except ImportError:
    matplotlib = None
    ListedColormap = None
    LinearSegmentedColormap = None
    mpl_cm = None

_HAS_MPL_COLORMAPS = bool(matplotlib) and hasattr(matplotlib, "colormaps")


def get_colormap(name: str):
    if matplotlib is None:
        raise RuntimeError("matplotlib is required for colormap rendering")
    try:
        if _HAS_MPL_COLORMAPS:
            return matplotlib.colormaps.get_cmap(name)
        if mpl_cm is not None:
            return mpl_cm.get_cmap(name)
    except Exception:
        pass
    fallback = "viridis"
    if _HAS_MPL_COLORMAPS:
        return matplotlib.colormaps.get_cmap(fallback)
    if mpl_cm is not None:
        return mpl_cm.get_cmap(fallback)
    raise RuntimeError("No matplotlib colormap access available")


def register_colormap(name: str, cmap) -> None:
    if matplotlib is None or cmap is None:
        return
    if _HAS_MPL_COLORMAPS:
        try:
            existing = getattr(matplotlib.colormaps, "names", None)
            if existing is not None and name in existing:
                return
            matplotlib.colormaps.register(cmap, name=name)
        except Exception:
            pass
    elif mpl_cm is not None:
        try:
            mpl_cm.get_cmap(name)
        except ValueError:
            mpl_cm.register_cmap(name=name, cmap=cmap)


ERROR_PERCENT_MAX = 20.0
ERROR_COLORMAP_NAME = "error_blue"
ERROR_COLORMAP_POINTS = [
    (0.0, "#ffffff"),                              # 0%
    (0.0015, "#f8fbff"),                           # 0.03%
    (0.015, "#e9f3ff"),                            # 0.3%
    (0.05, "#cde4ff"),                             # 1%
    (0.25, "#9bc7ff"),                             # 5%
    (0.75, "#4a76d6"),                             # 15%
    (1.0, "#0b2f78"),                              # ≥20%
]

ERROR_LEGEND_ENTRIES = [
    ("0 – 0.03%", "#ffffff"),
    ("0.03 – 1%", "#dcecff"),
    ("1 – 5%", "#aacdff"),
    ("5 – 20%", "#4a76d6"),
    ("≥ 20%", "#0b2f78"),
]

ERROR_LEGEND_FONT_SIZE = 8
ERROR_LEGEND_TEXT_COLOR = (0.0, 0.0, 0.0)
ERROR_LEGEND_USE_BACKGROUND = False
ERROR_LEGEND_BACKGROUND_COLOR = (0.97, 0.94, 0.83, 0.0)
ERROR_LEGEND_SHOW_BORDER = True
# ERROR_LEGEND_POSITION = (0.85, 0.78)  # Not used - legend uses default upper right position
ERROR_LEGEND_SIZE = (0.22, 0.18)

if matplotlib is not None and LinearSegmentedColormap is not None:
    try:
        _error_cmap = LinearSegmentedColormap.from_list(
            ERROR_COLORMAP_NAME,
            ERROR_COLORMAP_POINTS,
            N=256
        )
        register_colormap(ERROR_COLORMAP_NAME, _error_cmap)
    except Exception:
        _error_cmap = None
else:
    _error_cmap = None

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
    if matplotlib is not None and ListedColormap is not None:
        _SOLID_CMAPS = {
            "solid_cyan": ListedColormap([[0.0, 1.0, 1.0, 1.0]]),
            "solid_magenta": ListedColormap([[1.0, 0.0, 1.0, 1.0]]),
        }
        for _name, _cmap in _SOLID_CMAPS.items():
            register_colormap(_name, _cmap)
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: pyvista or pyvistaqt not installed.")
    print("Install with: pip install pyvista pyvistaqt")

from calibration_data import DEFAULT_LIDAR_TO_CAM, DEFAULT_LIDAR_TO_WORLD_v1, DEFAULT_CALIB
import math



def depth_map_to_point_cloud_vadas(
    depth_map: np.ndarray,
    intrinsic_params: list,
    image_size: tuple,
    max_depth: float = 100.0,
    min_depth: float = 0.1,
    allowed_mask: Optional[np.ndarray] = None,
    return_valid_mask: bool = False,
) -> np.ndarray:
    """
    Convert depth map to 3D point cloud using VADAS fisheye model.
    
    Args:
        depth_map: (H, W) depth values in meters
        intrinsic_params: VADAS intrinsic parameters (18 params)
        image_size: (width, height) of the image
    max_depth: Maximum valid depth
    min_depth: Minimum valid depth (values below are rejected)
        
    Returns:
        points_cam: (N, 3) array of 3D points in camera coordinates
        valid_mask: (H, W) boolean mask of pixels used (only if return_valid_mask=True)
    """
    h, w = depth_map.shape
    
    # Parse VADAS parameters (match ref_camera_lidar_projector.py)
    k = intrinsic_params[:7]  # Polynomial coefficients
    s = intrinsic_params[7]    # Scale factor
    div = intrinsic_params[8]  # Divisor
    ux = intrinsic_params[9]   # Principal point offset X
    uy = intrinsic_params[10]  # Principal point offset Y
    
    # Create pixel coordinates (match reference code)
    u_coords, v_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    
    # Flatten
    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()
    depth_flat = depth_map.flatten()
    
    # Filter valid depths
    valid_mask = (depth_flat > 0) & (depth_flat >= min_depth) & (depth_flat < max_depth)
    
    if allowed_mask is not None:
        if allowed_mask.shape != depth_map.shape:
            raise ValueError("allowed_mask must match depth_map shape")
        allowed_flat = allowed_mask.astype(bool).flatten()
        valid_mask &= allowed_flat
    u_valid = u_flat[valid_mask]
    v_valid = v_flat[valid_mask]
    depth_valid = depth_flat[valid_mask]
    
    if len(depth_valid) == 0:
        empty_points = np.zeros((0, 3), dtype=np.float32)
        if return_valid_mask:
            return empty_points, np.zeros_like(depth_map, dtype=bool)
        return empty_points
    
    # VADAS unprojection
    img_w_half = image_size[0] / 2.0
    img_h_half = image_size[1] / 2.0
    
    # Center coordinates
    u_centered = u_valid - img_w_half - ux
    v_centered = v_valid - img_h_half - uy
    
    # Radial distance in image plane
    rd = np.sqrt(u_centered**2 + v_centered**2)
    
    # Handle zero rd
    zero_rd_mask = rd < 1e-6
    
    # Compute angle phi
    cosPhi = np.zeros_like(rd)
    sinPhi = np.zeros_like(rd)
    non_zero_rd_mask = ~zero_rd_mask
    cosPhi[non_zero_rd_mask] = u_centered[non_zero_rd_mask] / rd[non_zero_rd_mask]
    sinPhi[non_zero_rd_mask] = v_centered[non_zero_rd_mask] / rd[non_zero_rd_mask]
    
    # Estimate theta from rd using Newton-Raphson
    theta = estimate_theta_from_rd(rd, k, s, div)
    
    valid_theta_mask = ~np.isnan(theta)
    
    # Calculate 3D coordinates in camera frame
    # X points forward (depth), Y right, Z down
    dist = np.zeros_like(depth_valid)
    calc_dist_mask = valid_theta_mask & non_zero_rd_mask
    dist[calc_dist_mask] = depth_valid[calc_dist_mask] * np.tan(theta[calc_dist_mask])
    
    X_cam = depth_valid        # Forward (depth direction)
    Y_cam = -dist * cosPhi     # Right (negated for correct orientation)
    Z_cam = -dist * sinPhi     # Down (negated for correct orientation)
    
    # Stack points
    points_3d_camera = np.full((len(u_valid), 3), np.nan, dtype=np.float32)
    final_valid_mask = valid_theta_mask & non_zero_rd_mask
    
    points_3d_camera[final_valid_mask, 0] = X_cam[final_valid_mask]
    points_3d_camera[final_valid_mask, 1] = Y_cam[final_valid_mask]
    points_3d_camera[final_valid_mask, 2] = Z_cam[final_valid_mask]
    
    # Handle zero rd cases (center pixels)
    points_3d_camera[zero_rd_mask, 0] = depth_valid[zero_rd_mask]
    points_3d_camera[zero_rd_mask, 1] = 0.0
    points_3d_camera[zero_rd_mask, 2] = 0.0
    
    # Filter out invalid points
    valid_rows = ~np.isnan(points_3d_camera).any(axis=1)
    valid_points = points_3d_camera[valid_rows]

    if return_valid_mask:
        full_mask_flat = np.zeros_like(depth_flat, dtype=bool)
        valid_indices = np.flatnonzero(valid_mask)
        if len(valid_indices) > 0 and len(valid_rows) == len(valid_indices):
            full_mask_flat[valid_indices[valid_rows]] = True
        full_mask = full_mask_flat.reshape(depth_map.shape)
        return valid_points, full_mask
    
    return valid_points


def estimate_theta_from_rd(rd_array: np.ndarray, k: list, s: float, div: float, max_iter: int = 10) -> np.ndarray:
    """
    Estimate theta from rd using Newton-Raphson iteration (vectorized).
    
    Args:
        rd_array: Radial distances in image plane
        k: Polynomial coefficients
        s: Scale factor
        div: Divisor
        max_iter: Maximum iterations
        
    Returns:
        theta: Angle from optical axis (radians)
    """
    if abs(div) < 1e-9:
        return np.full_like(rd_array, np.nan)
    
    # Initial guess
    theta = np.where(s != 0, rd_array / s, rd_array)
    
    converged_mask = np.zeros_like(rd_array, dtype=bool)
    
    for _ in range(max_iter):
        # Evaluate polynomial at theta*s
        xd = theta * s
        poly_val = poly_eval(k, xd)
        poly_deriv = poly_deriv_eval(k, xd)
        
        # Check for unstable derivative
        unstable_deriv_mask = np.abs(poly_deriv * s) < 1e-9
        
        # Compute predicted rd
        rd_pred = poly_val / div
        error = rd_pred - rd_array
        
        # Check convergence
        current_converged = np.abs(error) < 1e-6
        converged_mask = converged_mask | current_converged | unstable_deriv_mask
        
        # Update only non-converged points
        update_mask = ~converged_mask
        if not np.any(update_mask):
            break
        
        # Newton-Raphson update
        theta[update_mask] -= error[update_mask] / (poly_deriv[update_mask] * s / div)
        theta = np.clip(theta, 0, math.pi)
    
    # Set invalid values to NaN
    theta = np.where((theta >= 0) & (theta <= math.pi), theta, np.nan)
    
    return theta


def poly_eval(coeffs: list, x: np.ndarray) -> np.ndarray:
    """Evaluate polynomial (vectorized)."""
    result = np.zeros_like(x, dtype=np.float64)
    for c in reversed(coeffs):
        result = result * x + c
    return result


def poly_deriv_eval(coeffs: list, x: np.ndarray) -> np.ndarray:
    """Evaluate polynomial derivative (vectorized)."""
    if len(coeffs) <= 1:
        return np.zeros_like(x, dtype=np.float64)
    
    result = np.zeros_like(x, dtype=np.float64)
    for i, c in enumerate(reversed(coeffs[1:]), 1):
        result = result * x + c * i
    return result


def depth_map_to_point_cloud(depth_map: np.ndarray, intrinsic_params: list, max_depth: float = 100.0) -> np.ndarray:
    """
    Convert depth map to 3D point cloud in camera coordinates.
    
    Args:
        depth_map: (H, W) depth values in meters
        intrinsic_params: VADAS intrinsic parameters (18 params)
        max_depth: Maximum valid depth
        
    Returns:
        points_cam: (N, 3) array of 3D points in camera coordinates
    """
    h, w = depth_map.shape
    
    # Create pixel coordinates
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Flatten
    u_flat = u.flatten()
    v_flat = v.flatten()
    depth_flat = depth_map.flatten()
    
    # Filter valid depths
    valid_mask = (depth_flat > 0) & (depth_flat < max_depth)
    u_valid = u_flat[valid_mask]
    v_valid = v_flat[valid_mask]
    depth_valid = depth_flat[valid_mask]
    
    if len(depth_valid) == 0:
        return np.zeros((0, 3))
    
    # VADAS intrinsic unprojection
    # This is simplified - you may need the exact VADAS undistortion model
    # For now, use a simple pinhole approximation
    fx = intrinsic_params[9]  # Focal length approximation
    fy = intrinsic_params[9]
    cx = w / 2.0
    cy = h / 2.0
    
    # Back-project to 3D
    x_cam = (u_valid - cx) * depth_valid / fx
    y_cam = (v_valid - cy) * depth_valid / fy
    z_cam = depth_valid
    
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    
    return points_cam


def transform_points_to_world(points_cam: np.ndarray, T_cam_to_world: np.ndarray) -> np.ndarray:
    """
    Transform points from camera coordinates to world coordinates.
    
    Args:
        points_cam: (N, 3) points in camera frame
        T_cam_to_world: (4, 4) transformation matrix
        
    Returns:
        points_world: (N, 3) points in world frame
    """
    if len(points_cam) == 0:
        return np.zeros((0, 3))
    
    # Convert to homogeneous coordinates
    ones = np.ones((len(points_cam), 1))
    points_cam_hom = np.hstack([points_cam, ones])
    
    # Transform
    points_world_hom = (T_cam_to_world @ points_cam_hom.T).T
    
    # Convert back to 3D
    points_world = points_world_hom[:, :3]

    return points_world


class ProjectionComparisonWindow(QMainWindow):
    """Main window for comparing LiDAR projection results."""
    
    def __init__(self, results_dir: Path):
        super().__init__()
        self.results_dir = results_dir
        self.rgb_dir = results_dir / "rgb"
        self.gt_dir = results_dir / "gt"
        self.pred_dir = results_dir / "pred"
        self.default_camera = None
        self.keep_camera_state = False
        self._zoom_last_scale = None
        self.last_gt_depth = None
        self.last_pred_depth = None
        self.last_error_map = None
        self.show_error_checkbox = None
        self.error_legend_actor = None

        # Sequential/qualitative colormap presets to reduce desaturation toward white
        self.colormap_presets = [
            ("Viridis (balanced)", "viridis"),
            ("Viridis (reverse)", "viridis_r"),
            ("Turbo (vivid)", "turbo"),
            ("Turbo (reverse)", "turbo_r"),
            ("Magma (warm dark)", "magma"),
            ("Magma (reverse)", "magma_r"),
            ("Cividis (perceptual)", "cividis"),
            ("Cividis (reverse)", "cividis_r"),
            ("Plasma (warm)", "plasma"),
            ("Plasma (reverse)", "plasma_r"),
            ("Inferno (glow)", "inferno"),
            ("Inferno (reverse)", "inferno_r"),
            ("Spectral (diverging)", "Spectral"),
            ("Spectral (reverse)", "Spectral_r"),
        ]
        if matplotlib is not None:
            self.colormap_presets.extend([
                ("Solid Cyan", "solid_cyan"),
                ("Solid Magenta", "solid_magenta"),
            ])
        self.error_colormap_name = ERROR_COLORMAP_NAME if matplotlib is not None else "viridis"
        self.gt_colormap_name = self.colormap_presets[0][1]
        self.pred_colormap_name = self.colormap_presets[1][1]
        
        # Get list of available samples
        self.sample_files = sorted(list(self.rgb_dir.glob("*.png")))
        self.current_index = 0
        
        if not self.sample_files:
            raise ValueError(f"No PNG files found in {self.rgb_dir}")
        
        print(f"Found {len(self.sample_files)} samples")
        
        self.init_ui()
        self.load_current_sample()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("LiDAR-Camera Projection 3D Comparison")
        self.setGeometry(100, 100, 1100, 550)
        
        # Central widget
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left panel: Image stack
        left_panel = self.create_image_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # Right panel: 3D viewer
        if PYVISTA_AVAILABLE:
            right_panel = self.create_3d_panel()
            main_layout.addWidget(right_panel, stretch=3)
        else:
            placeholder = QLabel("PyVista not available.\nInstall with: pip install pyvista pyvistaqt")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("font-size: 16px; color: red;")
            main_layout.addWidget(placeholder, stretch=3)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def create_image_panel(self) -> QWidget:
        """Create left panel with RGB, GT, and Pred images."""
        panel = QWidget()
        panel.setMaximumWidth(280)
        layout = QVBoxLayout()
        
        # Control group
        control_group = QGroupBox("Sample Control")
        control_layout = QVBoxLayout()
        
        # Sample selector
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("Sample:"))
        self.sample_spinbox = QSpinBox()
        self.sample_spinbox.setMinimum(0)
        self.sample_spinbox.setMaximum(len(self.sample_files) - 1)
        self.sample_spinbox.valueChanged.connect(self.on_sample_changed)
        sample_layout.addWidget(self.sample_spinbox)
        
        self.sample_label = QLabel(f"/ {len(self.sample_files)}")
        sample_layout.addWidget(self.sample_label)
        sample_layout.addStretch()
        control_layout.addLayout(sample_layout)
        
        # File name display
        self.filename_label = QLabel("")
        self.filename_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        control_layout.addWidget(self.filename_label)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("◄ Previous")
        prev_btn.clicked.connect(self.prev_sample)
        next_btn = QPushButton("Next ►")
        next_btn.clicked.connect(self.next_sample)
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        control_layout.addLayout(nav_layout)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # RGB image
        rgb_group = QGroupBox("RGB Original")
        rgb_layout = QVBoxLayout()
        self.rgb_label = QLabel()
        self.rgb_label.setFixedSize(256, 154)
        self.rgb_label.setScaledContents(True)
        self.rgb_label.setStyleSheet("border: 2px solid #555; background-color: #000;")
        self.rgb_label.setAlignment(Qt.AlignCenter)
        rgb_layout.addWidget(self.rgb_label)
        rgb_group.setLayout(rgb_layout)
        layout.addWidget(rgb_group)
        
        # GT image
        gt_group = QGroupBox("Ground Truth Projection")
        gt_layout = QVBoxLayout()
        self.gt_label = QLabel()
        self.gt_label.setFixedSize(256, 154)
        self.gt_label.setScaledContents(True)
        self.gt_label.setStyleSheet("border: 2px solid #00aa00; background-color: #000;")
        self.gt_label.setAlignment(Qt.AlignCenter)
        gt_layout.addWidget(self.gt_label)
        gt_group.setLayout(gt_layout)
        layout.addWidget(gt_group)
        
        # Pred image
        pred_group = QGroupBox("Prediction Projection")
        pred_layout = QVBoxLayout()
        self.pred_label = QLabel()
        self.pred_label.setFixedSize(256, 154)
        self.pred_label.setScaledContents(True)
        self.pred_label.setStyleSheet("border: 2px solid #aa0000; background-color: #000;")
        self.pred_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.pred_label)
        pred_group.setLayout(pred_layout)
        layout.addWidget(pred_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_3d_panel(self) -> QWidget:
        """Create right panel with 3D viewer."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 3D viewer controls
        control_group = QGroupBox("3D View Controls")
        control_layout = QVBoxLayout()
        
        # Visibility checkboxes
        checkbox_layout = QHBoxLayout()

        self.show_gt_checkbox = QCheckBox("Show GT")
        self.show_gt_checkbox.setChecked(True)
        self.show_gt_checkbox.stateChanged.connect(self.on_visibility_changed)
        checkbox_layout.addWidget(self.show_gt_checkbox)

        self.show_pred_checkbox = QCheckBox("Show Pred")
        self.show_pred_checkbox.setChecked(True)
        self.show_pred_checkbox.stateChanged.connect(self.on_visibility_changed)
        checkbox_layout.addWidget(self.show_pred_checkbox)

        self.show_axes_checkbox = QCheckBox("Show Axes")
        self.show_axes_checkbox.setChecked(True)
        self.show_axes_checkbox.stateChanged.connect(self.on_visibility_changed)
        checkbox_layout.addWidget(self.show_axes_checkbox)

        self.match_pred_to_gt_checkbox = QCheckBox("Pred uses GT mask")
        self.match_pred_to_gt_checkbox.setChecked(True)
        self.match_pred_to_gt_checkbox.stateChanged.connect(self.on_visibility_changed)
        checkbox_layout.addWidget(self.match_pred_to_gt_checkbox)

        self.show_error_checkbox = QCheckBox("Show Error")
        self.show_error_checkbox.setChecked(False)
        self.show_error_checkbox.stateChanged.connect(self.on_show_error_changed)
        checkbox_layout.addWidget(self.show_error_checkbox)

        checkbox_layout.addStretch()
        control_layout.addLayout(checkbox_layout)

        # Colormap selectors
        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("GT Colormap:"))
        self.gt_cmap_combo = QComboBox()
        for label, cmap_name in self.colormap_presets:
            self.gt_cmap_combo.addItem(label, cmap_name)
        self.gt_cmap_combo.blockSignals(True)
        self.gt_cmap_combo.setCurrentIndex(0)
        self.gt_cmap_combo.blockSignals(False)
        self.gt_cmap_combo.currentIndexChanged.connect(self.on_gt_cmap_changed)
        self.gt_colormap_name = self.gt_cmap_combo.currentData()
        cmap_layout.addWidget(self.gt_cmap_combo)

        cmap_layout.addWidget(QLabel("Pred Colormap:"))
        self.pred_cmap_combo = QComboBox()
        for label, cmap_name in self.colormap_presets:
            self.pred_cmap_combo.addItem(label, cmap_name)
        # Choose a more vivid default for prediction contrast (Turbo)
        pred_default_index = next(
            (i for i, (label, _) in enumerate(self.colormap_presets) if "Turbo (vivid)" in label),
            1 if len(self.colormap_presets) > 1 else 0
        )
        self.pred_cmap_combo.blockSignals(True)
        self.pred_cmap_combo.setCurrentIndex(pred_default_index)
        self.pred_cmap_combo.blockSignals(False)
        self.pred_cmap_combo.currentIndexChanged.connect(self.on_pred_cmap_changed)
        self.pred_colormap_name = self.pred_cmap_combo.currentData()
        cmap_layout.addWidget(self.pred_cmap_combo)
        cmap_layout.addStretch()
        control_layout.addLayout(cmap_layout)
        
        # GT Transparency slider
        gt_transparency_layout = QHBoxLayout()
        gt_transparency_layout.addWidget(QLabel("GT Opacity:"))
        self.gt_opacity_slider = QSlider(Qt.Horizontal)
        self.gt_opacity_slider.setMinimum(0)
        self.gt_opacity_slider.setMaximum(100)
        self.gt_opacity_slider.setValue(70)
        self.gt_opacity_slider.valueChanged.connect(self.on_gt_opacity_changed)
        gt_transparency_layout.addWidget(self.gt_opacity_slider)
        self.gt_opacity_label = QLabel("70%")
        self.gt_opacity_label.setMinimumWidth(50)
        gt_transparency_layout.addWidget(self.gt_opacity_label)
        control_layout.addLayout(gt_transparency_layout)
        
        # Pred Transparency slider
        pred_transparency_layout = QHBoxLayout()
        pred_transparency_layout.addWidget(QLabel("Pred Opacity:"))
        self.pred_opacity_slider = QSlider(Qt.Horizontal)
        self.pred_opacity_slider.setMinimum(0)
        self.pred_opacity_slider.setMaximum(100)
        self.pred_opacity_slider.setValue(70)
        self.pred_opacity_slider.valueChanged.connect(self.on_pred_opacity_changed)
        pred_transparency_layout.addWidget(self.pred_opacity_slider)
        self.pred_opacity_label = QLabel("70%")
        self.pred_opacity_label.setMinimumWidth(50)
        pred_transparency_layout.addWidget(self.pred_opacity_label)
        control_layout.addLayout(pred_transparency_layout)

        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(0)
        self.zoom_slider.setMaximum(100)
        self.zoom_slider.setSingleStep(1)
        self.zoom_slider.setValue(50)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)
        self.zoom_label = QLabel("50%")
        self.zoom_label.setMinimumWidth(50)
        zoom_layout.addWidget(self.zoom_label)
        control_layout.addLayout(zoom_layout)
        self._zoom_last_scale = self.slider_value_to_scale(self.zoom_slider.value())
        
        # View buttons
        view_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset Camera")
        reset_btn.clicked.connect(self.reset_camera)
        view_layout.addWidget(reset_btn)
        
        top_view_btn = QPushButton("Top View")
        top_view_btn.clicked.connect(self.top_view)
        view_layout.addWidget(top_view_btn)
        
        side_view_btn = QPushButton("Side View")
        side_view_btn.clicked.connect(self.side_view)
        view_layout.addWidget(side_view_btn)
        
        screenshot_btn = QPushButton("Screenshot")
        screenshot_btn.clicked.connect(self.take_screenshot)
        view_layout.addWidget(screenshot_btn)
        control_layout.addLayout(view_layout)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # PyVista plotter
        self.plotter = QtInteractor(panel)
        self.plotter.setMinimumSize(800, 450)
        self.plotter.set_background('black')
        
        # Set text color to white for better visibility on black background
        pv.global_theme.font.color = 'white'
        pv.global_theme.font.family = 'courier'
        
        layout.addWidget(self.plotter.interactor)
        
        panel.setLayout(layout)
        return panel
    
    def load_current_sample(self):
        """Load and display current sample."""
        sample_file = self.sample_files[self.current_index]
        sample_name = sample_file.stem
        
        # Update filename label
        self.filename_label.setText(f"File: {sample_name}")
        
        # Load images
        rgb_path = self.rgb_dir / f"{sample_name}.png"
        gt_path = self.gt_dir / f"{sample_name}.png"
        pred_path = self.pred_dir / f"{sample_name}.png"
        
        # Update 2D image labels
        self.update_image_label(self.rgb_label, rgb_path)
        self.update_image_label(
            self.gt_label,
            gt_path,
            colormap_name=self.gt_colormap_name
        )

        self.update_image_label(
            self.pred_label,
            pred_path,
            colormap_name=self.pred_colormap_name
        )
        
        # Update 3D view
        if PYVISTA_AVAILABLE:
            self.keep_camera_state = False
            self.update_3d_view(gt_path, pred_path)
        
        # Update window title
        self.setWindowTitle(f"3D Comparison - {sample_name} ({self.current_index + 1}/{len(self.sample_files)})")
        self.refresh_depth_previews()
    
    def refresh_depth_previews(self):
        """Refresh GT/Pred depth previews using current colormaps."""
        if not self.sample_files:
            return

        sample_file = self.sample_files[self.current_index]
        sample_name = sample_file.stem
        gt_path = self.gt_dir / f"{sample_name}.png"
        pred_path = self.pred_dir / f"{sample_name}.png"

        gt_override = self.last_gt_depth if self.last_gt_depth is not None else None
        if gt_override is not None:
            gt_override = np.array(gt_override, copy=False)

        show_error_control = getattr(self, "show_error_checkbox", None)
        error_active = (
            show_error_control is not None
            and show_error_control.isChecked()
            and self.last_error_map is not None
        )
        if error_active:
            pred_override = self.last_error_map
        elif self.last_pred_depth is not None:
            pred_override = self.last_pred_depth
        else:
            pred_override = None

        if pred_override is not None:
            pred_override = np.array(pred_override, copy=False)

        self.update_image_label(
            self.gt_label,
            gt_path,
            colormap_name=self.gt_colormap_name,
            data_override=gt_override
        )
        pred_cmap_name = self.error_colormap_name if error_active else self.pred_colormap_name
        pred_value_range = (0.0, ERROR_PERCENT_MAX) if error_active else None

        self.update_image_label(
            self.pred_label,
            pred_path,
            colormap_name=pred_cmap_name,
            data_override=pred_override,
            value_range=pred_value_range
        )

    def update_image_label(
        self,
        label: QLabel,
        image_path: Path,
        colormap_name: Optional[str] = None,
        data_override: Optional[np.ndarray] = None,
        value_range: Optional[Tuple[float, float]] = None
    ):
        """Update QLabel with image, depth, or custom data visualization."""

        if data_override is not None and matplotlib is not None and colormap_name is not None:
            pixmap = self.depth_to_colormap_pixmap(data_override, colormap_name, value_range=value_range)
            if pixmap is not None:
                label.setPixmap(pixmap)
                label.setText("")
                return

        if not image_path.exists():
            label.setText(f"Not found:\n{image_path.name}")
            label.setStyleSheet(label.styleSheet() + " color: red;")
            return

        if colormap_name is not None and matplotlib is not None and data_override is None:
            depth_img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                pixmap = None
                if depth_img.dtype == np.uint16:
                    depth_m = depth_img.astype(np.float32) / 256.0
                    pixmap = self.depth_to_colormap_pixmap(depth_m, colormap_name, value_range=value_range)
                elif depth_img.dtype == np.float32:
                    pixmap = self.depth_to_colormap_pixmap(depth_img, colormap_name, value_range=value_range)

                if pixmap is not None:
                    label.setPixmap(pixmap)
                    label.setText("")
                    return

        try:
            pixmap = QPixmap(str(image_path))
            label.setPixmap(pixmap)
            label.setText("")
        except Exception as e:
            label.setText(f"Error loading:\n{image_path.name}\n{str(e)}")
            label.setStyleSheet(label.styleSheet() + " color: red;")
    
    def depth_to_colormap_pixmap(
        self,
        depth_map: np.ndarray,
        colormap_name: str,
        value_range: Optional[Tuple[float, float]] = None
    ) -> Optional[QPixmap]:
        """Convert a depth map to a QPixmap using the specified colormap."""
        if matplotlib is None:
            return None

        if value_range is not None:
            vmin, vmax = value_range
            mask = np.isfinite(depth_map)
            if vmin is not None:
                mask &= depth_map >= vmin
        else:
            mask = np.isfinite(depth_map) & (depth_map > 0)
        if not np.any(mask):
            return None

        valid = depth_map[mask]
        if value_range is not None:
            vmin, vmax = value_range
        else:
            vmin = np.percentile(valid, 2.0) if valid.size > 1 else float(valid.min())
            vmax = np.percentile(valid, 98.0) if valid.size > 1 else float(valid.max())
            if vmax <= vmin:
                vmax = vmin + 1e-6

        denom = (vmax - vmin) if (vmax is not None and vmin is not None) else None
        if denom is None or abs(denom) < 1e-12:
            denom = 1e-12

        norm = np.clip((depth_map - vmin) / (denom), 0.0, 1.0)
        norm[~mask] = 0.0

        try:
            cmap = get_colormap(colormap_name)
        except Exception:
            cmap = get_colormap('viridis')

        rgba = cmap(norm, bytes=True)
        rgba = np.ascontiguousarray(rgba)
        # Give non-valid pixels a dim appearance
        rgba[~mask] = np.array([70, 70, 70, 255], dtype=np.uint8)

        height, width, _ = rgba.shape
        qimage = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimage.copy())

    def update_3d_view(self, gt_path: Path, pred_path: Path):
        """Update 3D viewer with GT and Pred depth maps as 3D point clouds in World coordinates."""
        camera_state = None
        if self.keep_camera_state:
            camera_state = self.plotter.camera_position
            if camera_state is not None:
                camera_state = tuple(np.array(component, dtype=np.float64) for component in camera_state)

        self.plotter.clear()
        self.gt_actor = None
        self.pred_actor = None
        self.last_gt_depth = None
        self.last_pred_depth = None
        self.last_error_map = None

        # Load depth maps as 16-bit PNG
        gt_depth = None
        pred_depth = None

        if gt_path.exists():
            gt_img = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            if gt_img is not None:
                if gt_img.dtype == np.uint16:
                    gt_depth = gt_img.astype(np.float32) / 256.0
                else:
                    print(f"Warning: GT image is not 16-bit: {gt_img.dtype}")
                    gt_depth = gt_img.astype(np.float32) / 256.0

        if pred_path.exists():
            pred_img = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
            if pred_img is not None:
                if pred_img.dtype == np.uint16:
                    pred_depth = pred_img.astype(np.float32) / 256.0
                else:
                    print(f"Warning: Pred image is not 16-bit: {pred_img.dtype}")
                    pred_depth = pred_img.astype(np.float32) / 256.0

        if gt_depth is None and pred_depth is None:
            self.plotter.add_text(
                "No depth maps available",
                position='center',
                font_size=12,
                color='red',
                font='courier'
            )
            return

        if gt_depth is not None:
            self.last_gt_depth = gt_depth.copy()
        if pred_depth is not None:
            self.last_pred_depth = pred_depth.copy()

        error_mode = (
            self.show_error_checkbox.isChecked()
            and gt_depth is not None
            and pred_depth is not None
        )

        # Disable automatic rendering during scene construction
        self.plotter.render_window.SetDesiredUpdateRate(0.0001)
        
        try:

            intrinsic = DEFAULT_CALIB["a6"]["intrinsic"]

            T_lidar_to_cam = DEFAULT_LIDAR_TO_CAM
            T_cam_to_lidar = np.linalg.inv(T_lidar_to_cam)

            T_lidar_to_world = DEFAULT_LIDAR_TO_WORLD_v1

            T_cam_to_world = T_lidar_to_world @ T_cam_to_lidar
            cam_position_world = T_cam_to_world[:3, 3]
            lidar_origin_world = T_lidar_to_world[:3, 3]

            gt_points_cam = None
            gt_points_world = None
            pred_points_world = None
            gt_valid_mask = None
            pred_valid_mask = None
            pred_mask_mode = "All"
            gt_distances = None
            pred_distances = None
            pred_error_values = None

            gt_mask_needed = (
                gt_depth is not None
                and self.match_pred_to_gt_checkbox.isChecked()
                and pred_depth is not None
            )
            compute_gt_cloud = gt_depth is not None and (
                self.show_gt_checkbox.isChecked() or gt_mask_needed
            )

            if compute_gt_cloud:
                gt_image_size = (gt_depth.shape[1], gt_depth.shape[0])
                gt_points_cam, gt_valid_mask = depth_map_to_point_cloud_vadas(
                    gt_depth,
                    intrinsic,
                    gt_image_size,
                    return_valid_mask=True
                )

            if (
                self.show_gt_checkbox.isChecked()
                and gt_points_cam is not None
                and len(gt_points_cam) > 0
            ):
                gt_points_world = transform_points_to_world(gt_points_cam, T_cam_to_world)

                gt_offset_from_lidar = gt_points_world - lidar_origin_world[np.newaxis, :]
                gt_distances = np.linalg.norm(gt_offset_from_lidar, axis=1)
                if np.any(~np.isfinite(gt_distances)):
                    gt_distances = np.nan_to_num(gt_distances, nan=0.0, posinf=0.0, neginf=0.0)

                gt_cloud = pv.PolyData(gt_points_world)

                gt_colors = gt_distances
                
                # Determine colorbar range for GT
                gt_clim = [np.nanmin(gt_colors), np.nanmax(gt_colors)]

                self.gt_actor = self.plotter.add_points(
                    gt_cloud,
                    scalars=gt_colors,
                    point_size=4,
                    cmap=self.gt_colormap_name,
                    nan_color='black',
                    opacity=self.gt_opacity_slider.value() / 100.0,
                    render_points_as_spheres=True,
                    name="GT_points",
                    clim=gt_clim
                )

            if pred_depth is not None and self.show_pred_checkbox.isChecked():
                pred_image_size = (pred_depth.shape[1], pred_depth.shape[0])
                allowed_mask = None
                if (
                    self.match_pred_to_gt_checkbox.isChecked()
                    and gt_valid_mask is not None
                    and gt_valid_mask.shape == pred_depth.shape
                ):
                    allowed_mask = gt_valid_mask
                    pred_mask_mode = "GT mask"
                elif self.match_pred_to_gt_checkbox.isChecked() and gt_valid_mask is None:
                    pred_mask_mode = "All (no GT)"
                elif self.match_pred_to_gt_checkbox.isChecked() and gt_valid_mask is not None:
                    print("Warning: GT and Pred depth sizes differ; falling back to full Pred points")

                if error_mode:
                    pred_points_cam, pred_valid_mask = depth_map_to_point_cloud_vadas(
                        pred_depth,
                        intrinsic,
                        pred_image_size,
                        allowed_mask=allowed_mask,
                        return_valid_mask=True
                    )
                else:
                    pred_points_cam = depth_map_to_point_cloud_vadas(
                        pred_depth,
                        intrinsic,
                        pred_image_size,
                        allowed_mask=allowed_mask
                    )

                if len(pred_points_cam) > 0:
                    pred_points_world = transform_points_to_world(pred_points_cam, T_cam_to_world)

                    pred_offset_from_lidar = pred_points_world - lidar_origin_world[np.newaxis, :]
                    pred_distances = np.linalg.norm(pred_offset_from_lidar, axis=1)
                    if np.any(~np.isfinite(pred_distances)):
                        pred_distances = np.nan_to_num(pred_distances, nan=0.0, posinf=0.0, neginf=0.0)

                    pred_cloud = pv.PolyData(pred_points_world)

                    pred_colors = pred_distances

                    if error_mode and pred_valid_mask is not None:
                        error_map = np.full_like(pred_depth, np.nan, dtype=np.float32)
                        combined_mask = pred_valid_mask.copy()
                        combined_mask &= np.isfinite(pred_depth)
                        combined_mask &= np.isfinite(gt_depth)
                        if (
                            self.match_pred_to_gt_checkbox.isChecked()
                            and gt_valid_mask is not None
                            and gt_valid_mask.shape == combined_mask.shape
                        ):
                            combined_mask &= gt_valid_mask

                        combined_mask &= gt_depth > 0
                        valid_indices = np.flatnonzero(combined_mask)
                        if valid_indices.size > 0:
                            percent_error = np.zeros_like(pred_depth, dtype=np.float32)
                            percent_error[combined_mask] = (
                                np.abs(pred_depth[combined_mask] - gt_depth[combined_mask])
                                / gt_depth[combined_mask]
                            ) * 100.0
                            percent_error = np.clip(percent_error, 0.0, ERROR_PERCENT_MAX)
                            error_map[combined_mask] = percent_error[combined_mask]
                            pred_error_values = error_map[pred_valid_mask]
                            self.last_error_map = error_map

                            finite_error = pred_error_values[np.isfinite(pred_error_values)]
                            if finite_error.size > 0:
                                pred_colors = pred_error_values
                            else:
                                pred_error_values = None
                                self.last_error_map = None
                        else:
                            pred_error_values = None
                            self.last_error_map = None

                    cmap_name = self.error_colormap_name if error_mode else self.pred_colormap_name
                    add_kwargs = {}
                    
                    if error_mode:
                        # Error mode: always use 0-20% range
                        add_kwargs["clim"] = [0.0, ERROR_PERCENT_MAX]
                    else:
                        # Distance mode: use GT range if both visible, otherwise use Pred's own range
                        if self.show_gt_checkbox.isChecked() and gt_distances is not None:
                            # Both GT and Pred visible: use GT's range as reference
                            add_kwargs["clim"] = [np.nanmin(gt_distances), np.nanmax(gt_distances)]
                        else:
                            # Only Pred visible: use Pred's own range
                            add_kwargs["clim"] = [np.nanmin(pred_colors), np.nanmax(pred_colors)]

                    self.pred_actor = self.plotter.add_points(
                        pred_cloud,
                        scalars=pred_colors,
                        point_size=4,
                        cmap=cmap_name,
                        nan_color='black',
                        opacity=self.pred_opacity_slider.value() / 100.0,
                        render_points_as_spheres=True,
                        name="Pred_points",
                        **add_kwargs
                    )

            cam_origin = pv.Sphere(radius=0.03, center=cam_position_world)
            self.plotter.add_mesh(cam_origin, color='cyan', label='Camera')

            if self.show_axes_checkbox.isChecked():
                self.plotter.add_axes(
                    xlabel='X',
                    ylabel='Y',
                    zlabel='Z',
                    line_width=2,
                    color='white',
                    x_color='red',
                    y_color='green',
                    z_color='blue',
                    labels_off=False,
                    interactive=False,
                    viewport=(0, 0.85, 0.15, 1.0)
                )

            info_lines = ["World Coordinates"]
            info_lines.append(
                f"Camera at: [{cam_position_world[0]:.2f}, {cam_position_world[1]:.2f}, {cam_position_world[2]:.2f}]"
            )
            info_lines.append(
                f"LiDAR at: [{lidar_origin_world[0]:.2f}, {lidar_origin_world[1]:.2f}, {lidar_origin_world[2]:.2f}]"
            )

            if gt_points_world is not None:
                info_lines.append(f"GT points: {len(gt_points_world)}")
                if gt_distances is not None and len(gt_distances) > 0:
                    finite_gt = gt_distances[np.isfinite(gt_distances)]
                    if finite_gt.size > 0:
                        info_lines.append(
                            f"GT dist: {finite_gt.min():.2f}–{finite_gt.max():.2f} m (avg {finite_gt.mean():.2f} m)"
                        )

            if pred_points_world is not None:
                info_lines.append(f"Pred points: {len(pred_points_world)} ({pred_mask_mode})")
                if pred_distances is not None and len(pred_distances) > 0:
                    finite_pred = pred_distances[np.isfinite(pred_distances)]
                    if finite_pred.size > 0:
                        info_lines.append(
                            f"Pred dist: {finite_pred.min():.2f}–{finite_pred.max():.2f} m (avg {finite_pred.mean():.2f} m)"
                        )
                if (
                    pred_mask_mode == "GT mask"
                    and gt_distances is not None
                    and pred_distances is not None
                    and len(pred_distances) == len(gt_distances)
                    and len(pred_distances) > 0
                ):
                    distance_delta = np.abs(pred_distances - gt_distances)
                    finite_delta = distance_delta[np.isfinite(distance_delta)]
                    if finite_delta.size > 0:
                        info_lines.append(
                            f"Δdist avg: {finite_delta.mean():.3f} m (max {finite_delta.max():.3f} m)"
                        )
                if error_mode and pred_error_values is not None and pred_error_values.size > 0:
                    finite_error = pred_error_values[np.isfinite(pred_error_values)]
                    if finite_error.size > 0:
                        info_lines.append(
                            f"Error: {finite_error.min():.1f}–{finite_error.max():.1f}% (avg {finite_error.mean():.1f}%)"
                        )
            elif pred_depth is not None and self.show_pred_checkbox.isChecked():
                info_lines.append(f"Pred points: 0 ({pred_mask_mode})")

            if error_mode:
                info_lines.append("Error shown as % of GT depth")

            info_text = '\n'.join(info_lines)
            self.plotter.add_text(info_text, position='lower_left', font_size=6, color='white', font='courier')
            if self.show_gt_checkbox.isChecked():
                gt_label = "GT"
                if hasattr(self, 'gt_cmap_combo'):
                    gt_label = f"GT ({self.gt_cmap_combo.currentText()})"
                self.plotter.add_text(gt_label, position='upper_left', font_size=7, color='white', font='courier')
            if self.show_pred_checkbox.isChecked():
                pred_label = "Pred"
                if self.show_error_checkbox.isChecked():
                    pred_label = "Error (% of GT depth)"
                elif hasattr(self, 'pred_cmap_combo'):
                    pred_label = f"Pred ({self.pred_cmap_combo.currentText()})"
                self.plotter.add_text(pred_label, position='upper_right', font_size=7, color='white', font='courier')

            # Remove old error legend if exists
            if hasattr(self, 'error_legend_actor') and self.error_legend_actor is not None:
                try:
                    self.plotter.remove_actor(self.error_legend_actor)
                except Exception:
                    pass
                self.error_legend_actor = None
            
            # Add new error legend if in error mode
            if error_mode and pred_error_values is not None and pred_error_values.size > 0:
                self._add_error_legend()

            if camera_state is not None:
                try:
                    position, focal_point, view_up = camera_state
                    restored_state = (
                        np.array(position, dtype=np.float64),
                        np.array(focal_point, dtype=np.float64),
                        np.array(view_up, dtype=np.float64)
                    )
                    self.plotter.camera_position = tuple(component.tolist() for component in restored_state)

                    self.default_camera = restored_state
                    if hasattr(self, 'zoom_slider'):
                        self._zoom_last_scale = self.slider_value_to_scale(self.zoom_slider.value())
                except Exception as exc:
                    print(f"Warning: Failed to restore camera state ({exc}); falling back to reset.")
                    self.reset_camera()
                finally:
                    self.keep_camera_state = False
            else:
                self.reset_camera()
                self.keep_camera_state = False

        finally:
            # Restore normal rendering rate and render ONCE at the very end
            self.plotter.render_window.SetDesiredUpdateRate(30.0)
            self.plotter.render()
            self.keep_camera_state = False
    
    def on_sample_changed(self, value: int):
        """Handle sample spinbox change."""
        self.current_index = value
        self.load_current_sample()
    
    def prev_sample(self):
        """Go to previous sample."""
        if self.current_index > 0:
            self.current_index -= 1
            self.sample_spinbox.setValue(self.current_index)
    
    def next_sample(self):
        """Go to next sample."""
        if self.current_index < len(self.sample_files) - 1:
            self.current_index += 1
            self.sample_spinbox.setValue(self.current_index)
    
    def on_visibility_changed(self):
        """Handle visibility checkbox changes."""
        self.reload_current_view(preserve_camera=True)

    def on_gt_cmap_changed(self, index: int):
        """Apply new colormap for GT points."""
        _ = index
        self.gt_colormap_name = self.gt_cmap_combo.currentData()
        self.refresh_depth_previews()
        self.reload_current_view(preserve_camera=True)

    def on_pred_cmap_changed(self, index: int):
        """Apply new colormap for Pred points."""
        _ = index
        self.pred_colormap_name = self.pred_cmap_combo.currentData()
        self.refresh_depth_previews()
        self.reload_current_view(preserve_camera=True)

    def on_show_error_changed(self, state: int):
        _ = state
        if self.show_error_checkbox.isChecked() and self.show_gt_checkbox.isChecked():
            self.show_gt_checkbox.blockSignals(True)
            self.show_gt_checkbox.setChecked(False)
            self.show_gt_checkbox.blockSignals(False)
        self.reload_current_view(preserve_camera=True)

    def _add_error_legend(self):
        """Add error legend with transparent background and colored squares."""
        if not hasattr(self, "plotter"):
            return
        
        # Create legend with error ranges and colors
        legend_entries = [
            ("0 – 0.03%", "#ffffff"),
            ("0.03 – 1%", "#dcecff"),
            ("1 – 5%", "#aacdff"),
            ("5 – 20%", "#4a76d6"),
            ("≥ 20%", "#0b2f78"),
        ]
        
        try:
            # Add legend with PyVista
            legend_actor = self.plotter.add_legend(
                labels=legend_entries,
                face="rectangle",
                size=(0.15, 0.15),
                loc="upper right"
            )
            
            if legend_actor is not None:
                # Set transparent background
                try:
                    legend_actor.UseBackgroundOff()
                except Exception:
                    pass
                
                # Customize text properties
                try:
                    text_prop = legend_actor.GetEntryTextProperty()
                    if text_prop is not None:
                        text_prop.SetColor(1.0, 1.0, 1.0)  # White text
                        text_prop.SetFontSize(8)
                        text_prop.BoldOff()
                except Exception:
                    pass
                
                # Apply square symbols
                self._apply_square_symbols(legend_actor, len(legend_entries))
                
                self.error_legend_actor = legend_actor
                
        except Exception as e:
            print(f"Warning: Failed to create error legend: {e}")

    def _customize_error_legend(self, legend_actor, entry_count: int) -> None:
        if legend_actor is None or entry_count <= 0:
            return

        # Keep legend off-screen during customization
        # Position will be set to correct location just before final render

        text_property = None
        try:
            text_property = legend_actor.GetEntryTextProperty()
        except AttributeError:
            text_property = None

        if text_property is not None:
            try:
                text_property.SetColor(*ERROR_LEGEND_TEXT_COLOR)
            except Exception:
                pass
            try:
                text_property.SetFontSize(ERROR_LEGEND_FONT_SIZE)
            except Exception:
                pass
            try:
                text_property.BoldOff()
            except Exception:
                pass

        self._apply_square_symbols(legend_actor, entry_count)

        if ERROR_LEGEND_USE_BACKGROUND:
            try:
                legend_actor.UseBackgroundOn()
            except Exception:
                pass
            try:
                bg_color = ERROR_LEGEND_BACKGROUND_COLOR
                if len(bg_color) >= 3:
                    background_property = legend_actor.GetBackgroundProperty()
                    if background_property is not None:
                        background_property.SetColor(*bg_color[:3])
                        if len(bg_color) >= 4:
                            background_property.SetOpacity(bg_color[3])
            except Exception:
                pass
        else:
            try:
                legend_actor.UseBackgroundOff()
            except Exception:
                pass

        try:
            legend_actor.Modified()
        except Exception:
            pass

    def _apply_square_symbols(self, legend_actor, entry_count: int) -> None:
        if legend_actor is None or entry_count <= 0:
            return

        try:
            import vtk  # type: ignore
        except ImportError:
            return

        for idx in range(entry_count):
            square_poly = self._create_square_symbol()
            if square_poly is None:
                break
            try:
                legend_actor.SetEntrySymbol(idx, square_poly)
            except Exception:
                break

    @staticmethod
    def _create_square_symbol(edge: float = 0.6):
        try:
            import vtk  # type: ignore
        except ImportError:
            return None

        points = vtk.vtkPoints()
        half = edge / 2.0
        points.InsertNextPoint(-half, -half, 0.0)
        points.InsertNextPoint(half, -half, 0.0)
        points.InsertNextPoint(half, half, 0.0)
        points.InsertNextPoint(-half, half, 0.0)

        cell_array = vtk.vtkCellArray()
        cell_array.InsertNextCell(4)
        for i in range(4):
            cell_array.InsertCellPoint(i)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cell_array)
        return poly_data

    def reload_current_view(self, preserve_camera: bool = True):
        if not (PYVISTA_AVAILABLE and hasattr(self, 'plotter')):
            return
        if not self.sample_files:
            return

        self.keep_camera_state = preserve_camera
        sample_file = self.sample_files[self.current_index]
        sample_name = sample_file.stem
        gt_path = self.gt_dir / f"{sample_name}.png"
        pred_path = self.pred_dir / f"{sample_name}.png"
        
        # Refresh 2D previews BEFORE 3D update to avoid extra rendering
        self.refresh_depth_previews()
        self.update_3d_view(gt_path, pred_path)
    
    def on_gt_opacity_changed(self, value: int):
        """Handle GT opacity slider change."""
        opacity = value / 100.0
        self.gt_opacity_label.setText(f"{value}%")
        
        if PYVISTA_AVAILABLE and hasattr(self, 'plotter'):
            if hasattr(self, 'gt_actor') and self.gt_actor is not None:
                self.gt_actor.GetProperty().SetOpacity(opacity)
            self.plotter.render()
    
    def on_pred_opacity_changed(self, value: int):
        """Handle Pred opacity slider change."""
        opacity = value / 100.0
        self.pred_opacity_label.setText(f"{value}%")
        
        if PYVISTA_AVAILABLE and hasattr(self, 'plotter'):
            if hasattr(self, 'pred_actor') and self.pred_actor is not None:
                self.pred_actor.GetProperty().SetOpacity(opacity)
            self.plotter.render()

    def slider_value_to_scale(self, value: int) -> float:
        """Map 0-100 slider value to exponential zoom scale between 0.1× and 10×."""
        clamped = max(0, min(100, int(value)))
        return 10 ** ((clamped - 50) / 50.0)

    def on_zoom_changed(self, value: int):
        """Handle zoom slider change."""
        if not (PYVISTA_AVAILABLE and hasattr(self, 'plotter')):
            return
        if hasattr(self, 'zoom_label'):
            self.zoom_label.setText(f"{value}%")
        new_scale = self.slider_value_to_scale(value)
        if self._zoom_last_scale is None:
            self._zoom_last_scale = new_scale
            return

        camera_pos = self.plotter.camera_position
        if camera_pos is None:
            self._zoom_last_scale = new_scale
            return

        position = np.array(camera_pos[0], dtype=np.float64)
        focal_point = np.array(camera_pos[1], dtype=np.float64)
        view_up = np.array(camera_pos[2], dtype=np.float64)

        direction = position - focal_point
        distance = np.linalg.norm(direction)
        if distance < 1e-9:
            self._zoom_last_scale = new_scale
            return

        direction_norm = direction / distance
        target_distance = distance * (self._zoom_last_scale / new_scale)
        new_position = focal_point + direction_norm * target_distance

        self.plotter.camera_position = (new_position.tolist(), focal_point.tolist(), view_up.tolist())
        self.plotter.render()

        updated_state = self.plotter.camera_position
        if updated_state is not None:
            self.default_camera = tuple(np.array(component, dtype=np.float64) for component in updated_state)

        self._zoom_last_scale = new_scale
    
    def reset_camera(self):
        """Reset camera to default view."""
        if PYVISTA_AVAILABLE and hasattr(self, 'plotter'):
            self.plotter.reset_camera()
            self.plotter.view_isometric()
            self.plotter.camera.zoom(1.0)
            camera_pos = self.plotter.camera_position
            if camera_pos is not None:
                self.default_camera = tuple(np.array(component, dtype=np.float64) for component in camera_pos)
            if hasattr(self, 'zoom_slider'):
                self.zoom_slider.blockSignals(True)
                self.zoom_slider.setValue(50)
                self.zoom_slider.blockSignals(False)
            if hasattr(self, 'zoom_label'):
                self.zoom_label.setText("50%")
            self._zoom_last_scale = self.slider_value_to_scale(self.zoom_slider.value())
    
    def top_view(self):
        """Set camera to top view."""
        if PYVISTA_AVAILABLE and hasattr(self, 'plotter'):
            self.plotter.view_xy()
            self.plotter.camera.zoom(1.2)
            camera_pos = self.plotter.camera_position
            if camera_pos is not None:
                self.default_camera = tuple(np.array(component, dtype=np.float64) for component in camera_pos)
            if hasattr(self, 'zoom_slider'):
                self.zoom_slider.blockSignals(True)
                self.zoom_slider.setValue(50)
                self.zoom_slider.blockSignals(False)
            if hasattr(self, 'zoom_label'):
                self.zoom_label.setText("50%")
            self._zoom_last_scale = self.slider_value_to_scale(self.zoom_slider.value())
    
    def side_view(self):
        """Set camera to side view."""
        if PYVISTA_AVAILABLE and hasattr(self, 'plotter'):
            self.plotter.view_yz()
            self.plotter.camera.zoom(1.2)
            camera_pos = self.plotter.camera_position
            if camera_pos is not None:
                self.default_camera = tuple(np.array(component, dtype=np.float64) for component in camera_pos)
            if hasattr(self, 'zoom_slider'):
                self.zoom_slider.blockSignals(True)
                self.zoom_slider.setValue(50)
                self.zoom_slider.blockSignals(False)
            if hasattr(self, 'zoom_label'):
                self.zoom_label.setText("50%")
            self._zoom_last_scale = self.slider_value_to_scale(self.zoom_slider.value())
    
    def take_screenshot(self):
        """Save screenshot of 3D view."""
        if not PYVISTA_AVAILABLE or not hasattr(self, 'plotter'):
            return
        
        # Open file dialog
        default_name = f"screenshot_{self.sample_files[self.current_index].stem}.png"
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            str(self.results_dir / default_name),
            "PNG Images (*.png);;All Files (*)"
        )
        
        if filepath:
            self.plotter.screenshot(filepath)
            print(f"Screenshot saved to: {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="3D LiDAR-Camera Projection Comparison Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python visualize_3d_comparison.py --results_dir "output/ResNet-SAN_0.05to100_results"

Expected directory structure:
    results_dir/
        rgb/
            0000000278.png
            ...
        gt/
            0000000278.png
            ...
        pred/
            0000000278.png
            ...
        """
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to results directory containing rgb/, gt/, pred/ folders"
    )
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Check required subdirectories
    missing_dirs = []
    for subdir in ["rgb", "gt", "pred"]:
        if not (results_dir / subdir).exists():
            missing_dirs.append(subdir)
    
    if missing_dirs:
        print(f"Error: Required subdirectories not found: {', '.join(missing_dirs)}")
        print(f"Expected structure:")
        print(f"  {results_dir}/")
        print(f"    rgb/")
        print(f"    gt/")
        print(f"    pred/")
        sys.exit(1)
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    try:
        window = ProjectionComparisonWindow(results_dir)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error creating window: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
