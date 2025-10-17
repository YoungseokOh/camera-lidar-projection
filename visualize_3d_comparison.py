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
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFileDialog, QSpinBox, QGroupBox, QSlider,
    QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: pyvista or pyvistaqt not installed.")
    print("Install with: pip install pyvista pyvistaqt")

from calibration_data import DEFAULT_LIDAR_TO_CAM, DEFAULT_LIDAR_TO_WORLD_v1


class ProjectionComparisonWindow(QMainWindow):
    """Main window for comparing LiDAR projection results."""
    
    def __init__(self, results_dir: Path):
        super().__init__()
        self.results_dir = results_dir
        self.rgb_dir = results_dir / "rgb"
        self.gt_dir = results_dir / "gt"
        self.pred_dir = results_dir / "pred"
        
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
        self.setGeometry(100, 100, 2600, 1200)
        
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
        panel.setMaximumWidth(680)
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
        self.rgb_label.setFixedSize(640, 384)
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
        self.gt_label.setFixedSize(640, 384)
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
        self.pred_label.setFixedSize(640, 384)
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
        visibility_layout = QHBoxLayout()
        self.show_gt_checkbox = QCheckBox("Show GT")
        self.show_gt_checkbox.setChecked(True)
        self.show_gt_checkbox.stateChanged.connect(self.on_visibility_changed)
        visibility_layout.addWidget(self.show_gt_checkbox)
        
        self.show_pred_checkbox = QCheckBox("Show Pred")
        self.show_pred_checkbox.setChecked(True)
        self.show_pred_checkbox.stateChanged.connect(self.on_visibility_changed)
        visibility_layout.addWidget(self.show_pred_checkbox)
        
        self.show_axes_checkbox = QCheckBox("Show Axes")
        self.show_axes_checkbox.setChecked(True)
        self.show_axes_checkbox.stateChanged.connect(self.on_visibility_changed)
        visibility_layout.addWidget(self.show_axes_checkbox)
        
        visibility_layout.addStretch()
        control_layout.addLayout(visibility_layout)
        
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
        self.plotter.setMinimumSize(1920, 1000)
        self.plotter.set_background('black')
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
        self.update_image_label(self.gt_label, gt_path)
        self.update_image_label(self.pred_label, pred_path)
        
        # Update 3D view
        if PYVISTA_AVAILABLE:
            self.update_3d_view(gt_path, pred_path)
        
        # Update window title
        self.setWindowTitle(f"3D Comparison - {sample_name} ({self.current_index + 1}/{len(self.sample_files)})")
    
    def update_image_label(self, label: QLabel, image_path: Path):
        """Update QLabel with image from path."""
        if not image_path.exists():
            label.setText(f"Not found:\n{image_path.name}")
            label.setStyleSheet(label.styleSheet() + " color: red;")
            return
        
        try:
            pixmap = QPixmap(str(image_path))
            label.setPixmap(pixmap)
        except Exception as e:
            label.setText(f"Error loading:\n{image_path.name}\n{str(e)}")
            label.setStyleSheet(label.styleSheet() + " color: red;")
    
    def update_3d_view(self, gt_path: Path, pred_path: Path):
        """Update 3D viewer with GT and Pred as textured planes in World coordinates."""
        self.plotter.clear()
        
        # Load images as numpy arrays
        gt_img = None
        pred_img = None
        
        if gt_path.exists():
            gt_img = np.array(Image.open(gt_path).convert('RGB'))
        
        if pred_path.exists():
            pred_img = np.array(Image.open(pred_path).convert('RGB'))
        
        if gt_img is None and pred_img is None:
            self.plotter.add_text("No images available", position='center', font_size=20, color='red')
            return
        
        # Calculate camera position in world coordinates
        # Camera extrinsic gives us the camera pose in world frame
        # We'll place the image planes at the camera's position facing forward
        
        # For visualization, place image plane at camera position
        # Use DEFAULT_LIDAR_TO_WORLD_v1 to understand the world coordinate system
        # Camera is at approximately [0.037, 0.034, 0.77] in world coords
        camera_position = np.array([0.037, 0.034, 0.77])
        
        # Image plane normal (camera looks in -Z direction in camera frame)
        # In world frame, we need to rotate this by the camera rotation
        # For now, place planes perpendicular to camera's forward direction
        
        # Create textured planes at the same location (overlapped)
        # Both at camera position, facing camera's viewing direction
        if gt_img is not None and self.show_gt_checkbox.isChecked():
            gt_plane = self.create_textured_plane_world(
                gt_img, 
                position=camera_position, 
                name="GT"
            )
            self.gt_actor = self.plotter.add_mesh(
                gt_plane,
                name="GT_plane",
                opacity=self.gt_opacity_slider.value() / 100.0,
                show_edges=False
            )
        
        if pred_img is not None and self.show_pred_checkbox.isChecked():
            # Place Pred slightly offset in Z to avoid z-fighting
            pred_position = camera_position + np.array([0, 0, 0.01])
            pred_plane = self.create_textured_plane_world(
                pred_img, 
                position=pred_position, 
                name="Pred"
            )
            self.pred_actor = self.plotter.add_mesh(
                pred_plane,
                name="Pred_plane",
                opacity=self.pred_opacity_slider.value() / 100.0,
                show_edges=False
            )
        
        # Add LiDAR origin
        lidar_origin = pv.Sphere(radius=0.05, center=[0, 0, 0])
        self.plotter.add_mesh(lidar_origin, color='yellow', label='LiDAR Origin')
        
        # Add camera origin
        cam_origin = pv.Sphere(radius=0.03, center=camera_position)
        self.plotter.add_mesh(cam_origin, color='cyan', label='Camera')
        
        # Add coordinate axes at LiDAR origin
        if self.show_axes_checkbox.isChecked():
            self.plotter.add_axes_at_origin(
                xlabel='X (forward)',
                ylabel='Y (left)',
                zlabel='Z (up)',
                line_width=5,
                labels_off=False
            )
        
        # Add text labels
        if self.show_gt_checkbox.isChecked():
            self.plotter.add_text("GT (Green)", position='upper_left', font_size=14, color='green')
        if self.show_pred_checkbox.isChecked():
            self.plotter.add_text("Pred (Red)", position='upper_right', font_size=14, color='red')
        
        # Add world coordinate info
        info_text = f"World Coordinates (LiDAR frame)\nCamera at: [{camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}]"
        self.plotter.add_text(info_text, position='lower_left', font_size=10, color='white')
        
        # Reset camera to show the scene
        self.reset_camera()
    
    def create_textured_plane_world(self, image_np: np.ndarray, position: np.ndarray, name: str) -> pv.PolyData:
        """Create a textured plane from image numpy array in world coordinates."""
        h, w = image_np.shape[:2]
        aspect_ratio = w / h
        
        # Create plane representing the camera's image plane
        # Camera coordinate system: X right, Y down, Z forward
        # The image plane is perpendicular to Z axis (forward direction)
        
        # In world coordinates, we need to rotate the plane according to camera extrinsic
        # For visualization, create a plane facing the camera's viewing direction
        
        # Plane size in meters (approximate FOV visualization)
        plane_width = 2.0
        plane_height = plane_width / aspect_ratio
        
        # Create plane in local coordinates (perpendicular to Z-axis)
        plane = pv.Plane(
            center=[0, 0, 0],
            direction=[0, 0, 1],  # Normal pointing forward (Z-axis)
            i_size=plane_width,
            j_size=plane_height,
            i_resolution=10,
            j_resolution=10
        )
        
        # Apply rotation based on camera extrinsic
        # Camera looks in +Z direction in camera frame
        # Rotate plane to match camera orientation in world frame
        
        # For VADAS camera, approximate rotation (simplified)
        # Camera is roughly looking forward-down
        plane.rotate_x(-10, inplace=True)  # Tilt down slightly
        
        # Translate to camera position
        plane.translate(position, inplace=True)
        
        # Map texture coordinates
        plane.texture_map_to_plane(inplace=True)
        
        # Convert image to texture
        texture = pv.numpy_to_texture(image_np)
        plane.textures[name] = texture
        
        return plane
    
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
        if PYVISTA_AVAILABLE and hasattr(self, 'plotter'):
            # Reload the 3D view with new visibility settings
            sample_file = self.sample_files[self.current_index]
            sample_name = sample_file.stem
            gt_path = self.gt_dir / f"{sample_name}.png"
            pred_path = self.pred_dir / f"{sample_name}.png"
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
    
    def on_spacing_changed(self, value: int):
        """Handle spacing slider change."""
        spacing = value / 10.0
        self.spacing_label.setText(f"{spacing:.1f}")
        
        if PYVISTA_AVAILABLE and hasattr(self, 'plotter'):
            # Reload the 3D view with new spacing
            sample_file = self.sample_files[self.current_index]
            sample_name = sample_file.stem
            gt_path = self.gt_dir / f"{sample_name}.png"
            pred_path = self.pred_dir / f"{sample_name}.png"
            self.update_3d_view(gt_path, pred_path)
    
    def reset_camera(self):
        """Reset camera to default view."""
        if PYVISTA_AVAILABLE and hasattr(self, 'plotter'):
            self.plotter.reset_camera()
            # Set camera to view from behind and slightly above
            self.plotter.camera.position = (0, -2.0, 1.5)
            self.plotter.camera.focal_point = (0.037, 0.034, 0.77)
            self.plotter.camera.up = (0, 0, 1)
            self.plotter.camera.zoom(1.0)
    
    def top_view(self):
        """Set camera to top view."""
        if PYVISTA_AVAILABLE and hasattr(self, 'plotter'):
            self.plotter.view_xy()
            self.plotter.camera.zoom(1.2)
    
    def side_view(self):
        """Set camera to side view."""
        if PYVISTA_AVAILABLE and hasattr(self, 'plotter'):
            self.plotter.view_yz()
            self.plotter.camera.zoom(1.2)
    
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
