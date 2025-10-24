"""
VADAS 3D Interactive Depth Viewer (Complete Version)
"""

import os
import sys
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import colors # Add this line
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import argparse
import math

# 프로젝트 내 모듈 import
from ref.ref_calibration_data import DEFAULT_CALIB, DEFAULT_LIDAR_TO_CAM

# 테스트용 LiDAR-Camera 변환 행렬 (DEFAULT_LIDAR_TO_CAM 복사 후 Z축 변위 조정)
TEST_LIDAR_TO_CAM = np.array([
    [ 0.9456203,   0.01010798, -0.32511547,  0.293769   ],
    [-0.00881369,  0.99994629,  0.00545355, -0.0542026  ],
    [ 0.32515313, -0.00229153,  0.9456586,  -0.631615   ], # Z축 변위 조정
    [ 0.,          0.,          0.,          1.         ]
])

class VADASDepthMapConverter:
    """VADAS 카메라 모델을 사용한 깊이 맵 -> 3D 포인트 클라우드 변환기"""
    
    def __init__(self, vadas_intrinsic: List[float], image_size: Tuple[int, int], use_direct_lidar_to_cam: bool = True):
        """
        Args:
            vadas_intrinsic: VADAS 모델의 intrinsic 파라미터 (18개)
            image_size: (width, height) 이미지 크기
            use_direct_lidar_to_cam: True면 DEFAULT_LIDAR_TO_CAM 사용, False면 기존 World 경유 방식
        """
        from ref.ref_camera_lidar_projector import VADASFisheyeCameraModel
        
        self.vadas_model = VADASFisheyeCameraModel(vadas_intrinsic, image_size)
        self.image_size = image_size
        self.use_direct_transform = use_direct_lidar_to_cam
        
        if use_direct_lidar_to_cam:
            # 직접 LiDAR → Camera 변환 사용
            self.lidar_to_camera_transform = TEST_LIDAR_TO_CAM
            print(f"VADAS 모델 초기화 (Direct LiDAR→Camera Transform):")
        else:
            # 기존 방식: LiDAR → World → Camera
            from ref.ref_calibration_data import DEFAULT_LIDAR_TO_WORLD_v2
            from ref.ref_camera_lidar_projector import CalibrationDB
            
            calib_db = CalibrationDB(DEFAULT_CALIB, lidar_to_world=DEFAULT_LIDAR_TO_WORLD_v2)
            cam_extrinsic = calib_db.get('a6').extrinsic
            self.lidar_to_camera_transform = cam_extrinsic @ DEFAULT_LIDAR_TO_WORLD_v2
            print(f"VADAS 모델 초기화 (LiDAR→World→Camera Transform):")
        
        print(f"   이미지 크기: {image_size}")
        print(f"   Intrinsic 파라미터: k={vadas_intrinsic[:7]}, s={vadas_intrinsic[7]:.4f}")
        print(f"   Transform Matrix:")
        print(f"     {self.lidar_to_camera_transform[0]}")
        print(f"     {self.lidar_to_camera_transform[1]}")
        print(f"     {self.lidar_to_camera_transform[2]}")
        print(f"     {self.lidar_to_camera_transform[3]}")

    def depth_map_to_point_cloud(
        self,
        depth_map: np.ndarray,
        max_depth: float = 50.0,
        min_depth: float = 0.1,
        downsample_factor: int = 1,
        use_vadas_unprojection: bool = True
    ) -> o3d.geometry.PointCloud:
        """깊이 맵을 VADAS 모델로 3D 포인트 클라우드 변환"""
        h, w = depth_map.shape
        print(f"깊이 맵 변환 시작: {w}x{h} ({'Direct' if self.use_direct_transform else 'World경유'} transform)")
        
        # 다운샘플링 제거 (항상 원본 크기 사용)
        depth_map_small = depth_map
        
        # 유효한 깊이 값 필터링
        valid_mask = (depth_map_small > min_depth) & (depth_map_small < max_depth) & (depth_map_small != 0)
        valid_count = np.sum(valid_mask)
        
        if valid_count == 0:
            print("유효한 깊이 값이 없습니다.")
            return o3d.geometry.PointCloud()
        
        print(f"유효한 픽셀: {valid_count}/{h*w} ({valid_count/(h*w)*100:.1f}%)")
        
        # 픽셀 좌표 생성
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
        u_valid = u_coords[valid_mask]
        v_valid = v_coords[valid_mask]
        depth_valid = depth_map_small[valid_mask]
        
        points_3d = []
        
        if use_vadas_unprojection:
            print("VADAS 역투영 사용 중...")
            points_3d = self._vadas_unproject_approximate(u_valid, v_valid, depth_valid)
        else:
            print("표준 핀홀 모델 사용 중...")
            points_3d = self._pinhole_unproject_approximate(u_valid, v_valid, depth_valid)
        
        if len(points_3d) == 0:
            print("3D 포인트 생성 실패")
            return o3d.geometry.PointCloud()
        
        points_3d = np.array(points_3d)
        print(f"생성된 3D 포인트: {len(points_3d)}개")
        
        # LiDAR 좌표계로 변환
        if self.use_direct_transform:
            # DEFAULT_LIDAR_TO_CAM의 역변환으로 Camera → LiDAR 좌표계
            camera_to_lidar_transform = np.linalg.inv(self.lidar_to_camera_transform)
            
            # [DEBUG] 변환 행렬 출력
            print("\n--- [DEBUG] Coordinate Transformation ---")
            print("LiDAR → Camera Transform (Original):")
            print(self.lidar_to_camera_transform)
            print("\nCamera → LiDAR Transform (Inverse):")
            print(camera_to_lidar_transform)
            print("-------------------------------------\n")
            
            points_3d_hom = np.hstack([points_3d, np.ones((len(points_3d), 1))])
            points_lidar_hom = (camera_to_lidar_transform @ points_3d_hom.T).T
            points_3d_final = points_lidar_hom[:, :3]
            print(f"카메라 → LiDAR 좌표계 변환 적용 (Direct inverse)")
        else:
            points_3d_final = points_3d
            print(f"좌표계 변환 생략 (World 경유 방식)")
        
        # 포인트 클라우드 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d_final)
        
        return pcd

    def _vadas_unproject_approximate(self, u_coords: np.ndarray, v_coords: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """VADAS 모델의 근사적 역투영 (벡터화)"""
        img_w_half = self.image_size[0] / 2
        img_h_half = self.image_size[1] / 2
        
        u_centered = u_coords - img_w_half - self.vadas_model.ux
        v_centered = v_coords - img_h_half - self.vadas_model.uy
        
        rd = np.sqrt(u_centered**2 + v_centered**2)
        
        zero_rd_mask = rd < 1e-6
        
        cosPhi = np.zeros_like(rd)
        sinPhi = np.zeros_like(rd)
        non_zero_rd_mask = ~zero_rd_mask
        cosPhi[non_zero_rd_mask] = u_centered[non_zero_rd_mask] / rd[non_zero_rd_mask]
        sinPhi[non_zero_rd_mask] = v_centered[non_zero_rd_mask] / rd[non_zero_rd_mask]
        
        theta = self._estimate_theta_from_rd(rd)
        
        valid_theta_mask = ~np.isnan(theta)
        
        dist = np.zeros_like(depths)
        calc_dist_mask = valid_theta_mask & non_zero_rd_mask
        dist[calc_dist_mask] = depths[calc_dist_mask] * np.tan(theta[calc_dist_mask])
        
        Xc = depths
        Yc = -dist * cosPhi
        Zc = -dist * sinPhi
        
        points_3d_camera = np.full((len(u_coords), 3), np.nan, dtype=np.float64)

        final_valid_mask = valid_theta_mask & non_zero_rd_mask
        points_3d_camera[final_valid_mask, 0] = Xc[final_valid_mask]
        points_3d_camera[final_valid_mask, 1] = Yc[final_valid_mask]
        points_3d_camera[final_valid_mask, 2] = Zc[final_valid_mask]

        points_3d_camera[zero_rd_mask, 0] = depths[zero_rd_mask]
        points_3d_camera[zero_rd_mask, 1] = 0.0
        points_3d_camera[zero_rd_mask, 2] = 0.0

        points_3d_camera = points_3d_camera[~np.isnan(points_3d_camera).any(axis=1)]
        
        print(f"  VADAS 역투영 완료 (벡터화): {len(points_3d_camera)}/{len(u_coords)}개 성공")
        return points_3d_camera
    
    def _estimate_theta_from_rd(self, rd_array: np.ndarray, max_iter: int = 10) -> np.ndarray:
        """rd 배열에서 theta를 추정하는 반복 방법 (벡터화)"""
        if abs(self.vadas_model.div) < 1e-9:
            return np.full_like(rd_array, np.nan) # Return NaN for all if div is zero

        theta = np.where(self.vadas_model.s != 0, rd_array / self.vadas_model.s, rd_array)
        
        converged_mask = np.zeros_like(rd_array, dtype=bool)

        for _ in range(max_iter):
            xd = theta * self.vadas_model.s
            poly_val = self._poly_eval(self.vadas_model.k, xd)
            poly_deriv = self._poly_deriv_eval(self.vadas_model.k, xd)
            
            unstable_deriv_mask = np.abs(poly_deriv * self.vadas_model.s) < 1e-9
            
            rd_pred = poly_val / self.vadas_model.div
            error = rd_pred - rd_array
            
            current_converged = np.abs(error) < 1e-6
            converged_mask = converged_mask | current_converged | unstable_deriv_mask

            update_mask = ~converged_mask
            if not np.any(update_mask):
                break

            theta_update = np.zeros_like(theta)
            theta_update[update_mask] = error[update_mask] / (poly_deriv[update_mask] * self.vadas_model.s / self.vadas_model.div)
            
            theta = theta - theta_update
            theta = np.clip(theta, 0, math.pi)

        theta = np.where((theta >= 0) & (theta <= math.pi), theta, np.nan)
        return theta
    
    def _poly_eval(self, coeffs: List[float], x: np.ndarray) -> np.ndarray:
        """다항식 계산 (벡터화)"""
        result = np.zeros_like(x, dtype=np.float64)
        for c in reversed(coeffs):
            result = result * x + c
        return result
    
    def _poly_deriv_eval(self, coeffs: List[float], x: np.ndarray) -> np.ndarray:
        """다항식 도함수 계산 (벡터화)"""
        if len(coeffs) <= 1:
            return np.zeros_like(x, dtype=np.float64)
        
        result = np.zeros_like(x, dtype=np.float64)
        for i, c in enumerate(reversed(coeffs[1:]), 1):
            result = result * x + c * i
        return result
    
    def _pinhole_unproject_approximate(self, u_coords: np.ndarray, v_coords: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """표준 핀홀 카메라 모델을 사용한 근사 역투영 (벡터화)"""
        fx = fy = 1000.0
        cx = self.image_size[0] / 2
        cy = self.image_size[1] / 2
        
        x = (u_coords - cx) * depths / fx
        y = (v_coords - cy) * depths / fy
        z = depths
        
        points_3d = np.stack([z, x, y], axis=-1)
        
        return points_3d


class Interactive3DDepthViewer:
    """VADAS 모델을 사용한 3D 인터랙티브 깊이 뷰어"""
    
    def __init__(self, base_data_path: str, use_direct_lidar_to_cam: bool = True, pcd_frame: str = "lidar", scaled_depth_path: Optional[str] = None):
        self.base_data_path = Path(base_data_path)
        self.use_direct_transform = use_direct_lidar_to_cam
        self.pcd_frame = pcd_frame  # [MOD] 원본 PCD의 입력 프레임 지정
        self.vadas_converter = None
        self.scaled_depth_path = scaled_depth_path # 스케일 보정된 Depth 맵 경로
        self._initialize_vadas_system()
        
    def _initialize_vadas_system(self):
        """VADAS 시스템 초기화"""
        try:
            vadas_intrinsic = DEFAULT_CALIB['a6']['intrinsic']
            default_image_size = (1920, 1536)
            self.vadas_converter = VADASDepthMapConverter(
                vadas_intrinsic, 
                default_image_size, 
                use_direct_lidar_to_cam=self.use_direct_transform
            )
            
            print("VADAS 시스템 초기화 완료")
            print(f"   변환 방식: {'Direct LiDAR→Camera' if self.use_direct_transform else 'LiDAR→World→Camera'}")
            
        except Exception as e:
            print(f"VADAS 시스템 초기화 실패: {e}")
            sys.exit(1)

    def load_depth_map_png(self, file_path: str) -> Optional[np.ndarray]:
        """16비트 PNG 깊이 맵 로드"""
        try:
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Error: load_depth_map_png - Could not load image from {file_path}. Image is None.")
                return None
            
            if img.dtype != np.uint16:
                print(f"Warning: {file_path}는 16비트 이미지가 아닐 수 있습니다. dtype: {img.dtype}")
                img = img.astype(np.uint16)
            
            # 실제 이미지 크기로 VADAS 모델 업데이트
            actual_size = (img.shape[1], img.shape[0])
            if self.vadas_converter.image_size != actual_size:
                print(f"[DEBUG] Image size mismatch. Updating VADAS converter: {self.vadas_converter.image_size} -> {actual_size}")
                vadas_intrinsic = DEFAULT_CALIB['a6']['intrinsic']
                self.vadas_converter = VADASDepthMapConverter(
                    vadas_intrinsic, 
                    actual_size, 
                    use_direct_lidar_to_cam=self.use_direct_transform
                )
            
            depth_map_pixels = np.array(img, dtype=np.float32)
            depth_map_meters = depth_map_pixels / 256.0
            
            print(f"깊이 맵 로드 완료: {file_path}")
            print(f"  크기: {actual_size}")
            print(f"  깊이 범위: {depth_map_meters[depth_map_meters > 0].min():.2f} ~ {depth_map_meters.max():.2f} m")
            
            return depth_map_meters
            
        except Exception as e:
            print(f"Error: load_depth_map_png - An exception occurred while loading PNG depth map from {file_path}: {e}")
            return None

    # === Interactive3DDepthViewer.load_original_pcd 수정 ===
    def load_original_pcd(self, pcd_path: str) -> Optional[o3d.geometry.PointCloud]:
        """원본 PCD 파일 로드 후 LiDAR 좌표계로 강제 변환"""
        if not os.path.exists(pcd_path):
            print(f"PCD 파일을 찾을 수 없습니다: {pcd_path}")
            return None
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            if not pcd.has_points():
                print(f"PCD 파일에 포인트가 없습니다: {pcd_path}")
                return None

            original_points = np.asarray(pcd.points)
            print(f"원본 PCD 로드: {len(original_points)} 포인트, 입력 프레임='{self.pcd_frame}'")

            # [MOD] 입력 프레임별 → LiDAR 변환
            points_h = np.hstack([original_points, np.ones((len(original_points), 1))])

            if self.pcd_frame == "lidar":
                transformed_points = original_points  # 그대로
                print("[MOD] PCD 프레임이 LiDAR이므로 변환 없음")

            elif self.pcd_frame == "camera":
                # Camera → LiDAR = inv(LiDAR→Camera)
                cam_to_lidar = np.linalg.inv(self.vadas_converter.lidar_to_camera_transform)
                pts_lidar_h = (cam_to_lidar @ points_h.T).T
                transformed_points = pts_lidar_h[:, :3]
                print("[MOD] Camera→LiDAR 변환 적용")

            elif self.pcd_frame == "world":
                # World → LiDAR = inv(LiDAR→World)
                from ref.ref_calibration_data import DEFAULT_LIDAR_TO_WORLD_v2  # 지연 import
                world_to_lidar = np.linalg.inv(DEFAULT_LIDAR_TO_WORLD_v2)
                pts_lidar_h = (world_to_lidar @ points_h.T).T
                transformed_points = pts_lidar_h[:, :3]
                print("[MOD] World→LiDAR 변환 적용")

            else:
                print(f"[WARN] 알 수 없는 pcd_frame='{self.pcd_frame}', 변환 생략")
                transformed_points = original_points

            pcd.points = o3d.utility.Vector3dVector(transformed_points)

            # 변환 결과 출력
            original_center = np.mean(original_points, axis=0)
            transformed_center = np.mean(transformed_points, axis=0)
            print(f"  원본 PCD 중심({self.pcd_frame}): [{original_center[0]:.2f}, {original_center[1]:.2f}, {original_center[2]:.2f}]")
            print(f"  변환된 PCD 중심(LiDAR): [{transformed_center[0]:.2f}, {transformed_center[1]:.2f}, {transformed_center[2]:.2f}]")
            return pcd
        except Exception as e:
            print(f"PCD 로드 중 오류: {e}")
            return None

    def show_vadas_3d_viewer(
        self, 
        depth_map_path: str, 
        pcd_path: Optional[str] = None, 
        max_depth: float = 50.0, 
        use_vadas_unprojection: bool = True, 
        downsample_factor: int = 1
    ):
        """VADAS 3D 뷰어"""
        print(f"=== VADAS 3D 뷰어 ({'Direct' if self.use_direct_transform else 'World경유'} Transform) ===")
        
        depth_map = self.load_depth_map_png(depth_map_path)
        if depth_map is None:
            print("Depth map 로드 실패")
            return
        
        print(f"\n{'VADAS 역투영' if use_vadas_unprojection else '핀홀 근사'} 사용")
        pcd_from_depth = self.vadas_converter.depth_map_to_point_cloud(
            depth_map,
            max_depth=max_depth,
            downsample_factor=downsample_factor,
            use_vadas_unprojection=use_vadas_unprojection
        )
        
        if not pcd_from_depth.has_points():
            print("Depth map에서 포인트 생성 실패")
            return
        
        print(f"Depth map에서 생성된 포인트 수: {len(pcd_from_depth.points)}")
        
        geometries = [pcd_from_depth]
        
        # 원본 PCD 추가
        if pcd_path:
            original_pcd = self.load_original_pcd(pcd_path)
            if original_pcd is not None:
                original_pcd.paint_uniform_color([0.7, 0.7, 0.7])
                geometries.append(original_pcd)
        
        # 좌표계 표시
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        geometries.append(coordinate_frame)
        
        print("3D 뷰어 시작...")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"VADAS 3D Depth Viewer ({'Direct' if self.use_direct_transform else 'World경유'} {'VADAS' if use_vadas_unprojection else 'Pinhole'} projection)",
            width=1600,
            height=900
        )

    def show_interactive_advanced_viewer(
        self,
        depth_map_path: str,
        pcd_path: Optional[str] = None,
        max_depth: float = 50.0,
        downsample_factor: int = 1
    ):
        """고급 인터랙티브 뷰어 (키보드 제어 + 토글 기능)"""
        print("=== 고급 VADAS 인터랙티브 뷰어 ===")
        
        class VADASAdvancedViewer:
            def __init__(self, parent, max_depth: float):
                self.parent = parent
                self.vis = o3d.visualization.VisualizerWithKeyCallback()
                self.depth_map = None
                self.vadas_pcd = None
                self.original_pcd = None
                self.coordinate_frame = None
                self.scaled_depth_pcd = None # 스케일 보정된 Depth PCD
                self.max_depth = max_depth # Store max_depth
                
                # 표시 상태
                self.show_vadas = True
                self.show_original = True
                self.show_coordinate = True
                self.show_scaled_depth = False # 스케일 보정된 Depth PCD 표시 상태
                
                # 컬러맵 세트 정의
                self.colormap_sets = [
                    # Set 0: Default uniform colors
                    {'name': 'Uniform Colors', 'vadas': lambda x: np.array([1, 0, 0]), 'scaled': lambda x: np.array([0, 1, 0]), 'original': lambda x: np.array([0.7, 0.7, 0.7])},
                    # Set 1: Gradient colors (Jet, Viridis, Plasma)
                    {'name': 'Gradient Set 1 (Jet, Viridis, Plasma)', 'vadas': plt.cm.jet, 'scaled': plt.cm.viridis, 'original': plt.cm.plasma},
                    # Set 2: Gradient colors (Magma, Cividis, Greys)
                    {'name': 'Gradient Set 2 (Magma, Cividis, Greys)', 'vadas': plt.cm.magma, 'scaled': plt.cm.cividis, 'original': plt.cm.Greys},
                ]
                self.current_colormap_set_idx = 0
                
            def toggle_vadas(self, vis):
                print(f"VADAS PCD 토글: {'OFF' if self.show_vadas else 'ON'}")
                if self.vadas_pcd is not None:
                    if self.show_vadas:
                        vis.remove_geometry(self.vadas_pcd, reset_bounding_box=False)
                    else:
                        vis.add_geometry(self.vadas_pcd, reset_bounding_box=False)
                    self.show_vadas = not self.show_vadas
                return False
                
            def toggle_original(self, vis):
                print(f"원본 PCD 토글: {'OFF' if self.show_original else 'ON'}")
                if self.original_pcd is not None:
                    if self.show_original:
                        vis.remove_geometry(self.original_pcd, reset_bounding_box=False)
                    else:
                        vis.add_geometry(self.original_pcd, reset_bounding_box=False)
                    self.show_original = not self.show_original
                return False
                
            def toggle_scaled_depth(self, vis):
                print(f"스케일 보정된 Depth PCD 토글: {'OFF' if self.show_scaled_depth else 'ON'}")
                if self.scaled_depth_pcd is not None:
                    if self.show_scaled_depth:
                        vis.remove_geometry(self.scaled_depth_pcd, reset_bounding_box=False)
                    else:
                        vis.add_geometry(self.scaled_depth_pcd, reset_bounding_box=False)
                    self.show_scaled_depth = not self.show_scaled_depth
                return False
                
            def toggle_coordinate(self, vis):
                print(f"좌표계 토글: {'OFF' if self.show_coordinate else 'ON'}")
                if self.coordinate_frame is not None:
                    if self.show_coordinate:
                        vis.remove_geometry(self.coordinate_frame, reset_bounding_box=False)
                    else:
                        vis.add_geometry(self.coordinate_frame, reset_bounding_box=False)
                    self.show_coordinate = not self.show_coordinate
                return False
                
            def reset_view(self, vis):
                print("뷰 리셋")
                vis.reset_view_point(True)
                return False
                
            def change_background(self, vis):
                opt = vis.get_render_option()
                current_bg = opt.background_color
                if np.allclose(current_bg, [0.1, 0.1, 0.1]):
                    opt.background_color = np.asarray([1, 1, 1])
                    print("배경: 흰색")
                else:
                    opt.background_color = np.asarray([0.1, 0.1, 0.1])
                    print("배경: 어두운색")
                return False
            
            def _apply_colormap_to_pcd(self, pcd: o3d.geometry.PointCloud, colormap_func, min_depth: float, max_depth: float):
                if pcd is None or not pcd.has_points():
                    return

                points = np.asarray(pcd.points)
                distances = np.linalg.norm(points, axis=1)
                normalized_depth = (distances - min_depth) / (max_depth - min_depth)
                normalized_depth = np.clip(normalized_depth, 0, 1)

                normalized_depth = np.clip(normalized_depth, 0, 1)

                # Check if colormap_func is a matplotlib colormap instance
                if isinstance(colormap_func, colors.Colormap):
                    colors_array = colormap_func(normalized_depth)[:, :3]
                elif callable(colormap_func):
                    # It's a custom callable (e.g., a lambda for uniform color)
                    # Assume it returns a single color (RGB or RGBA)
                    single_color = np.asarray(colormap_func(0)) # Pass a dummy value, as it's for uniform color
                    if single_color.shape == (): # If it returns a scalar, assume grayscale
                        single_color = np.array([single_color, single_color, single_color])
                    elif single_color.shape == (4,): # If it returns RGBA, take RGB
                        single_color = single_color[:3]
                    
                    colors_array = np.tile(single_color, (len(points), 1))
                else:
                    # Fallback for unexpected colormap_func type, e.g., just use a default color
                    colors_array = np.tile(np.array([0.5, 0.5, 0.5]), (len(points), 1)) # Grey fallback
                
                pcd.colors = o3d.utility.Vector3dVector(colors_array)
                self.vis.update_geometry(pcd)
                self.vis.update_renderer()
                
            def cycle_colormaps(self, vis):
                self.current_colormap_set_idx = (self.current_colormap_set_idx + 1) % len(self.colormap_sets)
                current_set = self.colormap_sets[self.current_colormap_set_idx]
                print(f"컬러맵 세트 변경: {current_set['name']}")

                max_depth_val = self.max_depth
                min_depth_val = 0.1 # Hardcoded in depth_map_to_point_cloud, so use it here too.

                # Apply to VADAS PCD
                if self.vadas_pcd is not None:
                    self._apply_colormap_to_pcd(self.vadas_pcd, current_set['vadas'], min_depth=min_depth_val, max_depth=max_depth_val)
                
                # Apply to Scaled Depth PCD
                if self.scaled_depth_pcd is not None:
                    self._apply_colormap_to_pcd(self.scaled_depth_pcd, current_set['scaled'], min_depth=min_depth_val, max_depth=max_depth_val)

                # Apply to Original PCD
                if self.original_pcd is not None:
                    self._apply_colormap_to_pcd(self.original_pcd, current_set['original'], min_depth=min_depth_val, max_depth=max_depth_val)
                
                vis.update_renderer()
                return False
                
            def run(self):
                # 데이터 로드
                self.depth_map = self.parent.load_depth_map_png(depth_map_path)
                if self.depth_map is None:
                    print("Depth map 로드 실패")
                    return
                
                # 윈도우 생성
                self.vis.create_window("Advanced VADAS 3D Viewer", width=1800, height=1000)
                
                self.vis.register_key_callback(ord("1"), self.toggle_vadas)
                self.vis.register_key_callback(ord("2"), self.toggle_original)
                self.vis.register_key_callback(ord("3"), self.toggle_scaled_depth) # 새 토글
                self.vis.register_key_callback(ord("C"), self.toggle_coordinate)
                self.vis.register_key_callback(ord("R"), self.reset_view)
                self.vis.register_key_callback(ord("B"), self.change_background)
                self.vis.register_key_callback(ord("5"), self.cycle_colormaps) # New callback
                
                # VADAS PCD 생성 및 추가
                print("VADAS PCD 생성 중 (전체 해상도)...")
                self.vadas_pcd = self.parent.vadas_converter.depth_map_to_point_cloud(
                    self.depth_map, max_depth=max_depth, 
                    downsample_factor=1, use_vadas_unprojection=True
                )
                if self.vadas_pcd.has_points():
                    self._apply_colormap_to_pcd(self.vadas_pcd, self.colormap_sets[self.current_colormap_set_idx]['vadas'], min_depth=0.1, max_depth=max_depth) # Apply initial colormap
                    self.vis.add_geometry(self.vadas_pcd)
                    print(f"VADAS PCD 추가: {len(self.vadas_pcd.points)} 포인트")
                
                # 원본 PCD 로드
                if pcd_path:
                    self.original_pcd = self.parent.load_original_pcd(pcd_path)
                    if self.original_pcd is not None:
                        # Apply initial colormap for original PCD
                        self._apply_colormap_to_pcd(self.original_pcd, self.colormap_sets[self.current_colormap_set_idx]['original'], min_depth=0.1, max_depth=max_depth)
                        self.vis.add_geometry(self.original_pcd)
                        print(f"원본 PCD 추가: {len(self.original_pcd.points)} 포인트")
                
                # 스케일 보정된 Depth PCD 로드 및 추가
                if self.parent.scaled_depth_path:
                    print(f"[DEBUG] Attempting to load scaled depth map from: {self.parent.scaled_depth_path}")
                    scaled_depth_map = self.parent.load_depth_map_png(self.parent.scaled_depth_path)
                    if scaled_depth_map is not None:
                        print("스케일 보정된 Depth PCD 생성 중...")
                        self.scaled_depth_pcd = self.parent.vadas_converter.depth_map_to_point_cloud(
                            scaled_depth_map, max_depth=max_depth,
                            downsample_factor=1, use_vadas_unprojection=True
                        )
                        if self.scaled_depth_pcd.has_points():
                            self._apply_colormap_to_pcd(self.scaled_depth_pcd, self.colormap_sets[self.current_colormap_set_idx]['scaled'], min_depth=0.1, max_depth=max_depth) # Apply initial colormap
                            self.vis.add_geometry(self.scaled_depth_pcd)
                            print(f"스케일 보정된 Depth PCD 추가: {len(self.scaled_depth_pcd.points)} 포인트")
                        else:
                            print("스케일 보정된 Depth PCD에 포인트가 없습니다.")
                    else:
                        print("스케일 보정된 Depth 맵 로드 실패 (scaled_depth_map is None).")
                else:
                    print("scaled_depth_path가 설정되지 않았습니다. 스케일 보정된 Depth PCD를 로드하지 않습니다.")
                
                # 좌표계 추가
                self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4.0)
                self.vis.add_geometry(self.coordinate_frame)
                
                # 렌더링 옵션 설정
                opt = self.vis.get_render_option()
                opt.background_color = np.asarray([0.1, 0.1, 0.1])
                opt.point_size = 2.5
                opt.show_coordinate_frame = True
                
                # 초기 뷰 설정
                view_control = self.vis.get_view_control()
                view_control.set_front([1, 0, 0])
                view_control.set_lookat([0, 0, 0])
                view_control.set_up([0, 0, 1])
                view_control.set_zoom(0.3)
                
                print("\n=== 키보드 단축키 ===")
                print("1: VADAS PCD 토글 (빨간색)")
                print("2: 원본 PCD 토글 (회색)")
                print("3: 스케일 보정된 Depth PCD 토글 (초록색)")
                print("5: 컬러맵 세트 변경 (각 PCD별 다른 컬러맵)") # UPDATED
                print("C: 좌표계 토글")
                print("R: 뷰 리셋")
                print("B: 배경색 변경")
                print("ESC: 종료")
                print("==================")
                print("💡 각 PCD를 개별적으로 켜고 끌 수 있습니다!")
                
                # 메인 루프
                self.vis.run()
                self.vis.destroy_window()
        
        viewer = VADASAdvancedViewer(self, max_depth=max_depth)
        viewer.run()

    def show_transform_comparison_viewer(
        self,
        depth_map_path: str,
        pcd_path: Optional[str] = None,
        max_depth: float = 50.0,
        downsample_factor: int = 1
    ):
        """Direct vs World경유 변환 비교 뷰어"""
        print("=== Direct Transform vs World경유 Transform 비교 뷰어 (전체 해상도) ===")
        
        depth_map = self.load_depth_map_png(depth_map_path)
        if depth_map is None:
            print("Depth map 로드 실패")
            return
        
        # Direct Transform PCD 생성
        print("\n1. Direct LiDAR→Camera Transform (전체 해상도)...")
        direct_converter = VADASDepthMapConverter(
            DEFAULT_CALIB['a6']['intrinsic'], 
            (depth_map.shape[1], depth_map.shape[0]),
            use_direct_lidar_to_cam=True
        )
        direct_pcd = direct_converter.depth_map_to_point_cloud(
            depth_map,
            max_depth=max_depth,
            downsample_factor=1,
            use_vadas_unprojection=True
        )
        
        # World경유 Transform PCD 생성
        print("\n2. LiDAR→World→Camera Transform (전체 해상도)...")
        world_converter = VADASDepthMapConverter(
            DEFAULT_CALIB['a6']['intrinsic'],
            (depth_map.shape[1], depth_map.shape[0]),
            use_direct_lidar_to_cam=False
        )
        world_pcd = world_converter.depth_map_to_point_cloud(
            depth_map,
            max_depth=max_depth,
            downsample_factor=1,
            use_vadas_unprojection=True
        )
        
        geometries = []
        
        # Direct Transform PCD (빨간색)
        if direct_pcd.has_points():
            direct_pcd.paint_uniform_color([1, 0, 0])
            geometries.append(direct_pcd)
            print(f"Direct Transform PCD: {len(direct_pcd.points)} 포인트 (빨간색)")
        
        # World경유 Transform PCD (파란색)
        if world_pcd.has_points():
            world_pcd.paint_uniform_color([0, 0, 1])
            geometries.append(world_pcd)
            print(f"World경유 Transform PCD: {len(world_pcd.points)} 포인트 (파란색)")
        
        # 원본 PCD (회색)
        if pcd_path:
            original_pcd = self.load_original_pcd(pcd_path)
            if original_pcd is not None:
                original_pcd.paint_uniform_color([0.7, 0.7, 0.7])
                geometries.append(original_pcd)
                print(f"원본 LiDAR PCD: {len(original_pcd.points)} 포인트 (회색)")
        
        # 좌표계
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        geometries.append(coordinate_frame)
        
        print("\n색상 구분:")
        print("  빨간색: Direct LiDAR→Camera Transform")
        print("  파란색: LiDAR→World→Camera Transform") 
        print("  회색: 원본 LiDAR PCD")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Transform Method Comparison (Direct vs World경유) - Full Resolution",
            width=1600,
            height=900
        )

    def show_coordinate_analysis(
        self,
        depth_map_path: str,
        pcd_path: str,
        max_depth: float = 50.0
    ):
        """좌표계 분석 및 비교"""
        print("=== 좌표계 분석 ===")
        
        # Depth Map PCD 생성
        depth_map = self.load_depth_map_png(depth_map_path)
        if depth_map is None:
            print("Depth map 로드 실패")
            return
        
        depth_pcd = self.vadas_converter.depth_map_to_point_cloud(
            depth_map, max_depth=max_depth, downsample_factor=1, use_vadas_unprojection=True
        )
        
        # 원본 PCD 로드
        original_pcd = self.load_original_pcd(pcd_path)
        
        if not depth_pcd.has_points() or original_pcd is None or not original_pcd.has_points():
            print("PCD 로드 실패")
            return
        
        # 좌표 분석
        depth_points = np.asarray(depth_pcd.points)
        original_points = np.asarray(original_pcd.points)
        
        print("\n=== 좌표 통계 ===")
        print("Depth PCD:")
        print(f"  포인트 수: {len(depth_points)}")
        print(f"  중심: [{np.mean(depth_points, axis=0)}]")
        print(f"  X 범위: {depth_points[:, 0].min():.2f} ~ {depth_points[:, 0].max():.2f}")
        print(f"  Y 범위: {depth_points[:, 1].min():.2f} ~ {depth_points[:, 1].max():.2f}")
        print(f"  Z 범위: {depth_points[:, 2].min():.2f} ~ {depth_points[:, 2].max():.2f}")
        
        print("\n원본 PCD (변환 후):")
        print(f"  포인트 수: {len(original_points)}")
        print(f"  중심: [{np.mean(original_points, axis=0)}]")
        print(f"  X 범위: {original_points[:, 0].min():.2f} ~ {original_points[:, 0].max():.2f}")
        print(f"  Y 범위: {original_points[:, 1].min():.2f} ~ {original_points[:, 1].max():.2f}")
        print(f"  Z 범위: {original_points[:, 2].min():.2f} ~ {original_points[:, 2].max():.2f}")
        
        print("\n중심점 차이:")
        center_diff = np.mean(depth_points, axis=0) - np.mean(original_points, axis=0)
        print(f"  ΔX: {center_diff[0]:.3f}m")
        print(f"  ΔY: {center_diff[1]:.3f}m") 
        print(f"  ΔZ: {center_diff[2]:.3f}m")
        
        # 3D 시각화
        geometries = []
        
        # Depth PCD (빨간색)
        depth_pcd.paint_uniform_color([1, 0, 0])
        geometries.append(depth_pcd)
        
        # 원본 PCD (회색)
        original_pcd.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(original_pcd)
        
        # 좌표계
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        geometries.append(coordinate_frame)
        
        print("\n색상 구분:")
        print("  빨간색: Depth Map에서 생성된 PCD")
        print("  회색: 원본 LiDAR PCD")
        print("  좌표계: X(빨강), Y(초록), Z(파랑)")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Coordinate System Analysis - After Transform",
            width=1600,
            height=900
        )


def main():
    parser = argparse.ArgumentParser(description="VADAS 3D Interactive Depth Viewer with Coordinate Analysis")
    parser.add_argument("--base_path", 
                       default=r"C:\Users\seok436\Documents\VSCode\Projects\point-cloud-creation\point-cloud-creation\ncdb-cls-sample\synced_data",
                       help="Base data directory path")
    parser.add_argument("--filename", default="0000000931.pcd", 
                       help="PCD filename to visualize")
    parser.add_argument("--max_depth", type=float, default=50.0,
                       help="Maximum depth for visualization")
    parser.add_argument("--downsample", type=int, default=1,
                       help="Downsample factor for performance (1=no downsampling)")
    parser.add_argument("--mode", choices=["direct", "world", "comparison", "advanced", "analysis"], 
                       default="advanced", help="Viewer mode")
    parser.add_argument("--use_vadas", action="store_true", default=True,
                       help="Use VADAS unprojection (vs pinhole approximation)")
    parser.add_argument("--pcd_frame", choices=["lidar", "camera", "world"], 
                        default="world",
                        help="원본 PCD의 현재 좌표계(frames). 뷰어에서는 LiDAR로 통일됨.")
    parser.add_argument("--scaled_depth_path", type=str, default=None,
                       help="스케일 보정된 Depth 맵 PNG 파일 경로 (선택 사항)")

    
    args = parser.parse_args()
    
    # 경로 설정
    base_path = args.base_path
    pcd_filename = args.filename
    
    depth_map_path = os.path.join(base_path, "depth_maps", pcd_filename.replace(".pcd", ".png"))
    pcd_path = os.path.join(base_path, "pcd", pcd_filename)

    # scaled_depth_path 기본값 설정 (명시되지 않은 경우)
    if args.scaled_depth_path is None:
        # 현재 스크립트 파일의 디렉토리 (프로젝트 루트)를 기준으로 output 폴더의 특정 파일을 기본값으로 설정
        default_scaled_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "output", "distribution_analysis_0000000931", "matched_depth_0000000931.png"
        )
        args.scaled_depth_path = default_scaled_path
        print(f"[INFO] --scaled_depth_path가 설정되지 않아 기본값으로 설정되었습니다: {args.scaled_depth_path}")
    
    # 파일 존재 확인
    if not os.path.exists(depth_map_path):
        print(f"Depth map 파일을 찾을 수 없습니다: {depth_map_path}")
        sys.exit(1)
    
    print(f"--- [DEBUG] Main Function Paths ---")
    print(f"  Resolved Depth Map Path: {depth_map_path}")
    print(f"  Resolved PCD File Path: {pcd_path}")
    print(f"  Scaled Depth Path Argument: {args.scaled_depth_path}")
    print(f"-----------------------------------")

    print(f"Depth Map: {depth_map_path}")
    print(f"PCD File: {pcd_path if os.path.exists(pcd_path) else '없음'}")
    print(f"다운샘플링: {'없음 (전체 해상도)' if args.downsample == 1 else f'{args.downsample}x'}")
    
    try:
        if args.mode == "direct":
            viewer = Interactive3DDepthViewer(base_path, use_direct_lidar_to_cam=True, pcd_frame=args.pcd_frame, scaled_depth_path=args.scaled_depth_path)
            viewer.show_vadas_3d_viewer(
                depth_map_path, 
                pcd_path if os.path.exists(pcd_path) else None,
                args.max_depth,
                args.use_vadas,
                args.downsample
            )
        elif args.mode == "world":
            viewer = Interactive3DDepthViewer(base_path, use_direct_lidar_to_cam=False, pcd_frame=args.pcd_frame, scaled_depth_path=args.scaled_depth_path)
            viewer.show_vadas_3d_viewer(
                depth_map_path,
                pcd_path if os.path.exists(pcd_path) else None, 
                args.max_depth,
                args.use_vadas,
                args.downsample
            )
        elif args.mode == "comparison":
            viewer = Interactive3DDepthViewer(base_path, use_direct_lidar_to_cam=True, pcd_frame=args.pcd_frame, scaled_depth_path=args.scaled_depth_path)
            if not os.path.exists(pcd_path):
                print("비교 모드는 원본 PCD 파일이 필요합니다.")
                sys.exit(1)
            viewer.show_transform_comparison_viewer(
                depth_map_path,
                pcd_path,
                args.max_depth,
                args.downsample
            )
        elif args.mode == "advanced":
            viewer = Interactive3DDepthViewer(base_path, use_direct_lidar_to_cam=True, pcd_frame=args.pcd_frame, scaled_depth_path=args.scaled_depth_path)
            viewer.show_interactive_advanced_viewer(
                depth_map_path,
                pcd_path if os.path.exists(pcd_path) else None,
                args.max_depth,
                args.downsample
            )
        elif args.mode == "analysis":
            viewer = Interactive3DDepthViewer(base_path, use_direct_lidar_to_cam=True, pcd_frame=args.pcd_frame, scaled_depth_path=args.scaled_depth_path)
            if not os.path.exists(pcd_path):
                print("분석 모드는 원본 PCD 파일이 필요합니다.")
                sys.exit(1)
            viewer.show_coordinate_analysis(
                depth_map_path,
                pcd_path,
                args.max_depth
            )
            
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()