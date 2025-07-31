

import os
import json
import numpy as np
from PIL import Image
import open3d as o3d

def analyze_depth_map_png(file_path, output_stream):
    """16비트 그레이스케일 PNG 깊이 맵을 분석합니다."""
    try:
        img = Image.open(file_path)
        # KITTI 표준에 따라 16비트 그레이스케일 (I;16) 모드를 확인합니다.
        if img.mode != 'I;16':
            print(f"  경고: {file_path}는 16비트 그레이스케일 PNG가 아닐 수 있습니다. 현재 모드: {img.mode}", file=output_stream)
        
        # NumPy 배열로 변환합니다.
        pixel_values = np.array(img)

        # 0 값은 유효하지 않은 깊이를 나타내는 경우가 많으므로 분석에서 제외합니다.
        valid_pixels = pixel_values[pixel_values > 0]

        if valid_pixels.size == 0:
            print(f"  {file_path}: 유효한 픽셀 값이 없습니다 (모든 픽셀이 0).", file=output_stream)
            return

        min_pixel = np.min(valid_pixels)
        max_pixel = np.max(valid_pixels)
        mean_pixel = np.mean(valid_pixels)

        # pcd_depth_map_verfication.md에 언급된 스케일링 팩터 256.0을 적용하여 미터 단위로 변환합니다.
        min_depth_m = min_pixel / 256.0
        max_depth_m = max_pixel / 256.0
        mean_depth_m = mean_pixel / 256.0

        print(f"  PNG 깊이 맵 분석 ({file_path}):", file=output_stream)
        print(f"    픽셀 값 (0 제외): 최소={min_pixel:.2f}, 최대={max_pixel:.2f}, 평균={mean_pixel:.2f}", file=output_stream)
        print(f"    깊이 (미터, 픽셀/256.0): 최소={min_depth_m:.2f}m, 최대={max_depth_m:.2f}m, 평균={mean_depth_m:.2f}m", file=output_stream)

    except FileNotFoundError:
        print(f"  오류: {file_path} 파일을 찾을 수 없습니다.", file=output_stream)
    except Exception as e:
        print(f"  오류: {file_path} 분석 중 오류 발생: {e}", file=output_stream)

def analyze_pcd_file(file_path, output_stream):
    """PCD 파일의 깊이(Z-좌표) 분포를 분석합니다."""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)

        if points.shape[0] == 0:
            print(f"  {file_path}: 포인트 클라우드에 포인트가 없습니다.", file=output_stream)
            return

        # 사용자 좌표계에 따라 X-좌표가 깊이를 나타냅니다.
        x_coords = points[:, 0]

        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        mean_x = np.mean(x_coords)

        print(f"  PCD 파일 분석 ({file_path}):", file=output_stream)
        print(f"    X-좌표 (깊이, 미터): 최소={min_x:.2f}m, 최대={max_x:.2f}m, 평균={mean_x:.2f}m", file=output_stream)

    except FileNotFoundError:
        print(f"  오류: {file_path} 파일을 찾을 수 없습니다.", file=output_stream)
    except Exception as e:
        print(f"  오류: {file_path} 분석 중 오류 발생: {e}", file=output_stream)

def main(base_data_path, output_file_path=None):
    mapping_file = os.path.join(base_data_path, "mapping_data.json")
    
    if not os.path.exists(mapping_file):
        print(f"오류: {mapping_file} 파일을 찾을 수 없습니다. 데이터셋 경로를 확인해주세요.")
        return

    try:
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
    except json.JSONDecodeError:
        print(f"오류: {mapping_file} 파일이 유효한 JSON 형식이 아닙니다.")
        return

    pcd_files = mapping_data.get("pcd", [])
    
    if output_file_path:
        output_stream = open(output_file_path, 'w', encoding='utf-8')
    else:
        output_stream = sys.stdout # 기본적으로 콘솔로 출력

    print(f"데이터셋 경로: {base_data_path}\n", file=output_stream)

    for pcd_relative_path in pcd_files:
        # PCD 파일 경로에서 기본 파일 이름(예: "0000001996")을 추출합니다.
        base_filename = os.path.splitext(os.path.basename(pcd_relative_path))[0]
        
        # 해당 깊이 맵 PNG 파일과 PCD 파일의 전체 경로를 구성합니다.
        depth_map_png_path = os.path.join(base_data_path, "depth_maps", f"{base_filename}.png")
        pcd_full_path = os.path.join(base_data_path, pcd_relative_path)

        print(f"--- 파일 쌍 분석: {base_filename} ---", file=output_stream)
        analyze_depth_map_png(depth_map_png_path, output_stream)
        analyze_pcd_file(pcd_full_path, output_stream)
        print("-" * 30, file=output_stream)
    
    if output_file_path:
        output_stream.close()

if __name__ == "__main__":
    import sys # sys 모듈 추가
    # 여기에 데이터셋의 'synced_data' 폴더의 절대 경로를 입력하세요.
    # 예: "C:\Users\seok436\Documents\VSCode\Projects\Camera-LiDAR-Projection\ncdb-cls-sample\synced_data"
    base_data_directory = r"C:\Users\seok436\Documents\VSCode\Projects\Camera-LiDAR-Projection\ncdb-cls-sample\synced_data"
    
    # 분석 결과를 저장할 파일 경로
    output_results_file = r"C:\Users\seok436\Documents\VSCode\Projects\Camera-LiDAR-Projection\analysis_results.txt"
    
    main(base_data_directory, output_results_file)

