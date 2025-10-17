import os
import json
import numpy as np
import cv2
# from PIL import Image # Removed PIL import
import sys
from calibration_data import DEFAULT_CALIB

# 전역 변수 초기화
depth_map_display = None # 원본 깊이 맵 (미터 단위)
original_color_mapped_depth = None # 흰색 배경 처리된 원본 크기 컬러맵 이미지
display_image = None # 현재 화면에 표시될 (줌/패닝 적용된) 이미지

zoom_level = 1.0
pan_x = 0.0
pan_y = 0.0
drag_start_x = 0
drag_start_y = 0
is_dragging = False

# 고정된 창 크기 (동적으로 가져올 수 없으므로 기본값 설정)
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

def load_depth_map_png(file_path):
    """
    16비트 그레이스케일 PNG 깊이 맵을 로드하고 미터 단위로 변환합니다.
    KITTI 데이터셋 표준에 따라 16비트 픽셀 값을 256.0으로 나누어 실제 거리(미터)로 변환합니다.
    """
    try:
        # Use cv2.imread with IMREAD_UNCHANGED to load 16-bit image
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Could not load image from {file_path}", file=sys.stderr)
            return None
        
        # Ensure it's 16-bit (CV_16U)
        if img.dtype != np.uint16:
            print(f"Warning: {file_path}는 16비트 이미지가 아닐 수 있습니다. 현재 dtype: {img.dtype}", file=sys.stderr)
            # Attempt to convert if it's a different type, but warn
            img = img.astype(np.uint16)
        
        depth_map_pixels = np.array(img, dtype=np.float32)
        depth_map_meters = depth_map_pixels / 256.0
        return depth_map_meters
    except Exception as e:
        print(f"PNG 깊이 맵 로드 중 오류 발생 ({file_path}): {e}", file=sys.stderr)
        return None

def update_display_image():
    """
    현재 줌 레벨과 패닝 오프셋에 따라 화면에 표시될 이미지를 업데이트합니다.
    """
    global original_color_mapped_depth, display_image, zoom_level, pan_x, pan_y, WINDOW_WIDTH, WINDOW_HEIGHT

    if original_color_mapped_depth is None:
        return

    h_orig, w_orig, _ = original_color_mapped_depth.shape
    
    # 새로운 확대된 이미지 크기 계산
    new_w, new_h = int(w_orig * zoom_level), int(h_orig * zoom_level)
    
    # 이미지 확대
    zoomed_image = cv2.resize(original_color_mapped_depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # pan_x, pan_y 클램핑 (확대된 이미지가 창 경계를 벗어나지 않도록)
    # pan_x, pan_y는 확대된 이미지의 좌상단이 창의 좌상단으로부터 얼마나 떨어져 있는지를 나타냄
    if new_w < WINDOW_WIDTH:
        pan_x = (WINDOW_WIDTH - new_w) / 2 # 이미지가 창보다 작으면 중앙 정렬
    else:
        pan_x = max(min(pan_x, 0.0), -(new_w - WINDOW_WIDTH)) # 이미지가 창보다 크면 패닝 범위 제한

    if new_h < WINDOW_HEIGHT:
        pan_y = (WINDOW_HEIGHT - new_h) / 2 # 이미지가 창보다 작으면 중앙 정렬
    else:
        pan_y = max(min(pan_y, 0.0), -(new_h - WINDOW_HEIGHT)) # 이미지가 창보다 크면 패닝 범위 제한

    # 캔버스 생성 (창 크기와 동일, 흰색 배경)
    canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) + 255

    # 확대된 이미지에서 캔버스로 복사할 영역 계산
    # src_x1, src_y1: zoomed_image에서 복사 시작할 좌상단 좌표
    # dst_x1, dst_y1: canvas에 붙여넣기 시작할 좌상단 좌표
    
    src_x1 = max(0, -int(pan_x))
    src_y1 = max(0, -int(pan_y))
    src_x2 = min(new_w, src_x1 + WINDOW_WIDTH)
    src_y2 = min(new_h, src_y1 + WINDOW_HEIGHT)

    dst_x1 = max(0, int(pan_x))
    dst_y1 = max(0, int(pan_y))
    dst_x2 = min(WINDOW_WIDTH, dst_x1 + (src_x2 - src_x1))
    dst_y2 = min(WINDOW_HEIGHT, dst_y1 + (src_y2 - src_y1))

    # 이미지 복사
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = zoomed_image[src_y1:src_y2, src_x1:src_x2]

    display_image = canvas
    cv2.imshow("Depth Map (Hover for Value, Scroll to Zoom, Drag to Pan)", display_image)

def mouse_callback(event, x, y, flags, param):
    """마우스 이벤트 콜백 함수"""
    global depth_map_display, display_image, zoom_level, pan_x, pan_y, drag_start_x, drag_start_y, is_dragging

    # 텍스트를 그릴 현재 화면 이미지의 복사본
    current_display_image_for_text = display_image.copy() if display_image is not None else np.zeros((1,1,3), dtype=np.uint8)

    # 마우스 좌표 (창 기준)를 원본 깊이 맵 좌표로 변환
    original_x = int((x - pan_x) / zoom_level)
    original_y = int((y - pan_y) / zoom_level)

    # 깊이 값 표시
    if depth_map_display is not None and 0 <= original_y < depth_map_display.shape[0] and 0 <= original_x < depth_map_display.shape[1]:
        png_depth = depth_map_display[original_y, original_x]
        depth_text = f"Depth: {png_depth:.2f}m"
        
        text_pos = (x + 10, y - 10)
        font_scale = 0.8
        font_thickness = 2
        cv2.putText(current_display_image_for_text, depth_text, text_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

    # 마우스 휠 이벤트 (확대/축소)
    if event == cv2.EVENT_MOUSEWHEEL:
        zoom_factor = 1.1 if flags > 0 else 1 / 1.1
        new_zoom_level = max(0.1, min(10.0, zoom_level * zoom_factor))

        if new_zoom_level != zoom_level:
            # 줌의 중심을 마우스 커서 위치로 유지하기 위해 패닝 오프셋 조정
            orig_x_at_mouse = (x - pan_x) / zoom_level
            orig_y_at_mouse = (y - pan_y) / zoom_level

            zoom_level = new_zoom_level
            
            pan_x = x - (orig_x_at_mouse * zoom_level)
            pan_y = y - (orig_y_at_mouse * zoom_level)
            
            update_display_image()

    # 마우스 드래그 이벤트 (패닝)
    elif event == cv2.EVENT_LBUTTONDOWN:
        is_dragging = True
        drag_start_x = x
        drag_start_y = y
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        dx = x - drag_start_x
        dy = y - drag_start_y
        pan_x += dx
        pan_y += dy
        drag_start_x = x
        drag_start_y = y
        update_display_image()
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False
    
    # 마우스 움직임에 따른 깊이 값 표시 업데이트 (드래그 중이 아닐 때만)
    if not is_dragging and event == cv2.EVENT_MOUSEMOVE:
        cv2.imshow("Depth Map (Hover for Value, Scroll to Zoom, Drag to Pan)", current_display_image_for_text)
    # 다른 이벤트 (LBUTTONUP, MOUSEWHEEL)는 update_display_image()에서 이미 imshow를 호출합니다.
    # 따라서 여기에 추가적인 imshow는 필요 없습니다.

def main(base_data_path, pcd_filename):
    global depth_map_display, original_color_mapped_depth, display_image, zoom_level, pan_x, pan_y

    # 파일 경로 구성
    depth_map_png_path = os.path.join(base_data_path, "depth_maps", pcd_filename.replace(".pcd", ".png"))

    # 데이터 로드
    depth_map_meters = load_depth_map_png(depth_map_png_path)

    if depth_map_meters is None:
        print("깊이 맵 파일 로드에 실패했습니다. 프로그램을 종료합니다.")
        return

    image_height, image_width = depth_map_meters.shape[:2]

    # --- 시각화 ---
    # 1. 깊이 맵을 컬러맵으로 변환
    normalized_depth_display = cv2.normalize(depth_map_meters, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    color_mapped_depth = cv2.applyColorMap(normalized_depth_display, cv2.COLORMAP_JET)

    # 2. 깊이 값이 0인 부분을 흰색 배경으로 처리
    zero_depth_mask = (depth_map_meters == 0)
    color_mapped_depth[zero_depth_mask] = [255, 255, 255] # BGR: 흰색

    original_color_mapped_depth = color_mapped_depth.copy()

    # 초기 줌/패닝 상태 설정
    zoom_level = 1.0 # 초기 줌 레벨
    pan_x = 0.0
    pan_y = 0.0

    # OpenCV 창 설정
    cv2.namedWindow("Depth Map (Hover for Value, Scroll to Zoom, Drag to Pan)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth Map (Hover for Value, Scroll to Zoom, Drag to Pan)", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.setMouseCallback("Depth Map (Hover for Value, Scroll to Zoom, Drag to Pan)", mouse_callback)

    # 초기 이미지 표시 업데이트
    depth_map_display = depth_map_meters # 마우스 콜백에서 사용될 원본 깊이 맵 데이터
    update_display_image()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    base_data_directory = r"C:\Users\seok436\Documents\VSCode\Projects\Camera-LiDAR-Projection\ncdb-cls-sample\synced_data"
    pcd_to_visualize = "0000000931.pcd" 
    main(base_data_directory, pcd_to_visualize)