### 2025년 7월 29일 수정 내용 요약

오늘 LiDAR-카메라 투영 비교 도구와 관련된 여러 가지 개선 사항이 적용되었습니다.

**1. `calibration_data.py` 파일 생성:**
*   기존 `lidar_cam_calib.md`에 있던 보정 데이터(`DEFAULT_CALIB`, `DEFAULT_LIDAR_TO_WORLD_v1`, `DEFAULT_LIDAR_TO_WORLD_v2`)를 `calibration_data.py`라는 별도의 Python 파일로 분리했습니다. 이는 코드의 모듈성을 높이고 보정 데이터 관리를 용이하게 합니다.

**2. `compare_projections.py` 파일 수정:**
*   **보정 데이터 임포트:** 새로 생성된 `calibration_data.py`에서 보정 데이터를 임포트하도록 변경되었습니다.
*   **`CalibrationDB` 개선:** `CalibrationDB` 클래스에 `extrinsic_key` 인수가 추가되어 `extrinsic_v1` 또는 `extrinsic_v2` 중 어떤 외장(extrinsic) 매개변수를 사용할지 동적으로 선택할 수 있게 되었습니다.
*   **폴더 선택 UI 추가:** `--parent` 명령줄 인수가 제공되지 않을 경우, PyQt5의 `QFileDialog`를 사용하여 사용자가 데이터셋 상위 폴더를 그래픽 사용자 인터페이스(GUI)를 통해 선택할 수 있도록 했습니다.
*   **LiDAR 시각화 개선:**
    *   `LidarCameraProjector` 클래스의 `_get_color_from_distance` 메서드를 `sync_tool.py`에서 사용되는 JET-like 컬러맵으로 업데이트하여 LiDAR 포인트의 거리에 따른 색상 표현을 개선했습니다.
    *   `project_cloud_to_image` 메서드에서 LiDAR 포인트를 그릴 때 `point_radius`를 사용하여 점을 더 크게(기본값 2픽셀) 그려 시각적으로 더 잘 보이도록 했습니다.
    *   LiDAR 포인트 필터링 조건 중 `Xc >= 4.3` 또는 `Zc >= 3` 부분을 제거하여 더 많은 포인트가 투영되도록 변경했습니다.
*   **결과 저장 기능 추가:**
    *   투영된 비교 이미지를 화면에 직접 표시하는 대신, 지정된 상위 폴더 내의 `comparison_results`라는 새 폴더에 JPEG 파일로 저장하도록 변경되었습니다.
    *   저장되는 이미지는 원본 크기에서 너비 1280픽셀로 리사이즈되어 파일 크기를 줄이고 관리가 용이하도록 했습니다.

**3. `sync_tool.py` 파일 수정:**
*   **매핑 테이블 더블 클릭 기능 추가:** `MappingWindow`에서 매핑 테이블 항목을 더블 클릭했을 때, 해당 장면으로 메인 뷰어를 이동시키고 LiDAR 투영을 자동으로 활성화하는 기능을 추가했습니다.
    *   `MappingWindow` 클래스에 `item_double_clicked` PyQt 시그널을 정의했습니다.
    *   `MappingWindow`의 `doubleClicked` 시그널을 `_on_item_double_clicked` 메서드에 연결하여 `item_double_clicked` 시그널을 방출하도록 했습니다.
    *   `ImageSyncTool`에서 `MappingWindow`의 `item_double_clicked` 시그널을 받아 `_on_mapping_item_double_clicked` 메서드를 통해 뷰 모드 전환 및 장면 이동을 처리하도록 구현했습니다.

이러한 변경 사항은 LiDAR-카메라 투영 비교의 유연성, 시각적 명확성 및 사용 편의성을 향상시킵니다.

---

### Commit Message (English):

```
feat: Enhance LiDAR-Camera Projection Comparison Tool and Visualization

- Separated calibration data into `calibration_data.py` for improved modularity.
- Added folder selection GUI to `compare_projections.py` and set default `num_samples`.
- Applied JET-like colormap and point radius for better LiDAR point visualization.
- Changed projection results output to save images in `comparison_results` folder instead of displaying them.
- Resized saved images for better efficiency and management.
```
