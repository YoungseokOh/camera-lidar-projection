### `compare_projections.py` 코드 설명

이 Python 스크립트는 LiDAR 포인트 클라우드를 카메라 이미지에 투영하는 두 가지 다른 보정(calibration) 버전(v1 및 v2)을 비교하는 데 사용됩니다. 이 스크립트는 지정된 데이터셋 폴더에서 샘플 데이터를 로드하고, 각 보정 버전을 사용하여 투영을 수행한 다음, 결과를 시각적으로 나란히 표시하여 비교할 수 있도록 합니다.

**주요 구성 요소 및 기능:**

1.  **모듈 임포트 (Imports):**
    *   `sys`, `os`, `json`, `math`, `argparse`, `traceback`, `pathlib.Path`, `typing` (Dict, List, Tuple, Optional, Any): 표준 Python 라이브러리로 파일 시스템 상호 작용, JSON 처리, 수학 연산, 명령줄 인수 파싱, 오류 추적, 경로 처리, 타입 힌트에 사용됩니다.
    *   `numpy` (np): 수치 계산, 특히 행렬 연산에 사용됩니다.
    *   `PIL` (Pillow)의 `Image`, `ImageDraw`: 이미지 로드, 조작 및 포인트 그리기(투영된 LiDAR 포인트)에 사용됩니다.
    *   `cv2` (OpenCV): 이미지 표시 및 색상 공간 변환에 사용됩니다.
    *   `calibration_data`에서 `DEFAULT_CALIB`, `DEFAULT_LIDAR_TO_WORLD_v1`, `DEFAULT_LIDAR_TO_WORLD_v2`: 별도의 파일에 정의된 보정 데이터를 가져옵니다.
    *   `open3d` (o3d): PCD(Point Cloud Data) 파일을 로드하는 데 사용되지만, 설치되어 있지 않은 경우를 대비하여 대체(fallback) 로직이 포함되어 있습니다.

2.  **카메라 모델 클래스 (`CameraModelBase`, `VADASFisheyeCameraModel`):**
    *   `CameraModelBase`: 카메라 투영 모델의 기본 추상 클래스입니다. `project_point` 메서드를 정의합니다.
    *   `VADASFisheyeCameraModel`: VADAS 다항식 어안(Fisheye) 카메라 모델을 구현합니다. 카메라의 내장(intrinsic) 매개변수를 사용하여 3D 카메라 좌표(Xc, Yc, Zc)의 점을 2D 이미지 좌표(u, v)로 투영하는 로직을 포함합니다. C++ 참조 구현과 일치하도록 다항식 평가(`_poly_eval`) 및 특정 필터링 규칙을 따릅니다.

3.  **센서 정보 클래스 (`SensorInfo`):**
    *   카메라의 이름, 모델, 내장(intrinsic) 및 외장(extrinsic) 매개변수, 이미지 크기와 같은 정보를 캡슐화합니다.

4.  **보정 데이터베이스 클래스 (`CalibrationDB`):**
    *   다양한 카메라의 보정 데이터를 관리합니다.
    *   초기화 시 `calib_dict` (보정 데이터 딕셔너리), `lidar_to_world` (LiDAR-to-World 변환 행렬), `extrinsic_key` (사용할 외장 매개변수 버전, 예: "extrinsic_v1" 또는 "extrinsic_v2")를 받습니다.
    *   `_rodrigues_to_matrix`: 로드리게스 벡터(회전 및 변환)를 4x4 변환 행렬로 변환합니다.
    *   `get`: 특정 센서의 `SensorInfo`를 검색할 수 있습니다.

5.  **데이터 로드 함수 (`load_pcd_xyz`, `load_image`):**
    *   `load_pcd_xyz(path)`: PCD 파일에서 XYZ 포인트 클라우드 데이터를 NumPy 배열로 로드합니다. `open3d`가 설치되어 있으면 이를 사용하고, 그렇지 않으면 기본 ASCII PCD 파서를 사용합니다.
    *   `load_image(path)`: 지정된 경로에서 이미지 파일을 로드하여 PIL Image 객체를 반환합니다.

6.  **LiDAR-카메라 프로젝터 클래스 (`LidarCameraProjector`):**
    *   `CalibrationDB` 인스턴스를 사용하여 LiDAR 포인트 클라우드를 카메라 이미지에 투영하는 핵심 로직을 포함합니다.
    *   `_get_color_from_distance`: LiDAR 포인트의 카메라로부터의 거리에 따라 색상을 결정합니다. 이는 투영된 포인트의 깊이를 시각적으로 나타내는 데 사용됩니다.
    *   `project_cloud_to_image`: 주어진 카메라 이름, 포인트 클라우드, PIL 이미지에 대해 투영을 수행합니다.
        *   LiDAR 포인트 클라우드를 동차 좌표로 변환합니다.
        *   카메라 외장(extrinsic) 및 LiDAR-to-World 변환을 결합하여 LiDAR-to-Camera 변환 행렬을 계산합니다.
        *   LiDAR 포인트를 카메라 좌표계로 변환합니다.
        *   각 변환된 포인트에 대해 카메라 모델의 `project_point` 메서드를 호출하여 이미지 좌표를 얻습니다.
        *   유효한 투영된 포인트(카메라 앞에 있고 이미지 경계 내에 있는 포인트)를 이미지에 그립니다.

7.  **메인 함수 (`main`):**
    *   **명령줄 인수 파싱:**
        *   `--parent` (필수): `image_a6`, `pcd`, `synced_data` 폴더를 포함하는 상위 폴더의 경로를 지정합니다.
        *   `--cam` (선택 사항, 기본값: "a6"): 투영할 카메라 이름을 지정합니다.
        *   `--num_samples` (선택 사항, 기본값: 10): 비교할 샘플 이미지/PCD 쌍의 수를 지정합니다.
    *   **데이터 경로 설정:** 제공된 `--parent` 경로를 기반으로 `synced_data`, `mapping_data.json`, `pcd` 디렉토리의 경로를 구성합니다.
    *   **`mapping_data.json` 로드:** 이 파일은 이미지와 PCD 파일 간의 매핑 정보를 포함합니다. 파일이 없거나 비어 있으면 오류를 출력하고 종료합니다.
    *   **샘플 선택:** `mapping_data`에서 `num_samples`만큼 무작위로 샘플을 선택합니다.
    *   **투영 비교 루프:**
        *   각 샘플에 대해 이미지와 PCD 파일 경로를 결정하고 로드합니다.
        *   **v1 보정으로 투영:** `DEFAULT_CALIB`와 `DEFAULT_LIDAR_TO_WORLD_v1`, `extrinsic_key="extrinsic_v1"`을 사용하여 `CalibrationDB` 및 `LidarCameraProjector` 인스턴스를 생성하고 투영을 수행합니다.
        *   **v2 보정으로 투영:** `DEFAULT_CALIB`와 `DEFAULT_LIDAR_TO_WORLD_v2`, `extrinsic_key="extrinsic_v2"`를 사용하여 `CalibrationDB` 및 `LidarCameraProjector` 인스턴스를 생성하고 투영을 수행합니다.
        *   **이미지 결합:** v1 및 v2 투영 결과를 OpenCV 이미지로 변환한 다음, `np.hstack`을 사용하여 두 이미지를 가로로 결합하여 나란히 비교할 수 있도록 합니다.
        *   결합된 이미지를 `results_images` 리스트에 추가합니다.
    *   **결과 표시:** 모든 샘플에 대한 결합된 이미지를 OpenCV 창에 표시합니다. 각 창의 제목은 어떤 샘플인지, 왼쪽이 v1, 오른쪽이 v2 투영임을 나타냅니다. 사용자가 아무 키나 누르면 모든 창이 닫힙니다.
    *   **오류 처리:** 파일이 없거나 예기치 않은 오류가 발생하면 적절한 메시지를 출력하고 종료합니다.

**사용 방법:**

이 스크립트는 명령줄에서 실행됩니다. `--parent` 인수를 사용하여 데이터셋의 루트 폴더를 지정해야 합니다.

예시:
```bash
python compare_projections.py --parent "C:/Your/Dataset/Folder" --num_samples 10
```
