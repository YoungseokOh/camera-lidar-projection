import numpy as np
import math

# create_depth_maps.py에서 가져온 _rodrigues_to_matrix 함수
def _rodrigues_to_matrix(rvec_tvec: list) -> np.ndarray:
    """
    Rodrigues 벡터와 이동 벡터를 사용하여 4x4 변환 행렬을 생성합니다.
    rvec_tvec는 [tx, ty, tz, rx, ry, rz] 형식으로 가정합니다。
    """
    tvec = np.array(rvec_tvec[0:3]).reshape(3, 1)
    rvec = np.array(rvec_tvec[3:6])
    theta = np.linalg.norm(rvec)
    if theta < 1e-6:
        R = np.eye(3)
    else:
        r = rvec / theta
        K_skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        R = np.eye(3) + math.sin(theta) * K_skew + (1 - math.cos(theta)) * (K_skew @ K_skew)
    
    transform_matrix = np.eye(4)
    transform_matrix[0:3, 0:3] = R
    transform_matrix[0:3, 3:4] = tvec
    return transform_matrix

# calibration_data.py에서 가져온 extrinsic 값 (예시로 extrinsic_v2 사용)
# 사용자님의 필요에 따라 extrinsic_v1 또는 extrinsic_v2를 선택하세요.
extrinsic_data = [ 0.293769, -0.0542026, -0.631615, -0.00394431, -0.33116, -0.00963617 ]

# LiDAR-to-Camera 변환 행렬 계산
T_lidar_to_cam = _rodrigues_to_matrix(extrinsic_data)

print("계산된 LiDAR-to-Camera 변환 행렬 (T_lidar_to_cam):\n")
print(T_lidar_to_cam)

# 참고: 이 행렬은 LiDAR 좌표계의 점을 카메라 좌표계로 변환합니다.
# 사용자님의 좌표계 정의 (X=깊이, Y=왼쪽, Z=위쪽)에 따라
# 이 변환 행렬이 LiDAR 데이터를 카메라의 표준 좌표계(X=오른쪽, Y=아래쪽, Z=깊이)로
# 올바르게 매핑하도록 캘리브레이션이 수행되었어야 합니다.
# 만약 캘리브레이션이 다른 좌표계 정의를 따랐다면, 추가적인 좌표 변환이 필요할 수 있습니다.
