# Daily Modifications - 20250730

## 1. 빨간색 선 기반 마스크 생성 및 개선

- **목표:** 이미지 내 빨간색 선으로 표시된 영역을 기준으로 마스크를 생성하고, 선 안쪽은 유지하며 바깥쪽은 0으로 처리합니다.
- **초기 작업:** `create_mask.py` 스크립트를 작성하여 빨간색 픽셀을 식별하고 윤곽선을 채워 마스크를 생성했습니다.
- **경로 오류 수정:** Windows 경로의 백슬래시 문제로 인한 `SyntaxError`를 해결하기 위해 `create_mask.py` 파일 내 경로를 raw string으로 수정했습니다.
- **마스크 로직 개선:** "삐져나오는 부분"을 없애기 위해 `create_mask.py`의 마스크 생성 로직을 연결된 구성 요소 분석 방식으로 개선했습니다. 이를 통해 빨간색 선 바깥 영역이 확실하게 0으로 처리되도록 했습니다.
- **생성된 파일:**
    - `C:\Users\seok436\Documents\VSCode\Projects\Camera-LiDAR-Projection\mask\visual_mask.png` (시각적 확인용 마스크)
    - `C:\Users\seok436\Documents\VSCode\Projects\Camera-LiDAR-Projection\mask\binary_mask.png` (이진 마스크)
    - `C:\Users\seok436\Documents\VSCode\Projects\Camera-LiDAR-Projection\mask\masked_apache6_img.jpg` (마스크 적용된 원본 이미지)

## 2. 사용자 수정 마스크 기반 재처리

- **목표:** 사용자가 수정한 `visual_mask.png` (검은색/흰색)를 기반으로 새로운 이진 마스크를 생성하고, 이를 원본 이미지에 적용합니다.
- **작업:** `recreate_mask_from_visual.py` 스크립트를 작성하여 `visual_mask.png`의 검은색 영역을 1, 흰색 영역을 0으로 변환하여 새로운 이진 마스크를 만들고, 이를 원본 이미지에 적용했습니다.
- **생성된 파일:**
    - `C:\Users\seok436\Documents\VSCode\Projects\Camera-LiDAR-Projection\mask\new_binary_mask_from_visual.png` (수정된 visual_mask 기반 이진 마스크)
    - `C:\Users\seok436\Documents\VSCode\Projects\Camera-LiDAR-Projection\mask\remasked_apache6_img.jpg` (새로운 마스크가 적용된 원본 이미지)

## 3. 최종 시각적 마스크 생성

- **목표:** `new_binary_mask_from_visual.png`에 255를 곱하여 최종 시각적 마스크를 생성합니다.
- **작업:** `create_new_visual_mask.py` 스크립트를 작성하여 `new_binary_mask_from_visual.png`의 픽셀 값에 255를 곱한 이미지를 생성했습니다.
- **생성된 파일:**
    - `C:\Users\seok436\Documents\VSCode\Projects\Camera-LiDAR-Projection\mask\new_visual_mask.png` (최종 시각적 마스크)
