import os
from pathlib import Path

def verify_image_pcd_pairs(base_folder: str):
    """
    'image_a5' 폴더와 'pcd' 폴더의 파일 목록을 비교하여,
    이름(확장자 제외)이 일치하지 않는 파일을 찾아 보고합니다.
    """
    base_path = Path(base_folder)
    a5_dir = base_path / 'image_a5'
    pcd_dir = base_path / 'pcd'

    # 1. 두 폴더가 존재하는지 확인합니다.
    if not a5_dir.is_dir():
        print(f"❌ 오류: 'image_a5' 폴더를 찾을 수 없습니다: {a5_dir}")
        return
    if not pcd_dir.is_dir():
        print(f"❌ 오류: 'pcd' 폴더를 찾을 수 없습니다: {pcd_dir}")
        return

    # 2. 각 폴더에서 확장자를 제외한 파일 이름(basename) 목록을 추출합니다.
    #    - 숨김 파일이나 시스템 파일은 제외합니다.
    #    - sync_tool.py와 동일하게 png, jpg, jpeg 확장자를 이미지로 간주합니다.
    a5_basenames = {f.stem for f in a5_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', 'jpeg']}
    pcd_basenames = {f.stem for f in pcd_dir.iterdir() if f.is_file() and f.suffix.lower() == '.pcd'}

    # 3. 집합(set) 연산을 사용하여 차이를 계산합니다.
    matching_files = a5_basenames.intersection(pcd_basenames)
    a5_only = a5_basenames.difference(pcd_basenames)
    pcd_only = pcd_basenames.difference(a5_basenames)

    # 4. 결과를 출력합니다.
    print("--- 파일 쌍 검증 보고서 ---")
    print(f"기준 폴더: {base_folder}\n")

    print(f"'{a5_dir.name}' 폴더의 총 이미지 수: {len(a5_basenames)}")
    print(f"'{pcd_dir.name}' 폴더의 총 PCD 수:    {len(pcd_basenames)}")
    print(f"이름이 일치하는 파일 쌍의 수: {len(matching_files)}\n")

    if not a5_only and not pcd_only:
        print("✅ 완벽합니다! 모든 파일이 1:1로 정확하게 일치합니다.")
    else:
        print("--- 불일치 항목 ---")
        if a5_only:
            print(f"⚠️ 'image_a5' 폴더에만 존재하는 파일 ({len(a5_only)}개):")
            for name in sorted(list(a5_only)):
                print(f"  - {name}")
            print()

        if pcd_only:
            print(f"⚠️ 'pcd' 폴더에만 존재하는 파일 ({len(pcd_only)}개):")
            for name in sorted(list(pcd_only)):
                print(f"  - {name}")
            print()

if __name__ == "__main__":
    # 사용자로부터 상위 폴더 경로를 입력받습니다.
    folder_path = input("검사할 상위 폴더 경로를 입력하세요 (image_a5, pcd 폴더가 있는 곳): ").strip().strip('"')
    
    if not os.path.isdir(folder_path):
        print(f"❌ 오류: 입력한 경로가 올바른 폴더가 아닙니다: {folder_path}")
    else:
        verify_image_pcd_pairs(folder_path)
